"""model/hf_model.py -- THE ONLY FILE IN THIS REPOSITORY THAT IMPORTS HUGGINGFACE.

    from model.hf_model import load
    model, tok = load("checkpoints/pretrain/checkpoint_mira_llm-s_pretrain.pt")
    print(tok.decode(model.generate(**tok("hello", return_tensors="pt"), max_new_tokens=32)[0]))

THE DEPENDENCY POINTS ONE WAY AND ONLY ONE WAY. `model/mira_llm.py`, `tokenizer/bpe.py`,
`helpers/` and every stage import nothing from here and nothing from transformers; this file
imports them. Delete it and the project is exactly what it was. That is the whole design: the
architecture is ours and self-contained, and the bridge is a leaf.

WHAT IT BUYS. Anything that speaks the `PreTrainedModel` interface: `lm-evaluation-harness` on
every loglikelihood task, `Trainer` and TRL, PEFT adapters, bitsandbytes on the projections,
`save_pretrained`/`push_to_hub`, and -- through MiraCache below -- `generate()` with sampling,
beam search, streamers and stopping criteria.

WHAT IT DOES NOT BUY, and no wrapper can: vLLM, TGI, SGLang and llama.cpp reimplement an
architecture inside their own runtime and do not call this forward; and ONNX/torch.export cannot
trace the lattice, whose kernel is a cuLaunchKernel through ctypes.

CONVERSION HAPPENS ONCE, ON THE FIRST RUN. `load()` is given the project's own `.pt` and looks
beside it for `<name>-hf/`. If that directory exists it is loaded directly and nothing is
converted; otherwise the checkpoint's weights are written there as safetensors, its architecture
as config.json and its tokenizer as bpe.json, and the directory is loaded. The `.pt` is opened
read-only and is never modified, so every internal stage keeps reading exactly what it read
before.

THE TOKENIZER IS WRAPPED, NOT REWRITTEN. `MiraTokenizer` holds the project's own `BPETokenizer`
and asks it for every id, so ids are identical BY CONSTRUCTION rather than by a merge-table
conversion that has to be verified token by token. Getting that wrong would not raise -- it
would silently point every embedding row at a different token -- and the project's corpus
signature reads `256 + len(tok.merges)`, so a substituted tokenizer would also re-sign every
cached token stream and trigger a re-tokenisation of the whole corpus.
"""
from __future__ import annotations

import inspect
import json
import os

import torch
import torch.nn as nn

from transformers import PreTrainedModel, PretrainedConfig, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

import default_config as _project_config
from model import MiraLLM
from model import prefix_cache as _prefix_cache
from tokenizer import BPETokenizer

# GenerationMixin IS INHERITED EXPLICITLY, BECAUSE PreTrainedModel STOPPED PROVIDING IT.
# transformers 4.50 removed GenerationMixin from PreTrainedModel's bases, and can_generate()
# now tests for it directly -- so a model that only subclasses PreTrainedModel has no
# generate() at all on a current release. requirements.txt pins no version, so both are
# possible and the import is guarded rather than assumed.
try:                                                          # pragma: no cover - version shim
    from transformers import GenerationMixin as _GenerationMixin
except Exception:                                             # noqa: BLE001
    class _GenerationMixin:                                   # type: ignore[no-redef]
        pass

# THE CACHE BASE CLASS IS OPTIONAL, BECAUSE transformers MOVED IT. `Cache` lives in
# transformers.cache_utils in recent releases and did not exist at all in older ones. Subclassing
# it when it is there keeps every isinstance check in GenerationMixin happy; falling back to
# `object` keeps this file importable against a release that predates it, and MiraCache
# implements the methods either way.
try:                                                          # pragma: no cover - version shim
    from transformers.cache_utils import Cache as _HFCache
except Exception:                                             # noqa: BLE001
    _HFCache = object

WEIGHT_PREFIX = "mira."
HF_SUFFIX = "-hf"

MAX_CONTEXT = 262144


def _model_kwargs(model_cfg, log=print):
    """A recorded model_cfg, reduced to the arguments MiraLLM actually takes.

    THE SAME THREE CORRECTIONS EVERY OTHER LOADER MAKES, and they are not optional -- a benchmark
    checkpoint fails on each of them in turn:

    1. `experimental_arch` IS PROMOTED AND THE NAME IS CANONICALISED. baselines/__init__.py
       records `experimental_arch="mira-matrix"`; MiraLLM's own alias table has no entry for
       that spelling and refuses it, while config.canonical_arch maps it to "mira-lattice",
       which is what mirachat.py:293-303 does.
    2. `coefficient_bottleneck_factor` IS RECOMPUTED AGAINST n_embd. baselines/__init__.py:516
       writes d_e / coefficient_bottleneck = 32; MiraLLM requires n_embd / factor to equal the
       recorded bottleneck, which 768/32 = 24 does not. Only the divisor moves; the bottleneck
       is what the weights have and is passed through.
    3. EVERY FIELD THE CONSTRUCTOR DOES NOT TAKE IS DROPPED, by signature rather than by a list
       of names -- `lattice_state`, `benchmark_family`, `checkpoint_auxiliary`, the retired `pe`
       and whatever a later stage records next. MiraLLM.__init__ has no **kwargs, so one
       unrecognised key is a TypeError."""
    cfg = dict(model_cfg or {})
    cfg.pop("vocab_size", None)
    experimental = cfg.pop("experimental_arch", None)
    if experimental:
        cfg["arch"] = experimental
    if cfg.get("arch"):
        cfg["arch"] = _project_config.canonical_arch(cfg["arch"])
    # A cfg THAT NAMES NO ARCHITECTURE IS MIRA'S: there is one, so a MIRA cfg no longer records
    # the name. Everything else -- the rope ablation, every benchmark baseline -- still does.
    lattice = "lattice_order" in cfg
    if lattice:
        _project_config.mira_map_arch(cfg)
        if "arch" in cfg:
            raise ValueError("this MIRA checkpoint records arch instead of lattice_modulus_map and "
                             "lattice_argument_map, the format before those keys, and cannot be loaded")
    if lattice:
        cfg["lattice_initializer"] = _project_config.canonical_lattice_initializer(
            cfg.get("lattice_initializer", cfg.pop("lattice_init_method", "schur_init")), recorded=True)
    if "init_horizon" in cfg and "lattice_init_horizon" not in cfg:
        cfg["lattice_init_horizon"] = cfg.pop("init_horizon")
    if lattice and "lattice_init_horizon" not in cfg:
        cfg["lattice_init_horizon"] = _project_config.MODEL["lattice_init_horizon"]
    bottleneck = cfg.get("coefficient_bottleneck")
    factor = cfg.get("coefficient_bottleneck_factor")
    width = cfg.get("n_embd")
    if (bottleneck and factor and width and int(width) % int(bottleneck) == 0
            and int(bottleneck) != int(width) // int(factor)):
        cfg["coefficient_bottleneck_factor"] = int(width) // int(bottleneck)
        log(f"[hf] coefficient_bottleneck_factor {factor} recorded against d_e; using "
            f"{cfg['coefficient_bottleneck_factor']} = {width} / {bottleneck}, so the model "
            f"built is the one trained")
    accepted = set(inspect.signature(MiraLLM.__init__).parameters) - {"self", "vocab_size"}
    dropped = sorted(set(cfg) - accepted)
    if dropped:
        log(f"[hf] {len(dropped)} recorded field(s) MiraLLM does not take, dropped: "
            f"{', '.join(dropped)}")
    return {k: v for k, v in cfg.items() if k in accepted}


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
class MiraConfig(PretrainedConfig):
    """The architecture, carried verbatim rather than re-described.

    A MIRA CONFIGURATION IS NOT A TRANSFORMER CONFIGURATION, and `mira_cfg` is it, unchanged.
    What describes this model is the lattice geometry, in the project's own symbols:

        d      n_embd                        the residual width
        d_e    lattice_d_e                   the expansion width the lattice runs at
        d_r    coefficient_bottleneck        the coefficient bottleneck, d/16
        d_m    memory_dimension              the memory dimension of the state matrices
        K      lattice_order                 the lattice order, the number of stages
               lattice_expansion_factor      d_e = d * this
               coefficient_bottleneck_factor d_r = d / this
               n_layer                       depth
               block_size                    the length it was TRAINED at, not a limit

    THERE IS NO num_attention_heads AND THERE WILL NOT BE. This model has no queries, keys or
    values; a head count would be a number describing a mechanism that is absent. `hidden_size`
    and `num_hidden_layers` ARE meaningful -- a residual width and a depth exist here as they do
    anywhere -- so they are offered as read-only properties DERIVED from mira_cfg rather than
    stored beside it. Nothing is duplicated, so nothing can drift: config.json holds one
    description of this model and it is `mira_cfg`.

    max_position_embeddings IS A CEILING, NOT THE TRAINING WINDOW. `block_size` in mira_cfg is
    the length a checkpoint was trained at; nothing in this architecture refers to an absolute
    position, so no length limit follows from it. Writing the training window here would tell
    every HuggingFace caller that a 512-token model cannot be asked for more, which is false."""

    model_type = "mira"
    keys_to_ignore_at_inference = ["past_key_values"]

    # DERIVED, NEVER STORED. A property is not in __dict__, so to_dict() and config.json carry
    # `mira_cfg` alone and no second copy of the width or the depth exists to disagree with it.
    @property
    def hidden_size(self):
        return int(self.mira_cfg.get("n_embd", 0) or 0)

    @property
    def num_hidden_layers(self):
        return int(self.mira_cfg.get("n_layer", 0) or 0)

    def __init__(self, mira_cfg=None, vocab_size=50259, tie_word_embeddings=True,
                 max_position_embeddings=None, **kw):
        # A DERIVED FIELD THAT ARRIVES AS A KEYWORD IS DROPPED, NOT ASSIGNED. PretrainedConfig
        # setattr()s every unrecognised keyword, and a config.json written before these became
        # properties carries hidden_size and num_hidden_layers as plain values -- assigning to a
        # read-only property raises. Dropping them is correct rather than lossy: both are
        # recomputed from mira_cfg, which is the record that built the weights.
        for derived in ("hidden_size", "num_hidden_layers", "num_attention_heads"):
            kw.pop(derived, None)
        self.mira_cfg = dict(mira_cfg or {})
        self.vocab_size = int(self.mira_cfg.get("vocab_size", vocab_size) or vocab_size)
        # A value in config.json arrives as this argument and wins, which is what lets a
        # converted directory be retargeted by someone who does not have this repository.
        self.max_position_embeddings = int(
            MAX_CONTEXT if max_position_embeddings is None else max_position_embeddings)
        super().__init__(tie_word_embeddings=bool(tie_word_embeddings), **kw)


# --------------------------------------------------------------------------- #
# the decoding cache
# --------------------------------------------------------------------------- #
class MiraCache(_HFCache):
    """The lattice state, wearing the shape `generate()` expects.

    IT IS NOT KEYS AND VALUES AND IT DOES NOT GROW. What a MIRA layer carries between calls is
    V_t^(0..K-1) -- K matrices of [d_e, d_m] per layer -- which is the same size at token 10 and
    at token 10 million. So `get_seq_length` answers from a counter rather than from the width of
    a stored tensor, and `crop`, which drops the newest keys, has nothing to drop: the state is a
    summary of every token so far, folded together, with no per-token part to remove.

    IT SPEAKS BOTH INTERFACES ON PURPOSE. `get`/`put`/`advance`/`within_budget` are what
    model/mira_mixer.py calls, so this is a drop-in for model.prefix_cache.PrefixCache
    inside the trunk; `get_seq_length`, `update`, `reorder_cache` and `to_legacy_cache` are what
    GenerationMixin calls. One object, so there is nothing to translate at the boundary."""

    def __init__(self, layers, model, budget=None):
        try:                                                  # pragma: no cover - version shim
            super().__init__()
        except Exception:                                     # noqa: BLE001
            pass
        self.model = model
        self._inner = _prefix_cache.PrefixCache(int(layers), budget=budget)
        # WHICH CACHED KEYS STAND FOR A REAL TOKEN. MiraLLM._trunk:642-659 reads and writes this
        # attribute directly, not through getattr with a default, the moment an attention mask
        # reaches it -- which is every padded batch HuggingFace produces. Without it a batched
        # generate() dies on AttributeError rather than on anything to do with MIRA.
        self.key_valid = None

    # ---- the interface the MIRA trunk uses -------------------------------- #
    def __len__(self):
        """TOKENS, NOT LAYERS, AND THE TWO INTERFACES GENUINELY DISAGREE HERE.
        MiraLLM._trunk:656 reads `len(cache)` as the number of tokens already absorbed;
        transformers' own DynamicCache reads it as the layer count. The trunk's use is a hard
        error if it is wrong and HuggingFace's is not, so the trunk wins and `get_seq_length`
        below is what HF is expected to call."""
        return self._inner.tokens

    def get(self, layer):
        return self._inner.get(layer)

    def put(self, layer, state):
        return self._inner.put(layer, state)

    def advance(self, tokens):
        return self._inner.advance(tokens)

    def clear_states(self):
        self._inner.clear_states()
        self.key_valid = None

    def reset(self):
        self._inner.reset(self.model)
        self.key_valid = None

    def within_budget(self):
        return self._inner.within_budget()

    def nbytes(self):
        return self._inner.nbytes()

    # ---- the interface transformers uses ---------------------------------- #
    def get_seq_length(self, layer_idx=0):
        del layer_idx
        return self._inner.tokens

    def get_max_cache_shape(self):
        """None means UNBOUNDED, which here is a fact rather than a convenience: the state does
        not grow with the sequence, so there is no length at which it stops fitting."""
        return None

    def get_max_length(self):                                  # older transformers spelling
        return None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """A no-op that exists because GenerationMixin may call it.

        THE TRUNK WRITES THIS CACHE ITSELF, through put(), inside the mixer -- there are no keys
        and values passing through here to store. Returning what was handed in keeps any caller
        that uses the return value correct."""
        del layer_idx, cache_kwargs
        return key_states, value_states

    def reorder_cache(self, beam_idx):
        """Beam search reorders the batch; the state's batch axis is dim 0.

        EVERY LAYER MOVES TOGETHER, which is what makes this correct for a recurrence: a beam is
        an entire history, so its state at every depth travels with it."""
        for layer in range(len(self._inner.state)):
            state = self._inner.get(layer)
            if state is not None:
                self._inner.put(layer, state.index_select(0, beam_idx.to(state.device)))
        return self

    def to_legacy_cache(self):
        return self

    def crop(self, max_length):
        """REFUSED, LOUDLY, rather than silently doing nothing.

        Cropping drops the most recent entries of a key/value cache. This state has no most
        recent part: every token so far is folded into the same K matrices. A caller that crops
        is assuming an architecture this is not, and returning silently would give it a state it
        believes is shorter than it is."""
        raise NotImplementedError(
            "MiraCache cannot be cropped: the lattice state is a summary of every token so far "
            f"folded into K matrices, with no per-token part to drop (asked for {max_length})")


# --------------------------------------------------------------------------- #
# the tokenizer
# --------------------------------------------------------------------------- #
class MiraTokenizer(PreTrainedTokenizer):
    """The project's own BPETokenizer, answering HuggingFace's questions.

    WRAPPED RATHER THAN CONVERTED, and the reason is the failure mode. A merge-table conversion
    to tokenizer.json that got one id wrong would not raise; it would point every embedding row
    at a different token, and the model would produce fluent nonsense. Asking the original for
    every id makes that class of error impossible rather than unlikely.

    IT SAVES bpe.json, WHICH IS THE FORMAT THE PROJECT ALREADY HAS. `save_pretrained` writes it
    into the HF directory and `from_pretrained` reads it back, so the HF directory is
    self-contained without a second vocabulary existing anywhere."""

    vocab_files_names = {"bpe_file": "bpe.json"}
    model_input_names = ["input_ids"]

    def __init__(self, bpe_file=None, bpe=None, **kw):
        if bpe is None:
            if not bpe_file:
                raise ValueError("MiraTokenizer needs bpe_file or bpe")
            bpe = BPETokenizer.load(bpe_file)
        self._bpe = bpe
        self._bpe_file = bpe_file
        eos = bpe.id2special.get(bpe.eos_token_id) if hasattr(bpe, "id2special") else None
        super().__init__(eos_token=eos, bos_token=eos, pad_token=eos, unk_token=eos, **kw)

    @property
    def vocab_size(self):
        return len(self._bpe)

    def get_vocab(self):
        return self._bpe.get_vocab()

    def _tokenize(self, text, **kw):
        """HF tokenises to STRINGS and then maps them to ids; this one goes straight to ids and
        carries them as decimal strings, because a byte-level BPE has tokens that are not valid
        UTF-8 on their own and a string round trip would lose them."""
        del kw
        return [str(i) for i in self._bpe.encode_ordinary(text)]

    def _convert_token_to_id(self, token):
        return int(token)

    def _convert_id_to_token(self, index):
        return str(int(index))

    def convert_tokens_to_string(self, tokens):
        return self._bpe.decode([int(t) for t in tokens], skip_special_tokens=False)

    def decode(self, token_ids, skip_special_tokens=False,
               clean_up_tokenization_spaces=None, **kw):
        """HF's OWN DEFAULT IS False AND IT IS KEPT. Flipping it would give every harness that
        relies on the documented default a different string, silently."""
        del kw, clean_up_tokenization_spaces
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self._bpe.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        name = (filename_prefix + "-" if filename_prefix else "") + "bpe.json"
        path = os.path.join(save_directory, name)
        self._bpe.save(path)
        return (path,)


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class MiraForCausalLM(PreTrainedModel, _GenerationMixin):
    """MiraLLM behind the PreTrainedModel interface, and nothing else changed.

    THE TRUNK IS HELD, NOT SUBCLASSED. `self.mira` is the real model, built from the checkpoint's
    own model_cfg, so the architecture is decided by the same record every internal stage reads.
    Weights therefore carry a `mira.` prefix here; `export_checkpoint` writes them that way and
    the project's own `.pt` is left exactly as it is."""

    config_class = MiraConfig
    base_model_prefix = "mira"
    supports_gradient_checkpointing = True
    # THE CLASS NAMES accelerate MATCHES ON, and they are the names of the classes actually
    # built: MiraBlock for arch="mira-lattice" (model/mira_llm.py) and Block for arch="rope".
    # An entry must equal the class's own __class__.__name__, never an alias, or it cannot match.
    _no_split_modules = ["MiraBlock"]
    # THE HEAD IS `head` AND IT LIVES UNDER THE `mira.` PREFIX. There is no module called
    # lm_head anywhere in this model; the entry is matched against parameter names, so the old
    # spelling matched nothing and save_pretrained fell back to guessing which alias to drop.
    _tied_weights_keys = ["mira.tok.weight"]

    def __init__(self, config):
        super().__init__(config)
        # THROUGH THE SAME REDUCTION THE CONVERTER USED, so a config.json written by an older
        # export -- or edited by hand -- is corrected here too rather than only on the way in.
        self.mira = MiraLLM(vocab_size=config.vocab_size,
                            **_model_kwargs(config.mira_cfg, log=lambda *a: None))
        self.post_init()

    # ---- the accessors resize_token_embeddings and tie_weights need -------- #
    def get_input_embeddings(self):
        return self.mira.tok

    def set_input_embeddings(self, value):
        self.mira.tok = value

    def get_output_embeddings(self):
        return self.mira.head

    def set_output_embeddings(self, value):
        self.mira.head = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        del gradient_checkpointing_kwargs
        self.mira.grad_checkpoint = True

    def gradient_checkpointing_disable(self):
        self.mira.grad_checkpoint = False

    # ---- generation -------------------------------------------------------- #
    def _new_cache(self):
        """The cache THIS trunk needs, or a refusal that says which switch turned it off.

        TWO ARCHITECTURES, TWO CACHES. arch="mira-lattice" carries a lattice state and takes a
        MiraCache; arch="rope" builds attention whose blocks call `cache.k[layer]` and
        `cache.append(...)` (model/mira_llm.py:143-179), which is helpers.kv_cache.Cache and
        which MiraCache does not implement. Handing the wrong one over is an AttributeError on
        the first cached forward.

        AND INFERENCE.prefix_cache = 0 IS ANSWERED HERE, WHERE IT CAN BE EXPLAINED. Without this
        the mixer refuses the cache several frames down with "streaming cache is not wired into
        mira-dsp", which names nothing the caller can act on."""
        if not _prefix_cache.enabled():
            raise ValueError(
                "use_cache was asked for but INFERENCE.prefix_cache is 0, so the lattice state "
                "cannot be carried between calls. Set INFERENCE.prefix_cache=1, or pass "
                "use_cache=False and accept that the prefix is re-run at every step.")
        return MiraCache(len(self.mira.blocks), model=self.mira)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, use_cache=True, **kw):
        """Feed the whole prompt once, then one token at a time.

        THE CACHE COUNTS WHAT IT HAS ABSORBED, so what is new is everything past that count. A
        recurrence cannot be handed a token twice: unlike attention, which would merely recompute
        a key it already has, folding a token in again advances the state by a token that was
        never said."""
        del kw
        seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        if seen and input_ids.shape[1] > seen:
            input_ids = input_ids[:, seen:]
        # THE CALLER'S use_cache IS HONOURED. Forcing it True here made `generate(use_cache=False)`
        # impossible to ask for, which is the only way to run a trunk whose cache is switched off.
        return {"input_ids": input_ids, "past_key_values": past_key_values,
                "attention_mask": attention_mask, "use_cache": bool(use_cache)}

    def _reorder_cache(self, past_key_values, beam_idx):
        return past_key_values.reorder_cache(beam_idx)

    @staticmethod
    def _real_mask(attention_mask):
        """An all-ones mask is NO mask, and collapsing it is the most important line in this file.

        The mixer takes the CUDA kernel only when `valid is None`
        (model/mira_mixer.py); any mask tensor sends it to the PyTorch reference
        instead. Every HuggingFace caller passes an all-ones attention_mask as a matter of
        course, so forwarding it unexamined would put the whole ecosystem on the slow path --
        silently, since nothing errors and only a profile would ever show it."""
        if attention_mask is None:
            return None
        return None if bool(attention_mask.all()) else attention_mask

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                past_key_values=None, use_cache=None, inputs_embeds=None,
                output_attentions=None, output_hidden_states=None,
                return_dict=None, **kw):
        del kw
        if inputs_embeds is not None:
            raise NotImplementedError(
                "MiraForCausalLM takes input_ids; the trunk embeds them itself")
        if output_attentions:
            raise ValueError(
                "this architecture has no attention: MIRA-Lattice carries a per-channel lattice "
                "and has no queries, keys or values to report")
        if use_cache and past_key_values is None:
            past_key_values = self._new_cache()
        hidden = self.mira.hidden_states(
            input_ids=input_ids, attention_mask=self._real_mask(attention_mask),
            cache=past_key_values)
        logits = self.mira.head(hidden)
        loss = None
        if labels is not None:
            from model.lm import normalised_nll
            tgt = labels[:, 1:]
            rm = (tgt != -100).to(logits.dtype)
            tgt = tgt.masked_fill(tgt == -100, 0)
            lp = nn.functional.log_softmax(logits.float(), -1)[:, :-1]
            lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            loss = normalised_nll(lp, rm)
        if return_dict is False:
            out = (logits,) if loss is None else (loss, logits)
            return out
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=past_key_values,
            hidden_states=(hidden,) if output_hidden_states else None, attentions=None)


# --------------------------------------------------------------------------- #
# conversion, once, on the first run
# --------------------------------------------------------------------------- #
def hf_dir_for(checkpoint):
    """Where the converted form of `checkpoint` lives: `<name>-hf/` beside it.

    BESIDE IT, SO THE TWO TRAVEL TOGETHER. A converted directory that lived under a central cache
    would outlive the checkpoint it came from and be loaded against the wrong weights the first
    time a checkpoint was replaced in place."""
    stem = os.path.splitext(str(checkpoint))[0]
    return stem + HF_SUFFIX


def export_checkpoint(checkpoint, out_dir=None, max_context=None, log=print):
    """Write `checkpoint` as an HF directory: safetensors, config.json, bpe.json. Returns the dir.

    THE .pt IS OPENED READ-ONLY AND IS NEVER MODIFIED. Every internal stage keeps reading exactly
    what it read before, and the converted directory is an addition rather than a migration.

    THE KEYS GAIN THE `mira.` PREFIX HERE, because MiraForCausalLM holds the trunk rather than
    being it. Nothing else about them changes, so the tensors in the safetensors file are the
    tensors in the .pt, name for name under the prefix."""
    from safetensors.torch import save_file
    out_dir = hf_dir_for(checkpoint) if out_dir is None else out_dir
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck or "model_cfg" not in ck:
        raise SystemExit(
            f"[hf] {checkpoint} has no model/model_cfg; it cannot be converted. Every checkpoint "
            f"this project writes carries the architecture that built its weights.")
    cfg = _model_kwargs(ck["model_cfg"], log=log)
    os.makedirs(out_dir, exist_ok=True)

    blob = ck.get("bpe")
    if blob is None:
        raise SystemExit(
            f"[hf] {checkpoint} carries no tokenizer. Its ids cannot be checked against a "
            f"vocabulary, and a wrong vocabulary loads and talks nonsense rather than failing.")
    bpe = BPETokenizer.loads(blob) if isinstance(blob, str) else BPETokenizer.loads(
        json.dumps(blob))
    tok = MiraTokenizer(bpe=bpe)
    tok.save_pretrained(out_dir)

    config = MiraConfig(mira_cfg=cfg, vocab_size=len(bpe), max_position_embeddings=max_context)
    config.save_pretrained(out_dir)

    # THE TIED PAIR IS WRITTEN ONCE. fp32_state() calls state_dict(), and a tied embedding
    # appears twice in it -- as tok.weight and as head.weight, one Parameter under two names.
    # safetensors refuses two entries that share storage outright when they really do share it
    # (a checkpoint saved from CPU fp32), and silently duplicates the whole vocabulary matrix
    # when the save-time .cpu()/.float() happened to copy them apart. Dropping the alias avoids
    # both; MiraLLM re-ties on load (model/mira_llm.py:520) and _tied_weights_keys names it, so
    # from_pretrained restores it rather than reporting it missing.
    state = {WEIGHT_PREFIX + k: v.contiguous() for k, v in ck["model"].items()}
    tied = WEIGHT_PREFIX + "tok.weight"
    if tied in state and WEIGHT_PREFIX + "head.weight" in state:
        state.pop(tied)
    save_file(state, os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt", "source": os.path.basename(str(checkpoint))})
    log(f"[hf] converted {checkpoint} -> {out_dir}")
    log(f"[hf]   {len(state):,} tensor(s), vocabulary {len(bpe):,}, "
        f"{config.num_hidden_layers} layer(s) of width {config.hidden_size}")
    return out_dir


def load(checkpoint, device=None, max_context=None, log=print):
    """(model, tokenizer) for a project checkpoint, converting it on the first run only.

    THE SECOND RUN CONVERTS NOTHING. `<name>-hf/` is looked for first and loaded directly when it
    is there; conversion happens only when it is absent. A checkpoint that has been retrained
    into the same path is the one case this cannot see, so `export_checkpoint` is callable on its
    own to force it."""
    checkpoint = str(checkpoint)
    out_dir = hf_dir_for(checkpoint)
    if not os.path.isfile(os.path.join(out_dir, "config.json")):
        export_checkpoint(checkpoint, out_dir, max_context=max_context, log=log)
    else:
        log(f"[hf] loading the converted checkpoint at {out_dir}")
    model = MiraForCausalLM.from_pretrained(out_dir)
    tok = MiraTokenizer.from_pretrained(out_dir)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, tok
