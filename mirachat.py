"""
mirachat.py -- load a trained checkpoint and chat with it, to qualitatively test what a
pretrain / SFT / RLHF / DPO model actually generates.

Loading is by --checkpoint ONLY: point it at any stage's

    checkpoints/<stage>/<run tag>_<stage-label>/checkpoint.pt

(a dict with a 'model' entry) or at a raw state_dict.

IT BUILDS MiraLLM AND NOTHING ELSE. The model is rebuilt from the ARCHITECTURE STORED IN THE
CHECKPOINT, with the tokenizer that travels inside it. A benchmark baseline -- mha, rwkv_v6,
rwkv_v7, mamba_v1, mamba_v2 and the rest -- is built by baselines.build_lm and its weights do
not fit this constructor; a distilled gpt2 student is not this architecture at all. Both are
refused by name rather than half-loaded.

    # interactive REPL on the most aligned checkpoint present (the default)
    python mirachat.py
    python mirachat.py --checkpoint checkpoints/instruct_sft/mira-s_llm-ultrachat/checkpoint.pt

    # one-shot, greedy
    python mirachat.py --checkpoint checkpoints/rlhf/mira-s_instruct-rlhf/checkpoint.pt \
        --prompt "Where?" --temperature 0

IT IS A CONVERSATION, NOT A SEQUENCE OF PROMPTS. Every turn is prompted with the whole
transcript so far, wrapped in the training format -- the message delimiters and role prefixes
stage 7 is fine-tuned on -- so a follow-up like "and why?" has something to refer
to. `reset` clears it, `history` prints it. The transcript is left-truncated only when it would
leave the reply less than half the window.

THE WRAPPER DEPENDS ON WHAT THE CHECKPOINT WAS TRAINED WITH, and a pretrain or benchmark model
was trained with nothing. The message delimiters are registered by stage 7, so a checkpoint from
an earlier stage does not carry them and is prompted with the same roles written as plain text
-- `User:` and `Assistant:`, one turn a paragraph -- which is a shape it has read. Its reply is
cut where it starts writing the user's next turn, because nothing else would stop it. See
TEMPLATES and STOP_TEXT.

IT STREAMS, token by token, in colour when the output is a terminal. And it has NO LENGTH CAP:
a reply ends at <|endoftext|> or at the context window, and at nothing else.

Decoding is incremental through model/prefix_cache.py: the lattice state, K matrices per layer,
constant in the length of the conversation. INFERENCE.prefix_cache = 0 turns that off and the
prefix is re-run at every step instead.
"""
import os
import argparse
import sys

import torch
import torch.nn.functional as F

import default_config as config
from tokenizer import BPETokenizer
from model import MiraLLM

# THE PROMPT SHAPES. "chat" IS THE ONE STAGE 7 TRAINS, and its DELIMITERS are registered specials
# rather than text: <|startofmessage|> and <|endofmessage|> are one id each, so a prompt
# containing those characters cannot forge a message boundary, and the reply ends at
# <|endofmessage|>, which generate() stops on because it is a token rather than a string.
#
# THE ROLE IS TEXT, NOT A TOKEN. `User:\n` and `Assistant:\n` are ordinary text, exactly as
# helpers.chat_index writes them at batch time -- see helpers.utils.CHAT_USER_PREFIX.
#
# THE PROMPT ENDS AFTER `Assistant:\n`, which is the cue and is the last thing the model is
# given. What follows it is the answer. Stage 7 supervises the assistant's WHOLE message,
# including the <|startofmessage|> and the cue that this template supplies here, so the model can
# also produce them itself when it is left to continue on its own.
#
# A CHECKPOINT WITHOUT THE SPECIALS KEEPS ITS OWN SHAPE -- the template is chosen from the
# tokenizer that came with the weights, so a pretrain or pre-registration checkpoint is prompted
# as it was trained.
TEMPLATES = {
    "none": "{p}",
    # THE SAME ROLE TEXT AS "chat", WITH THE DELIMITERS TAKEN OUT. A pretrain or benchmark
    # checkpoint has never seen the message specials -- stage 7 registers them -- so prompting it
    # with them hands it subwords of "<", "|", "startofmessage". What it HAS seen is ordinary
    # dialogue, so the roles are written as text and the turn is closed by a newline. The role
    # spelling is helpers.utils.CHAT_USER_PREFIX's, so the text a pretrain model is prompted with
    # is the text an instruct model is prompted with, differing only in the delimiters.
    "text": "User:\n{p}\nAssistant:\n",
    "chat": "<|startofmessage|>User:\n{p}<|endofmessage|><|startofmessage|>Assistant:\n",
    "hh":   "\n\nHuman: {p}\n\nAssistant:",
    "shp":  "POST: {p}\n\nRESPONSE: ",
}

# WHAT CLOSES A COMPLETED MESSAGE, which is not what opens one. A training document is
#
#     <|startofmessage|> User:\n question <|endofmessage|>
#     <|startofmessage|>* Assistant:\n* answer* <|endofmessage|>*
#
# repeated, where * marks what stage 7 supervises -- helpers.chat_index builds exactly that at
# batch time. <|endofmessage|> follows the ANSWER, so it closes a message that is finished and
# never the one being asked.
TURN_END = {"chat": "<|endofmessage|>",  # closes a COMPLETED turn in the transcript, never the prompt
            "text": "\n\n"}             # a blank line, which is what separates turns in text

# WHAT ENDS A REPLY WHEN THE BOUNDARY IS TEXT RATHER THAN A TOKEN. "chat" ends at
# <|endofmessage|>, one id, and generate() stops on it. "text" has no such id, so nothing stops a
# model from writing the next turn itself -- and a pretrain checkpoint, which has never been
# taught to yield, reliably does: it answers, writes "User:" and holds both sides of the
# conversation. These are the strings that mean it has started one; the reply is cut before them
# and the rest is never printed, so the transcript keeps only what the model said as itself.
STOP_TEXT = {"text": ("\nUser:", "\nAssistant:")}


def cut_at(text, markers):
    """`text` up to the first marker in it. Returns the text unchanged when there is none."""
    hits = [text.index(marker) for marker in markers if marker in text]
    return text[:min(hits)] if hits else text


def held_back(text, markers):
    """How many trailing characters might still be growing into a marker, so must not print.

    A stop marker arrives a token at a time, and "\nUs" is not yet "\nUser:" -- printing it and
    cutting afterwards would leave it on the reader's screen. This is the same rule the U+FFFD
    hold-back below follows: print only the prefix that cannot change."""
    return max((n for marker in markers
                for n in range(min(len(marker), len(text)), 0, -1)
                if text.endswith(marker[:n])), default=0)


class C:
    """ANSI colours, or nothing at all.

    OFF WHENEVER THE OUTPUT IS NOT A TERMINAL, and off when NO_COLOR is set (the convention:
    any value, however empty, means no colour). `python mirachat.py > transcript.txt` must produce a
    transcript and not a file full of escape sequences, and that decision belongs here rather
    than at each print -- a colour that has to be remembered at forty call sites is a colour
    that leaks into a log."""
    _on = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None \
        and os.environ.get("TERM", "") != "dumb"
    YOU = "\033[1;36m" if _on else ""          # bold cyan  -- what you typed
    MODEL = "\033[0;32m" if _on else ""        # green      -- what the model said
    INFO = "\033[2;37m" if _on else ""         # dim grey   -- the harness talking
    OFF = "\033[0m" if _on else ""

    @classmethod
    def info(cls, m):
        print(f"{cls.INFO}{m}{cls.OFF}", flush=True)


def _stage_checkpoints(stage="*"):
    """Every checkpoint whose directory is a `stage`, oldest first, at any depth.

    ONE NAME IS GLOBBED, AT ANY DEPTH, because the configuration a checkpoint carries is now
    in its DIRECTORY and not in its filename: every checkpoint is `checkpoint.pt` inside a
    directory that names the run -- checkpoints/pretrain/<model>_<pe>_<stage-label>/ -- so the
    file is predictable and the identity is still there, one level out. See
    helpers.utils.artefact_dir. A tree written under the former filename convention is not
    looked for here.

    THE STAGE DIRECTORY IS NOT ALWAYS DIRECTLY UNDER checkpoints/, WHICH IS WHY THIS RECURSES.
    A single-model run writes checkpoints/<stage>/, but the benchmark writes
    checkpoints/benchmark/<dataset>/<family>/<stage>/ and the ablation
    checkpoints/ablation/<dataset>/<variant>/<stage>/ -- three levels deeper, and on a machine
    that has only ever run the benchmark those are the ONLY checkpoints that exist. Globbing one
    level found none of them and mirachat.py reported "not found" beside a full checkpoints/ tree.
    `**` matches zero or more directories, so the flat layout is still found by the same pattern."""
    import glob
    roots = [name for names in _DEFAULT_ORDER for name in names if stage in ("*", name)]
    pats = [os.path.join(config.CHECKPOINT_DIR, name, "**", "checkpoint.pt")
            for name in roots]
    return sorted({p for pat in pats for p in glob.glob(pat, recursive=True)},
                  key=os.path.getmtime)


# WHERE A DEFAULT CHECKPOINT IS LOOKED FOR, IN ORDER, AND THE ORDER IS GIVEN:
#
#     "the search path is:
#      instruct_rlhf --> instruct_dpo --> instruct_sft --> pretrain --> benchmark"  -- 2026-09-05
#
# Most aligned first, and the benchmark -- which is a controlled comparison, not a model anyone
# means to talk to -- last. Not simply "the newest file", because re-running an early stage would
# then silently demote the default.
#
# EACH RUNG IS EVERY SPELLING THAT RUNG'S DIRECTORY HAS, because the tree does not spell them all
# one way. Stage 7 is handed checkpoints/instruct_sft/<scheme>_<dataset>/ (helpers.utils.sft_dir),
# while helpers.utils.stage_dirname gives stages 8-10 their bare stage key -- checkpoints/rlhf/,
# checkpoints/dpo/. Naming one spelling per rung is what left the previous list globbing for
# checkpoints/sft/, a directory that has never existed, so `python mirachat.py` walked past a
# finished instruct-sft checkpoint and answered from a benchmark pretrain one instead. Listing
# both costs one glob and cannot be wrong in either tree.
#
# `cot`, `ablation` and `experimental` ARE NOT RUNGS -- the search path above names five, and
# these are not among them. They are still loadable by --checkpoint and are still listed by the
# not-found notice; they are simply never chosen for the reader.
#
# `distill` IS NOT HERE EITHER: that stage writes a gpt2 student, which this file refuses.
_DEFAULT_ORDER = (
    ("instruct_rlhf",),
    ("instruct_dpo",),
    ("instruct_sft",),
    ("pretrain",),
    ("benchmark",),
)

# THE BENCHMARK TREE IS EXCLUDED FROM EVERY RUNG ABOVE ITS OWN, and that is what makes the last
# rung mean anything. checkpoints/benchmark/<dataset>/<family>/pretrain/ ended in `pretrain`, so
# the fourth rung matched it and the fifth could never be reached: a benchmark checkpoint would
# have been chosen one rung early, ahead of nothing but itself, and a real standalone pretrain
# run would have had to race it on mtime. `pretrain` now means a pretrain of its own.
_BENCHMARK_TREE = "benchmark"


def _names_mira(path):
    """Whether this checkpoint's own path says it holds a MIRA model.

    The benchmark, the ablation and the experimental stage each write one directory per family
    at <tree>/<dataset>/<family>/<stage>/<file>, and the run tag is in the filename beside it --
    `mira-matrix/checkpoint_benchmark-mira-s_pretrain.pt` against
    `rwkv_v7/checkpoint_benchmark-rwkv_v7_pretrain.pt`. No family this repository builds through
    baselines.build_lm carries `mira` in its name, so the substring separates the two without
    importing the baselines package, which pulls in every mixer and torch with it.

    A FALSE NEGATIVE COSTS NOTHING, which is why the test may be this cheap: default_checkpoint
    prefers what this accepts and falls back to the whole list, so an ablation variant directory
    named for its axis rather than its family is chosen exactly as it is today."""
    return "mira" in os.path.relpath(path, config.CHECKPOINT_DIR).lower()


def default_checkpoint():
    """The checkpoint to open when --checkpoint is not given. Falls back to a path that does
    NOT exist when nothing has been trained, so read_checkpoint prints its 'not found' notice
    with the list of alternatives instead of quietly loading an untrained model.

    A MIRA checkpoint is preferred over a baseline at the same stage. On a machine that has run
    only the benchmark, five of the six checkpoints are foreign families that read_checkpoint
    refuses by name, and taking the newest file made the default whichever family happened to
    finish last -- rwkv_v7, and `python mirachat.py` refused to start."""
    for rung, names in enumerate(_DEFAULT_ORDER):
        found = [p for name in names for p in _stage_checkpoints(name)]
        if rung < len(_DEFAULT_ORDER) - 1:
            found = [p for p in found
                     if _BENCHMARK_TREE not in
                     os.path.relpath(p, config.CHECKPOINT_DIR).split(os.sep)]
        if found:
            found.sort(key=os.path.getmtime)
            return ([p for p in found if _names_mira(p)] or found)[-1]
    return os.path.join(config.CHECKPOINT_DIR, "instruct_rlhf",
                        "<model>_instruct-rlhf", "checkpoint.pt")


DEFAULT_CKPT = default_checkpoint()


def parse_args():
    ap = argparse.ArgumentParser(description="chat with a trained checkpoint")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT,
                    help="weights to load: any stage's "
                         "checkpoints/<stage>/<model>_<pe>_<stage-label>/checkpoint.pt or a "
                         "raw state_dict. Empty = base (untrained) mira_llm. "
                         f"Default: {os.path.relpath(DEFAULT_CKPT, config.ROOT)}")
    # PER REPLY, NOT THE CONTEXT. The window is what the whole conversation may grow to;
    # this is what ONE answer may be. A reply is still cut short by the window when the
    # transcript has less room left than this, and by <|endoftext|> whenever the model emits it.
    ap.add_argument("--response_length", type=int, default=4096, metavar="N",
                    help="tokens ONE reply may generate (default: 4096). Not the context "
                         "window: the transcript may be far longer. A reply ends at "
                         "<|end-of-session|>, at N tokens, or when the window is full")
    ap.add_argument("--dataset", default="hh",
                    help="only used to choose the prompt template")
    ap.add_argument("--gpu", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--prompt", default="", help="one-shot prompt; if empty, start an interactive REPL")
    ap.add_argument("--template", default="", choices=["", "none", "text", "chat", "hh", "shp"],
                    help="prompt wrapper; '' = auto (chat when the checkpoint's tokenizer "
                         "carries the role specials, else hh/shp by dataset, else text). "
                         "'none' prompts with the bare text, which is raw continuation")
    # THERE IS NO LENGTH CAP. The model stops when it emits <|endoftext|> or when the context
    # window is full, and nothing else stops it -- the same rule the CoT stage was put on. A cap
    # here was a number invented by the harness, and every reply that ran into it was cut in the
    # middle of a sentence and read as the model losing its thread.
    # "--temp" is spelled out as an alias: argparse's prefix matching makes the abbreviation
    # ambiguous against --template, and failing on `--temp 1` is a poor greeting.
    ap.add_argument("--temperature", "--temp", "-t", type=float, default=1.0,
                    help="0 = greedy (argmax)")
    ap.add_argument("--top_k", type=int, default=50, help="0 = disabled")
    ap.add_argument("--top_p", type=float, default=0.95, help="nucleus; 1.0 = disabled")
    ap.add_argument("--max_len", type=int, default=0, help="0 = auto per model")
    # A CHAT SESSION IS RANDOM BY DEFAULT, WHICH IS WHAT A CHAT SESSION IS. The sampler below is
    # ordinary multinomial sampling with temperature, top-k and top-p -- but main() seeded torch
    # with this value, and a fixed default of 0 seeded every session identically, so the same
    # prompt returned the same reply word for word however many times it was asked. That reads
    # as a greedy decoder and is why sampling looked broken. -1 seeds from the operating system
    # instead; any value >= 0 is passed to torch.manual_seed and reproduces a session exactly.
    ap.add_argument("--seed", type=int, default=-1,
                    help="-1 = fresh randomness each session; >= 0 reproduces a session")
    # THE CEILING ON THE PREFIX CACHE, IN GiB, and it is the same number the yaml carries. It is
    # written into config.INFERENCE in main() before anything reads it, because
    # model/prefix_cache.py reads the ceiling from there at call time.
    ap.add_argument("--prefix_cache_size", type=float,
                    default=config.INFERENCE["prefix_cache_size"],
                    help="ceiling on the MIRA inference state cache, in GiB")
    return ap.parse_args()


def resolve_device(choice):
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def _available():
    """Stage checkpoints that DO exist, so a missing one can suggest the alternatives."""
    return sorted(os.path.relpath(p, config.ROOT) for p in _stage_checkpoints())


def read_checkpoint(ckpt):
    """The state dict inside `ckpt` (a stage checkpoint or a raw state_dict), or None when no
    (existing) checkpoint was given -- the caller then falls back to the base model.

    A MISSING checkpoint is shouted about rather than mentioned. An untrained MiraLLM samples
    uniformly from a 50k vocabulary, so it emits fluent-looking word salad -- which reads
    exactly like a broken tokenizer, and is the single most confusing thing this script can
    do quietly."""
    if not ckpt:
        print("[chat] no checkpoint given -> UNTRAINED model; output will be random text",
              flush=True)
        return None, None, None
    if not os.path.isfile(ckpt):
        have = _available()
        print("\n" + "!" * 78, flush=True)
        print(f"[chat] CHECKPOINT NOT FOUND: {ckpt}", flush=True)
        print("[chat] Falling back to an UNTRAINED model. Its output is random text drawn "
              "from the\n[chat] vocabulary -- nothing is wrong with the tokenizer or the "
              "decoder.", flush=True)
        print(("[chat] Available checkpoints:\n[chat]   "
               + "\n[chat]   ".join(have)) if have else
              "[chat] No stage checkpoints exist yet: run a stage first, e.g.\n"
              "        ./old_scripts/stage6_pretrain.sh",
              flush=True)
        print("!" * 78 + "\n", flush=True)
        return None, None, None
    obj = torch.load(ckpt, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        # The whole record is returned, not just the weights: a checkpoint carries the
        # TOKENIZER it was trained with, and reading the weights while leaving the vocabulary
        # behind is how a model comes to talk fluent nonsense with nothing in the log to say so.
        return obj["model"], obj.get("model_cfg"), obj
    return obj, None, None


def build_from_checkpoint(sd, cfg, args, device, ck=None, path=""):
    """(model, tok, max_len) matching the checkpoint's architecture.

    'transformer.*' keys mean the distilled gpt2-small student. Otherwise it is a MiraLLM, and
    the ARCHITECTURE RECORDED IN THE CHECKPOINT is used -- a model trained at a different
    width or depth must be rebuilt as it was, and rebuilding it from whatever default_config.py currently
    says would fail to load (or load something subtly different). config.MODEL is only the
    fallback for checkpoints written before the architecture was stored."""
    if sd is not None and any(k.startswith("transformer.") for k in sd):
        raise SystemExit(
            "[chat] this checkpoint holds a distilled gpt2 student ('transformer.*' keys), not "
            "a MiraLLM. mirachat.py builds MiraLLM and nothing else. Point --checkpoint at a MIRA "
            "checkpoint.")
    # MiraLLM + THE TOKENIZER THE CHECKPOINT WAS TRAINED WITH, which travels inside it. A
    # checkpoint written before that, or one that could not embed its tokenizer, falls back to
    # checkpoints/bpe/bpe.json -- announced as a fallback, because it is the only candidate
    # rather than a verified match. This is what lets a lang_sft checkpoint, whose vocabulary
    # was EXTENDED for another language, decode correctly without any of it being configured.
    # IMPORTED HERE, NOT AT THE TOP, AND THE REASON IS NARROWER THAN IT WAS. helpers.common
    # imports this module at module scope, so `from helpers import common` here would close a
    # cycle. `import helpers` alone would not -- helpers/__init__ pulls in only helpers.utils,
    # which never reaches helpers.common -- but the deferred form costs nothing and keeps the
    # question from having to be re-asked.
    from helpers.utils import bpe_from_ckpt
    tok = bpe_from_ckpt(ck, fallback=expected_vocabulary(path), log=print,
                        checkpoint_path=path) if ck is not None else None
    # THE CHECKPOINT'S OWN TOKENIZER WINS, AND THE NESTING IS WHY. Written flat as
    # `if tok is None and os.path.isfile(...): ... else: raise`, a tokenizer FOUND in the
    # checkpoint fails the first condition and falls into the else -- so the one case that is
    # entirely correct was the one that stopped the session. The three outcomes are: found in
    # the checkpoint, found on disk, or refused.
    if tok is None:
        # saved_models/ FIRST, THEN checkpoints/ -- see tokenizer.run.base_path.
        from tokenizer.run import base_path as _base_path
        _bpe = _base_path()
        if os.path.isfile(_bpe):
            tok = BPETokenizer.load(_bpe)
        else:
            raise SystemExit(
                f"[chat] no tokenizer: this checkpoint carries none and {_bpe} does "
                f"not exist. A vocabulary cannot be reconstructed from the weights, and a "
                f"rebuilt one would assign different ids to the same tokens. Run "
                f"./stage3_train_tokenizer.sh, or point --checkpoint at a checkpoint that "
                f"carries its tokenizer.")
    if sd is not None and not isinstance(cfg, dict):
        raise SystemExit("[chat] checkpoint architecture metadata is missing")
    if cfg:
        m_cfg = {str(k).lower(): v for k, v in cfg.items()
                 if str(k).lower() != "vocab_size"}
        family = str(m_cfg.get("benchmark_family") or "")
        # A cfg THAT NAMES NOTHING IS MIRA'S. There is one architecture, so a MIRA cfg stopped
        # recording `arch`; a baseline or the rope ablation still records what it is, which is
        # what this check is here to refuse.
        arch = config.canonical_arch(m_cfg.get("experimental_arch") or m_cfg.get("arch")
                                     or "mira-lattice")
        if not config.is_lattice_arch(arch):
            raise SystemExit(
                f"[chat] this checkpoint is {family or arch or 'an unrecorded architecture'!r}, "
                f"not MIRA-Lattice. mirachat.py builds mira-lattice and nothing else -- a "
                f"benchmark baseline comes from baselines.build_lm and does not fit this "
                f"constructor, and the rope ablation is a different model. Point --checkpoint "
                f"at a MIRA checkpoint.")
        experimental_arch = m_cfg.pop("experimental_arch", None)
        m_cfg.pop("benchmark_architecture", None)
        m_cfg.pop("benchmark_family", None)
        if experimental_arch:
            m_cfg["arch"] = experimental_arch
        # THROUGH THE ONE ALIAS TABLE, exactly as helpers.common.translate_cfg does it. A
        # checkpoint may record the package directory ("mira-matrix") where MiraLLM wants the
        # architecture ("mira-lattice"), and config.canonical_arch is where that is resolved for
        # the whole pipeline.
        if "lattice_order" in m_cfg:
            config.mira_map_arch(m_cfg)
            if "arch" in m_cfg:
                raise ValueError("this MIRA checkpoint records arch instead of lattice_modulus_map and "
                                 "lattice_argument_map, the format before those keys, and cannot be loaded")
        elif m_cfg.get("arch"):
            m_cfg["arch"] = config.canonical_arch(m_cfg["arch"])
    else:
        m_cfg = dict(config.MODEL)
        m_cfg.pop("arch", None)
        m_cfg.setdefault("block_size", config.context_window(config.DEFAULT_SCHEME))
        if sd is not None:                            # infer at least the depth
            layers = {int(k.split(".")[1]) for k in sd if k.startswith("blocks.")}
            if layers:
                m_cfg["n_layer"] = max(layers) + 1
    import inspect
    accepted = set(inspect.signature(MiraLLM.__init__).parameters) - {"self", "vocab_size"}
    ignored = sorted(set(m_cfg) - accepted)
    if ignored:
        print(f"[chat] ignoring {len(ignored)} field(s) this model no longer takes: "
              f"{', '.join(ignored)}", flush=True)
        m_cfg = {k: v for k, v in m_cfg.items() if k in accepted}
    # THE BOTTLENECK FACTOR IS RECORDED AGAINST d_e AND VALIDATED AGAINST d. baselines.build_lm
    # writes coefficient_bottleneck_factor = lattice_d_e / coefficient_bottleneck
    # (baselines/__init__.py:516) -- 1536/48 = 32 at the benchmark shape -- and MiraLLM requires
    # coefficient_bottleneck == n_embd / coefficient_bottleneck_factor (model/mira_llm.py:380),
    # which is 768/32 = 24 and refuses the checkpoint. Every benchmark MIRA checkpoint on disk
    # records it that way, so this is what opening one costs.
    #
    # ONLY THE DIVISOR IS RECOMPUTED, NEVER THE WIDTH. coefficient_bottleneck is recorded
    # explicitly and is what the weights actually have, so it is passed through untouched; the
    # factor is redundant beside it and exists only for the constructor to derive a bottleneck
    # that is already given. Recomputing it as n_embd / coefficient_bottleneck therefore builds
    # the same model, and both values are printed so the substitution is not silent.
    #
    # HERE AND NOT IN baselines/, WHICH IS SHARED. The recorded value is what every ablation
    # checkpoint on disk carries, and changing the writer makes those checkpoints refuse to
    # resume; instruct_sft/run.py:model_from resolves it locally for the same reason.
    _b = m_cfg.get("coefficient_bottleneck")
    _f = m_cfg.get("coefficient_bottleneck_factor")
    _d = m_cfg.get("n_embd")
    if (_b and _f and _d and int(_d) % int(_b) == 0
            and int(_b) != int(_d) // int(_f)):
        print(f"[chat] coefficient_bottleneck_factor recorded as {_f} against d_e; MiraLLM reads "
              f"it against n_embd, where {_d} / {_f} = {int(_d) // int(_f)} is not the recorded "
              f"coefficient_bottleneck {_b}", flush=True)
        print(f"[chat] using coefficient_bottleneck_factor {int(_d) // int(_b)} = {_d} / {_b}; "
              f"the bottleneck itself is unchanged, so the model built is the one trained",
              flush=True)
        m_cfg["coefficient_bottleneck_factor"] = int(_d) // int(_b)
    if "init_horizon" in m_cfg and "lattice_init_horizon" not in m_cfg:
        m_cfg["lattice_init_horizon"] = m_cfg.pop("init_horizon")
    if (config.canonical_arch(m_cfg.get("arch", "mira-lattice")) == "mira-lattice"
            and "lattice_init_horizon" not in m_cfg):
        m_cfg["lattice_init_horizon"] = config.MODEL["lattice_init_horizon"]
    if config.canonical_arch(m_cfg.get("arch", "mira-lattice")) == "mira-lattice":
        m_cfg["lattice_initializer"] = config.canonical_lattice_initializer(
            m_cfg.get("lattice_initializer", m_cfg.pop("lattice_init_method", "schur_init")), recorded=True)
    m_cfg.pop("lattice_norm_module", None)      # lattice_norm_module (retired 2026-10-05; MIRA is RMSNorm only)
    model = MiraLLM(vocab_size=len(tok), **m_cfg).to(device)
    print(f"[chat] architecture: MIRA-Lattice, {len(model.blocks)} layers, no positional "
          f"encoding and no attention", flush=True)
    # A LATTICE TRUNK IS NOT HELD TO WHAT IT WAS TRAINED AT. block_size is the training length;
    # nothing in the model refers to an absolute position, so nothing about it caps what it can
    # be asked at inference, and truncating a transcript to 512 threw away conversation the model
    # could read. MAX_CONTEXT is the ceiling instead. An explicit --max_len still wins, for a
    # session that wants to hold the model to a length on purpose.
    return model, tok, args.max_len or MAX_CONTEXT


def summarise(model, tok, ck, cfg, path, device, max_len, args, log):
    """The loaded model, as a table, BEFORE the first prompt.

    WHAT IS ANSWERED HERE is the question every confusing chat session ends up asking: which
    model is this, actually. A checkpoint's filename carries its configuration, but a filename
    is not read carefully at midnight, and the two mistakes that waste an evening -- talking to
    the pretrain checkpoint while believing it is the aligned one, and a vocabulary that does
    not match the weights -- are both invisible in the output. They are both in this table.

    EVERYTHING IS READ FROM WHAT WAS LOADED, never from default_config: the architecture comes
    from the checkpoint's own model_cfg, the parameter counts from the built modules, the
    vocabulary from the tokenizer that came with the weights. A summary that described the
    configuration file rather than the object in memory would agree with itself while being
    wrong about the model."""
    import helpers                       # HERE, not at the top: helpers.common imports this
    rows, cfg = [], dict(cfg or {})      # module, so a module-level import closes the cycle
    if path:
        how = ("--checkpoint" if path != DEFAULT_CKPT else
               "default (" + " > ".join(names[0] for names in _DEFAULT_ORDER) + ")")
        rows.append(("checkpoint", os.path.relpath(path, config.ROOT)))
        rows.append(("  chosen by", how))
    else:
        rows.append(("checkpoint", "(none)"))
    if path and os.path.isfile(path):
        # BYTES, not helpers.human -- that formats COUNTS, and a 2.1 GiB file rendered as
        # "2.15B" reads as two billion of something rather than as gigabytes.
        b = os.path.getsize(path)
        rows.append(("size on disk", next(f"{b / d:.2f} {u}" for u, d in
                                          (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10),
                                           ("B", 1)) if b >= d or d == 1)))
    if ck:
        step, total = ck.get("step"), ck.get("total")
        if step is not None:
            rows.append(("trained to", f"step {step:,}" + (f" of {total:,}" if total else "")
                         + ("  (finished)" if ck.get("done") else "  (unfinished)")))
        if ck.get("eval"):
            rows.append(("recorded eval", str(ck["eval"])))
    rows.append(("", ""))

    n_par = sum(p.numel() for p in model.parameters())
    emb = getattr(model, "tok", None)
    n_emb = emb.weight.numel() if emb is not None and hasattr(emb, "weight") else 0
    tied = (emb is not None and getattr(model, "head", None) is not None
            and emb.weight is model.head.weight)
    if cfg:
        d = cfg.get("n_embd", 0)
        ffn_hidden = cfg.get("ffn_hidden") or 4 * d
        # `pe` is what a checkpoint written before the third architecture calls this
        arch = helpers.arch_of(cfg)
        rows += [
            ("layers", helpers.count(cfg.get("n_layer", len(getattr(model, "blocks", []))))),
            ("d_model (d)", helpers.count(d)),
            ("expansion (d_e)", helpers.count(cfg.get("lattice_d_e", 0))),
            ("memory (d_m)", helpers.count(cfg.get("memory_dimension", 0))),
            ("bottleneck (d_r)", helpers.count(cfg.get("coefficient_bottleneck", 0))),
            ("lattice order (K)", helpers.count(cfg.get("lattice_order", 0))),
            ("ffn", f"{ffn_hidden / d:g}x  ({ffn_hidden:,})" if d else str(ffn_hidden)),
            ("positional encoding", str(arch)),
            # TRAINED AT, NOT LIMITED TO. The two are the same number for an attention trunk and
            # are not the same fact for a lattice one, which has no length at which anything
            # about it changes.
            ("context window", f"{cfg.get('block_size', max_len):,} trained, "
                               f"{max_len:,} at inference"
                               if max_len != cfg.get("block_size", max_len)
                               else f"{max_len:,}"),
        ]
    else:
        rows.append(("architecture", f"{type(model).__name__} (no model_cfg in the checkpoint)"))
    rows.append(("", ""))
    rows += [
        ("parameters", f"{helpers.human(n_par)}  ({n_par:,})"),
        ("  embedding", f"{helpers.human(n_emb)}" + ("  tied" if tied else "")),
        ("  everything else", helpers.human(n_par - n_emb)),
        ("vocabulary", helpers.count(len(tok)) if tok is not None else "?"),
        ("tokenizer", _rel(getattr(tok, "loaded_from", "")) or "(none)"),
        ("  source", getattr(tok, "loaded_how", "") or "(unknown)"),
        # WHICH TURN MARKERS THIS CHECKPOINT HAS, BY NAME AND BY ID. The size alone does not say
        # it: 50,262 is the base vocabulary plus three and 58,262 is the coder vocabulary plus
        # three, and which three they are decides whether the prompt shape above is the one the
        # model was trained in. Read from the tokenizer that came with the weights, so it
        # describes the model in memory rather than the configuration in force.
        ("chat markers", ", ".join(f"{t}={i}" for t, i in _chat_markers(tok)) or "none"),
        ("", ""),
        ("device", str(device)),
        ("dtype", str(next(model.parameters()).dtype).replace("torch.", "")),
        ("sampling", f"temperature {args.temperature}, top_k {args.top_k or 'off'}, "
                     f"top_p {args.top_p}"),
        # THE SEED IS PART OF THE SAMPLING CONFIGURATION AND IT IS THE PART THAT DECIDES WHETHER
        # A REPLY CAN BE REPEATED. A session sampling freely and a session pinned to a seed
        # produce the same table otherwise, and the difference is the whole reason two runs of
        # one prompt differ.
        ("seed", f"{args.seed} (--seed)" if args.seed >= 0
                 else f"{torch.initial_seed()} (from the OS)"),
        ("reply length", f"{args.response_length:,} (--response_length)"),
    ]
    # THE DECODING CACHE, WHICH IS WHAT DECIDES WHETHER A REPLY COSTS n TOKEN-FORWARDS OR n^2.
    # Asked of the model rather than named from the architecture, exactly as new_cache decides
    # it, so the table cannot disagree with what the session actually uses.
    from model import prefix_cache
    shape = geometry(model)
    if shape is None:
        rows.append(("prefix cache", "not applicable"))
    elif not prefix_cache.enabled():
        rows.append(("prefix cache", "disabled (INFERENCE.prefix_cache = 0)"))
    else:
        rows.append(("prefix cache", f"enabled -- {prefix_cache.describe(*shape)}"))
        rows.append(("  ceiling", f"{args.prefix_cache_size} GiB (--prefix_cache_size)"))
    helpers.table("loaded model", rows, out=log)


_INSTRUCT_DIRS = ("instruct_sft", "instruct_rlhf", "instruct_dpo", "instruct_reward",
                  "rlhf", "dpo", "reward")
INSTRUCT_VOCAB, BASE_VOCAB = "bpe-instruct", "bpe"


def expected_vocabulary(path):
    """The bpe.json a checkpoint at `path` should be read with, when it carries none itself.

    THIS IS THE LAST RESORT AND IT IS A GUESS, which is why it is a guess made from the one
    thing that is always true of a checkpoint: where the stage that wrote it puts its files.
    bpe_from_ckpt tries the EMBEDDED copy first -- the only one known to match the weights --
    and then the vocabulary the checkpoint NAMES; this answers only for a file that carries
    neither, which means one written before either was recorded.

    Returns a PATH, because that is what bpe_from_ckpt's fallback takes, and saved_models/ is
    preferred over checkpoints/ for the same reason everywhere else: a published vocabulary
    arrives there whole, with the sidecars that say what is appended to it.

    THE LOOKUP IS tokenizer.run.resolve_named'S, NOT A SECOND ONE WRITTEN HERE.

    This built `saved_models/tokenizer-{name}/bpe.json` itself, and for the BASE vocabulary that
    is `tokenizer-bpe` -- A DIRECTORY THAT DOES NOT EXIST. The base is the one repository
    predating the tokenizer-<name> convention and is published as
    config.BPE_DOWNLOAD_NAME, `tokenizer-bpe-fineweb-edu-2BT`, so every place that spelled the
    directory out found nothing and fell through to checkpoints/. It also tried only the
    bpe.json-at-the-root shape. resolve_named knows both, and knows the base's real name."""
    from tokenizer.run import resolve_named
    parts = {p.lower() for p in os.path.normpath(os.path.abspath(path or "")).split(os.sep)}
    name = INSTRUCT_VOCAB if (parts & set(_INSTRUCT_DIRS)) else BASE_VOCAB
    tok, found = resolve_named(name, log=lambda _m: None)
    return found if tok is not None else config.BPE_PATH


def _rel(path):
    """A path under the project root, shortened to it; anything else as it stands."""
    if not path:
        return ""
    try:
        rel = os.path.relpath(os.path.abspath(path), config.ROOT)
    except ValueError:                                         # a different drive on Windows
        return str(path)
    return str(path) if rel.startswith(os.pardir) else rel


def _chat_markers(tok):
    """[(token, id)] for the turn markers this tokenizer actually carries, in registration order.

    Reported rather than assumed: a checkpoint written under an earlier scheme carries a
    different three, and the vocabulary SIZE does not distinguish them."""
    try:
        from helpers.utils import CHAT_SPECIALS
        have = getattr(tok, "special2id", None) or {}
        return [(t, have[t]) for t in CHAT_SPECIALS if t in have]
    except Exception:                                          # noqa: BLE001
        return []


def decode(tok, ids):
    return tok.decode(ids, skip_special_tokens=True)


MAX_CONTEXT = 65536


def new_cache(model):
    """The decoding cache, or None when the trunk has no blocks.

    ONE ARCHITECTURE, ONE CACHE. A MIRA-Lattice block has no queries, keys or values; what it
    carries between calls is the persistent lattice state V_t^(0..K-1), K matrices per layer that
    are the same size at 512 tokens as at 32,768. That is model.prefix_cache.PrefixCache, and
    since this file builds nothing else there is nothing to choose between.

    INFERENCE.prefix_cache = 0 RETURNS None and the caller decodes without a cache, re-running
    the prefix at every step. That is what the setting is for."""
    from model import prefix_cache
    blocks = getattr(model, "blocks", None)
    if not blocks or not prefix_cache.enabled():
        return None
    mixer = next((getattr(b, "mixer", None) for b in blocks
                  if hasattr(getattr(b, "mixer", None), "state_values")), None)
    if mixer is None:
        return None
    shape = geometry(model)
    fits, why = prefix_cache.affordable(*shape)
    if not fits:
        raise SystemExit(f"[chat] {why}")
    return prefix_cache.PrefixCache(len(blocks))


def geometry(model):
    """(layers, batch, K, d_e, d_m) -- what a PrefixCache over `model` holds.

    Known here and nowhere earlier, and constant in the context length, which is the whole
    property this cache has: one line built from it answers "how much memory does talking to this
    model cost" for every prompt length at once. Shared by new_cache, announce_cache and the
    loaded-model table so the three cannot describe different caches."""
    blocks = getattr(model, "blocks", None) or []
    mixer = next((getattr(b, "mixer", None) for b in blocks
                  if hasattr(getattr(b, "mixer", None), "state_values")), None)
    if mixer is None:
        return None
    return (len(blocks), 1, mixer.order, mixer.inner, mixer.memory_dimension)


@torch.no_grad()
def cache_supported(model, device):
    """Whether this model can take a streaming cache, decided by asking it once.

    MIRA-MATRIX REFUSES RATHER THAN IGNORING, and that is the right choice. Its mixer carries a
    recurrent state that incremental decoding would have to thread between calls; it does not
    implement that, so it raises instead of quietly returning an answer computed from a state that
    was never advanced. benchmark/run.py records the same fact about the other recurrent families,
    which is why the benchmark runs no generation previews at all.

    ONE TOKEN THROUGH THE TRUNK IS THE WHOLE TEST, and it is asked once per session rather than
    guessed from the architecture name -- a family that gains incremental decoding later starts
    using it here without this file being touched.

    THE PROBE USES THE CACHE THE GENERATION WILL USE, from new_cache. Probing with one kind and
    decoding with another would answer a question nobody asked: a lattice mixer refuses an
    attention cache and accepts a prefix cache, so the two give opposite answers.
    """
    try:
        with torch.no_grad():
            model.hidden_states(
                input_ids=torch.zeros((1, 1), dtype=torch.long, device=device),
                cache=new_cache(model))
        return True
    except Exception:                                          # noqa: BLE001
        return False


def announce_cache(model, device):
    """Say which cache this session decodes with, once, BEFORE the first prompt.

    IT BELONGS AT STARTUP AND NOT IN generate(). Which cache a session got decides whether a
    reply costs n token-forwards or n^2 of them, so it is worth two lines -- but generate() is
    called after "Model: " has already been written, so a line printed from there lands inside
    the model's reply and reads as the answer. Worse, generate() is not chat's alone: every
    trainer's generation preview calls it through helpers.common, so a line printed from there
    appears in a TRAINING log wearing a [chat] prefix. Both lines are said here instead.
    """
    from model import prefix_cache
    cache = new_cache(model)
    if cache is None:
        C.info("[chat] no prefix cache (INFERENCE.prefix_cache = 0)")
    elif not cache_supported(model, device):
        C.info("[chat] no incremental decoding in this architecture; no prefix cache")
    else:
        shape = geometry(model)
        if shape is not None:
            C.info(f"[chat] prefix cache: {prefix_cache.describe(*shape)}")
        C.info(f"[chat] decoding with {type(cache).__name__}")


def generate(model, prompt_ids, device, max_new, temperature, top_k, top_p, eos_id, max_len,
             desc="", on_token=None, stop_text=(), detok=None):
    """Autoregressive sampling, CACHED, and REPORTED while it runs.

    IT RECOMPUTED THE WHOLE PREFIX AT EVERY STEP -- O(n^2). That was tolerable while a preview
    was sixty tokens; it stopped being tolerable the moment the chain-of-thought previews began
    filling the context window, because one 7,800-token sample is then 30,423,900 token-forwards
    instead of 7,800, and five of them are 152 million. The cache keeps each layer's
    lattice state -- K matrices, constant in the length of the conversation -- exactly as the
    GRPO rollout does.

    AND IT PRINTED NOTHING WHILE IT RAN. `desc` puts a self-erasing bar on the token loop, so a
    long generation shows progress and leaves no trace in the block afterwards -- which is what
    the surrounding preview needs: something to read, not a bar wedged between its lines. A
    generation that takes minutes with no output is indistinguishable from a hung process, and
    this project's rule that every long-running loop reports is there for exactly that reason.

    `on_token(ids)` IS THE OTHER WAY OF SATISFYING THAT RULE, for a reader rather than a log:
    it is called after every accepted token with the ids generated SO FAR, and whatever it
    returns is ignored. The whole list is passed rather than the one new id because a
    byte-level BPE token is not a character -- a multi-byte character spans several tokens, and
    an accented letter decoded one token at a time comes out as replacement characters. The
    caller decodes the list and prints only what has grown, which is correct at every prefix.
    Pass `on_token` OR `desc`, not both: a bar and a stream on one line fight for it.

    `stop_text` ENDS THE REPLY AT A STRING, for a checkpoint whose turn boundary is text rather
    than a registered id -- see STOP_TEXT. It is matched on the DECODED reply, because a
    byte-level BPE token is not a character and the ids that spell "\nUser:" here need not be the
    ids that spell it anywhere else. `detok(ids) -> str` supplies the decoding; without it there
    is nothing to match and the reply ends at eos_id alone, as it always did. The marker is left
    in the returned ids and the caller cuts it, so what stopped the reply is visible rather than
    silently removed."""
    from helpers.utils import progress
    cur, out = list(prompt_ids), []
    cache = new_cache(model)
    # A TRUNK THAT REFUSES A CACHE IS DECODED WITHOUT ONE, which is slower and correct rather than
    # fast and impossible. The prefix is re-run at every step, so a reply of n tokens costs O(n^2)
    # token-forwards; at a short window that is seconds per reply, and it is the
    # only way to talk to a family whose recurrence is not threaded between calls.
    # NOTHING IS PRINTED FROM HERE ON. announce_cache() below says which cache a session got,
    # once, at startup. Saying it here put it AFTER "Model: " had been written and INSIDE the
    # model's reply:
    #     Model: [chat] prefix cache: 12 layer(s) x batch 1 ...
    # which is what the reader sees instead of an answer.
    uncached = cache is not None and not cache_supported(model, device)
    if uncached:
        cache = None
    fed = 0                                   # how much of `cur` the cache has already seen
    steps = progress(range(max_new), desc=desc) if desc else range(max_new)
    for _ in steps:
        ctx = cur[-max_len:] if max_len else cur
        if cache is not None:
            x = torch.tensor([cur[fed:] if fed else ctx], device=device)
            h = model.hidden_states(input_ids=x, cache=cache)
            logits = model.head(h[:, -1])[0].float()
            fed = len(cur)
            # TRIMMED ONLY WHERE THERE IS SOMETHING TO TRIM. Attention's keys and values belong
            # to individual tokens, and the ones that have left the window are dropped. The
            # lattice state belongs to no token -- it is the summary of every token so far
            # folded into K matrices -- so a prefix cache has no trim, and calling one would be
            # asking to discard the prefix this cache exists to keep.
            if max_len and hasattr(cache, "trim"):
                cache.trim(max_len)
        elif uncached:
            x = torch.tensor([ctx], device=device)
            h = model.hidden_states(input_ids=x, attention_mask=None)
            logits = model.head(h[:, -1])[0].float()
        else:
            x = torch.tensor([ctx], device=device)
            logits = model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k > 0:
                thresh = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            if 0 < top_p < 1:
                sp, si = torch.sort(probs, descending=True)
                keep = (torch.cumsum(sp, dim=-1) - sp) <= top_p     # always keep the top token
                sp = sp * keep
                probs = torch.zeros_like(probs).scatter(0, si, sp)
                probs = probs / probs.sum().clamp_min(1e-9)
            nxt = int(torch.multinomial(probs, 1))
        if eos_id is not None and nxt == eos_id:
            break
        cur.append(nxt); out.append(nxt)
        if on_token is not None:
            on_token(out)
        if stop_text and detok is not None:
            grown = detok(out)
            if any(grown.endswith(marker) for marker in stop_text):
                break
    if hasattr(steps, "close"):
        steps.close()
    return out


def main():
    args = parse_args()
    device = resolve_device(args.gpu)
    # SEEDED ONLY WHEN A SEED WAS ASKED FOR. Seeding unconditionally is what made every session
    # identical; leaving torch's own entropy alone is what makes two runs of the same prompt two
    # different replies, which is what sampling is for. The seed actually used is printed, so a
    # session worth keeping can be reproduced with --seed <that number>.
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        C.info(f"[chat] seed {args.seed} (--seed)")
    else:
        C.info(f"[chat] seed {torch.initial_seed()} (from the OS)")
    # THE CEILING REACHES model/prefix_cache.py THROUGH config.INFERENCE, which is where it
    # reads it from at call time -- before new_cache() sizes anything.
    config.INFERENCE["prefix_cache_size"] = args.prefix_cache_size

    sd, cfg, ck = read_checkpoint(args.checkpoint)
    model, tok, max_len = build_from_checkpoint(sd, cfg, args, device, ck,
                                               path=args.checkpoint)
    if sd is not None:
        miss, unexp = model.load_state_dict(sd, strict=False)
        C.info(f"[chat] loaded {args.checkpoint} "
               f"(missing={len(miss)} unexpected={len(unexp)})")
    model.eval()
    eos_id = getattr(tok, "eos_token_id", None)
    # <|end|> ENDS A REPLY, AND <|endoftext|> ENDS A DOCUMENT. Stage 7 supervises <|end|> at the
    # close of every assistant turn, so it is the token a fine-tuned model emits to say it has
    # finished; a pretrain checkpoint has neither trained it nor, before the specials existed,
    # any id for it, and keeps stopping where it always did.
    from helpers.utils import stop_token_id
    eos_id = stop_token_id(tok)

    # resolve the prompt template
    tmpl = args.template
    if not tmpl:
        # THE FORMAT THE CHECKPOINT WAS TRAINED IN, ASKED OF THE TOKENIZER THAT CAME WITH IT.
        # A model fine-tuned under the message delimiters must be prompted with them; one
        # trained before they existed has no such ids and falls through to what it did have.
        # ALL FOUR OR NONE. The template writes <|startofmessage|> and <|endofmessage|>, so a
        # tokenizer carrying neither encodes them AS ORDINARY TEXT -- the subwords of "<", "|",
        # "startofmessage", "|", ">" -- and the model is handed a cue it has never seen,
        # silently. The whole set is required rather than the two the template uses, because
        # CHAT_SPECIALS is what stage 7 registers: a tokenizer with some of it is not a
        # tokenizer this scheme wrote, and is prompted as whatever it was trained as instead.
        #
        # WHAT A CHECKPOINT WITHOUT THEM FALLS TO IS "text", NOT "none". A pretrain or benchmark
        # model has no instruction tuning, but it has read dialogue, and a bare prompt gives it
        # nothing to tell it that a reply is wanted rather than more of the same document --
        # "What is a lattice?" continues as another question. The text template supplies the turn
        # structure the specials would have supplied. `--template none` is still there for raw
        # continuation, which is a different thing to want and is now asked for rather than
        # arrived at.
        from helpers.utils import CHAT_SPECIALS
        have = getattr(tok, "special2id", None) or {}
        tmpl = ("chat" if all(t in have for t in CHAT_SPECIALS)
                else args.dataset if args.dataset in ("hh", "shp") else "text")

    # THE CONVERSATION, not the last turn of it. `history` is [(you, model), ...] and every
    # prompt is built from the WHOLE of it: a model asked "and why?" with only those two words
    # for context has been given an unanswerable question, and answering it anyway is the
    # failure that reads as the model being stupid rather than as the harness being wrong.
    history = []

    def wrap(text):
        """The full transcript, ending at the point the model is to continue from.

        THE SHAPE IS THE ONE THE CHECKPOINT WAS TRAINED IN, which is not a detail: stage 7
        trains on it, stage 8 scores it, stages 9 and 11 roll out from it. Prompting here in any
        other shape would be asking the model a question in a format it was trained out of. A
        template of "none" keeps the turns separated by blank lines, which is the most a model
        with no instruction tuning can be given."""
        # A COMPLETED TURN IS CLOSED, THE OPEN ONE IS NOT: TURN_END goes after the assistant's
        # answer in every PAST turn and never after the prompt, which is where the training
        # document has it.
        close = TURN_END.get(tmpl, "")
        # WHAT SEPARATES THE CUE FROM THE ANSWER, AND IT IS THE TRAINING DOCUMENT'S ANSWER. The
        # "chat" template ends at `Assistant:\n` and helpers.chat_index puts the reply straight
        # after that prefix with nothing in between, so a space here would prompt in a shape
        # stage 7 never trained. The text templates end at a bare "Assistant:" and do take one.
        sep = "" if tmpl in ("chat", "text") else " "
        past = "".join(TEMPLATES[tmpl].format(p=u) + sep + a + close for u, a in history)
        return past + TEMPLATES[tmpl].format(p=text)

    def fit(ids):
        """The transcript, LEFT-TRUNCATED only if it would leave the reply no room.

        The window is finite and a conversation is not, so eventually something has to go, and
        it is the OLDEST turns -- dropping the end would discard the question being asked.
        Truncation is by token from the left rather than by whole turns, which can cut a turn in
        half; that is the cheap approximation, and it degrades an old turn rather than losing
        the new one.

        THE TRANSCRIPT MAY TAKE AT MOST HALF THE WINDOW, so the reply always has the other half.
        That is a derived rule and not a knob: there is no length cap to set here any more, and
        a conversation that has grown to fill the context would otherwise leave the model a
        handful of tokens to answer in and look as though it had stopped talking. Reported when
        it happens, because a model that has quietly forgotten the start of the conversation
        looks exactly like a model that is ignoring it."""
        if not max_len or len(ids) <= max_len // 2:
            return ids
        room = max(max_len // 2, 1)
        C.info(f"[chat] transcript {len(ids):,} tokens; keeping the last {room:,} "
               f"of a {max_len:,} window")
        return ids[-room:]

    def reply(text):
        """Generate, PRINTING AS IT GOES, and return what was said.

        THE BUDGET IS --response_length, WHICH IS PER REPLY AND NOT THE CONTEXT.
        The model stops at <|endoftext|>, at that many new tokens, or when the context is full,
        whichever comes first.

        The stream decodes the whole generation each step and prints only the growth. Decoding
        the one new token instead would be cheaper and wrong: a byte-level BPE splits a
        multi-byte character across tokens, so "é" arrives as two ids and would print as two
        replacement characters."""
        ids = fit(tok(wrap(text), add_special_tokens=False)["input_ids"])
        stops = STOP_TEXT.get(tmpl, ())
        room = max(max_len - len(ids), 1) if max_len else args.response_length
        budget = max(1, min(int(args.response_length), room))
        shown = 0

        def emit(text):
            # COLOURED PER CHUNK rather than opened once and closed at the end: an interrupted
            # generation would otherwise leave the terminal green.
            print(f"{C.MODEL}{text}{C.OFF}", end="", flush=True)

        def stream(out):
            """Print what has become STABLE, which is not the same as what has grown.

                b'\xe4'         -> '\ufffd'
                b'\xe4\xbd'     -> '\ufffd'
                b'\xe4\xbd\xa0' -> '你'

            So a TRAILING run of U+FFFD is held back until the next token resolves it. What is
            printed is always a prefix that will not change; the tail is flushed once the
            generation ends, where a genuinely invalid byte finally does print as U+FFFD."""
            nonlocal shown
            s = decode(tok, out)
            safe = len(s)
            while safe > 0 and s[safe - 1] == "\ufffd":
                safe -= 1
            safe = max(0, safe - held_back(s[:safe], stops))
            if safe > shown:
                emit(s[shown:safe])
                shown = safe

        print(f"{C.MODEL}Model: {C.OFF}", end="", flush=True)
        gen = generate(model, ids, device, budget, args.temperature,
                       args.top_k, args.top_p, eos_id, max_len, on_token=stream,
                       stop_text=stops, detok=lambda out: decode(tok, out))
        # THE TAIL, WHICH THE STREAM WAS HOLDING BACK. Anything still unresolved at the end is
        # not going to resolve, so it prints as it decodes -- including a real U+FFFD. The turn
        # the model started writing for the user is cut here and was never printed.
        final = cut_at(decode(tok, gen), stops)
        if len(final) > shown:
            emit(final[shown:])
        print(flush=True)
        # STORED AS GENERATED, NOT STRIPPED. `history` is what `wrap` rebuilds the transcript
        # from and what the `history` command prints, so stripping here would make the model's
        # own past turns a slightly different text from the one it produced -- and would drop a
        # trailing blank line from a reply that had ended a code block or a list with one.
        answer = final
        history.append((text, answer))
        return answer

    print()
    announce_cache(model, device)
    summarise(model, tok, ck, cfg, args.checkpoint, device, max_len, args,
              log=lambda m: print(f"{C.INFO}{m}{C.OFF}", flush=True))
    C.info(f"[chat] prompt template: {tmpl}")
    if args.prompt:
        print(f"\n{C.YOU}You:   {args.prompt}{C.OFF}", flush=True)
        reply(args.prompt)                     # prints itself, token by token
        return
    C.info("Type a prompt. 'reset' forgets the conversation, 'history' shows it, "
           "Ctrl-D or 'exit'/'quit' leaves.")
    while True:
        try:
            text = input(f"\n{C.YOU}You:   ").strip()
        except (EOFError, KeyboardInterrupt):
            print(C.OFF)
            break
        finally:
            sys.stdout.write(C.OFF); sys.stdout.flush()
        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            break
        # A CONVERSATION HAS TO BE ENDABLE WITHOUT ENDING THE PROCESS. Otherwise the only way to
        # ask an unrelated question is to reload the checkpoint, which on a big model is a
        # minute of waiting to clear four lines of context.
        if text.lower() == "reset":
            history.clear()
            C.info("[chat] conversation cleared")
            continue
        if text.lower() == "history":
            if not history:
                C.info("[chat] nothing yet")
            for i, (u, a) in enumerate(history, 1):
                print(f"{C.INFO}[chat] {i}.{C.OFF} {C.YOU}You:   {u}{C.OFF}", flush=True)
                print(f"{C.INFO}[chat]    {C.OFF}{C.MODEL}Model: {a}{C.OFF}", flush=True)
            n = len(tok(wrap(""), add_special_tokens=False)["input_ids"])
            C.info(f"[chat] {len(history)} turns, {n:,} tokens of a {max_len:,} window")
            continue
        reply(text)


if __name__ == "__main__":
    main()
