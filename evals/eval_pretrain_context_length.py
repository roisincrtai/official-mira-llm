"""
evals/eval_pretrain_context_length.py -- does a PRETRAINED model actually use a long context,
and does it still work past the length it was trained at?

    python evals/eval_pretrain_context_length.py
    python evals/eval_pretrain_context_length.py --only mira_llm
    python evals/eval_pretrain_context_length.py --no-resume     # measure everything again
    python evals/eval_pretrain_context_length.py --plot_only

    outputs/plots/evaluation/pretrain_context_generalization.pdf
    outputs/eval/pretrain_context_generalization.json
    cache/evals/eval_context_length/<model>.json                      # measured points, for resume

MEMORY IS BUDGETED, NOT HOPED FOR. --max_vram (20 GiB) is what the whole thing may occupy:
weights, the KV cache, and the pass in flight. The sequence goes through the cache in chunks,
and the CHUNK IS SOLVED FOR THE BUDGET AT EACH LENGTH -- smaller at 32k than at 512, because by
then the cache has taken the room. A length whose cache ALONE exceeds the budget is reported
unreachable before anything is allocated, rather than after the allocator says so.

IT RESUMES, and it is built to: the sweep runs to 32k and takes hours. Every measured point is written to
cache/evals/eval_context_length/ the moment it exists, and a second run continues from there. The
cache key is the checkpoint's NAME and step, the corpus's NAME, the probe grids and the
sampling settings: no path, no device, no timestamp, so two machines agree on what a cached
point means. Change any of them and the file is discarded rather than merged into.

FOR BASE MODELS, WHICH MEANS EVERYTHING IS SCORED BY LIKELIHOOD AND NOTHING IS GENERATED. A
pretrained model cannot be asked a question -- it has never been shown that answering is what
one does with a question -- so every long-context benchmark built on instructions (LongBench,
InfiniteBench, RULER's QA half) measures instruction tuning and reports the absence of it as
the absence of context. The three probes here ask only what a language model is: how surprised
is it by these exact bytes.

    1. CONTEXT CURVE     the same target span, preceded by 512, 1k, 2k ... 32k TOKENS of its
                         own document. If the curve keeps falling, the model is USING the extra
                         context; if it flattens at 512 while being fed 8,192, it merely
                         ACCEPTS it. This is the whole distinction, and one number cannot make
                         it -- only the same targets under a growing prefix can.

    2. COPY AT DISTANCE  a random passage, then d tokens of filler, then THE SAME PASSAGE. The
                         second copy is free information: a model that can reach back d tokens
                         pays almost nothing for it, one that cannot pays full price. No
                         semantics, no vocabulary effects, no corpus confound -- it isolates
                         retrieval from language modelling, which is exactly what a positional
                         scheme is or is not providing.

    3. PASSKEY           a five-digit key buried at a known depth, and the continuation
                         "The pass key is". Scored as the likelihood of the digits, never as
                         generated text: a weak model that puts 30% on the right key scores 30%
                         here and zero under exact match, and the graded number is the one that
                         still means something at 133M.

THE COMPARISON SET IS THE README'S TABLE -- every compact model in it that exists on the Hub,
smallest first: TinyStories 1M/8M/33M, Pythia-70M, GPT-2 small, SmolLM2-135M, Gemma 3 270M,
Qwen3-0.6B-Base, Qwen2.5-0.5B, TinyLlama v1.1. So the table stops being a literature survey
and becomes a measurement.

EVERY FIGURE IS NATS PER BYTE OVER AN IDENTICAL SPAN OF CHARACTERS. Those models have as many
tokenizers between them, and perplexity is per TOKEN: a vocabulary that cuts text into fewer,
longer pieces reports a lower perplexity for identical predictions. Nats per byte divides that
out -- the same sentence is the same number of bytes to everyone -- and it is the only unit in
which these rows may share an axis.

THREE ANSWERS TO ONE QUESTION, which is why the set is worth its cost. A LEARNED POSITION TABLE
IS A WALL: GPT-2 has 1,024 rows and no row 1,025, so past it the model cannot be evaluated at
all, and the sweep reports that rather than a bad number. RoPE IS A SLOPE: the rotary models
run past their training length and degrade. MiraLLM refers to no position index anywhere, so
its sweep ends where memory ends and not where the architecture does.

THE CORPUS IS A HELD-OUT SPLIT, AND ONLY A HELD-OUT SPLIT. --data_dir defaults to
data/download/llm-pretrain-wikitext103, of which only validation-*.parquet and
test-*.parquet are read; a corpus with no such files is REFUSED rather than scored. It is not
the pretraining corpus, deliberately: FineWeb-Edu ships as one undifferentiated set of shards
and every one of them is trained on, so scoring there would measure memorisation and report it
as generalisation. That the held-out text comes from a different distribution is a feature --
the claim under test is about LENGTH, not about domain.

AND THE SAME THREE NUMBERS ARE REPORTED ON EACH EVALUATION TARGET -- llm-hellaswag,
llm-lambada and llm-pretrain-wikitext103 -- by the same scoring code, at whatever length each
target's own documents are. That is a FLAT reading, not a curve: one perplexity, one token
accuracy and one nats-per-byte per (model, target), which is what "report perplexity and token
accuracy on these three sets" asks for and all three corpora can support. The context curve
above needs documents long enough to fill 32,768 tokens and only a long-document corpus has
those; HellaSwag items are two sentences.

WHAT THAT NUMBER IS, ON THE TWO NEW TARGETS, SO NOBODY COMPARES IT TO A LEADERBOARD. HellaSwag
is published as ACCURACY OVER FOUR CANDIDATE ENDINGS and LAMBADA as ACCURACY ON THE FINAL WORD;
neither is what is computed here. What is computed here is TEXT PERPLEXITY -- and token
accuracy, and nats per byte -- over the text those datasets ship, exactly as over any other
corpus: HellaSwag's `ctx` premises and LAMBADA's `text` passages, scored whole by likelihood.
It is a language-modelling measurement on those distributions and it is NOT the HellaSwag or
LAMBADA benchmark figure. A model's number here cannot be put beside a published one.

A TARGET THAT IS NOT ON THIS MACHINE IS REPORTED AND SKIPPED, never fatal -- the same rule
every baseline already follows. The held-out split is used where the corpus HAS one; where it
has none, the whole corpus is, and only because these targets appear in no scheme's
PRETRAIN_CORPUS and are therefore unseen by construction. A corpus that IS pretrained on and
ships no split is refused, for the reason the paragraph above gives.

DOCUMENTS ARE USED WHOLE AND NEVER CONCATENATED, and a document that cannot fill the context is
not scored at that length: stitching short articles into a long one manufactures a context
whose distant half cannot be exploited by anyone, and a curve drawn over it is flat for reasons
that have nothing to do with the models. Where no document is long enough the sweep stops and
says that the CORPUS ended it.

Nothing here is fetched by this script except the comparison weights, through transformers'
own cache (config.MODEL_DIR). A model that cannot be loaded is SKIPPED AND REPORTED, never
fatal: an evaluation that dies because one baseline is missing has wasted the others.
"""
import os
import sys

# RUNNABLE AS A SCRIPT, not only as a module. `python evals/<name>.py` is what a person
# types, and without this it fails on `import default_config` -- the project root is on
# the path when python is given a module (-m) and is NOT when it is given a file.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import argparse
import gc
import json
import math
import random

import torch

import default_config as config
from model import prefix_cache
from helpers.utils import display_name, model_name, progress
import helpers.utils as helpers          # spec_mira_llm calls helpers.arch_of
from model import MiraLLM
from tokenizer import BPETokenizer

# THE COMPARISON SET IS THE README'S TABLE, every row of it that exists on the Hub -- so the
# table stops being a literature survey and becomes a measurement. SMALLEST FIRST, because the
# cheap rows then land in the figure within minutes and a run stopped early still compares
# something.
#
# BASE CHECKPOINTS ONLY. An instruct model is a different experiment, and a chat model's
# likelihoods are shaped by a template that is not present here -- scoring raw prose against a
# model tuned to expect a chat header measures the header.
#
# `Baby GPT (character)` is in the table and NOT here: it is nanoGPT's character-level demo,
# trained per-reader rather than published, so there is nothing to fetch. Gemma 3 is gated on
# the Hub and will be skipped with a report unless `huggingface-cli login` has been run --
# which is exactly what a skipped baseline is for.
# THE TINYSTORIES MODELS ARE NOT IN HERE, and that is a judgement about what a baseline IS.
# They were trained on synthetic three-year-old's stories with a vocabulary of a few thousand
# words -- so on WikiText they sit at 1.9-2.1 nats/byte and 4-9% accuracy, FLAT at every
# context length, with copy gains that scatter around zero and passkey scores worse than a
# uniform guess. None of that measures a positional scheme; it measures a distribution
# mismatch, and a curve that cannot move cannot show whether context is being used.
#
# A long-context comparison needs models that read the text at all. Their numbers stay in the
# cache and in the README's table; they are not drawn, and they are not measured again.
HIDDEN = ("tinystories-1m", "tinystories-8m", "tinystories-33m")

# A BASELINE IS STRONG HERE IF IT CONTESTS THE CLAIM, which means two things: it must read
# English web text well enough for its curve to MOVE, and its positional scheme must be one
# somebody would defend. Sorted by that second axis, the set covers all three:
#
#   learned absolute table   gpt2 -- a wall at 1,024, and what interpolating it costs
#   RoPE                     pythia 70m and 160m, SmolLM2, Qwen, Gemma, TinyLlama
#   ALiBi                    bloom-560m -- THE SCHEME DESIGNED TO EXTRAPOLATE
#
# ALiBi IS THE ONE THAT MATTERS. Beating a learned table past its rows is arithmetic, and
# beating RoPE past its base is well documented; ALiBi biases attention by distance precisely
# so that a model trained at n keeps working past n, and it is the only rival that claims what
# MiraLLM claims. BLOOM is multilingual, so its LEVEL on English is inflated -- but this figure
# compares SLOPES past a training window, and a level offset does not touch a slope.
#
# pythia-160m is the exact size match for mira-s AND the same family as pythia-70m, so data,
# tokenizer and recipe are held constant across a size step -- the cleanest RoPE control there
# is. SmolLM2-360M is the strongest sub-500M model published, trained on 4T tokens at a native
# 8,192: the baseline that is not supposed to be caught.
BASELINES = {
    "pythia-70m": "EleutherAI/pythia-70m",               # 70.4M,  RoPE,      2,048
    "gpt2": "gpt2",                                      # 124M,   learned,   1,024
    "pythia-160m": "EleutherAI/pythia-160m",             # 162M,   RoPE,      2,048
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",        # 134.5M, RoPE,      8,192
    "gemma3-270m": "google/gemma-3-270m",                # 268.1M, RoPE,      32,768
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",        # 362M,   RoPE,      8,192
    "qwen": "Qwen/Qwen2.5-0.5B",                         # 0.5B,   RoPE,      32,768
    "bloom-560m": "bigscience/bloom-560m",               # 559M,   ALiBi,     2,048
    "qwen3-0.6b": "Qwen/Qwen3-0.6B-Base",                # 0.6B,   RoPE,      32,768
    "tinyllama": "TinyLlama/TinyLlama_v1.1",             # 1.1B,   RoPE,      2,048
}
# THE NATIVE CONTEXT WINDOW OF EACH BASELINE -- where its circle goes. Normally this comes off
# the loaded model's config and is stored with its points; this table is what a row rebuilt
# from a bucket written before that used instead of having no circle at all. It is the same
# number the README's table gives.
# THE POSITIONAL SCHEME OF EACH BASELINE -- what goes in its legend entry. Normally read off
# the loaded model; this is the fallback for a row rebuilt from a bucket written before the
# scheme was stored, and the authority for ALiBi, which no config announces in a single field.
NATIVE_PE = {
    "pythia-70m": "rope", "gpt2": "learned", "pythia-160m": "rope", "smollm2-135m": "rope",
    "gemma3-270m": "rope", "smollm2-360m": "rope", "qwen": "rope", "bloom-560m": "alibi",
    "qwen3-0.6b": "rope", "tinyllama": "rope",
}
NATIVE_CONTEXT = {
    "pythia-70m": 2048, "gpt2": 1024, "pythia-160m": 2048, "smollm2-135m": 8192,
    "gemma3-270m": 32768, "smollm2-360m": 8192, "qwen": 32768, "bloom-560m": 2048,
    "qwen3-0.6b": 32768, "tinyllama": 2048,
}
OUT_PDF = os.path.join(config.PLOT_DIR, "evaluation", "pretrain_context_generalization.pdf")
OUT_JSON = os.path.join(config.OUTPUT_DIR, "eval", "pretrain_context_generalization.json")

# THE PALETTE. Eleven lines on one axis need to be told apart by MORE THAN HUE, so each model
# gets a colour, a marker and a dash pattern, and keeps all three in every panel -- a reader who
# has found a line in one panel finds the same line in the next without re-reading the legend.
#
# A COLORMAP WAS THE WRONG INSTRUMENT. Sampling `cividis` at eleven points is a SEQUENTIAL scale
# cut into categories: neighbouring models came out as neighbouring shades of the same
# blue-to-yellow ramp, indistinguishable in the middle and ugly throughout. A sequential
# colormap encodes an ordered quantity; these are eleven different objects, which is what a
# QUALITATIVE palette is for.
#
# The colours are Okabe-Ito's colourblind-safe set extended with four more of the same
# saturation, chosen to stay separable under deuteranopia and to survive being printed in
# greyscale -- where the markers and dashes carry the distinction on their own.
OWN = "#12386B"            # MiraLLM: a deep navy nothing else is near, drawn heavier and on top
PALETTE = [
    "#E69F00",             # orange
    "#56B4E9",             # sky blue
    "#009E73",             # bluish green
    "#D55E00",             # vermillion
    "#CC79A7",             # reddish purple
    "#6A51A3",             # violet
    "#8C6D31",             # olive brown
    "#17A2A2",             # teal
    "#B03060",             # maroon
    "#5A5A5A",             # graphite
]
MARKERS = ["s", "^", "v", "D", "P", "X", "<", ">", "*", "h"]

# THE LINE STYLE IS NOT IDENTITY, IT IS MEANING. Solid up to a model's own training length,
# dashed past it, and a ring on the boundary. Every curve then says where it stops being
# interpolation and starts being extrapolation, without the reader holding eleven training
# lengths in mind or counting dotted verticals. Identity is carried by colour and marker, which
# is enough for eleven lines and leaves the dashes free to mean something.
SOLID, BEYOND = "-", (0, (4.5, 1.8))


def style_of(key, i):
    """Colour, marker and dash for one model -- the same three in every panel.

    MiraLLM is solid, navy, round-markered and on top; a baseline takes the next entry of each
    cycle. Three channels rather than one means the figure survives a colourblind reader, a
    greyscale printer, and two lines that happen to cross."""
    if key == "mira_llm":
        return {"color": OWN, "marker": "o", "lw": 2.6, "ms": 5.0, "zorder": 5,
                "markeredgecolor": "white", "markeredgewidth": 0.6}
    j = i % len(PALETTE)
    return {"color": PALETTE[j], "marker": MARKERS[j], "lw": 1.3, "ms": 3.4, "zorder": 2}


CACHE_DIR = os.path.join(config.CACHE_DIR, "evals", "eval_context_length")

# WHAT AN ALLOCATOR SAYS WHEN IT CANNOT ALLOCATE, on each backend. There is no common exception
# type and no common message: CUDA raises OutOfMemoryError and says "out of memory", MPS raises
# a plain RuntimeError and says "Invalid buffer size: 19.44 GiB", and the CPU allocator says
# "can't allocate memory" or raises MemoryError outright.
#
# THIS LIST IS NOT COSMETIC. Attention materialises a (heads x T x T) matrix, so the sweep is
# GUARANTEED to hit the ceiling at some length -- that is what the last point of every curve
# IS. Failing to recognise the message turns the expected end of a curve into a traceback that
# loses the whole run, which is exactly what happened at 27k tokens on MPS: 8 x 27k x 27k x 4
# bytes is 19.44 GiB, reported in words this did not know.
_ALLOC_MSGS = ("out of memory", "invalid buffer size", "can't allocate", "cannot allocate",
               "failed to allocate", "not enough memory", "insufficient memory",
               "mps backend out of memory")


def is_alloc_error(e):
    return isinstance(e, MemoryError) or any(m in str(e).lower() for m in _ALLOC_MSGS)


# --------------------------------------------------------------------------- #
# the models, behind one interface
# --------------------------------------------------------------------------- #
def _finish(spec, model, log):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    spec["model"] = model
    spec["params"] = sum(p.numel() for p in model.parameters())
    log(f"[context] {spec['key']:<10} {spec['params'] / 1e6:7.1f}M params  "
        f"trained at {spec['train_len']:,}  "
        f"{'hard position limit ' + format(spec['max_pos'], ',') if spec['max_pos'] else 'no position limit'}")
    return spec


def spec_mira_llm(args, device, dtype, log):
    """The pipeline's own checkpoint, rebuilt from the architecture stored inside it."""
    path = args.checkpoint
    if not os.path.isfile(path):
        log(f"[context] mira_llm skipped: no checkpoint at {path}")
        return None
    # saved_models/ FIRST, THEN checkpoints/ -- tokenizer.run.base_path. config.BPE_PATH
    # names only the checkpoints/ copy, and the base vocabulary is published as
    # tokenizer-bpe-fineweb-edu-2BT, which is where stage 1 puts it.
    from tokenizer.run import base_path
    bpe_file = base_path(log)
    if not os.path.isfile(bpe_file):
        log(f"[context] mira_llm skipped: no tokenizer at {bpe_file} (run stage 3)")
        return None
    tok = BPETokenizer.load(bpe_file)
    # A CHECKPOINT THAT WILL NOT LOAD IS A SKIPPED MODEL, not a dead run. Every baseline is
    # already treated that way and there is no reason ours should be the one that takes the
    # evaluation down -- least of all when the likeliest cause is a TRAINING RUN STILL WRITING
    # IT, which is a matter of waiting rather than of anything being wrong.
    try:
        ck = torch.load(path, map_location="cpu")
    except Exception as e:                                        # noqa: BLE001
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        import time
        age = time.time() - os.path.getmtime(path) if size else 0
        log(f"[context] mira_llm skipped: {os.path.relpath(path, config.ROOT)} would not load")
        log(f"[context]   {type(e).__name__}: {str(e).splitlines()[0]}")
        log(f"[context]   {size / 2**30:.2f} GiB, last written {age / 60:.1f} minutes ago")
        if age < 600:
            log(f"[context]   THAT IS RECENT. A pretraining run writes this file every "
                f"--checkpoint_every_steps; the write is atomic (helpers._atomic_torch_save "
                f"renames a temporary over it), so a complete file is always there -- but a "
                f"copy taken while the rename lands can still be short. Try again, or")
        # THE SNAPSHOT GOES IN THE PROJECT'S OWN CACHE, not in /tmp. cache/ is where this
        # project puts everything it can rebuild -- it is on the same filesystem as the
        # checkpoint, it survives a reboot, and it is covered by the same rules as the token
        # streams and these very results. /tmp is a machine's business, not a run's.
        snap = os.path.relpath(os.path.join(config.CACHE_DIR, "evals", "checkpoint_snapshot.pt"),
                               config.ROOT)
        log(f"[context]   evaluate a snapshot, which a live trainer cannot touch:")
        log(f"[context]       mkdir -p {os.path.dirname(snap)} && "
            f"cp {os.path.relpath(path, config.ROOT)} {snap}")
        log(f"[context]       python evals/eval_pretrain_context_length.py --checkpoint {snap}")
        log(f"[context]   If a copy fails the same way the file itself is damaged; "
            f"checkpoints/pretrain/ keeps only the newest, so the run must be resumed from "
            f"its history.")
        return None
    saved = ck.get("model_cfg")
    if not isinstance(saved, dict):
        log("[context] checkpoint architecture metadata is missing")
        return None
    cfg = {str(k).lower(): v for k, v in saved.items()
           if str(k).lower() != "vocab_size"}
    if cfg.get("benchmark_family"):
        log("[context] benchmark checkpoints must use benchmark.evaluate_context_length")
        return None
    experimental_arch = cfg.pop("experimental_arch", None)
    cfg.pop("benchmark_architecture", None)
    cfg.pop("benchmark_family", None)
    if experimental_arch:
        cfg["arch"] = experimental_arch
    if "lattice_order" in cfg:
        config.mira_map_arch(cfg)
        if "arch" in cfg:
            raise ValueError("this MIRA checkpoint records arch instead of lattice_modulus_map and "
                             "lattice_argument_map, the format before those keys, and cannot be loaded")
    else:
        cfg.setdefault("arch", cfg.pop("pe", None) or "ssm")
    if "init_horizon" in cfg and "lattice_init_horizon" not in cfg:
        cfg["lattice_init_horizon"] = cfg.pop("init_horizon")
    if helpers.arch_of(cfg) == "mira-lattice" and "lattice_init_horizon" not in cfg:
        cfg["lattice_init_horizon"] = config.MODEL["lattice_init_horizon"]
    if helpers.arch_of(cfg) == "mira-lattice":
        cfg["lattice_initializer"] = config.canonical_lattice_initializer(
            cfg.get("lattice_initializer", cfg.pop("lattice_init_method", "schur_init")), recorded=True)
    cfg.pop("lattice_norm_module", None)        # lattice_norm_module (retired 2026-10-05; MIRA is RMSNorm only)
    model = MiraLLM(vocab_size=len(tok), **cfg).to(device=device, dtype=dtype)
    model.load_state_dict(ck["model"])
    lattice = helpers.arch_of(cfg) == "mira-lattice"
    if lattice:
        cache_type = prefix_cache.PrefixCache
        new_state = lambda: cache_type(len(model.blocks))
        fixed_state_bytes = sum(
            prefix_cache.state_bytes(1, 1, block.mixer.order, block.mixer.inner,
                                     block.mixer.memory_dimension)
            for block in model.blocks)
    else:
        from helpers.kv_cache import Cache
        cache_type = Cache
        new_state = lambda: cache_type(len(model.blocks))
        fixed_state_bytes = 0
    return _finish({
        "key": "mira_llm",
        # THE SCHEME'S NAME, not its dimensions. `24L-512d` is a description of a model, and
        # this project already has a NAME for that model, derived from default_config.SCHEMES.
        # Two vocabularies for one object is one too many, and the dimensions are in the startup
        # table for anyone who wants them.
        #
        # THE DISPLAY FORM, BECAUSE THIS ONE IS READ BY A PERSON. `mira-s` is the IDENTIFIER
        # -- what model_name returns, and what every checkpoint and history file is named after --
        # but a figure legend is not a filename, and the codebase's name for itself is noise in
        # one. display_name maps it to `Mira-S`; anything no scheme describes comes back
        # unchanged, so a divergence still shows rather than being dressed up.
        "name": f"{display_name(cfg)}"
                + ("" if cfg.get("arch", "ssm") == "ssm" else f" ({cfg.get('arch')})"),
        "encode": lambda t: tok(t, add_special_tokens=False)["input_ids"],
        # MAX_POS = 0 MEANS "NO ARCHITECTURAL LIMIT", which is the claim under test and not a
        # missing value: there is no position table to run out of and no rotary base to
        # extrapolate, so the only ceiling is the memory of the machine.
        # THE WINDOW IT HAS ACTUALLY TRAINED AT, which is what the circle on the curve means.
        # `block_size` is the LONGEST window of the scheme's schedule -- where the run will
        # end, not where it is -- so a checkpoint 82,875 steps into a schedule that is still
        # at 2,048 was being drawn as though it had trained at 8,192, and every point beyond
        # 2,048 was being counted as inside the training window when it is extrapolation.
        # save_ckpt records the scheduled window now; older checkpoints have no such field and
        # fall back to what was assumed before.
        "max_pos": 0,
        "train_len": (args.base or int(ck.get("ctx_window", 0) or 0)
                      or int(cfg.get("block_size", 512))),
        "step": ck.get("step"), "total": ck.get("total"),
        "checkpoint": os.path.relpath(path, config.ROOT),
        "positional": {"ssm": "nope/ssa",
                       "mira-lattice": "nope/mira"}.get(helpers.arch_of(cfg), "rope"),
        # ONE CHUNK OF TOKENS, ATTENDING TO EVERYTHING ALREADY CACHED. helpers.kv_cache.Cache
        # carries four tensors per layer -- attention's keys and values, and the state space
        # module's convolution window and recurrence state -- because a block is SSM ->
        # attention -> FFN and the recurrence would otherwise be re-run over the whole prefix.
        "shape": {"n_layer": int(cfg["n_layer"]),
                  "n_head": 0 if lattice else int(cfg["n_head"]),
                  # no grouped-query attention here: every head keeps its own K and V
                  "kv_per_token": 0 if lattice else 2 * int(cfg["n_embd"]),
                  "fixed_state_bytes": fixed_state_bytes, "vocab": len(tok)},
        "ensure": lambda T: None,      # nothing to grow: no position index exists
        "new_state": new_state,
        "forward_chunk": lambda x, st: (
            model.head(model.hidden_states(input_ids=x, cache=st)),
            st),
# ---- COMMENTED OUT: token-wise positional residue ---------------------------------- #
# Kept, not deleted. Uncomment this block and the four other blocks marked
# 'COMMENTED OUT: token-wise positional residue' to put the probe back.
#         # THE TRUNK'S OUTPUT, BEFORE THE VOCABULARY PROJECTION -- what the residue is measured
#         # on. The head is a fixed linear map of it, so it carries the same positional content
#         # in d numbers per token instead of |V|; at 32,768 tokens that is the difference
#         # between 268 MB and 6 TB, and the residue is a ratio, so the projection would cancel
#         # in the leading order anyway.
#         "hidden_chunk": lambda x, st: (model.hidden_states(input_ids=x, cache=st), st),
    }, model, log)


def extend_positions(model, target, log):
    """Stretch a learned position table to `target` rows, so a model with a wall can be run past
    it. Returns the new limit, or 0 when there is nothing to stretch.

    WHY NOT JUST REFUSE. A learned table stops the sweep at 1,024 and the curve ends there,
    which is true but uninformative: it says the model was BUILT for 1,024, not what happens to
    a learned scheme asked for more. Interpolating the table is the standard way to ask, and it
    is the comparison that matters -- MiraLLM needs no such operation, and the point is what the
    operation COSTS.

    THE ROWS ARE INTERPOLATED, NOT APPENDED. Appending fresh rows would hand the model positions
    it has never seen, which is not an extension but a partial re-initialisation. Linear
    interpolation instead RESCALES the table: old row i lands at i x target / n, and the
    positions in between are averages of learned neighbours. Position 4,000 of 8,192 then means
    what position 500 of 1,024 meant -- the same fraction of the way through -- which is
    Position Interpolation's idea, applied to a table rather than to a rotation.

    THE CAUSAL MASK, WHERE THERE IS ONE. GPT-2 and GPT-Neo carry a lower-triangular `bias`
    buffer sized to the original length, and it must match the new one or the forward fails on
    its own shapes:

        RuntimeError: The size of tensor a (2048) must match the size of tensor b (2560)

    So the models are loaded with attn_implementation="sdpa", where the mask is a FLAG to
    scaled_dot_product_attention and no buffer exists to go stale. Only where sdpa is
    unavailable is a buffer rebuilt, and then as BOOL -- one byte an entry rather than four.

    THAT REBUILD IS QUADRATIC AND IS NOT IN THE MEMORY BUDGET. At 32,768 a bool mask is 1 GiB
    per buffer, and a model with one per layer pays that per layer, on top of everything
    plan_chunk accounted for. It is reported when it happens. On the sdpa path -- which is every
    model in BASELINES on a current transformers -- none of it occurs.

    THIS CHANGES THE MODEL, and the run says so in the log every time it happens. It is not put
    in the legend: the legend identifies a row, and a note about method in that slot is both
    longer than the panel and the same note for every learned table."""
    import torch.nn as nn
    emb = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding) and name.split(".")[-1] in (
                "wpe", "embed_positions", "position_embeddings"):
            emb = (name, mod)
            break
    if emb is None:
        return 0
    name, mod = emb
    n_old, d = mod.weight.shape
    if target <= n_old:
        return n_old
    # A DENSE CAUSAL MASK IS REBUILT, NOT USED AS A REASON TO REFUSE. With sdpa there is none --
    # the mask is a flag to scaled_dot_product_attention. An eager path indexes a real buffer,
    # so it is rebuilt as BOOL (one byte an entry, not four) at the length being asked for.
    impl = str(getattr(getattr(model, "config", None), "_attn_implementation", "")).lower()
    dense = [(m, bn, b) for m in model.modules()
             for bn, b in m.named_buffers(recurse=False)
             if b is not None and b.dim() == 4 and b.shape[-1] == n_old]
    if dense and impl not in ("sdpa", "flash_attention_2"):
        for m, bn, b in dense:
            m.register_buffer(bn, torch.tril(torch.ones(
                target, target, dtype=torch.bool, device=b.device)).view(1, 1, target, target))
        log(f"[context]            {len(dense)} dense causal mask(s) rebuilt at {target:,} "
            f"as bool ({target * target * len(dense) / 2**30:.2f} GiB)")
    w = mod.weight.data.detach().float().t().unsqueeze(0)          # (1, d, n_old)
    w = torch.nn.functional.interpolate(w, size=target, mode="linear", align_corners=True)
    # DEVICE AND DTYPE BOTH. nn.Embedding is created in the default fp32; moving only the
    # device would leave an fp32 table inside a bf16 model, and the first forward would fail on
    # a dtype it never had to see. --dtype bf16 was the only run that would have found this.
    new = nn.Embedding(target, d).to(device=mod.weight.device, dtype=mod.weight.dtype)
    new.weight.data.copy_(w.squeeze(0).t().to(mod.weight.dtype))
    parent = model
    for part in name.split(".")[:-1]:
        parent = getattr(parent, part)
    setattr(parent, name.split(".")[-1], new)

    if hasattr(model, "config"):
        model.config.max_position_embeddings = target
        if hasattr(model.config, "n_positions"):
            model.config.n_positions = target
    log(f"[context]            position table {n_old:,} -> {target:,} rows by linear "
        f"interpolation" + ("" if dense and impl not in ("sdpa", "flash_attention_2")
                            else f" ({impl or 'eager'}: no dense causal mask to rebuild)"))
    return target


def spec_hf(key, repo, device, dtype, log, args):
    """A Hugging Face base model, or None. A baseline that cannot be loaded is reported and
    dropped -- the run continues with whichever rows it has."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:                                        # noqa: BLE001
        log(f"[context] {key} skipped: transformers not importable ({e})")
        return None
    try:
        tk = AutoTokenizer.from_pretrained(repo, cache_dir=args.model_dir)
        # SDPA WHERE IT EXISTS. scaled_dot_product_attention takes the causal mask as a flag,
        # so no (T x T) buffer is built and no (T x T) score matrix is materialised either --
        # which is what makes a 32k forward possible at all. Older architectures without an
        # sdpa path fall back to eager, and extend_positions then declines to stretch them.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo, cache_dir=args.model_dir, attn_implementation="sdpa")
        except Exception:                                          # noqa: BLE001
            model = AutoModelForCausalLM.from_pretrained(repo, cache_dir=args.model_dir)
    except Exception as e:                                        # noqa: BLE001
        log(f"[context] {key} skipped: could not load {repo!r} ({e})")
        return None
    tk.model_max_length = int(1e9)          # it warns per sequence otherwise; long is the point
    model = model.to(device=device, dtype=dtype)
    c = model.config
    n_pos = int(getattr(c, "n_positions", 0) or getattr(c, "max_position_embeddings", 0) or 0)
    # A LEARNED POSITION TABLE IS FOUND BY LOOKING FOR ONE, not by recognising a family name.
    # This tested `model_type.startswith("gpt2")`, which is false for GPT-Neo -- and TinyStories
    # is GPT-Neo, with a `wpe` table of 2,048 rows exactly like GPT-2's. It was therefore
    # treated as rotary, run out to 2,560, and died inside its own causal mask:
    #     RuntimeError: The size of tensor a (2048) must match the size of tensor b (2560)
    # A parameter named wpe / embed_positions / position_embeddings IS the table, whatever the
    # architecture is called, and every rotary model lacks all three.
    learned = any(("wpe" in n) or ("embed_positions" in n) or ("position_embeddings" in n)
                  for n, _ in model.named_parameters())
    # AND ALiBi IS NOT RoPE. "no learned table" was being read as rotary, which is right for
    # nine of these and wrong for BLOOM: ALiBi has no table AND no rotation, it biases attention
    # by distance -- and it is the one scheme in the set that claims to extrapolate, so calling
    # it RoPE in the legend would hide the comparison the baseline was added for.
    mt = str(getattr(c, "model_type", "")).lower()
    pe = ("learned" if learned else
          "alibi" if (mt in ("bloom", "mpt") or bool(getattr(c, "alibi", False)))
          else NATIVE_PE.get(key, "rope"))
    native = n_pos                     # what it was TRAINED at, whatever is done to it after
    state = {"rows": n_pos}
    grows = bool(learned and args.extend_positions)

    def ensure(T):
        """Make the model able to take T tokens. ON DEMAND, and never a refusal.

        AN AUTOREGRESSIVE MODEL HAS NO INHERENT LENGTH. A learned position table is a table and
        a table can be stretched; refusing the length instead was a limit this script invented
        and then went on to measure. Every length asked for is attempted, and the table grows to
        meet it -- to the next power of two past T, so a sweep of doubling lengths interpolates
        once per doubling rather than once per point."""
        if not grows or T <= state["rows"]:
            return
        # NO CAP. A cap is a limit this script would be imposing on an autoregressive model,
        # which has none of its own; the table grows to the next power of two past whatever is
        # asked for, and the only thing that can stop a curve is memory or the corpus.
        want = 1 << max(T - 1, 1).bit_length()
        state["rows"] = max(state["rows"], extend_positions(model, want, log) or state["rows"])
    trunk = getattr(model, "transformer", None) or getattr(model, "model", None)
    return _finish({
        "key": key,
        # THE NAME IS THE MODEL'S NAME. I had it carrying "(positions interpolated)", which is
        # a note about method in a slot that exists to identify a row -- and at eleven rows it
        # was longer than the panel. What was done to a learned table is the same for every
        # learned table, is stated in the log when it happens, and belongs in the caption.
        "name": repo.split("/")[-1],
        "encode": lambda t: tk(t, add_special_tokens=False)["input_ids"],
        # NOTHING IS REFUSED FOR ITS LENGTH. A learned table is stretched by `ensure` when a
        # longer sequence arrives and a rotary model needs nothing, so the only things that can
        # end a curve are MEMORY and the CORPUS -- both of which are facts about the run rather
        # than limits this script imposed. There is no switch to put the wall back: refusing a
        # length would be measuring this script rather than the model.
        #
        # TRAIN_LEN IS THE NATIVE WINDOW AND STAYS THERE. Stretching a table does not train a
        # model; reporting "trained at 32,768" of a model trained at 2,048 would put the
        # extrapolation marker in the wrong place and hide the entire finding.
        "max_pos": 0 if grows else (n_pos if learned else 0),
        "train_len": native or 2048, "ensure": ensure,
        # A SECOND TRUNCATION, AND THE LAST ONE. score_ids treats a forward that raises BEYOND
        # this length as the model refusing it -- a fact to plot -- and re-raises the same
        # error inside it as the bug it is. Held at the ORIGINAL 2,048 it went stale the moment
        # the table grew: a genuine failure at 8,192 was then swallowed as "past its declared
        # 2,048" and the curve stopped there, on a model that had just been given 8,192 rows.
        # A model that grows declares NOTHING, so any failure is either memory (reported) or a
        # bug (raised), and neither is quietly turned into a shorter curve.
        "declared_max": 0 if grows else n_pos,
        "step": None, "total": None, "checkpoint": repo,
        "positional": pe,
        # transformers' OWN incremental path: past_key_values in, past_key_values out. Same
        # reason as above -- the attention matrix becomes (chunk x T) instead of (T x T).
        "shape": {
            "n_layer": int(getattr(c, "num_hidden_layers", None) or getattr(c, "n_layer", 12)),
            "n_head": int(getattr(c, "num_attention_heads", None) or getattr(c, "n_head", 12)),
            # GROUPED-QUERY ATTENTION MAKES THE CACHE SMALLER THAN THE HEAD COUNT SUGGESTS:
            # Qwen2.5-0.5B has 14 query heads and 2 key/value heads, so its cache is a seventh
            # of what num_attention_heads would imply. Reading num_key_value_heads is the
            # difference between a usable estimate and a seven-fold overestimate.
            "kv_per_token": 2 * int(getattr(c, "num_key_value_heads", None)
                                    or getattr(c, "num_attention_heads", None)
                                    or getattr(c, "n_head", 12))
            * int((getattr(c, "hidden_size", None) or getattr(c, "n_embd", 768))
                  // max(int(getattr(c, "num_attention_heads", None)
                             or getattr(c, "n_head", 12)), 1)),
            "vocab": int(getattr(c, "vocab_size", 50257))},
        "new_state": lambda: None,
        "forward_chunk": lambda x, st: (lambda o: (o.logits, o.past_key_values))(
            model(input_ids=x, past_key_values=st, use_cache=True)),
# ---- COMMENTED OUT: token-wise positional residue ---------------------------------- #
# Kept, not deleted. Uncomment this block and the four other blocks marked
# 'COMMENTED OUT: token-wise positional residue' to put the probe back.
#         # The LAST hidden state, which is the trunk's output: the same object MiraLLM's
#         # hidden_chunk returns, so the residue compares like with like across architectures
#         # whose vocabularies have nothing in common.
#         "hidden_chunk": lambda x, st: (lambda o: (o.hidden_states[-1], o.past_key_values))(
#             model(input_ids=x, past_key_values=st, use_cache=True,
#                   output_hidden_states=True)),
    }, model, log)


# --------------------------------------------------------------------------- #
# scoring: nats over an exact span, and the bytes that span covers
# --------------------------------------------------------------------------- #
def plan_chunk(spec, T, budget_bytes, itemsize, cap, log=None):
    """How many tokens may go through one forward pass, so that the whole thing fits `budget`.

    THE CHUNK IS NOT A CONSTANT, because what it has to fit is not. Three things are resident
    while a chunk is scored, and only the first two grow with the CONTEXT:

        weights            n parameters x itemsize                  fixed
        the KV cache       2 x layers x T x d_kv x itemsize         LINEAR IN T, and unavoidable
        the chunk itself   heads x chunk x T x itemsize (scores)    linear in BOTH
                           + chunk x vocab x 4        (logits)

    So the budget left for a pass shrinks as T grows, and a chunk of 512 that is comfortable at
    8k is four times too large at 32k. Solving for the chunk:

        chunk = (budget - weights - cache) / (2 x heads x T x itemsize + 4 x vocab)

    with the 2 because the scores and their softmax are alive together. THE CACHE IS THE HARD
    FLOOR: at 32k it is gigabytes on its own, and when it alone exceeds the budget no chunk
    size helps -- 0 is returned and the length is reported unreachable BEFORE anything is
    allocated, rather than after the allocator says so.

    Returns (chunk, reason). `chunk` is 0 when nothing fits, and `reason` says WHICH THING RAN
    OUT -- which matters because they are not the same finding:

        "weights"   the model does not fit the budget AT ANY LENGTH. Nothing to do with
                    context; it would fail identically at 512. Reported once and the model
                    skipped, NOT drawn as a curve that ends early.
        "cache"     the KV cache for THIS length exceeds what is left. A genuine limit of the
                    length, and the point the curve should stop at.
        ""          it fits.

    CHUNKING IS A TECHNIQUE, NOT A CEILING. The chunk decides how much is resident during one
    pass; it decides nothing about how long a context may be. Conflating the two would let a
    budget too small for the WEIGHTS be drawn as a model that cannot handle long contexts."""
    sh = spec.get("shape") or {}
    n_layer = sh.get("n_layer", 12)
    n_head = sh.get("n_head", 12)
    kv_tok = sh.get("kv_per_token", 2 * 768)
    vocab = sh.get("vocab", 50257)
    weights = spec.get("params", 0) * itemsize
    cache = sh.get("fixed_state_bytes", 0) + n_layer * T * kv_tok * itemsize
    per_token = 2 * n_head * T * itemsize + 4 * vocab
    if weights >= budget_bytes:
        if log is not None:
            log(f"[context] {spec['key']:<10} plan   weights alone are "
                f"{weights / 2**30:.2f} GiB against a {budget_bytes / 2**30:.2f} GiB budget: "
                f"this model does not fit AT ANY LENGTH")
        return 0, "weights"
    room = budget_bytes - weights - cache
    chunk = int(room // max(per_token, 1))
    if log is not None:
        log(f"[context] {spec['key']:<10} plan   T={T:,}  weights {weights / 2**30:.2f} + "
            f"cache {cache / 2**30:.2f} GiB, {room / 2**30:.2f} left -> chunk "
            f"{max(min(chunk, cap), 0) if chunk > 0 else 0}"
            + ("  (does not fit: the cache alone exceeds the budget)" if chunk <= 0 else
               f", pass ~{min(chunk, cap) * per_token / 2**30:.2f} GiB"))
    return (max(min(chunk, cap), 0), "" if chunk > 0 else "cache")


@torch.no_grad()
def score_ids(spec, pid, sid, device, chunk, desc=""):
    """Score `sid` given `pid`, FEEDING THE SEQUENCE THROUGH THE MODEL'S KV CACHE IN CHUNKS.

    THIS IS WHAT MAKES 32k REACHABLE. A single forward over T tokens materialises attention as
    (heads x T x T): at T = 27,000 and 8 heads that is 19.44 GiB, which is the allocation that
    ended the first run. Fed `chunk` tokens at a time with the keys and values of everything
    before them already in the cache, the matrix is (heads x chunk x T) instead -- at chunk 512
    and T = 32,768, 0.5 GiB. LINEAR IN T RATHER THAN QUADRATIC, for identical arithmetic: each
    position still attends to the whole prefix, it is simply not asked to do so all at once.

    The result is exactly the single-pass result. Causal attention means a position's output
    depends only on positions at or before it, so splitting the sequence changes what is
    resident, never what is computed.

    Returns (nats, tokens, correct, T, ok)."""
    T = len(pid) + len(sid)
    # GROW THE MODEL TO THE LENGTH rather than turn the length away. A no-op for anything with
    # no position index, and for a table already large enough.
    if spec.get("ensure"):
        spec["ensure"](T)
    if spec["max_pos"] and T > spec["max_pos"]:
        return None, len(sid), 0, T, False
    # THE CHUNK IS PLANNED FOR THIS LENGTH, not carried over from a shorter one. `chunk` may
    # arrive as a plan already made (the context curve makes one per length and logs it) or as
    # a cap to plan under.
    if isinstance(chunk, tuple):
        chunk, budget, itemsize = chunk
    else:
        budget, itemsize = 0, 4
    if budget:
        chunk, _why = plan_chunk(spec, T, budget, itemsize, chunk)
        if chunk <= 0:
            return None, len(sid), 0, T, False
    ids_all = pid + sid
    n_pre = len(pid)
    state = spec["new_state"]()
    total, correct = 0.0, 0
    # RULE1: A LOOP THAT CAN TAKE MORE THAN A SECOND REPORTS WHILE IT RUNS. At 32,768 tokens
    # and a chunk of 512 this is 64 forward passes, each one slower than the last because the
    # cache it attends to keeps growing -- minutes for ONE document, in silence, which is
    # indistinguishable from a hang. The bar is self-erasing (progress passes leave=False), so
    # it shows the work and leaves no trace between the lines of the result.
    steps = progress(range(0, T, chunk), desc=desc or f"[context] {spec['key']} {T:,} tok",
                     total=-(-T // chunk))
    try:
        for a in steps:
            b = min(a + chunk, T)
            x = torch.tensor([ids_all[a:b]], device=device)
            logits, state = spec["forward_chunk"](x, state)
            # positions lo..hi-1 of THIS chunk have a next token AND lie in the span
            lo, hi = max(n_pre - 1, a), min(b, T - 1)
            if lo >= hi:
                del logits
                continue
            lg = logits[:, lo - a:hi - a].float()
            tgt = torch.tensor([ids_all[lo + 1:hi + 1]], device=device)
            total += float(-lg.log_softmax(-1).gather(-1, tgt.unsqueeze(-1)).sum())
            correct += int((lg.argmax(-1) == tgt).sum())
            n_done = hi - max(n_pre - 1, 0)
            if hasattr(steps, "set_postfix_str") and n_done:
                steps.set_postfix_str(f"{n_done:,} scored, "
                                      f"{total / max(n_done, 1):.3f} nats/tok")
            del lg, tgt, logits
    except (RuntimeError, MemoryError, IndexError) as e:
        # TWO EXPECTED ENDS OF A CURVE, and everything else is a bug.
        #
        # RUNNING OUT OF MEMORY: even chunked, the cache itself grows with T.
        #
        # AND RUNNING PAST WHAT THE MODEL DECLARES. An implementation may hold a causal mask or
        # a position table sized to max_position_embeddings and fail on its own shapes rather
        # than on ours -- which is a FACT ABOUT THE LENGTH and belongs in the figure. It counts
        # only BEYOND the declared length: the same error inside it is a real bug, and is
        # raised. That distinction is the whole guard; without it this would swallow the next
        # genuine shape error and report it as a context limit.
        dm = spec.get("declared_max") or 0
        beyond = dm and T > dm
        if not (is_alloc_error(e) or beyond):
            raise
        if beyond and not is_alloc_error(e):
            spec["_note"] = (f"refused {T:,} tokens past its declared "
                             f"{dm:,}: {type(e).__name__}")
        empty_cache()
        return None, len(sid), 0, T, False
    finally:
        if hasattr(steps, "close"):
            steps.close()
    return total, len(sid), correct, T, True


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.no_grad()
def score_span(spec, prefix, span, device, chunk=512, desc=""):
    """score_ids, for callers that hold text rather than ids.

    Returns (nats, bytes, n_tokens, n_correct, ctx_tokens, ok)."""
    nb = len(span.encode("utf-8"))
    sid = spec["encode"](span)
    if not sid:
        return 0.0, 0, 0, 0, 0, False
    pid = spec["encode"](prefix) if prefix else []
    n, t, c, T, ok = score_ids(spec, pid, sid, device, chunk, desc=desc)
    return n, nb, t, c, T, ok


def npb(nats, nbytes):
    """Nats over a span -> NATS PER BYTE, the one unit every tokenizer shares.

    NATS, NOT BITS. The loss is in nats, so dividing by the byte count and stopping there is
    the whole conversion; going on to divide by ln 2 introduces a second unit for no gain and
    makes every figure disagree with every log line. One unit, and it is the one the objective
    is already written in."""
    return None if nats is None or not nbytes else nats / nbytes


def metrics(nats, nbytes, n_tok, n_cor, ok):
    """ONE ACCUMULATION OF NATS -> THE FOUR NUMBERS EVERY PROBE REPORTS. There is exactly one
    perplexity and one token accuracy in this file, and this is it.

        nats per byte     comparable ACROSS models -- the only unit every tokenizer shares
        perplexity        the number everyone quotes, exp(nats per token), PER TOKENIZER and so
                          NOT comparable between models whose vocabularies differ
        nats per token    the same thing before the exponential, where small differences read
        token accuracy    how often the argmax is right, which moves for different reasons than
                          perplexity does -- a model can sharpen a distribution it already had
                          roughly right and gain a lot of one and none of the other

    NATS PER BYTE IS REPORTED BESIDE THE PERPLEXITY EVERYWHERE THE PERPLEXITY IS, and that is
    why this returns four numbers rather than one. A vocabulary that cuts text into fewer,
    longer pieces reports a lower perplexity for identical predictions: a perplexity may be read
    DOWN a column of one model's targets and must not be read ACROSS a row of different models.
    Nats per byte may.

    `ok` counts the measurements that succeeded. When it is falsy nothing was measured at all,
    and every field is None rather than zero -- a floor of zero drawn from no data is the one
    thing a figure cannot show."""
    npt = (nats / n_tok) if (ok and n_tok) else None
    return {"npb": npb(nats, nbytes) if ok else None,
            "nats_per_token": npt,
            "ppl": math.exp(npt) if npt is not None else None,
            "acc": (n_cor / n_tok) if (ok and n_tok) else None}


# --------------------------------------------------------------------------- #
# probe 1 -- the same targets under a growing prefix
# --------------------------------------------------------------------------- #
def long_documents(data_dir, want_chars, log, limit=64):
    """The longest documents available locally, as text. NOTHING IS CONCATENATED.

    Stitching short documents into a long one manufactures a context whose distant half is
    genuinely irrelevant to the current token: no model can exploit it, every model scores the
    same, and the flat curve reads as success. So documents are used whole, the longest first,
    and if none is long enough the run SAYS SO rather than quietly measuring nothing."""
    from helpers import corpus_files
    from helpers.corpus_id import _split_token
    from helpers.utils import _pack
    # THE HELD-OUT SPLIT, AND NOTHING ELSE. This is an EVALUATION: scoring a model on the text
    # it was trained on measures how much of it the model memorised, and a context curve
    # measured that way says nothing about generalisation. corpus_files' `exclude_dirs` is the
    # wrong instrument -- it drops directories named test/valid/validation, which a corpus of
    # flat `<name>_00000.parquet` shards does not have, so it excluded NOTHING and the sweep
    # read the training set. The split is selected here by name instead, and its ABSENCE IS
    # FATAL rather than a silent fall back to training data.
    every = corpus_files(data_dir, ())
    files = [f for f in every
             if _split_token(os.path.basename(f)) in ("validation", "valid", "test", "eval")]
    if not every:
        raise SystemExit(f"[context] no corpus under {data_dir}\n"
                         f"          ./run_scripts/run_download_data.sh")
    if not files:
        wiki = config.dataset_dir("llm-pretrain-wikitext103")
        raise SystemExit(
            f"[context] {os.path.relpath(data_dir, config.ROOT)} has no held-out split: all "
            f"{len(every):,} of its files are training data.\n"
            f"          Evaluating on them would measure memorisation, so this refuses "
            f"rather than report it as generalisation.\n"
            f"          Point --data_dir at a corpus that ships one, e.g.\n"
            f"              --data_dir {os.path.relpath(wiki, config.ROOT)}\n"
            f"          (its validation-*.parquet is genuinely unseen), or hold shards of "
            f"this corpus out of pretraining\n"
            f"          by naming them validation-*.parquet -- PRETRAIN.exclude_dirs already "
            f"keeps such a split out of training.")
    log(f"[context] held-out split: {len(files):,} of {len(every):,} files under "
        f"{os.path.relpath(data_dir, config.ROOT)} "
        f"({', '.join(sorted({_split_token(os.path.basename(f)) for f in files}))})")
    files = files[:limit]
    docs = []
    for fp in progress(files, desc="[context] scanning for long documents", total=len(files)):
        try:
            for d in _pack(fp, 10 ** 9, config.PRETRAIN["text_column"]):
                if len(d) > 2000:
                    docs.append(d)
        except Exception:                                          # noqa: BLE001
            continue
        if len(docs) > 20000:
            break
    docs.sort(key=len, reverse=True)
    if not docs:
        raise SystemExit(f"[context] no documents over 2,000 characters under {data_dir}")
    log(f"[context] {len(docs):,} documents, longest {len(docs[0]):,} characters")
    if len(docs[0]) < want_chars:
        log(f"[context] WARNING: the longest document is {len(docs[0]):,} characters and the "
            f"sweep wants {want_chars:,}.")
        log(f"[context]          The context curve is therefore capped by the CORPUS, not by "
            f"the models. Point --data_dir at a corpus of long documents (PG-19, arXiv) for "
            f"the full sweep; probes 2 and 3 are synthetic and unaffected.")
    return docs


def prefix_ids(spec, text, want, chars_per_token=4.5):
    """The LAST `want` token ids of `text`, tokenising ONLY AS MUCH TEXT AS THAT NEEDS.

    THIS IS WHY THERE IS NO PRE-TOKENISATION PASS. Every model here has its own tokenizer, so
    nothing can be shared between them, and tokenising a whole 130,000-character document to
    use the last 512 tokens of it is ten minutes of work thrown away -- which is exactly what
    the first version did, at 74 seconds a document before a single forward pass had run.

    A token is about 4.5 characters, so `want` tokens need about 4.5 x want characters. The
    estimate is checked rather than trusted: if the slice came out short, the window is doubled
    and the slice retaken, until it is long enough or the document is used up. Overshooting
    costs a little tokenising; undershooting would silently measure a shorter context than the
    x axis claims, which is the failure worth spending a retry to avoid.

    Taking the last N CHARACTERS and then cutting to the last N TOKENS leaves the left edge on
    whatever boundary the merges land on -- but the ids are cut from the left anyway, so that
    edge is arbitrary in both cases and nothing is lost by not seeing the whole document."""
    if want <= 0:
        return []
    take = min(len(text), int(want * chars_per_token) + 64)
    while True:
        ids = spec["encode"](text[len(text) - take:])
        if len(ids) >= want or take >= len(text):
            return ids[-want:]
        take = min(len(text), take * 2)


def probe_context_curve(spec, docs, args, device, log, live, cache, out):
    """The FIXED target span at CONTEXT LENGTHS OF 512, 1k, 2k, ... 32k TOKENS.

    THE X AXIS IS EXACTLY 2^n FOR EVERY MODEL. The document is tokenised once per model, the
    target is a fixed span of CHARACTERS -- the same text for everyone -- and the prefix is then
    truncated BY TOKEN ID to whatever leaves the total at L. Truncating ids rather than
    characters is what makes the length exact: a character budget lands on a different token
    count in each of four vocabularies, and the curves would not share an x value anywhere.

    FOUR CURVES OUT OF ONE SWEEP, because they are four readings of the same logits and they
    fail in different ways:

        nats per byte     comparable ACROSS models -- the only unit every tokenizer shares
        perplexity        the number everyone quotes, exp(nats per token), per-tokenizer
        nats per token    the same thing before the exponential, where small differences read
        token accuracy    how often the argmax is right, which moves for different reasons than
                          perplexity does -- a model can sharpen a distribution it already had
                          roughly right and gain a lot of one and none of the other

    The target never moves. Only how much of the document precedes it changes, so a curve that
    falls is the model extracting something from text it did not have before, and a curve that
    flattens is the model having stopped reading.

    THE PDF IS REWRITTEN AFTER EVERY POINT, and every point is cached. A sweep to 32k is hours,
    and a result that exists only in memory until the end is a result an interruption throws
    away."""
    tail = args.target_chars
    chosen = [d for d in docs[:args.documents] if len(d) > tail]
    if not chosen:
        log(f"[context] {spec['key']:<10} curve  no document longer than the target span")
        return out
    # NOTHING IS TOKENISED IN ADVANCE, NOT EVEN THE TARGET. Every model here has its own
    # vocabulary, so no ids can be shared between them and none are worth keeping: a token
    # cache would be per-model, per-text, and correct only until a tokenizer changed under it.
    # The target span is re-encoded where it is used, inside the loop, at a cost of two
    # thousand characters -- and that is the whole of the rule, applied without exception so
    # there is no second place for a stale id to come from.
    for L in args.context_lengths:
        hit = cache.get(f"curve:{L}")
        if hit is not None:
            out.append(hit)
            log(f"[context] {spec['key']:<10} curve  ctx {L:>7,} tok   npb {hit['npb']:.4f}   "
                f"ppl {hit['ppl']:8.2f}   nats/tok {hit['nats_per_token']:.4f}   "
                f"acc {hit['acc']:.4f}   (cached)")
            live()
            continue
        chunk, why = plan_chunk(spec, L, args._budget_bytes, args._itemsize,
                                args.chunk_tokens, log)
        if why == "weights":
            # NOT A CONTEXT FINDING. The model is too big for the budget and would fail the
            # same way at 512; drawing a curve that stops here would say the model cannot
            # handle long contexts, which is not what was measured. Nothing is recorded.
            log(f"[context] {spec['key']:<10} curve  SKIPPED: the model does not fit "
                f"{args.max_vram:g} GiB at any length -- raise --max_vram or use "
                f"--dtype bf16")
            break
        if chunk <= 0:
            log(f"[context] {spec['key']:<10} curve  ctx {L:>7,} tok   the KV cache for this "
                f"LENGTH exceeds {args.max_vram:g} GiB -- a limit of the length, not of "
                f"the chunk")
            live()
            break
        nats = 0.0
        nbytes = n_tok = n_cor = ctx = n_ok = short = 0
        docs_bar = progress(list(enumerate(chosen, 1)),
                            desc=f"[context] {spec['key']} ctx {L:,}", total=len(chosen))
        for i_doc, d in docs_bar:
            # ON THE FLY, here, every time. See above: no ids are kept anywhere.
            span_text = d[len(d) - tail:]
            span_ids = spec["encode"](span_text)
            span_bytes = len(span_text.encode("utf-8"))
            room = L - len(span_ids)
            if room < 0:                       # the target alone is longer than this context
                continue
            pre_ids = prefix_ids(spec, d[:len(d) - tail], room)
            # THE DOCUMENT MUST BE ABLE TO FILL THE CONTEXT. prefix_ids returns what it has,
            # so a short article would be scored at its own length while the axis said 32,768 --
            # a curve whose x values are fiction. Short documents are skipped at this length,
            # and if none is long enough the sweep stops and says which.
            if len(pre_ids) < room:
                short += 1
                continue
            n, t, c, T, ok = score_ids(spec, pre_ids, span_ids, device, chunk,
                                       desc=f"[context] {spec['key']} ctx {L:,} "
                                            f"doc {i_doc}/{len(chosen)}")
            if not ok:
                n_ok = 0
                break
            nats += n; nbytes += span_bytes; n_tok += t; n_cor += c; ctx += T; n_ok += 1
            if hasattr(docs_bar, "set_postfix_str"):
                docs_bar.set_postfix_str(f"npb {npb(nats, nbytes):.4f}, "
                                         f"acc {n_cor / max(n_tok, 1):.4f}")
        if hasattr(docs_bar, "close"):
            docs_bar.close()
        rec = {"context_length": L, "documents": n_ok,
               "ctx_tokens": round(ctx / n_ok) if n_ok else L,
               **metrics(nats, nbytes, n_tok, n_cor, n_ok)}
        out.append(rec)
        log(f"[context] {spec['key']:<10} curve  ctx {L:>7,} tok   "
            + (f"npb {rec['npb']:.4f}   ppl {rec['ppl']:8.2f}   "
               f"nats/tok {rec['nats_per_token']:.4f}   acc {rec['acc']:.4f}   "
               f"({n_ok} docs" + (f", {short} too short)" if short else ")")
               if rec["npb"] is not None else
               (f"no document reaches {L:,} tokens -- the CORPUS ends the sweep, "
                f"not the model" if short and not n_ok else "unreachable")))
        if rec["npb"] is not None:
            cache.put(f"curve:{L}", rec)
        live()
        if rec["npb"] is None:
            break
    return out


# --------------------------------------------------------------------------- #
# the evaluation targets -- one perplexity, one token accuracy, one npb each
# --------------------------------------------------------------------------- #
def target_column(path, name, log):
    """WHICH COLUMN OF A TARGET HOLDS THE TEXT, decided by reading the file's schema.

    The pretraining corpora all keep their prose in `text` and PRETRAIN["text_column"] names
    it, which is why nothing else in this project has to ask. THE EVALUATION TARGETS DO NOT ALL
    AGREE: LAMBADA ships `text` and `domain`, WikiText-103 ships `text`, and HellaSwag ships
    `ind, activity_label, ctx_a, ctx_b, ctx, endings, source_id, split, split_type, label` and
    no `text` at all. Handing `text` to the corpus reader there is not a wrong number, it is a
    SystemExit naming the columns present -- which is the right behaviour for a pretraining
    corpus that was misconfigured and the wrong one for an evaluation target whose layout is
    simply its own.

    So the configured column wins where it exists and a short ordered list of the spellings
    these datasets actually use is tried after it. WHICH COLUMN WAS SCORED IS LOGGED, always,
    because it decides what the perplexity is a perplexity OF: on HellaSwag it is `ctx`, the
    premise, and the four `endings` and the `label` that picks the right one are NOT read --
    see the module docstring, this is text perplexity and not the HellaSwag benchmark.

    Returns "" when no column can be scored, and the caller then skips the target."""
    if os.path.splitext(path)[1].lower() != ".parquet":
        return config.PRETRAIN["text_column"]        # one document per file; no column to pick
    try:
        import pyarrow.parquet as pq
        names = list(pq.ParquetFile(path).schema_arrow.names)
    except Exception as e:                                        # noqa: BLE001
        log(f"[context] {name}: cannot read the parquet schema of "
            f"{os.path.basename(path)} ({type(e).__name__}: {e}) -- skipped")
        return ""
    want = config.PRETRAIN["text_column"]
    for c in (want,) + tuple(c for c in TARGET_TEXT_COLUMNS if c != want):
        if c in names:
            log(f"[context] {name}: scoring column {c!r}"
                + ("" if c == want else f" ({want!r} is not in this dataset)")
                + f"; columns present: {', '.join(names)}")
            return c
    log(f"[context] {name}: no text column among ({', '.join(names)}) -- skipped. Add its "
        f"spelling to TARGET_TEXT_COLUMNS if this dataset should be scored.")
    return ""


def target_documents(corpus_dir, want, log):
    """(documents, column) for one evaluation target -- or (None, "") if it cannot be scored.

    NOT DOWNLOADED IS NOT AN ERROR HERE. Every baseline that will not load is reported and
    skipped, and a target that is not on this machine is the same kind of fact: an evaluation
    that dies because the third corpus is missing has wasted the first two. The line says which
    corpus and what would fetch it.

    THE HELD-OUT SPLIT WHERE THERE IS ONE. Files whose leading name token is validation / valid
    / test / eval are the split, exactly as the context curve selects it. Where a target has no
    such file the WHOLE corpus is scored, and that is sound for these targets and only for
    these: a corpus that appears in no scheme's PRETRAIN_CORPUS has never been trained on, so
    every byte of it is held out by construction. A corpus that IS pretrained on and ships no
    split is refused instead -- scoring a model on its own training text measures memorisation,
    and reporting that as a perplexity is the one thing this file has always refused to do.

    Documents are whole rows, never concatenated and never repacked: `max_words` is set past
    any document so `_pack` yields one document per parquet row and one per plain file, which
    is what the corpus reader does for every other stage."""
    from helpers import corpus_files
    from helpers.corpus_id import _split_token
    from helpers.utils import _pack
    name = os.path.basename(os.path.normpath(corpus_dir))
    every = corpus_files(corpus_dir, ())
    if not every:
        log(f"[context] {name}: not on this machine "
            f"({os.path.relpath(corpus_dir, config.ROOT)}) -- skipped. "
            f"./run_scripts/run_download_data.sh")
        return None, ""
    held = [f for f in every
            if _split_token(os.path.basename(f)) in ("validation", "valid", "test", "eval")]
    if held:
        files = held
    elif any(os.path.abspath(d) == os.path.abspath(corpus_dir)
             for ds in config.PRETRAIN_CORPUS.values() for d in ds):
        log(f"[context] {name}: PRETRAINED ON and ships no held-out split -- refused, not "
            f"skipped. Scoring it would measure memorisation and report it as a perplexity.")
        return None, ""
    else:
        files = every
        log(f"[context] {name}: no held-out split; scoring all {len(every):,} file(s) -- this "
            f"corpus is in no scheme's PRETRAIN_CORPUS, so none of it has been trained on")
    column = target_column(files[0], name, log)
    if not column:
        return None, ""
    docs, failed = [], ""
    for fp in files:
        try:
            for d in _pack(fp, 10 ** 9, column):
                if d.strip():
                    docs.append(d)
                if len(docs) >= want:
                    break
        except (Exception, SystemExit) as e:                       # noqa: BLE001
            failed = failed or f"{os.path.basename(fp)}: {type(e).__name__}: {e}"
            continue
        if len(docs) >= want:
            break
    if failed:
        log(f"[context] {name}: {failed} -- that file was skipped")
    if not docs:
        log(f"[context] {name}: no readable documents in {len(files):,} file(s) -- skipped")
        return None, ""
    log(f"[context] {name}: {len(docs):,} document(s) from {len(files):,} file(s), "
        f"{'held-out split' if held else 'whole corpus'}, column {column!r}")
    return docs, column


def probe_targets(spec, args, device, log, live, cache_meta, out):
    """PERPLEXITY, TOKEN ACCURACY AND NATS PER BYTE ON EACH EVALUATION TARGET, whole documents.

    THE SAME SCORING CODE AS EVERY OTHER PROBE. `score_span` feeds the document through the
    model's KV cache in chunks planned for the memory budget and returns the nats, the bytes,
    the tokens and the argmax hits; `metrics` turns one accumulation of those into the four
    numbers. Nothing about perplexity or accuracy is computed twice in this file.

    ONE POINT PER (MODEL, TARGET), NOT A CURVE. Each document is scored at its own length with
    no prefix, which is what a perplexity ON A DATASET means; the sweep above is the separate
    question of what a growing prefix buys. HellaSwag's items are two sentences and could not
    fill a 512-token context, let alone 32,768.

    THE CACHE HANDLES THE THREE TARGETS WITHOUT A SINGLE NEW MECHANISM: the corpus is already
    part of `cache_signature`, so each target's points land in their OWN bucket of the model's
    file, beside the curve's, and a bucket is never overwritten. The document count is in the
    point id rather than in the signature -- putting it in the signature would orphan every
    curve point already measured, which is exactly the failure the Cache docstring records."""
    for name in args.eval_targets:
        corpus = config.dataset_dir(name)
        cache = Cache(spec["key"], cache_signature(spec, args, corpus), log,
                      enabled=args.resume)
        # the same meta the curve's bucket carries, so this bucket can be redrawn alone
        cache.describe(cache_meta)
        pid = f"target:{name}:{args.eval_documents}"
        hit = cache.get(pid)
        if hit is not None:
            out.append(hit)
            log(f"[context] {spec['key']:<10} target {name:<26} npb {hit['npb']:.4f}   "
                f"ppl {hit['ppl']:8.2f}   nats/tok {hit['nats_per_token']:.4f}   "
                f"acc {hit['acc']:.4f}   (cached)")
            live()
            continue
        docs, column = target_documents(corpus, args.eval_documents, log)
        if not docs:
            continue
        nats = 0.0
        nbytes = n_tok = n_cor = scored = empty = 0
        stopped = ""
        bar = progress(list(enumerate(docs, 1)), total=len(docs),
                       desc=f"[context] {spec['key']} {name}")
        for i_doc, text in bar:
            n, b, t, c, _T, ok = score_span(
                spec, "", text, device,
                (args.chunk_tokens, args._budget_bytes, args._itemsize),
                desc=f"[context] {spec['key']} {name} doc {i_doc}/{len(docs)}")
            if not ok:
                if not t:
                    empty += 1          # a row that tokenises to nothing; not a failure
                    continue
                # THE SAME TWO ENDS score_ids KNOWS: the budget, or a model refusing a length.
                # Whatever was scored before it still stands, so it is reported rather than
                # thrown away -- with the count it was measured over, which is what makes it
                # readable as a partial figure.
                stopped = (f"stopped at document {i_doc}: the pass did not fit "
                           f"{args.max_vram:g} GiB, or the model refused the length")
                break
            nats += n; nbytes += b; n_tok += t; n_cor += c; scored += 1
            if hasattr(bar, "set_postfix_str"):
                bar.set_postfix_str(f"npb {npb(nats, nbytes):.4f}, "
                                    f"acc {n_cor / max(n_tok, 1):.4f}")
        if hasattr(bar, "close"):
            bar.close()
        rec = {"corpus": name, "documents": scored, "tokens": n_tok, "column": column,
               **metrics(nats, nbytes, n_tok, n_cor, scored)}
        out.append(rec)
        log(f"[context] {spec['key']:<10} target {name:<26} "
            + (f"npb {rec['npb']:.4f}   ppl {rec['ppl']:8.2f}   "
               f"nats/tok {rec['nats_per_token']:.4f}   acc {rec['acc']:.4f}   "
               f"({scored:,} docs, {n_tok:,} tokens"
               + (f", {empty} empty" if empty else "") + ")"
               if rec["ppl"] is not None else "nothing scored"))
        if stopped:
            log(f"[context] {spec['key']:<10} target {name:<26} {stopped}")
        if rec["ppl"] is not None:
            cache.put(pid, rec)
        live()
    return out


# ---- COMMENTED OUT: token-wise positional residue ---------------------------------- #
# # --------------------------------------------------------------------------- #
# # probe 2 -- token-wise positional residue
# # --------------------------------------------------------------------------- #
# def hidden_all(spec, ids, device, chunk, desc=""):
#     """F(x) for the whole sequence: a (T, d) float32 tensor ON THE CPU, or None.
#
#     Chunked through the KV cache exactly as score_ids is, and for the same reason -- the
#     attention matrix is (heads x chunk x T) instead of (T x T) -- and exact for the same
#     reason: causality means a position's output depends only on what precedes it, so where the
#     sequence is cut changes what is resident and not what is computed.
#
#     Kept on the CPU because the residue needs the WHOLE of F(x) at once, at arbitrary
#     positions, to compare against F(pi x) position by position: (32,768 x 2,048) in float32 is
#     268 MB, twice over, which is a great deal of a card and nothing at all of a host."""
#     T = len(ids)
#     if spec.get("ensure"):
#         spec["ensure"](T)
#     if spec["max_pos"] and T > spec["max_pos"]:
#         return None
#     if isinstance(chunk, tuple):
#         chunk, budget, itemsize = chunk
#     else:
#         budget, itemsize = 0, 4
#     if budget:
#         chunk, _why = plan_chunk(spec, T, budget, itemsize, chunk)
#         if chunk <= 0:
#             return None
#     state = spec["new_state"]()
#     parts = []
#     # RULE1: sixty-four forward passes, each slower than the last, is not something to do in
#     # silence.
#     steps = progress(range(0, T, chunk), desc=desc or f"[context] {spec['key']} F(x) {T:,} tok",
#                      total=-(-T // chunk))
#     try:
#         for a in steps:
#             x = torch.tensor([ids[a:min(a + chunk, T)]], device=device)
#             h, state = spec["hidden_chunk"](x, state)
#             parts.append(h[0].detach().float().cpu())
#             del h, x
#     except (RuntimeError, MemoryError, IndexError) as e:
#         dm = spec.get("declared_max") or 0
#         if not (is_alloc_error(e) or (dm and T > dm)):
#             raise
#         empty_cache()
#         return None
#     finally:
#         if hasattr(steps, "close"):
#             steps.close()
#     return torch.cat(parts, 0) if parts else None
#
#
# def probe_residue(spec, docs, args, device, log, live, cache, out):
#     """rho(T), the TOKEN-WISE POSITIONAL RESIDUE.
#
#         rho(T) = E_{pi ~ Pi} E_x [ || F(pi x) - pi F(x) ||_2 / || F(x) ||_2 / T ]
#
#     WHAT IT MEASURES. F is the model read as a map from a sequence of T tokens to a sequence of
#     T output vectors. pi is a permutation of the T positions, drawn uniformly. `pi x` permutes
#     the INPUT; `pi F(x)` permutes the OUTPUT of the unpermuted input. A model that used no
#     positional information whatsoever -- a bag of tokens -- would commute with pi exactly and
#     score zero. Everything above zero is output that depends on WHERE a token sat rather than
#     on WHICH token it was: the positional content of the representation, measured without
#     reference to any positional encoding, which is what makes it comparable between a model
#     that has a position table, one that rotates its queries, and one that has neither.
#
#     It is a residue rather than a score: nothing here says a larger value is better. It says
#     how much of what the model computes is positional, and at which context lengths that holds
#     up -- a model whose residue collapses towards zero as T grows is one whose representation
#     is becoming order-blind out there, whatever its perplexity says.
#
#     Norms are over the whole (T, d) output, the ratio makes it scale-free, and the 1/T makes it
#     per token -- all three exactly as written above."""
#     for L in args.context_lengths:
#         hit = cache.get(f"residue:{L}")
#         if hit is not None:
#             out.append(hit)
#             log(f"[context] {spec['key']:<10} residue {L:>7,} tok  rho {hit['rho']:.3e}   "
#                 f"({hit['perms']} permutations x {hit['docs']} documents)  (cached)")
#             live()
#             continue
#         acc, n = 0.0, 0
#         plan = (args.chunk_tokens, args._budget_bytes, args._itemsize)
#         texts = docs[:args.residue_documents]
#         pairs = progress(range(len(texts) * args.permutations),
#                          total=len(texts) * args.permutations,
#                          desc=f"[context] {spec['key']} residue {L:,}")
#         broke = False
#         for k in pairs:
#             d, p = divmod(k, args.permutations)
#             ids = prefix_ids(spec, texts[d], L)
#             if ids is None or len(ids) < L:
#                 continue                      # this document cannot fill T; it is not a failure
#             ids = ids[:L]
#             H = hidden_all(spec, ids, device, plan,
#                            desc=f"[context] {spec['key']} residue {L:,} doc {d + 1} F(x)")
#             if H is None:
#                 broke = True
#                 break
#             den = float(torch.linalg.vector_norm(H))
#             if den <= 0:
#                 del H
#                 continue
#             g = torch.Generator().manual_seed(args.seed * 7919 + L * 13 + d * 101 + p)
#             pi = torch.randperm(L, generator=g)
#             # (pi x)_t = x_{pi(t)} and (pi F(x))_t = F(x)_{pi(t)} -- one permutation applied to
#             # the input in the first term and to the output in the second, which is what makes
#             # the difference a failure of EQUIVARIANCE rather than of anything else.
#             Hp = hidden_all(spec, [ids[i] for i in pi.tolist()], device, plan,
#                             desc=f"[context] {spec['key']} residue {L:,} doc {d + 1} F(pi x)")
#             if Hp is None:
#                 del H
#                 broke = True
#                 break
#             acc += float(torch.linalg.vector_norm(Hp - H[pi])) / den / L
#             n += 1
#             del H, Hp
#             empty_cache()
#             if hasattr(pairs, "set_postfix_str"):
#                 pairs.set_postfix_str(f"rho {acc / max(n, 1):.3e}")
#         if hasattr(pairs, "close"):
#             pairs.close()
#         rec = {"context": L, "docs": args.residue_documents,
#                "perms": args.permutations, "samples": n,
#                "rho": (acc / n) if n else None}
#         out.append(rec)
#         log(f"[context] {spec['key']:<10} residue {L:>7,} tok  "
#             + (f"rho {rec['rho']:.3e}   over {n} (document, permutation) pairs"
#                if rec["rho"] is not None else "unreachable"))
#         if rec["rho"] is not None:
#             cache.put(f"residue:{L}", rec)
#         live()
#         if broke and rec["rho"] is None:
#             break
#     return out


# --------------------------------------------------------------------------- #
# probe 3 -- copy at distance
# --------------------------------------------------------------------------- #
def _filler_words(rng, n):
    """Neutral filler. Common English words, so the filler is cheap to model and whatever the
    second copy costs is about RETRIEVAL and not about the filler being surprising."""
    bank = ("the of and to in a is that for it as with was on be by at this from or an "
            "which we can has been are but not they their more when there all would some "
            "such time other into over after where these two may than first also its").split()
    return " ".join(rng.choice(bank) for _ in range(n))


def probe_copy(spec, args, device, log, live, cache, out):
    """A passage, d words of filler, then THE SAME PASSAGE. Nats per byte on each copy.

    THE SECOND COPY IS FREE INFORMATION. A model that can reach back over the filler pays
    almost nothing for it; one that cannot pays what it paid the first time. The GAIN --
    first minus second -- is the retrieval signal, in nats per byte, and it is the cleanest
    statement of what a positional mechanism buys: no semantics, no vocabulary, no corpus.

    The passage is drawn from the same word bank, so the first copy is not itself hard; what
    is being measured is the DIFFERENCE, and a difficult passage would only add noise to both
    ends of it."""
    for dist in args.copy_distances:
        hit = cache.get(f"copy:{dist}")
        if hit is not None:
            out.append(hit)
            log(f"[context] {spec['key']:<10} copy   filler {dist:>7,} words  "
                f"1st {hit['npb_first']:.3f} -> 2nd {hit['npb_second']:.3f} npb   "
                f"gain {hit['gain']:+.3f}   (cached)")
            live()
            continue
        first_n = first_b = second_n = second_b = 0.0
        reached = 0
        trials = progress(range(args.trials), total=args.trials,
                          desc=f"[context] {spec['key']} copy filler {dist:,}")
        for trial in trials:
            rng = random.Random(args.seed * 1009 + dist * 31 + trial)
            passage = _filler_words(rng, args.copy_words)
            filler = _filler_words(rng, dist)
            tag = f"[context] {spec['key']} copy filler {dist:,} trial {trial + 1}"
            n1, b1, _, _, _, ok1 = score_span(spec, "", passage, device,
                                              (args.chunk_tokens, args._budget_bytes,
                                               args._itemsize), desc=tag + " (1st copy)")
            n2, b2, _, _, _, ok2 = score_span(spec, passage + " " + filler + " ", passage,
                                              device, (args.chunk_tokens,
                                              args._budget_bytes, args._itemsize),
                                              desc=tag + " (2nd copy)")
            if not (ok1 and ok2):
                break
            first_n += n1; first_b += b1; second_n += n2; second_b += b2; reached += 1
            if hasattr(trials, "set_postfix_str"):
                trials.set_postfix_str(
                    f"gain {npb(first_n, first_b) - npb(second_n, second_b):+.3f} npb")
        if hasattr(trials, "close"):
            trials.close()
        rec = {"filler_words": dist, "trials": reached,
               "npb_first": npb(first_n, first_b) if reached else None,
               "npb_second": npb(second_n, second_b) if reached else None}
        rec["gain"] = (None if rec["npb_first"] is None
                       else rec["npb_first"] - rec["npb_second"])
        out.append(rec)
        log(f"[context] {spec['key']:<10} copy   filler {dist:>7,} words  "
            + (f"1st {rec['npb_first']:.3f} -> 2nd {rec['npb_second']:.3f} npb   "
               f"gain {rec['gain']:+.3f}" if rec["gain"] is not None else "unreachable"))
        if rec["gain"] is not None:
            cache.put(f"copy:{dist}", rec)
        live()
        if rec["gain"] is None:
            break
    return out


# --------------------------------------------------------------------------- #
# probe 4 -- passkey, scored as a likelihood
# --------------------------------------------------------------------------- #
def probe_passkey(spec, args, device, log, live, cache, out):
    """A five-digit key at a known depth, scored as -log P(the digits | everything before).

    NEVER GENERATED, NEVER EXACT-MATCHED. A 133M model that puts a third of its mass on the
    right key has found it; exact match would call that a zero and the whole probe would read
    as a flat floor for every small model. The likelihood is graded, and it is the same
    quantity for a model that would have got it right."""
    for total in args.passkey_lengths:
        for depth in args.passkey_depths:
            pid = f"key:{total}:{depth:.2f}"
            hit = cache.get(pid)
            if hit is not None:
                out.append(hit)
                log(f"[context] {spec['key']:<10} key    filler {total:>7,} words  depth "
                    f"{depth:.2f}  {hit['nats_per_key']:.2f} nats/key   (cached)")
                live()
                continue
            nats = nbytes = 0.0
            reached = 0
            trials = progress(range(args.trials), total=args.trials,
                              desc=f"[context] {spec['key']} key {total:,} @ {depth:.2f}")
            for trial in trials:
                rng = random.Random(args.seed * 7919 + total * 17 + int(depth * 100) + trial)
                key = f"{rng.randrange(10000, 100000)}"
                before = int(total * depth)
                text = (_filler_words(rng, before)
                        + f" The pass key is {key}. Remember it. "
                        + _filler_words(rng, total - before)
                        + " The pass key is")
                n, b, _, _, _, ok = score_span(
                    spec, text, f" {key}", device,
                    (args.chunk_tokens, args._budget_bytes, args._itemsize),
                    desc=f"[context] {spec['key']} key {total:,} @ {depth:.2f} "
                         f"trial {trial + 1}")
                if not ok:
                    break
                nats += n; nbytes += b; reached += 1
                if hasattr(trials, "set_postfix_str"):
                    trials.set_postfix_str(f"{nats / reached:.2f} nats/key")
            if hasattr(trials, "close"):
                trials.close()
            rec = {"filler_words": total, "depth": depth, "trials": reached,
                   "npb": npb(nats, nbytes) if reached else None,
                   # per-key nats is the readable form: -log P over the five digits, so 0 is
                   # certainty and ln(10^5) = 11.5 is a model guessing uniformly at random
                   "nats_per_key": (nats / reached) if reached else None}
            out.append(rec)
            log(f"[context] {spec['key']:<10} key    filler {total:>7,} words  depth "
                f"{depth:.2f}  "
                + (f"{rec['nats_per_key']:.2f} nats/key (random = 11.51)"
                   if rec["nats_per_key"] is not None else "unreachable"))
            if rec["nats_per_key"] is not None:
                cache.put(pid, rec)
            live()
    return out


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #
def split_at(pts, x0):
    """A polyline cut at x0 into (within, beyond), SHARING an interpolated point at the cut.

    Sharing the boundary point is what keeps the line continuous: drawn as two polylines a gap
    would open at exactly the length the reader is being asked to look at. Where x0 falls
    between two measured points the y there is interpolated."""
    pts = sorted(pts)
    if not pts or not x0 or x0 <= pts[0][0]:
        return [], pts
    if x0 >= pts[-1][0]:
        return pts, []
    within = [q for q in pts if q[0] <= x0]
    beyond = [q for q in pts if q[0] > x0]
    # INTERPOLATE ONLY WHEN x0 IS NOT ALREADY A MEASURED POINT -- but SHARE the boundary in
    # BOTH cases, which is where this went wrong. When a training length coincides with a
    # measured length, as 2,048 does for TinyStories and Pythia, the interpolation branch was
    # skipped and so was the sharing, so `beyond` began at the NEXT length and the figure had a
    # visible gap between 2^11 and 2^12 -- exactly at the boundary the panel exists to show.
    if within[-1][0] != x0:
        (xa, ya), (xb, yb) = within[-1], beyond[0]
        within = within + [(x0, ya + (yb - ya) * (x0 - xa) / (xb - xa))]
    return within, [within[-1]] + beyond


def ring_at(pts, x0):
    """The point on the curve at x0, interpolated if need be -- or None if x0 is off the curve.

    Unlike split_at this does not care whether the boundary divides the data. A model whose
    training length equals its LAST measured length still has a training length, and the ring
    is how the panel says so."""
    pts = sorted(pts)
    if not pts or not x0 or x0 < pts[0][0] or x0 > pts[-1][0]:
        return None
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if xa <= x0 <= xb:
            return (x0, ya if xa == xb else ya + (yb - ya) * (x0 - xa) / (xb - xa))
    return pts[-1] if x0 == pts[-1][0] else None


def _curve_panel(ax, models, field, xlabel, ylabel, title, logy=False):
    """One metric of probe 1 against CONTEXT LENGTH IN TOKENS.

    x is each model's OWN token count, not a shared axis of characters, because perplexity,
    nats per token and accuracy are per-token quantities: plotting them against a character
    count would divide by one thing and index by another. What is held identical across models
    is the TARGET -- the same span of text is being predicted in every case -- and the x
    positions differ only because the tokenizers do."""
    drew = False
    for i, m in enumerate(models):
        pts = [(r.get("context_length") or r["ctx_tokens"], r[field])
               for r in m["curve"] if r.get(field) is not None]
        if not pts:
            continue
        st = style_of(m["key"], i)
        within, beyond = split_at(pts, m.get("train_len"))
        if within:
            ax.plot(*zip(*within), linestyle=SOLID, label=m["name"], **st)
        if beyond:
            ax.plot(*zip(*beyond), linestyle=BEYOND,
                    **({**st, "label": m["name"]} if not within else st))
        # EVERY MODEL GETS ITS RING, at ITS OWN training length. This was drawn only when the
        # curve had points on BOTH sides of the boundary, so a model whose training length is
        # the last measured length -- Qwen and Gemma 3 at 32,768, the very models the sweep was
        # extended to reach -- got no ring at all, and the panel silently stopped marking the
        # thing it exists to mark. The ring is a property of the model, not of whether the
        # sweep happened to pass it.
        ring = ring_at(pts, m.get("train_len"))
        if ring:
            # A CIRCLE, FOR EVERY MODEL. One symbol means one thing -- "this is where this
            # model's context window ends" -- and it means it identically on all eleven curves.
            # WHAT DIFFERS IS WHERE IT SITS: 1,024 for GPT-2, 2,048 for TinyStories and
            # TinyLlama, 8,192 for SmolLM2, 32,768 for Qwen and Gemma. The position carries the
            # information; the shape must not, or the reader is decoding a second alphabet to
            # learn something the x axis already said.
            ax.plot([ring[0]], [ring[1]], marker="o", markersize=11.0, markerfacecolor="none",
                    markeredgecolor=st["color"], markeredgewidth=1.6, linestyle="none",
                    zorder=st["zorder"] + 1)
        drew = True
        # WHERE A MODEL STOPPED, and that it did. A line ending because the architecture
        # refused the length must LOOK like it ended -- an x, not a line run to the axis edge.
        if any(r.get(field) is None for r in m["curve"]):
            ax.plot(pts[-1][0], pts[-1][1], "x", color=st["color"], ms=8, mew=1.8,
                    zorder=st["zorder"] + 1)
        # NO VERTICAL RULES. Each curve carries its own boundary as a ring, so eleven of them
        # would repeat what the lines already say and turn the panel into a grid.
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9.5)
    if drew:
        # MODEL NAMES ONLY. The solid/dashed/ring convention belongs in the caption, not in the
        # legend: three grey rows explaining notation crowd out the eleven rows a reader is
        # actually looking for.
        ax.legend(fontsize=6.2, ncol=2, loc="best", handlelength=2.6)


def figure(res, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 200, "font.size": 8.5, "font.family": "serif",
        "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6))
    ax = axes.ravel()
    models = res["models"]

    # ---- row 1: three readings of the same sweep, against context length --------------- #
    # SOLID WITHIN THE TRAINED CONTEXT, DASHED BEYOND IT, RINGED AT THE BOUNDARY. Everything
    # drawn dashed is extrapolation, which is the whole question, and it is visible without
    # arithmetic and without a rule per model.
    #
    _curve_panel(ax[0], models, "npb", "context length (tokens)", "nats / byte",
                 "average encoding length per byte (nats / byte)", logy=True)
    _curve_panel(ax[1], models, "ppl", "context length (tokens)", "perplexity",
                 "perplexity", logy=True)
    _curve_panel(ax[2], models, "acc", "context length (tokens)", "accuracy",
                 "next-token prediction accuracy")

    _curve_panel(ax[3], models, "nats_per_token", "context length (tokens)", "nats / token",
                 "average encoding length per token")

# ---- COMMENTED OUT: token-wise positional residue ---------------------------------- #
#     # HOW MUCH OF WHAT THE MODEL COMPUTES IS POSITIONAL.
#     #
#     #     rho(T) = E_{pi ~ Pi} E_x [ || F(pi x) - pi F(x) ||_2 / || F(x) ||_2 / T ]
#     #
#     # Zero is a model that commutes with every permutation of its input -- a bag of tokens,
#     # order-blind. The distance from zero is positional content, measured without asking the
#     # model what positional encoding it has, which is the only way to put a learned table, a
#     # rotation and a state-space recurrence on one axis. Log y: the values run over decades.
#     drew = False
#     for i, m in enumerate(models):
#         pts = [(r["context"], r["rho"]) for r in m.get("residue", []) if r.get("rho")]
#         if not pts:
#             continue
#         st = style_of(m["key"], i)
#         # the same solid / dashed / ring convention as row 1, and for the same reason: past a
#         # model's training length this is extrapolation, whatever the quantity being plotted.
#         within, beyond = split_at(pts, m.get("train_len"))
#         if within:
#             ax[3].plot(*zip(*within), linestyle=SOLID, label=m["name"], **st)
#         if beyond:
#             ax[3].plot(*zip(*beyond), linestyle=BEYOND,
#                        **({**st, "label": m["name"]} if not within else st))
#         ring = ring_at(pts, m.get("train_len"))
#         if ring:
#             ax[3].plot([ring[0]], [ring[1]], marker="o", markersize=11.0,
#                        markerfacecolor="none", markeredgecolor=st["color"],
#                        markeredgewidth=1.6, linestyle="none", zorder=st["zorder"] + 1)
#         drew = True
#     ax[3].set_xscale("log", base=2)
#     ax[3].set_yscale("log")
#     ax[3].set_xlabel("context length (tokens)")
#     ax[3].set_ylabel(r"$\rho(T)$")
#     ax[3].set_title("token-wise positional residue", loc="left", fontsize=9.5)
#     if drew:
#         ax[3].legend(fontsize=6.2, ncol=2, loc="best", handlelength=2.6)

    for i, m in enumerate(models):
        pts = [(r["filler_words"], r["gain"]) for r in m["copy"] if r.get("gain") is not None]
        if pts:
            ax[4].plot(*zip(*pts), label=m["name"], **style_of(m["key"], i))
    ax[4].axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax[4].set_xscale("log", base=2)
    ax[4].set_xlabel("filler words between the two copies")
    ax[4].set_ylabel("nats / byte saved on the second copy")
    ax[4].set_title("encoding length saved by a repeated passage", loc="left", fontsize=9.5)
    if ax[4].get_legend_handles_labels()[0]:
        ax[4].legend(fontsize=6.2, ncol=2, loc="best", handlelength=2.6)

    for i, m in enumerate(models):
        by_len = {}
        for r in m["passkey"]:
            if r.get("nats_per_key") is not None:
                by_len.setdefault(r["filler_words"], []).append(r["nats_per_key"])
        pts = sorted((k, sum(v) / len(v)) for k, v in by_len.items())
        if pts:
            ax[5].plot(*zip(*pts), label=m["name"], **style_of(m["key"], i))
    ax[5].axhline(math.log(1e5), color="k", lw=0.8, ls=":", alpha=0.6)
    ax[5].text(0.02, math.log(1e5), " uniform guess", va="bottom", fontsize=7,
               transform=ax[5].get_yaxis_transform())
    ax[5].set_xscale("log", base=2)
    ax[5].set_xlabel("filler words around the key")
    ax[5].set_ylabel("nats to name the key")
    ax[5].set_title("encoding length of a buried key", loc="left", fontsize=9.5)
    if ax[5].get_legend_handles_labels()[0]:
        ax[5].legend(fontsize=6.2, ncol=2, loc="best", handlelength=2.6)

    # NO SUPTITLE. The panels carry their own titles and a figure that goes into a paper is
    # captioned there; a banner across the top is duplication in the document and noise in the
    # figure. The progress note lives in the log, where a progress note belongs.
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # ATOMIC. The figure is rewritten after every measured point, so a reader may well have it
    # open; replacing it whole means they never catch it half-written.
    #
    # FORMAT IS PASSED EXPLICITLY, because matplotlib infers it from the EXTENSION and the
    # temporary file ends in `.part` -- which it rejects as an unknown format. Without this the
    # write raised every time, the guard around the redraw swallowed it, and the pdf silently
    # stopped updating after the first point. THE ONE THING THIS FEATURE IS FOR.
    #
    # AND THE FIGURE IS CLOSED IN A `finally`: closing after a save that raised leaks it, and a
    # hundred live updates then leak a hundred figures.
    tmp = out_path + ".part"
    try:
        fig.savefig(tmp, format="pdf", bbox_inches="tight")
    finally:
        plt.close(fig)
    os.replace(tmp, out_path)
    return out_path


class Cache:
    """Measured points, on disk, so a crashed or changed sweep never measures twice.

        cache/evals/eval_context_length/<model>.json

    NOTHING IS EVER OVERWRITTEN. THIS IS THE RULE THE FILE IS BUILT AROUND, and it was broken
    twice: a signature that did not match left `points` empty and the first put() then wrote
    that empty dict over the file, destroying every measurement it had just declined to use.
    --no-resume did the same. Hours of forwards, gone, to record that a setting had changed.

    SO THE FILE HOLDS BUCKETS, one per signature:

        {"buckets": {<sig hash>: {"signature": {...}, "points": {...}}, ...}}

    THE SIGNATURE IS MACHINE-INDEPENDENT -- the checkpoint's NAME and step, the corpus's NAME,
    the sampling settings; never a path, a device or a time -- and it is compared only on the
    keys the CURRENT version declares, so narrowing it later cannot invalidate anything.

    ONLY SUCCESSFUL POINTS ARE KEPT. An allocation failure is a fact about one machine, and a
    machine-independent key must not carry it."""

    def __init__(self, key, signature, log, enabled=True):
        self.path = os.path.join(CACHE_DIR, f"{key}.json")
        self.sig, self.log, self.enabled, self.points = signature, log, enabled, {}
        self.hash = _sig_hash(signature)
        buckets = self._read()
        mine, why = _match(buckets, signature)
        if mine and why:
            log(f"[context] {key}: {why}")
        if not enabled:
            if mine:
                log(f"[context] {key}: --no-resume, re-measuring "
                    f"{len(mine.get('points', {})):,} points (they are KEPT on disk until "
                    f"each is replaced)")
            return
        if mine:
            self.points = dict(mine.get("points", {}))
            if self.points:
                log(f"[context] {key}: resuming, {len(self.points):,} points already measured")
        elif buckets:
            log(f"[context] {key}: no bucket matches these settings; "
                f"{len(buckets)} other set(s) on disk are left untouched")

    def _read(self):
        """Every bucket in the file, keyed by signature hash. A file in the old flat shape
        becomes one bucket; an unreadable file becomes none, and is not written over until a
        point is actually measured."""
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return {}
        if "buckets" in d:
            return d["buckets"]
        if "signature" in d:                        # the older, single-signature layout
            return {_sig_hash(d["signature"]): {"signature": d["signature"],
                                                "points": d.get("points", {})}}
        return {}

    def get(self, pid):
        return self.points.get(pid) if self.enabled else None

    def describe(self, meta):
        """What the FIGURE needs to draw this model, stored beside its points.

        Without it a cached model can only be redrawn by loading its weights again, which is
        absurd for a row whose numbers are already on disk -- and impossible for a model whose
        checkpoint has since gone bad. Name, parameter count, training length and positional
        scheme are a few hundred bytes and they make the cache self-sufficient."""
        self.meta = dict(meta)

    def put(self, pid, rec):
        """Written THE MOMENT the point is measured, atomically, and MERGED rather than
        replacing: the file is re-read first, so buckets this run knows nothing about survive,
        and so does a bucket written by a run in another terminal."""
        self.points[pid] = rec
        buckets = self._read()
        buckets[self.hash] = {"signature": self.sig, "meta": getattr(self, "meta", {}),
                              "points": self.points}
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = self.path + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"buckets": buckets}, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)


def _sig_hash(sig):
    """A stable id for a signature. Sorted keys so two runs agree, and short so the file stays
    readable by eye."""
    import hashlib
    return hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:12]


def _match(buckets, signature):
    """The bucket whose signature agrees with `signature` ON THE KEYS IT DECLARES, and a note
    when it was written by a version that keyed on more than this one does."""
    for b in buckets.values():
        old = b.get("signature", {})
        if any(old.get(k) != v for k, v in signature.items()):
            continue
        extra = [k for k in old if k not in signature]
        return b, (f"cache written by an older version ({', '.join(sorted(extra))} no longer "
                   f"part of the key) -- kept" if extra else "")
    return None, ""


# THE KEYS THAT BELONG TO THE RUN rather than to a model. A cached row can be redrawn without
# loading the model at all, but only if it was measured under the same corpus and settings --
# these are what that means, and the model's own keys (which checkpoint, which step) are
# deliberately not among them because a row is being IDENTIFIED, not re-verified.
RUN_KEYS = ("corpus", "target_chars", "documents", "trials", "copy_words", "seed", "dtype",
            "unit")


def label_for(key, name, params, pe=""):
    """The legend entry for one model: its name, and its positional scheme.

        gpt2 (learned)   pythia-70m (rope)   bloom-560m (alibi)   mira-s (nope/ssa)

    THE SCHEME IS WHAT THE FIGURE IS ABOUT, so it belongs where the eye already is. Eleven
    curves sorted by size tell a reader nothing about why two of them behave differently past a
    training window; the one word in brackets does, and it turns the legend into the
    experiment's design rather than a list of downloads.

    Ours is named by its SCHEME -- mira-s, from default_config.SCHEMES via helpers.model_name
    -- which is the name its checkpoints and figures are already filed under.

    Appended AFTER clean_name, so the parenthetical that is stripped and this one cannot be
    confused: one was a note about a method applied to a table, this is the method itself."""
    base = clean_name(name)
    tag = str(pe or "").strip().lower()
    return f"{base} ({tag})" if tag else base


def normalise_pe(text):
    """The scheme as one lower-case word, whatever an older bucket wrote there.

    Buckets exist carrying "RoPE", "learned absolute table" and "none (state space recurrence)"
    -- the long forms this used before the legend needed a word. They are the same three facts."""
    t = str(text or "").strip().lower()
    if not t:
        return ""
    if "none" in t or t.startswith("nope"):
        return "nope/ssa"
    if "learned" in t:
        return "learned"
    if "alibi" in t:
        return "alibi"
    return "rope" if "rope" in t else t


def clean_name(name):
    """A model's name, with any parenthetical stripped."""
    return str(name).split(" (")[0].strip() or str(name)


def rebuild_rows(points):
    """Cached points -> the (curve, residue, copy, passkey, target) lists a row reads, in order.

    Keyed on the point id's prefix, so a bucket written before the residue existed simply
    yields an empty residue list and every other panel of that row draws exactly as it did --
    and the same holds for a bucket written before the evaluation targets existed.

    `target:` HAS ITS OWN BRANCH RATHER THAN FALLING THROUGH TO THE PASSKEY LIST, which is
    where an unrecognised prefix lands. A target record carries no `nats_per_key`, so the
    passkey panel would have filtered it out and drawn correctly anyway -- and the summary
    would have counted it as a passkey trial that scored nothing. A list is not a default."""
    curve, res, cp, pk, tg = [], [], [], [], []
    for pid, rec in points.items():
        # residue points already on disk are still READ and still sorted, so a cache written
        # while the probe was active is not lost and comes straight back when it is uncommented
        (curve if pid.startswith("curve:") else
         res if pid.startswith("residue:") else
         cp if pid.startswith("copy:") else
         tg if pid.startswith("target:") else pk).append(rec)
    curve.sort(key=lambda r: r.get("context_length") or r.get("ctx_tokens") or 0)
    res.sort(key=lambda r: r.get("context") or 0)
    cp.sort(key=lambda r: r.get("filler_words") or 0)
    pk.sort(key=lambda r: (r.get("filler_words") or 0, r.get("depth") or 0))
    tg.sort(key=lambda r: str(r.get("corpus") or ""))
    return curve, res, cp, pk, tg


def cached_models(args, corpus, log):
    """Every model with measurements on disk, as figure rows, WITHOUT LOADING A SINGLE WEIGHT.

    A bucket carries its own `meta` (name, parameters, training length, positional scheme), so
    a row is redrawn from the cache alone -- including for a model whose checkpoint has since
    become unreadable, which is exactly when it matters most."""
    out = {}
    if not os.path.isdir(CACHE_DIR):
        return out
    run = {k: v for k, v in cache_signature({"key": "", "checkpoint": "", "step": None},
                                            args, corpus).items() if k in RUN_KEYS}
    for fn in sorted(os.listdir(CACHE_DIR)):
        if not fn.endswith(".json"):
            continue
        key = fn[:-len(".json")]
        if key in HIDDEN:                  # measured once, kept on disk, never drawn again
            continue
        try:
            with open(os.path.join(CACHE_DIR, fn), "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        buckets = d.get("buckets") or ({"flat": d} if "signature" in d else {})
        row = {"key": key, "name": key, "params": 0, "train_len": 0, "max_pos": 0,
               "positional": "", "checkpoint": "", "step": None, "total": None}
        points, meta = {}, {}
        # THE EVALUATION TARGETS LIVE IN THEIR OWN CORPUS'S BUCKET, which is what keeps a
        # perplexity on LAMBADA from being mistaken for one on WikiText-103. So the corpus is
        # dropped from the match for `target:` points ALONE, and each record names its own
        # corpus -- without this, --plot_only would summarise only the target that happens to
        # share the corpus the curve was measured on, and silently omit the other two.
        run_any_corpus = {k: v for k, v in run.items() if k != "corpus"}
        for b in buckets.values():
            sig = b.get("signature", {})
            if any(sig.get(k) != v for k, v in run.items()):
                if all(sig.get(k) == v for k, v in run_any_corpus.items()):
                    points.update({p: r for p, r in (b.get("points") or {}).items()
                                   if p.startswith("target:")})
                continue
            points.update(b.get("points") or {})
            meta.update({k: v for k, v in (b.get("meta") or {}).items() if v is not None})
            row["checkpoint"] = sig.get("checkpoint", row["checkpoint"])
            row["step"] = sig.get("step", row["step"])
        if not points:
            continue
        row.update(meta)
        row["positional"] = normalise_pe(row.get("positional")) or NATIVE_PE.get(key, "")
        row["name"] = label_for(key, row.get("name") or key, row.get("params"),
                                row["positional"])
        # AND A CIRCLE EVEN WITHOUT META. A window is a property of the model, not of when its
        # points happened to be written; the table above supplies it when the bucket cannot.
        if not row.get("train_len"):
            row["train_len"] = (args.base if key == "mira_llm" else 0) or \
                NATIVE_CONTEXT.get(key, 0)
        (row["curve"], row["residue"], row["copy"], row["passkey"],
         row["targets"]) = rebuild_rows(points)
        out[key] = row
    if out:
        log(f"[context] redrawing {len(out)} model(s) from the cache: "
            + ", ".join(f"{k} ({len(v['curve'])}+{len(v['copy'])}+{len(v['passkey'])} pts)"
                        for k, v in out.items()))
    return out


def cache_signature(spec, args, corpus):
    """What a cached POINT depends on -- and nothing else.

    THE GRID IS NOT IN HERE, and putting it in was the bug: `context_lengths` was part of the
    signature, so adding a length to the sweep discarded every point already measured at 512
    through 32,768. Those points had not changed. `curve:512` means "the nats per byte at
    context 512, for this checkpoint, on this corpus, over these documents" -- which of the
    other lengths were also asked for is a property of the RUN, not of the measurement. The same
    went for the copy and passkey grids.

    NOR IS ANYTHING THAT ONLY AFFECTS HOW THE NUMBER IS COMPUTED. --chunk_tokens and
    --max_vram decide how much is resident during a forward pass; they do not change the
    logits, so a cache must survive a change to either. Wiping hours of measurement because the
    memory budget moved is exactly the failure this cache exists to prevent.

    WHAT IS LEFT is what would make the number different: which model (spec["name"] carries
    "positions interpolated" when the table was stretched, so an extended model is a different
    model here), which checkpoint and step, which corpus, how much text is scored, how many
    documents and trials, the seed, the precision, and the unit.

    NO PATH, NO DEVICE, NO TIME -- so two machines with the same checkpoint agree on what a
    cached point means."""
    return {
        # THE LABEL IS NOT IN THE KEY, and now the label does not vary either. It once carried
        # "(positions interpolated)", so turning the extension on discarded every point already
        # measured -- for a change that alters nothing below the native window and nothing
        # above it either, since a refused length was never cached.
        "model": spec["key"],
        "checkpoint": os.path.basename(str(spec["checkpoint"]).rstrip("/")),
        "step": spec.get("step"),
        "corpus": os.path.basename(str(corpus).rstrip("/")),
        "target_chars": args.target_chars, "documents": args.documents,
        "trials": args.trials, "copy_words": args.copy_words,
        "seed": args.seed, "dtype": args.dtype, "unit": "nats_per_byte",
    }


class Live:
    """The figure and the json, rewritten AFTER EVERY MEASURED POINT.

    NEVER BUFFERED TO THE END. This sweep is four models over six context lengths plus two
    synthetic probes -- hours, most of it in forwards at 32k -- and a result that exists only in
    memory until the last line is a result that any interruption destroys. Writing costs about
    a second against minutes of compute, so the trade is not close.

    It is also the only way to WATCH the thing: open the pdf and the curves grow. A run that
    produces nothing until it finishes is indistinguishable from a run that has hung."""

    def __init__(self, res, json_path, pdf_path, log):
        self.res, self.json_path, self.pdf_path, self.log = res, json_path, pdf_path, log
        self.n = 0
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

    def __call__(self, note=""):
        self.n += 1
        self.res["progress"] = note or f"{self.n} points measured"
        tmp = self.json_path + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.res, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())          # on disk, not in the page cache, before the rename
        os.replace(tmp, self.json_path)
        try:
            figure(self.res, self.pdf_path)
        except Exception as e:             # noqa: BLE001
            # A FIGURE MUST NEVER KILL A MEASUREMENT. The numbers are the expensive part and
            # they are already on disk; a broken draw is reported and the sweep continues.
            self.log(f"[context] [plot] skipped this update: {type(e).__name__}: {e}")


def summarise(res, log):
    from helpers import table
    rows = []
    for m in res["models"]:
        curve = [r for r in m["curve"] if r["npb"] is not None]
        gains = [r["gain"] for r in m["copy"] if r.get("gain") is not None]
        keys = [r["nats_per_key"] for r in m["passkey"] if r.get("nats_per_key") is not None]
        far = max((r["filler_words"] for r in m["copy"]
                   if (r.get("gain") or 0) > 0.05), default=0)
        rows += [
            (m["name"], ""),
            ("  positional", m["positional"]),
            ("  parameters", f"{m['params'] / 1e6:.1f}M"),
            ("  trained at", f"{m['train_len']:,} tokens"),
            ("  best npb / at", f"{min(r['npb'] for r in curve):.4f} at "
                                f"{max(r.get('context_length') or 0 for r in curve):,} tokens"
                                if curve else "-"),
            ("  furthest copy", f"{far:,} filler words still worth over 0.05 npb"),
            ("  best key", f"{min(keys):.2f} nats (uniform = 11.51)" if keys else "-"),
        ]
        # THE EVALUATION TARGETS, ONE ROW EACH, PERPLEXITY AND TOKEN ACCURACY AND NATS PER BYTE.
        # The perplexity is per tokenizer: read it down this model's three targets, never
        # across two models. The npb beside it is the figure that may be read across.
        for r in m.get("targets", []):
            if r.get("ppl") is None:
                continue
            rows += [(f"  {r['corpus']}", f"ppl {r['ppl']:.2f}, acc {r['acc']:.4f}, "
                                          f"{r['npb']:.4f} npb over {r['documents']:,} "
                                          f"documents")]
        rows += [("", "")]
    table("context generalization", rows, out=log)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# THE SETTINGS. NOT ARGUMENTS.
#
# There are two flags: --checkpoint and --no-resume. Everything else that governs this
# benchmark is a CONSTANT, here, where it is read once and cannot be varied per invocation.
#
# A knob is a promise that a number is a choice, and every one of these is not: the context
# lengths are what the sweep IS, the target span and the document count are what makes two runs
# comparable, the chunk and the budget are how the arithmetic is made to fit and change no
# result. Exposing them invites a figure produced under settings nobody recorded, which is the
# one thing a benchmark must never allow -- and it lets the cache's signature drift for reasons
# that have nothing to do with the models.
# --------------------------------------------------------------------------- #
CORPUS_DIR = config.dataset_dir("llm-pretrain-wikitext103")
# Its held-out split, and no other corpus: this is the only downloaded PRETRAINING corpus that
# holds anything back, and scoring a model on text it was trained on measures memorisation.

# THE EVALUATION TARGETS: the corpora every model is scored on for perplexity, token accuracy
# and nats per byte, by config/download/config_download.yaml's own names. WikiText-103 is here as well as
# the context curve's corpus, because the flat number and the curve are two different readings
# and the flat one is what the other two targets can be compared against.
#
# WHAT MAKES THIS HONEST IS THE HELD-OUT SPLIT, NOT THE CHOICE OF CORPUS. HellaSwag and LAMBADA
# are in no scheme's PRETRAIN_CORPUS, so stage 4 never tokenises them -- it tokenises the
# scheme's list, and `--only llm-hellaswag` reaches them there when a token stream of one is
# actually wanted. WikiText-103 IS a pretraining corpus: it is the whole of mira_llm-baby's
# PRETRAIN_CORPUS and the second entry for -s, -m and -l. It is scored anyway because
# PRETRAIN["exclude_dirs"] = ["test", "valid", "validation"] keeps those files out of training
# and this evaluator reads exactly those files -- see score_target below, which selects the
# split by leading name token and REFUSES a pretrained-on corpus that ships none. Trained-on
# text is never scored; held-out text from a trained-on distribution is, deliberately.
#
# AND THE NUMBER ON THE FIRST TWO IS TEXT PERPLEXITY, NOT THE PUBLISHED BENCHMARK. HellaSwag is
# scored as accuracy over four candidate endings and LAMBADA as accuracy on a final word;
# neither is computed here. See the module docstring: what is computed is the likelihood of the
# text those datasets ship, which is a language-modelling measurement on their distributions
# and is not comparable to any leaderboard figure.
EVAL_TARGETS = ["llm-hellaswag", "llm-lambada", "llm-pretrain-wikitext103"]
# DOCUMENTS PER TARGET. Every one is a forward pass, over eleven models and three targets, so
# this is what the addition costs: 256 short documents is minutes per model and is already far
# past the point where the mean stops moving -- LAMBADA's validation split is 4,869 passages
# and HellaSwag's 10,042 items, and a perplexity over 256 of them is stable to well under the
# gaps this figure exists to show.
EVAL_DOCUMENTS = 256
# WHERE THE TEXT IS, IN THE ORDER THE COLUMN IS LOOKED FOR. PRETRAIN["text_column"] is tried
# first and is right for LAMBADA and WikiText-103; HellaSwag has no `text` column at all and
# its premise is `ctx`. The rest are the spellings a plain-text evaluation set is likely to
# use, and a target whose column is none of them is skipped with its columns printed rather
# than taking the whole run down.
TARGET_TEXT_COLUMNS = ("text", "ctx", "sentence", "content", "document", "passage")

# CONTEXT LENGTHS ARE POWERS OF TWO IN TOKENS, so every model is measured at the same x and the
# doubling is legible on a log axis.
CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
COPY_DISTANCES = [16, 64, 256, 1024, 4096, 16384]
PASSKEY_LENGTHS = [64, 256, 1024, 4096, 16384]
PASSKEY_DEPTHS = [0.1, 0.5, 0.9]

DOCUMENTS = 8              # averaged per context length
TARGET_CHARS = 2000        # the fixed span every context length is scored on
TRIALS = 4                 # repeats per synthetic point
# ---- COMMENTED OUT: token-wise positional residue ---------------------------------- #
# # THE EXPECTATION OVER Pi IS SAMPLED, NOT ENUMERATED. Pi is the uniform distribution over all
# # T! permutations of the positions, and T! is not a number anything can be summed over -- at
# # T = 512 it already exceeds the atoms in the observable universe. rho is an EXPECTATION, so it
# # is estimated the way an expectation is: draw permutations and average. Five draws per
# # document is the default, --num_permut_sampling.
# #
# # Each draw is a full forward pass over T tokens, and none of it can be shared between context
# # lengths, so this is the term that sets what the probe costs: documents x (1 + draws) passes
# # per length per model.
# RESIDUE_DOCUMENTS = 2      # documents per context length for rho(T)
# PERMUTATIONS = 5           # default draws from Pi per document; --num_permut_sampling
COPY_WORDS = 48            # length of the copied passage
CHUNK_TOKENS = 512         # upper bound on tokens per forward pass; the chunk is planned
DTYPE = "fp32"             # the eval reports likelihoods; nothing here is trained
SEED = 0


def parse_args():
    p = argparse.ArgumentParser(
        description="does a pretrained model use, and generalise, a long context")
    p.add_argument("--checkpoint", default=os.path.join(
        config.CHECKPOINT_DIR, "pretrain", "checkpoint_mira_llm-s_ssm_pretrain.pt"),
        help="the MiraLLM checkpoint to measure")
    p.add_argument("--only", action="append", choices=["mira_llm"] + list(BASELINES),
                   help="measure just this model; repeatable. The figure still redraws every "
                        "model in the cache, so this narrows the WORK and not the result")
    p.add_argument("--data_dir", default=CORPUS_DIR,
                   help="corpus whose HELD-OUT split is scored (validation/test by name); a "
                        "corpus with none is refused. WikiText articles do not reach 32k -- "
                        "point this at PG-19 or arXiv for the full sweep")
    p.add_argument("--base", type=int, default=0,
                   help="MiraLLM's training length, which sets where its curve turns from solid "
                        "to dashed; 0 = the checkpoint's own block_size. A checkpoint part-way "
                        "through the context schedule has been trained SHORTER than its "
                        "block_size says, and only you know how far it got")
    # RESUME IS THE DEFAULT, because this sweep is hours long and the normal case is a second
    # run. --no-resume re-measures every point; nothing on disk is deleted either way.
    p.add_argument("--no-resume", "--no_resume", dest="resume", action="store_false",
                   default=True,
                   help=f"measure every point again instead of reusing "
                        f"{os.path.relpath(CACHE_DIR, config.ROOT)}/")
    # ---- COMMENTED OUT: token-wise positional residue ---------------------------------- #
#     p.add_argument("--num_permut_sampling", type=int, default=PERMUTATIONS, metavar="N",
#                    help=f"permutations drawn per document when estimating the positional "
#                         f"residue. Pi is uniform over all T! orderings and rho is an "
#                         f"expectation over it, so it is SAMPLED: N draws, averaged "
#                         f"(default: {PERMUTATIONS})")
    p.add_argument("--plot_only", action="store_true",
                   help="redraw the pdf from the cache and stop; loads no model")
    p.add_argument("--gpu", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--max_vram", type=float, default=20.0, metavar="GiB",
                   help="what the whole thing may occupy: weights + KV cache + the pass in "
                        "flight. The chunk per forward pass is solved for this, and a length "
                        "whose KV CACHE ALONE exceeds it is reported before anything is "
                        "allocated (default: 20)")
    a = p.parse_args()
    # THE DIRECTORIES COME FROM default_config, ASSIGNED AFTER PARSING -- which is what
    # helpers.common.parse_args does (args.model_dir = config.MODEL_DIR, args.data_dir =
    # config.DOWNLOAD_DIR). A path that the configuration already fixes is not a per-run
    # choice, and one place deciding it is what keeps a figure and a checkpoint in the same
    # tree.
    a.model_dir = config.MODEL_DIR                      # saved_models/, the Hub cache
    a.plot_dir = os.path.join(config.PLOT_DIR, "evaluation")
    a.out = os.path.join(a.plot_dir, "pretrain_context_generalization.pdf")
    a.json = os.path.join(config.OUTPUT_DIR, "eval", "pretrain_context_generalization.json")
    # and the settings, which are not arguments -- see above
    a.documents, a.target_chars = DOCUMENTS, TARGET_CHARS
    a.trials, a.copy_words, a.seed = TRIALS, COPY_WORDS, SEED
    a.eval_targets, a.eval_documents = list(EVAL_TARGETS), EVAL_DOCUMENTS
    # ---- COMMENTED OUT: token-wise positional residue ---------------------------------- #
#     a.residue_documents = RESIDUE_DOCUMENTS
#     a.permutations = a.num_permut_sampling
    a.chunk_tokens, a.dtype = CHUNK_TOKENS, DTYPE
    a.extend_positions = True         # not a flag: a length is never refused, ever
    a.context_lengths = list(CONTEXT_LENGTHS)
    a.copy_distances = list(COPY_DISTANCES)
    a.passkey_lengths = list(PASSKEY_LENGTHS)
    a.passkey_depths = list(PASSKEY_DEPTHS)
    a._budget_bytes = int(a.max_vram * 2 ** 30)
    a._itemsize = {"fp32": 4, "bf16": 2, "fp16": 2}[a.dtype]
    return a


def main():
    args = parse_args()
    def log(m): print(m, flush=True)

    if args.plot_only:
        # STRAIGHT FROM THE CACHE, which is where the measurements live. There is no separate
        # json to be stale against it, and no model is loaded to redraw a figure.
        seeded = cached_models(args, args.data_dir, log)
        if not seeded:
            raise SystemExit("[context] --plot_only: nothing measured yet under these settings")
        order = ["mira_llm"] + list(BASELINES)
        res = {"models": [seeded[k] for k in order if k in seeded],
               "corpus": os.path.relpath(args.data_dir, config.ROOT),
               "target_chars": args.target_chars, "trials": args.trials,
               "copy_words": args.copy_words, "dtype": args.dtype, "seed": args.seed,
               "progress": "redrawn"}
        Live(res, args.json, args.out, log)("redrawn from the cache")
        summarise(res, log)
        log(f"[context] [figure] {os.path.relpath(args.out, config.ROOT)}")
        return

    from helpers import resolve_device
    device = resolve_device(args.gpu)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    log(f"[context] device={device} dtype={args.dtype} seed={args.seed}")

    want = list(args.only) if args.only else ["mira_llm"] + list(BASELINES)
    data_dir = args.data_dir
    # WHAT IS ALREADY ON DISK, FIRST. Rows are rebuilt from the cache before a single model is
    # loaded, so the figure carries every measurement ever taken under these settings -- and a
    # model that cannot be loaded at all keeps the curve it earned earlier.
    seeded = cached_models(args, data_dir, log)

    def load(key):
        return (spec_mira_llm(args, device, dtype, log) if key == "mira_llm"
                else spec_hf(key, BASELINES[key], device, dtype, log, args))
    # a token is ~4 characters, so the longest context wants roughly 4x its tokens in text
    docs = long_documents(data_dir, max(args.context_lengths) * 4 + args.target_chars, log)

    # THE RESULT OBJECT STARTS FROM THE CACHE, in the canonical size order, so the very first
    # figure of a resumed run already carries everything measured before it. A model being
    # measured now REPLACES its seeded row rather than adding a second one -- the probes replay
    # their cached points into the fresh list, and two rows for one model would draw it twice.
    order = ["mira_llm"] + list(BASELINES)
    res = {"models": [seeded[k] for k in order if k in seeded],
           "corpus": os.path.relpath(data_dir, config.ROOT),
           "target_chars": args.target_chars, "trials": args.trials,
           "copy_words": args.copy_words, "dtype": args.dtype, "seed": args.seed,
           "progress": "starting"}
    live = Live(res, args.json, args.out, log)
    live("redrawn from the cache")             # the pdf is correct before anything is measured
    log(f"[context] writing {os.path.relpath(args.out, config.ROOT)} after every point")

    loaded = 0
    for i, key in enumerate(want, 1):
        spec = load(key)                       # loaded HERE, measured, and dropped below
        if spec is None:
            continue                           # already said why; the cached row still draws
        loaded += 1
        log(f"\n[context] === {spec['name']}  ({i}/{len(want)}) ===")
        m = {k: spec[k] for k in ("key", "name", "params", "train_len", "max_pos",
                                  "positional", "checkpoint", "step", "total")}
        m["name"] = label_for(m["key"], m["name"], m.get("params"),
                              m.get("positional"))
        m["curve"], m["copy"], m["passkey"], m["targets"] = [], [], [], []
        m["residue"] = []          # left in place: the figure reads it with .get and skips
        at = next((j for j, r in enumerate(res["models"]) if r["key"] == spec["key"]), None)
        if at is None:
            res["models"].append(m)
        else:
            res["models"][at] = m
        tag = f"{spec['key']} ({i}/{len(want)})"
        cache = Cache(spec["key"], cache_signature(spec, args, data_dir), log,
                      enabled=args.resume)
        # WHAT THE FIGURE NEEDS TO REDRAW THIS ROW WITHOUT THE MODEL, stored beside its points.
        cache.describe({k: m[k] for k in ("name", "params", "train_len", "max_pos",
                                          "positional", "step", "total")})
        # THE PROBES APPEND TO THE VERY LISTS THE FIGURE READS. Assigning their return value
        # instead -- m["curve"] = probe_...(...) -- binds only when the probe RETURNS, so every
        # live() call inside it redrew a figure whose curve was still empty and the pdf gained
        # nothing until a whole model was finished. The list has to be shared, not returned.
        probe_context_curve(spec, docs, args, device, log,
                            lambda: live(f"{tag}: context curve"), cache, m["curve"])
        # ---- COMMENTED OUT: token-wise positional residue ---------------------------------- #
#         probe_residue(spec, docs, args, device, log,
#                       lambda: live(f"{tag}: positional residue"), cache, m["residue"])
        probe_copy(spec, args, device, log, lambda: live(f"{tag}: copy"), cache, m["copy"])
        probe_passkey(spec, args, device, log, lambda: live(f"{tag}: passkey"), cache,
                      m["passkey"])
        # THE EVALUATION TARGETS, LAST AND WITH THEIR OWN CACHES. Each target is a corpus, and a
        # corpus is already part of the cache signature, so each one writes into its own bucket
        # of this model's file -- `cache` above belongs to the curve's corpus and is not the one
        # to write a LAMBADA perplexity into.
        probe_targets(spec, args, device, log, lambda: live(f"{tag}: eval targets"),
                      {k: m[k] for k in ("name", "params", "train_len", "max_pos",
                                         "positional", "step", "total")}, m["targets"])
        # AND NOW ACTUALLY LET IT GO. `spec["model"] = None` did not: forward_chunk,
        # hidden_chunk and ensure are closures over the local `model` in spec_mira_llm /
        # spec_hf, and a closure holds its cell however the dict is edited. The weights stayed
        # alive behind those three lambdas for the whole run. Dropping the callables drops the
        # cells, and the collection is forced rather than left to a later allocation -- CUDA
        # memory is not returned by empty_cache() until Python has released the tensors, so
        # the order of these three lines is the point of them.
        spec["model"] = None
        for k in ("forward_chunk", "hidden_chunk", "ensure", "new_state", "encode"):
            spec.pop(k, None)
        del spec
        gc.collect()
        empty_cache()
        live(f"{tag} complete")

    if not loaded and not seeded:
        raise SystemExit("[context] no model could be loaded and nothing is cached")
    if not loaded:
        log("[context] no model could be loaded; the figure is what was already cached")
    live("complete")
    log(f"\n[context] [json] {os.path.relpath(args.json, config.ROOT)}")
    summarise(res, log)
    log(f"[context] [figure] {os.path.relpath(args.out, config.ROOT)}")


if __name__ == "__main__":
    main()
