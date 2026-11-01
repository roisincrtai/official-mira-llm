"""
instruct_sft/run.py -- stage 7 entry point: domain-adaptive fine-tuning on the fine-tuning data.

SFT MUST be based on the pretrained model: without a pretrain checkpoint this exits.

    python -m instruct_sft.run [--sft_steps N --sft_lr LR]
    ./old_scripts/stage7_instruct_sft.sh
"""
import glob
import inspect
import os
import sys

import default_config as config
import helpers
from helpers import common

from . import instruct_sft as sft


def _short(path):
    """A path as the user typed it: relative to the project root when it is under it."""
    text = str(path)
    return os.path.relpath(text, config.ROOT) if os.path.isabs(text) else text


def pretrain_checkpoints(ckdir):
    """Every checkpoint under checkpoints/pretrain/, newest last."""
    root = helpers.stage_dir(ckdir, "pretrain")
    found = set()
    for pattern in ("checkpoint.pt", os.path.join("*", "checkpoint.pt")):
        found.update(glob.glob(os.path.join(root, pattern)))
    return sorted(found, key=os.path.getmtime)


def benchmark_search():
    """The glob patterns this stage looks for a benchmark MIRA checkpoint under.

    BOTH PARTS OF THE PATH ARE CONFIGURED, AND NEITHER MAY BE GUESSED HERE.

    THE DIRECTORY COMPONENT NAMING THE STAGE is BENCHMARK.benchmark_dir, which a yaml may set to
    benchmark_v2 or anything else; this asked for a literal "benchmark" under CHECKPOINT_DIR and
    so looked in a directory the benchmark had not written to.

    THE UNIT IS NAMED AFTER THE GEOMETRY, NOT AFTER THE ARCHITECTURE. A baseline writes under its
    family -- rwkv_v6, mamba_v1 -- but MIRA writes under BENCHMARK.mira_checkpoint_name, which is
    mira-s_K2_d768_de1536_dr48_dm16 at the shipped configuration, because one repository
    benchmarks several MIRA shapes and the shape is what tells their checkpoints apart. So that
    directory name is not an architecture name and canonical_arch leaves it alone: a filter
    asking `canonical_arch(<directory>) == "mira-lattice"` rejects the one MIRA checkpoint there
    is. benchmark/run.py::checkpoint_unit is where the rule lives; the key is read here rather
    than the rule restated, so a renamed unit moves the search with it.

    THE DATASET LEVEL IS A WILDCARD because a machine may have benchmarked more than one corpus
    and any of them can start stage 7. benchmark_checkpoint_root builds the path by joining the
    dataset on, so passing "*" gives the pattern without a second copy of how that path is made.

    THE LEGACY PATTERN IS KEPT because a tree written before the geometry naming has the family
    name at that level, and a checkpoint that exists must not stop being found."""
    root = config.benchmark_checkpoint_root("*")
    return [*(os.path.join(root, unit, "checkpoint.pt")
              for unit in config.mira_checkpoint_units()),
            os.path.join(root, "*", "checkpoint.pt")]


def benchmark_mira_checkpoints():
    """Every MIRA checkpoint under the benchmark tree, newest last.

    THE BENCHMARK IS THE FALLBACK BECAUSE IT IS OFTEN THE ONLY THING TRAINED. Only MIRA's own
    unit holds a MiraLLM -- every other directory there comes from baselines.build_lm and cannot
    be loaded by this stage -- so the first pattern names that unit exactly and the second keeps
    the old family-named layout by testing the name it would have had."""
    *exact, legacy = benchmark_search()
    found = set(path for pattern in exact for path in glob.glob(pattern))
    found.update(p for p in glob.glob(legacy)
                 if config.canonical_arch(
                     os.path.basename(os.path.dirname(p))) == "mira-lattice")
    return sorted(found, key=os.path.getmtime)


def named_checkpoint(argv=None):
    """The path given as --checkpoint, popped off `argv`, or "" when it was not given.

    POPPED, BECAUSE THE SHARED PARSER DOES NOT KNOW THIS FLAG. common.parse_args builds one
    parser for every stage, and adding --checkpoint there would give it to all of them; this
    stage is the one that was asked for it, so it is read and removed here, before parse_args
    sees the list. `--checkpoint PATH` and `--checkpoint=PATH` are both accepted.

    THE YAML'S sft_checkpoint IS THE OTHER WAY IN, and the flag wins over it, exactly as a flag
    wins over a yaml value everywhere else in this pipeline."""
    argv = sys.argv if argv is None else argv
    for index, token in enumerate(list(argv)):
        if token == "--checkpoint" and index + 1 < len(argv):
            value = argv[index + 1]
            del argv[index:index + 2]
            return value
        if token.startswith("--checkpoint="):
            del argv[index]
            return token.split("=", 1)[1]
    return ""


def initial_checkpoint(ctx, flag=""):
    """(path, where it came from) for the weights this stage starts from.

    THE ORDER IS --checkpoint, THEN THE YAML, THEN THE TWO DIRECTORIES. checkpoints/pretrain/
    is what stage 6 writes and is tried first; checkpoints/benchmark/ is the fallback for a
    machine that has run nothing but the benchmark. Where a directory holds several, the newest
    is taken and the log says which -- the printed path is what makes that checkable."""
    named = str(flag or config.SFT.get("checkpoint") or "").strip()
    if named:
        path = named if os.path.isabs(named) else os.path.join(config.ROOT, named)
        if not os.path.isfile(path):
            raise SystemExit(f"[instruct-sft] {'--checkpoint' if flag else 'SFT.checkpoint'} names "
                             f"{named}, which does not exist")
        return path, "--checkpoint" if flag else "SFT.checkpoint"
    exact = helpers.ckpt_path(ctx["ckdir"], "pretrain")
    if os.path.isfile(exact):
        return exact, "checkpoints/pretrain/"
    found = pretrain_checkpoints(ctx["ckdir"])
    if found:
        return found[-1], "checkpoints/pretrain/"
    found = benchmark_mira_checkpoints()
    if found:
        return found[-1], _short(
            os.path.dirname(config.benchmark_checkpoint_root("*"))) + "/"
    # WHERE IT LOOKED, NOT JUST THAT IT FAILED. A search that names no path leaves the reader
    # with nothing to check against their own tree, which is what made the geometry-named
    # benchmark directory take a session to find.
    searched = "\n".join(f"                  {_short(p)}" for p in benchmark_search())
    raise SystemExit(
        f"[instruct-sft] no checkpoint to start from: {_short(exact)} does not "
        f"exist and no MIRA checkpoint matched\n{searched}\n"
        f"Run stage 6, or set sft_checkpoint in the yaml, or pass --checkpoint PATH.")


def stage_batch(args):
    """This stage's own batch and micro-batch, unless the run asked for others.

    parse_args has already applied SCHEME_BATCH, which is sized for a PRETRAINING step where
    every sequence fills the context window; an instruction demonstration is a few hundred
    tokens, so that batch would leave the card almost empty and make one epoch ten times longer.
    The yaml's SFT_BATCH and SFT_MICRO_BATCH become --batch and --micro_batch for this stage
    (tools/config_file.SPEC), so the configured path never reaches this; it exists so that
    `python -m instruct_sft.run --config /dev/null` -- and the Readme's bare `python -m
    instruct_sft.run` -- really is equivalent, rather than quietly training at a third of the
    batch and a tenth of the epoch.

    An explicit flag, or a value in the yaml, still wins: both arrive as the same flag and are
    tested exactly as parse_args tests them, so nothing anybody set is replaced here either."""
    if not common._was_given("--batch", "batch"):
        args.batch = int(config.SFT["batch"])
    if not common._was_given("--micro_batch", "micro_batch"):
        args.micro_batch = int(config.SFT["micro_batch"])
    return args


def model_from(ctx, path, train_mode=True):
    """A model built from the architecture recorded in `path` and loaded with its weights.

    THE CHECKPOINT'S OWN ARCHITECTURE WINS, exactly as common.load_stage_model has it: a
    checkpoint trained at a different width or depth must be rebuilt as it was. This differs
    from that function only in taking a PATH, because the fallback checkpoint lives outside the
    stage tree it looks in."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    saved = ck.get("model_cfg") if isinstance(ck, dict) else None
    if not isinstance(saved, dict) or "model" not in ck:
        raise SystemExit(f"[instruct-sft] {_short(path)} has no architecture "
                         f"metadata; it cannot be rebuilt")
    family = str(saved.get("benchmark_family") or "")
    if family and "lattice_order" not in saved:
        raise SystemExit(f"[instruct-sft] {_short(path)} is the {family!r} "
                         f"baseline, not a MiraLLM; it cannot be fine-tuned here")
    # THE FAMILY KEY IS DROPPED AFTER THE CHECK ABOVE HAS USED IT. common.translate_cfg refuses
    # every checkpoint carrying `benchmark_family` outright, which is right for the stages that
    # cannot tell a MIRA one from a baseline; the line above has just told them apart, so what
    # reaches it here is a MiraLLM and the key would only make it refuse the checkpoint this
    # stage was pointed at on purpose. translate_cfg discards the same key itself once past
    # that guard, so nothing downstream sees a different dict from the one it would have built.
    if family:
        saved = {k: v for k, v in saved.items() if str(k).lower() != "benchmark_family"}
    cfg = common.translate_cfg(saved)
    # ONLY WHAT THE CONSTRUCTOR TAKES IS PASSED, AND EVERY DROPPED FIELD IS NAMED WITH ITS
    # VALUE. model_cfg is splatted into MiraLLM, so one recorded field the constructor does not
    # have stops the stage: a benchmark checkpoint records `lattice_state` (benchmark/run.py:174)
    # and MiraLLM.__init__ does not accept it -- it fixes lattice_state="matrix" itself when it
    # builds the blocks (model/mira_llm.py:446), which is the same value, so dropping it cannot
    # change the model that is built. The values are PRINTED rather than dropped quietly because
    # this is a trainer: a field that did not match what the constructor assumes would be a
    # different architecture, and the reader has to be able to see it.
    # tools/convert_checkpoints.py --apply removes these from the file itself.
    from model import MiraLLM
    accepted = set(inspect.signature(MiraLLM.__init__).parameters) - {"self", "vocab_size"}
    ignored = sorted(set(cfg) - accepted)
    if ignored:
        log = ctx.get("log", print)
        log(f"[instruct-sft] {len(ignored)} recorded field(s) this model does not take, dropped:")
        for key in ignored:
            log(f"       {key} = {cfg[key]!r}")
        cfg = {k: v for k, v in cfg.items() if k in accepted}
    # THE BOTTLENECK FACTOR IS RECORDED AGAINST d_e AND VALIDATED AGAINST d. baselines.build_lm
    # writes coefficient_bottleneck_factor = d_e / d_r (baselines/__init__.py:516) -- 1536/48 =
    # 32 at the benchmark shape -- and MiraLLM requires coefficient_bottleneck == n_embd /
    # coefficient_bottleneck_factor, which is 768/32 = 24 and refuses the checkpoint.
    #
    # ONLY THE DIVISOR IS RECOMPUTED, NEVER THE WIDTH. coefficient_bottleneck is recorded
    # explicitly and is what the weights actually have, so it is passed through untouched; the
    # factor is redundant beside it and exists for the constructor to derive a bottleneck that
    # is already given. Recomputing it as n_embd / coefficient_bottleneck therefore builds the
    # same model, and both values are printed so the substitution is not silent.
    bottleneck = cfg.get("coefficient_bottleneck")
    factor = cfg.get("coefficient_bottleneck_factor")
    width = cfg.get("n_embd")
    if (bottleneck and factor and width and int(width) % int(bottleneck) == 0
            and int(bottleneck) != int(width) // int(factor)):
        corrected = int(width) // int(bottleneck)
        log = ctx.get("log", print)
        log(f"[instruct-sft] coefficient_bottleneck_factor recorded as {factor} against d_e; MiraLLM "
            f"reads it against n_embd, where {width} / {factor} = {int(width) // int(factor)} "
            f"is not the recorded coefficient_bottleneck {bottleneck}")
        log(f"[instruct-sft] using coefficient_bottleneck_factor {corrected} = {width} / {bottleneck}; "
            f"the bottleneck itself is unchanged, so the model built is the one trained")
        cfg["coefficient_bottleneck_factor"] = corrected
    m = common.build_model(ctx["tok"], ctx["device"], model_cfg=cfg, args=ctx["args"])
    common.adopt_context(ctx, saved, "pretrain")
    common.load_growing(m, ck["model"], log=ctx.get("log", print))
    m.train() if train_mode else m.eval()
    return m


def register_chat_specials(ctx):
    """Register the four instruct markers on THIS stage's tokenizer, and keep it.

    THE SET IS helpers.CHAT_SPECIALS AND IT IS THE SIDECAR'S. It must match
    tools/create_instruct_bpe.SPECIALS in content and order, because both register onto the same
    base and the ORDER is what fixes the ids: a stage that registered a different four would read
    every marker as something else. <startofthink> and <endofthink> are registered here although
    nothing emits them, so that the chain-of-thought stage does not have to grow the vocabulary
    of a checkpoint this stage already wrote.

    STAGE 7 AND NOT default_config.EXTRA_SPECIAL_TOKENS, because that list is registered on every
    tokenizer this project loads -- pretraining, the benchmark and the ablation would each grow a
    vocabulary for turns they do not have.

    NOTHING ALREADY TRAINED MOVES. The specials sit last, so a new one takes the next id: the 256
    bytes and 50,000 merges keep the ids they had, every cached token stream stays valid, and the
    pretrain checkpoint is short by three embedding rows, which common.load_growing adds. No
    corpus is re-read and no stream is rebuilt.

    IDEMPOTENT. register_special returns the existing id for a token already present, so a
    resumed run -- whose tokenizer came out of its own checkpoint with the three already in it --
    registers nothing and reports the same ids.

    THE CHECKPOINT CARRIES THIS TOKENIZER, not the file on disk. set_bpe_source is called again
    with the live object, so helpers.utils._bpe_blob serialises what is in memory: a checkpoint
    whose embedding has the three rows and whose embedded vocabulary does not would decode its
    own output as something else."""
    tok, log = ctx["tok"], ctx["log"]
    if not hasattr(tok, "register_special"):
        log("[instruct-sft] this tokenizer cannot register specials; turns are not marked")
        return tok
    before = len(tok)
    ids = {t: tok.register_special(t) for t in helpers.CHAT_SPECIALS}
    helpers.set_bpe_source(helpers.get_bpe_source(), log=None, tokenizer=tok)
    log(f"[instruct-sft] chat specials: "
        + ", ".join(f"{t}={i}" for t, i in ids.items())
        + f"; vocabulary {before:,} -> {len(tok):,}")
    return tok


def main():
    flag = named_checkpoint()                       # popped before the shared parser sees it
    args = stage_batch(common.parse_args(stage="instruct_sft"))
    ctx = common.setup(args, need_pairs=False)      # the fine-tuning corpus and nothing else
    # THE ARTEFACT NAME IS SET BEFORE ANYTHING IS NAMED FROM IT, as lang_sft does it.
    # artefact_tag is f"{run tag}_{stage label}", so a run tag of "<scheme>-instruct" and the
    # "sft" label give <scheme>-instruct_sft, which is the rule's artefact name.
    helpers.set_run_tag(f"{helpers.sft_scheme_name()}-instruct",
                        scheme=helpers.sft_scheme_name())
    register_chat_specials(ctx)
    path, source = initial_checkpoint(ctx, flag)
    ctx["log"](f"[instruct-sft] initial checkpoint: {_short(path)}")
    ctx["log"](f"[instruct-sft] chosen by:          {source}")
    ctx["log"](f"[instruct-sft] fine-tuning corpus: {_short(args.sft_dir)}")
    ctx["log"](f"[instruct-sft] steps {args.sft_steps:,} at batch/GPU {args.batch}")
    corpus, n_files, n_tok = sft.load_corpus(ctx["tok"], args.sft_dir, ctx["log"])
    if not corpus:
        ctx["log"](f"no corpus files under {args.sft_dir}: nothing to fine-tune on")
        return
    init = model_from(ctx, path, train_mode=True)
    sft.run(ctx, init, corpus)


if __name__ == "__main__":
    try:
        main()
    finally:
        from model import parallel
        parallel.dist_close()
