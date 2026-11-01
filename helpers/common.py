"""
common.py -- the setup every stage shares, and nothing stage-specific.

This module knows about arguments, devices, the tokenizer, the encoder, the model factory,
figures and generation previews. It does NOT know which stages exist, what they optimise or
in what order they run: each stage package (pretrain/, instruct_sft/, instruct_rlhf/, instruct_dpo/,
distill/) owns its own algorithm, corpus and reporting, and asks setup() only for what it
consumes. Adding or removing a stage therefore does not touch this file.

ALL configuration lives in default_config.py; parse_args() exposes every knob as a CLI flag whose
default comes from config, so a flag present in argv wins.

THREE LAYERS AND NO MORE, and this file owns the middle one's application:

    default_config.py   <   the yaml named by --config   <   the flags on the command line

    parse_args(stage=...)     argparse over default_config.py's values, under the yaml
    setup(args)               -> ctx dict (device, log, tok, enc, model factory, monitor,
                                 sampler, data, dirs); draws bpe_dynamics only when draw_bpe=True
    frozen(model)             a detached, eval-mode deep copy (reference / teacher)
    load_stage_model(ctx, s)  a fresh model with stage s's saved weights loaded (errors if missing)

Outputs: checkpoints/<stage>/{checkpoint_,history_}<model>_<pe>_<stage-label>.{pt,json},
         outputs/plots/<stage>/dynamics_<model>_<pe>_<stage-label>.pdf
"""
import argparse
import copy
import inspect
import os
import random
import sys

import torch

import default_config as config
from tools import config_file as cfgfile
from . import dataset_helpers as dsets
from . import utils as helpers
from .utils import progress, tag
from mirachat import generate, decode
from model import MiraLLM, parallel
from . import visualization as viz

from tokenizer import run as train_bpe    # stage 3's tokenizer builder (also draws bpe_dynamics)

ROOT = config.ROOT
CKDIR = config.CHECKPOINT_DIR
PLOTDIR = config.PLOT_DIR
BPE_PATH = config.BPE_PATH
N_SAMPLES = config.N_SAMPLES


def plot_dir(stage):
    """Figures are organised outputs/plots/<stage>/*.pdf -- one directory per stage.

    A stage whose artefacts live elsewhere (cot_sft writes into cot/) takes its figures there
    too, through the SAME mapping the checkpoints use, so a stage's two kinds of output cannot
    end up in differently-named directories."""
    # STAGE 8 SUPPLIES ITS OWN, and it is asked for it rather than guessed at: its figures go
    # to outputs/plots/instruct_reward/<scheme>_<dataset>/, the same two leaf levels its
    # checkpoints get, so the two kinds of output cannot end up in differently-named
    # directories. helpers.reward_plot_dir composes both.
    if stage == "reward":
        return helpers.reward_plot_dir()
    # STAGE 11 SUPPLIES ITS OWN, the same two leaf levels its checkpoints get.
    if stage == "dpo":
        return helpers.dpo_plot_dir()
    # An empty mapping means the stage supplies its own directory (lang_sft, whose figures go
    # under outputs/plots/lang-sft/<dataset>/); fall back to the stage LABEL rather than to
    # nothing, or every such stage would write into outputs/plots/ itself.
    sub = helpers.stage_dirname(stage) or helpers.stage_label(stage)
    return os.path.join(PLOTDIR, sub)


# lattice_norm_module (2026-10-04, retired 2026-10-05): MIRA's lattice input norm is RMSNorm only, and
# a cfg recorded while the key existed carries "rmsnorm" or the retired LayerNorm choices.
RETIRED_CFG_KEYS = ("use_ssm", "use_rope", "lattice_norm_module")


def translate_cfg(saved):
    """A checkpoint's model_cfg in today's vocabulary."""
    saved = {str(k).lower(): v for k, v in saved.items()}
    if saved.get("benchmark_family"):
        raise SystemExit(
            "[checkpoint] benchmark checkpoints must be restored by benchmark evaluators")
    if saved.get("checkpoint_wrapper"):
        raise SystemExit(
            "[checkpoint] wrapped checkpoints must be restored by their dedicated loader")
    cfg = {k: v for k, v in saved.items() if k not in RETIRED_CFG_KEYS and k != "vocab_size"}
    experimental_arch = cfg.pop("experimental_arch", None)
    cfg.pop("benchmark_architecture", None)
    cfg.pop("benchmark_family", None)
    cfg.pop("checkpoint_auxiliary", None)
    if experimental_arch:
        cfg["arch"] = experimental_arch
    if "pe" not in cfg and "use_ssm" in saved:
        cfg["pe"] = "ssm" if saved["use_ssm"] else "rope"
    # `pe` BECAME `arch` WHEN A THIRD ARCHITECTURE ARRIVED. Every checkpoint written before
    # that carries only `pe`, and it means exactly what `arch` means for the two values it
    # could hold, so it is promoted here rather than at each of the dozen places that read it.
    if "arch" not in cfg and "pe" in cfg:
        cfg["arch"] = cfg["pe"]
    if "arch" in cfg:
        cfg["arch"] = config.canonical_arch(cfg["arch"])
    if "lattice_order" in cfg:
        config.mira_map_arch(cfg)
        if "arch" in cfg:
            raise ValueError("this MIRA checkpoint records arch instead of lattice_modulus_map and "
                             "lattice_argument_map, the format before those keys, and cannot be loaded")
    # THROUGH arch_of RATHER THAN OFF THE KEY, because a MIRA cfg no longer records `arch`:
    # there is one architecture, so the name said nothing and went with the other ten fields
    # config.json does not write. arch_of reads the name when a checkpoint has one and answers
    # "mira-lattice" when it has none, so both generations take this branch and every baseline
    # still takes its own.
    if helpers.arch_of(cfg) == "mira-lattice":
        cfg["lattice_initializer"] = config.canonical_lattice_initializer(
            cfg.get("lattice_initializer", cfg.pop("lattice_init_method", "schur_init")), recorded=True)
    if "init_horizon" in cfg and "lattice_init_horizon" not in cfg:
        cfg["lattice_init_horizon"] = cfg.pop("init_horizon")
    if helpers.arch_of(cfg) == "mira-lattice" and "lattice_init_horizon" not in cfg:
        cfg["lattice_init_horizon"] = config.MODEL["lattice_init_horizon"]
    cfg.pop("pe", None)                 # one name below this line
    return cfg


def model_cfg_from_args(args, overrides=None):
    """default_config.MODEL with the command line applied.

    --mira_config picks depth, width and the scheme's own context window together, because
    those three are not independent choices: a scheme is a point on a size ladder, and mixing
    one scheme's depth with another's width produces a model that no reported number describes.
    --context_window then overrides the context alone, which is the one part of a scheme it is
    legitimate to vary on its own -- it is a training choice, not an architectural one.
    --arch selects the architecture, and with it the single source of position: "mira-lattice"
    (the selective lattice recurrence is itself the position, no encoding) or "rope" (the
    ablation control)."""
    cfg = dict(config.MODEL)
    if args is not None and getattr(args, "model_scheme", None):
        cfg.update(config.scheme(args.model_scheme))
    if args is not None and getattr(args, "context_window", ""):
        # A LIST HERE TOO: --context_window 1024,4096 pins the whole schedule, and a single
        # number pins one window. block_size is the largest of them, because that is the widest
        # sequence the model will ever be handed.
        cfg["block_size"] = max(config.windows(args.context_window))
    elif args is not None and getattr(args, "model_scheme", None):
        cfg["block_size"] = config.context_window(args.model_scheme)
    elif "block_size" not in cfg:
        cfg["block_size"] = config.context_window(config.DEFAULT_SCHEME)
    if args is not None and (getattr(args, "arch", None) or getattr(args, "pe", None)):
        cfg["arch"] = getattr(args, "arch", None) or args.pe
        cfg.pop("pe", None)
    cfg.update(overrides or {})
    if config.is_lattice_arch(cfg.get("arch")):
        config.mira_map_arch(cfg)
        cfg.pop("arch")
    return cfg


def _say(m):
    """print(m, flush=True), FROM THE MAIN PROCESS ALONE.

    THE ONE GATE FOR EVERY ROUTINE LINE THIS PIPELINE EMITS. Under torchrun there are four
    processes writing to one terminal, and four copies of every line do not merely repeat: they
    interleave mid-line and the log stops being readable at the moment it is most needed. Every
    stage's `log` is this (see setup), and so is the model banner and the [gpu]/[parallel]
    lines, so a stage never has to know whether it is a rank.

    WHAT IS NOT GATED, DELIBERATELY. Exceptions, tracebacks and warnings.warn go to stderr
    through machinery this does not touch, so a crash on rank 3 still says so -- a rank that
    fails silently is a rank nobody can debug. Only routine progress and information is here.

    THE FORMAT IS UNTOUCHED. Off DDP is_main() is true and this is print(m, flush=True), which
    is what every one of these call sites already was; the output of a one-GPU run is
    byte-identical to what it was before this existed."""
    if parallel.is_main():
        print(m, flush=True)


def build_model(tok, device, model_cfg=None, args=None):
    """The from-scratch MiraLLM (random init). This is the ONLY place a model is constructed,
    so every stage trains the same architecture."""
    m_cfg = model_cfg_from_args(args, model_cfg) if args is not None else model_cfg_from_args(None)
    if model_cfg and args is None:
        m_cfg.update(model_cfg)
    m_cfg.pop("vocab_size", None)
    for k in RETIRED_CFG_KEYS:
        m_cfg.pop(k, None)
    # A KEY THE CONSTRUCTOR DOES NOT TAKE IS DROPPED, BY SIGNATURE RATHER THAN BY A LIST OF
    # NAMES. MiraLLM.__init__ has no **kwargs, so one unrecognised key is a TypeError, and a
    # checkpoint's model_cfg records more than the constructor's arguments: `lattice_state` is
    # written into model.cfg by mira_llm itself and is not a parameter, which is what stopped
    # stage 11 loading the instruct-sft checkpoint. A list of names would have to be extended
    # for every such key that ever existed; the signature is the question being asked.
    accepted = set(inspect.signature(MiraLLM.__init__).parameters) - {"self", "vocab_size"}
    dropped = sorted(set(m_cfg) - accepted)
    if dropped:
        _say(f"[checkpoint] {len(dropped)} recorded field(s) MiraLLM does not take, dropped: "
             f"{', '.join(dropped)}")
        m_cfg = {k: v for k, v in m_cfg.items() if k in accepted}
    m = MiraLLM(vocab_size=len(tok), **m_cfg)
    # NOT A cfg KEY, so it is not written into the checkpoint and two runs that differ only
    # in it produce identical weights. It is a property of the card, set per run.
    m.grad_checkpoint = bool(getattr(args, "grad_checkpoint", False)) if args is not None \
        else bool(config.TRAIN.get("grad_checkpoint", False))
    # THE MODEL SUMMARY, AND THIS IS THE ONE PLACE IT IS PRINTED FROM.
    #
    #     "for all trainers, it must print the model summary, including, layers, heads,
    #      d_model, and trainable parameter size! humans need rich information!!!!"
    #
    # build_model is what every stage of the pipeline reaches -- pretrain and the CoT SFT and
    # lang_sft through ctx["new_model"](), the SFT, reward, RLHF, DPO and distill stages through
    # load_stage_model, which rebuilds at the checkpoint's own architecture and comes back here.
    # So the block is emitted once, here, rather than pasted into ten trainers; helpers.utils
    # owns the format, and the two trainers that build a model WITHOUT this function --
    # scaling_laws (its own loop over a grid of sizes) and benchmark (baselines.build_lm) -- call
    # the same function with their own prefix.
    #
    # THROUGH _say, so it is rank 0 alone under DDP. The counts inside it are measured from
    # model.parameters(), not read from the configuration, so a stage holding a frozen model
    # reports what it will actually update.
    helpers.model_summary(m, log=_say)
    _say(f"[model {helpers.model_name(m.cfg)}] positions: " +
         {"mira-lattice": "implicit (the lattice recurrence is the position, no encoding)",
          "ssm": "implicit (state space module before attention, no encoding)",
          "rope": "rotary inside attention (ABLATION: no state space module)",
          }.get(helpers.arch_of(m.cfg), helpers.arch_of(m.cfg)))
    _say(f"[model mira_llm] cfg={m.cfg}")
    if device.type in ("cuda", "npu") or parallel.dist_env()[2] > 1:
        mode = (getattr(args, "distributed_parallelism", None)
                or config.TRAIN["distributed_parallelism"])
        m = parallel.distribute(m, mode, device=device, log=_say)
    else:
        m = m.to(device)
    return m


def parse_args(argv=None, stage=None):
    """Every trainer knob as a CLI flag, defaulting to default_config.py's values, with the
    yaml named by --config read in between.

        default_config.py   <   --config's yaml   <   the flags in argv

    THE YAML'S TOKENS GO FIRST and the caller's argv second, because argparse keeps the LAST
    occurrence of a repeated option: that one ordering is the entire precedence rule, and it is
    the same one `python -m pretrain.run $COMMON_FLAGS $PRETRAIN_FLAGS "$@"` relied on."""
    t = config.TRAIN
    argv = list(sys.argv[1:] if argv is None else argv)
    from_yaml, cfg_path = cfgfile.resolve(argv, stage)
    from_environment = []
    if os.environ.get("DISTRIBUTED_PARALLELISM"):
        from_environment = ["--distributed_parallelism",
                            os.environ["DISTRIBUTED_PARALLELISM"]]
    # WHAT THE RUN WAS ACTUALLY GIVEN, remembered for _was_given: a scheme default must not
    # quietly replace a value the yaml or the person set, and after this line the two are one
    # command line.
    global _EFFECTIVE_ARGV
    _EFFECTIVE_ARGV = from_yaml + from_environment + argv
    ap = argparse.ArgumentParser(description="mira_llm pipeline stage")
    # THE ONLY FLAG THAT IS NOT A SETTING: it says which file the settings came from. Declared
    # so that argparse accepts it and `--help` lists it; its value was read above, before
    # parsing, because what it names decides what argparse is handed.
    ap.add_argument("--config", default=cfg_path, metavar="FILE",
                    help="the yaml this run's settings come from (default: "
                         f"{os.path.relpath(cfg_path, config.ROOT)}). Written by "
                         "`python -m tools.config_wizard`. A flag given here beats it; a "
                         "missing default file is not an error, a missing named one is")
    ap.add_argument("--dataset", default=t["dataset"],
                    help="\"hh\" for the downloaded rlhf_hh tree, or a PATH to your "
                         "own folder of json/jsonl preference records (absolute, or "
                         "relative to --data_dir); the layout is detected, not declared")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="SEC.key=v",
                    help="override any value in default_config.py's dictionaries, e.g. "
                         "--set BPE.min_freq=3 --set RLHF.kl_coef=0.02. Repeatable. "
                         "Sections: BPE PRETRAIN SFT REWARD RLHF COT DPO DISTILL TRAIN "
                         "SCALING MODEL")
    ap.add_argument("--gpu", choices=["auto", "cuda", "npu", "mps", "cpu"], default=t["gpu"])
    ap.add_argument("--seed", type=int, default=t["seed"])
    ap.add_argument("--batch", "--batch_size", type=int, default=t["batch"], dest="batch",
                    help="sequences per GPU per optimiser step; the global batch is this "
                         "value multiplied by the number of data-parallel GPUs. --batch_size is "
                         "the same flag and overrides the yaml's BATCH")
    ap.add_argument("--micro_batch", type=int, default=t["micro_batch"],
                    help="sequences per GPU per forward pass; 0 = the whole batch at once "
                         "(default). "
                         "Splits one step into several passes whose gradients accumulate, so "
                         "the optimiser sees the same step at a fraction of the activations.")
    # CHUNKED LOSS, ON BY DEFAULT. The vocabulary projection is the largest tensor in the step
    # at any context worth the name, and slicing it changes the memory and nothing else -- the
    # positions, the targets and the normalisation are identical. --no_chunked_loss returns to
    # projecting the whole sequence at once, which is what to do when comparing against an
    # older run or when the sequences are short enough that the slicing only costs time.
    ap.add_argument("--loss_chunk", type=int, default=t["loss_chunk"],
                    help="positions per slice of the vocabulary projection; 0 disables chunking")
    # LAYERS ACROSS THE GPUs, ON BY DEFAULT. With more than one CUDA device the blocks are
    # divided into contiguous runs, one per device, so a model or a context window that does
    # not fit one card fits several. It buys MEMORY, not speed -- one device is busy at a time
    # -- so --no-tensor_parallel is the right choice whenever the run already fits one card.
    ap.add_argument("--distributed_parallelism", choices=parallel.MODES,
                    default=t["distributed_parallelism"])
    # MIXED PRECISION, ON BY DEFAULT WHEREVER THE HARDWARE HAS IT. fp32 weights and fp32
    # optimiser state, bfloat16 activations and matmuls: the standard recipe every modern
    # pretrain uses, and the reason full-fp32 training is obsolete at scale -- twice the memory
    # and slower, for no quality gain. It is resolved against the DEVICE in setup(), so this
    # flag asks for it and the hardware decides; on CPU and MPS it silently stays off, because
    # bf16 without the hardware paths is emulated and slower than the fp32 it replaced.
    ap.add_argument("--mixed_precision", dest="mixed_precision", action="store_true",
                    default=True,
                    help="bfloat16 activations with fp32 weights and optimiser state, on CUDA "
                         "(default)")
    ap.add_argument("--no-mixed_precision", "--no_mixed_precision", dest="mixed_precision",
                    action="store_false",
                    help="train in full fp32 everywhere")
    # GRADIENT CHECKPOINTING. Stores each block's INPUT and recomputes its interior during
    # the backward pass, so what is resident is one block's activations rather than all L.
    # It is what makes the long windows reachable at all: a single sequence at 32,768 tokens
    # is 176 GiB of block activations at -l without it, and no micro-batch can reduce one
    # sequence. Costs one extra forward pass per step, about 30% of step time.
    ap.add_argument("--grad_checkpoint", dest="grad_checkpoint", action="store_true",
                    default=bool(t.get("grad_checkpoint", True)),
                    help="recompute each block in the backward pass instead of storing it "
                         "(default): ~L times less activation memory for ~30%% more time")
    ap.add_argument("--no-grad_checkpoint", "--no_grad_checkpoint", dest="grad_checkpoint",
                    action="store_false",
                    help="store every block's activations; faster, and needs the memory")
    # INCREMENTAL DECODING for the stages that generate. On by default: without it, producing
    # n tokens costs O(n^2) forward work, which the context schedule made intolerable.
    ap.add_argument("--kv_cache", dest="kv_cache", action="store_true", default=True,
                    help="cache keys, values and the recurrence state while generating (default)")
    ap.add_argument("--no-kv_cache", "--no_kv_cache", dest="kv_cache", action="store_false",
                    help="recompute the whole prefix at every generated token")
    ap.add_argument("--kv_cache_size", type=float, default=config.COT["kv_cache_size"],
                    metavar="GiB",
                    help="what the prefix cache may hold, in GiB; a rollout larger than this "
                         "is decoded in groups that fit (default: "
                         f"{config.COT['kv_cache_size']})")
    ap.add_argument("--mira_cuda_bin", choices=config.MIRA_CUDA_BINS,
                    default=(os.environ.get("MIRA_CUDA_BIN")
                             or cfgfile.read(cfg_path).get("mira_cuda_bin", config.MIRA_CUDA_BIN)),
                    help="MIRA CUDA backend package to load")
    ap.add_argument("--mira_config", dest="model_scheme", default=config.PRETRAIN["model_scheme"],
                    choices=list(config.SCHEMES) + list(config.LEGACY_SCHEME_ALIASES),
                    help="which MIRA scheme to build (mira-xs, mira-s, mira-m, mira-l): its "
                         "architecture from config/default/config_mira.yaml")
    ap.add_argument("--pretrain_dir", default="",
                    help="pretraining corpus directories, comma-separated; each is tokenised "
                         "and cached on its own and they are mixed in proportion to their "
                         "tokens. Empty = the scheme's own list from "
                         "default_config.PRETRAIN_CORPUS, which is set for every scheme. "
                         "Read locally: no stage downloads, see tools/download_data.py")
    ap.add_argument("--sft_dir", default="",
                    help="fine-tuning data directory; empty = default_config.SFT_DIR, "
                         "UltraChat. A conversation mixture, an hh tree or your own "
                         "folder of json records: the layout is detected and reported")
    ap.add_argument("--instruct_dir", default="",
                    help="root of the instruction-tuning tree (alpaca_gpt4/, rlhf_hh/) used "
                         "by stages 7, 8, 9 and 11, and by distillation; empty = "
                         "default_config.INSTRUCT_DIR")
    ap.add_argument("--context_window", default=config.PRETRAIN["context_window"],
                    help="context window in tokens; 0 = the scheme's own (512 for tiny and "
                         "-s, 1024 for -m and -l). Not an architectural limit -- the model "
                         "refers to no "
                         "absolute position -- but what it is TRAINED at, and what the "
                         "checkpoint records as its block_size")
    ap.add_argument("--beta", type=float, default=t["beta"])
    ap.add_argument("--val_frac", type=float, default=t["val_frac"])
    ap.add_argument("--limit", type=int, default=t["limit"], help="cap #pairs loaded (0 = all)")
    ap.add_argument("--lr_schedule", default=t["lr_schedule"], choices=["cosine", "constant"])
    ap.add_argument("--last_lr", type=float, default=t["last_lr"],
                    help="cosine floor: the final lr the schedule decays to, an absolute value")
    ap.add_argument("--warmup_steps", type=int, default=t["warmup_steps"],
                    help="linear warmup: the rate ramps to the stage lr over the first N "
                         "steps, and the cosine spans the steps that remain (0 = no warmup)")
    ap.add_argument("--checkpoint_every_steps", type=int, default=t["checkpoint_every_steps"])
    # ONE CADENCE FOR BOTH. Redraws the figures AND profiles the model on the same step, so
    # a figure is never drawn from statistics measured a cadence earlier. 0 turns off both.
    ap.add_argument("--plot_every_steps", type=int, default=t["plot_every_steps"],
                    help="redraw the figures and profile the model every N steps (0 = off). "
                         "The profile is per layer -- memory horizons, decays, gates and "
                         "per-bank state of both recurrences, and under arch=mira the "
                         "attention scores, distances and metric spectrum -- drawn in the "
                         "dynamics figure and recorded in full to "
                         "checkpoints/<stage>/dynamics.jsonl")
    ap.add_argument("--profile_every_steps", type=int, default=t["profile_every_steps"],
                    help="measure the per-layer lattice diagnostics every N steps (0 = off). "
                         "Statistics only -- memory horizons, pole radii and angles, the "
                         "fraction of coefficients at the activation ceiling, the ladder and "
                         "gate magnitudes -- appended to checkpoints/<stage>/dynamics.jsonl. "
                         "No coefficient tensor is stored")
    ap.add_argument("--print_samples_every_steps", type=int,
                    default=t["print_samples_every_steps"],
                    help=f"print {N_PREVIEW} generation examples every N steps, in every stage "
                         f"that generates (0 = off; default {t['print_samples_every_steps']})")
    ap.add_argument("--arch", "--pe", dest="arch", default=config.MODEL["arch"],
                    type=config.canonical_arch,
                    choices=["cauchy_x_mobius", "cauchy_x_hyperbolic", "cauchy_x_identity", "gaussian_x_mobius", "gaussian_x_hyperbolic", "gaussian_x_identity", "expexp_x_mobius", "expexp_x_hyperbolic", "expexp_x_identity", "expsoftplus_x_mobius", "expsoftplus_x_hyperbolic", "expsoftplus_x_identity", "hill_x_mobius", "hill_x_hyperbolic", "hill_x_identity", "mira-lattice"],
                    help="the MIRA lattice parameterization; the recurrence carries position "
                         "without positional encoding. "
                         "'mira', 'cdsa', 'mira-softmax' and 'mira-ln' are accepted as former "
                         "names and resolve to 'mira-lattice'. "
                         "--pe is the former name and "
                         "is still accepted")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.set_defaults(resume=t["resume"])
    # held-out probe (used by the DPO stage)
    ap.add_argument("--eval_every", type=int, default=t["eval_every"])
    ap.add_argument("--eval_pairs", type=int, default=t["eval_pairs"])
    ap.add_argument("--n_hist", type=int, default=t["n_hist"])
    # per-stage budgets / learning rates
    ap.add_argument("--bpe_merges", type=int, default=config.BPE["num_merges"])
    ap.add_argument("--pretrain_steps", type=int, default=config.PRETRAIN["steps"])
    ap.add_argument("--pretrain_epochs", type=int, choices=(1,), default=1)
    ap.add_argument("--peak_lr", type=float, default=config.PRETRAIN["lr"])
    ap.add_argument("--sft_steps", type=int, default=config.SFT["steps"])
    ap.add_argument("--sft_lr", type=float, default=config.SFT["lr"])
    ap.add_argument("--reward_steps", type=int, default=config.REWARD["steps"])
    ap.add_argument("--reward_lr", type=float, default=config.REWARD["lr"])
    ap.add_argument("--rlhf_steps", type=int, default=config.RLHF["steps"])
    ap.add_argument("--rlhf_lr", type=float, default=config.RLHF["lr"])
    ap.add_argument("--cot_steps", type=int, default=config.COT["steps"])
    ap.add_argument("--cot_lr", type=float, default=config.COT["lr"])
    # THE REASONING SFT THAT PRECEDES GRPO. Same problems, but supervised: the dataset's own
    # reference trace, rewritten into this pipeline's <think>/<answer> form. It teaches the
    # FORMAT, which GRPO cannot teach a base model that never samples it -- a reward for
    # something no completion in the group produces is a reward with no gradient.
    ap.add_argument("--cot_sft_steps", type=int, default=config.COT["sft_steps"])
    ap.add_argument("--cot_sft_lr", type=float, default=config.COT["sft_lr"])
    ap.add_argument("--cot_sft", dest="cot_sft", action="store_true",
                    default=config.COT["sft"],
                    help="fine-tune on the reference traces before GRPO (default)")
    ap.add_argument("--no-cot_sft", "--no_cot_sft", dest="cot_sft", action="store_false",
                    help="skip the reasoning SFT: GRPO straight from --cot_init "
                         "(the R1-Zero setting)")
    ap.add_argument("--cot_init", default=config.COT["init_stage"],
                    choices=["pretrain", "sft", "rlhf", "dpo"],
                    help="checkpoint the reasoning SFT starts from -- and, with --no-cot_sft, "
                         "the checkpoint GRPO itself starts from; 'pretrain' is the R1-Zero "
                         "setting (no supervised reasoning anywhere before the RL)")
    ap.add_argument("--cot_samples_every", type=int, default=config.COT["samples_every"],
                    help="print the GRPO rollout samples every N steps; 0 = follow "
                         "--print_samples_every_steps. Separate because a GRPO step costs a "
                         "whole group of rollouts and there are far fewer of them")
    ap.add_argument("--cot_batch", type=int, default=config.COT["batch"],
                    help="problems per GRPO step; the rollout is this x --cot_group. Separate "
                         "from --batch because a GRPO step is that many SEQUENCES of "
                         "context-window length, not that many training examples")
    ap.add_argument("--cot_group", type=int, default=config.COT["group_size"],
                    help="completions sampled per problem; the group is GRPO's baseline "
                         "and must be >= 2")
    # stages 12 and 13: vocabulary adaptation, then language adaptation
    # --only AND --tokenizer ARE SHARED BY BOTH, which is what lets them compose. Stage 12
    # extends a vocabulary for a corpus and writes it under a name; stage 13 trains on that
    # corpus with that vocabulary. Naming the pair the same way twice is what makes the second
    # stage able to find what the first one wrote, including when the name was chosen by hand.
    ap.add_argument("--only", default="", metavar="DATASET",
                    help="the corpus these stages act on, overriding --lang_sft_dataset: a "
                         "name under data/download/, or a path")
    ap.add_argument("--benchmark_dataset", default=config.BENCHMARK["dataset"],
                    metavar="NAME",
                    help="the benchmark's corpus by name -- one of "
                         + ", ".join(sorted(config.BENCHMARK_CORPORA))
                         + ". Selecting it also sets the text column that corpus keeps its "
                           "text in, which differs between them")
    ap.add_argument("--tokenizer", default="", metavar="NAME",
                    help="directory under checkpoints/ holding the adapted vocabulary "
                         "(default: bpe-sft-<dataset>). Stage 12 writes NAME/bpe.json; "
                         "stage 13 reads it")
    ap.add_argument("--lang_sft_dataset", default=config.LANG_SFT["dataset"],
                    help="corpus to adapt to: a name under data/download/, or a path")
    ap.add_argument("--lang_sft_short", default=config.LANG_SFT["short"],
                    metavar="NAME",
                    help="the label in every stage-11 filename, e.g. `gaelic`")
    ap.add_argument("--lang_sft_merges", type=int, default=config.LANG_SFT["extra_merges"],
                    help="merges to ADD to the base vocabulary for the new language")
    ap.add_argument("--lang_sft_steps", type=int, default=config.LANG_SFT["steps"])
    ap.add_argument("--lang_sft_lr", type=float, default=config.LANG_SFT["lr"])
    ap.add_argument("--checkpoint", default="", metavar="FILE",
                    help="stage 11 (DPO) only: the checkpoint its policy and its frozen "
                         "reference both start from. Default: the newest one under "
                         "checkpoints/instruct_sft/, and the stage exits if there is none")
    ap.add_argument("--dpo_steps", type=int, default=config.DPO["steps"])
    ap.add_argument("--dpo_lr", type=float, default=config.DPO["lr"])
    ap.add_argument("--distill_steps", type=int, default=config.DISTILL["steps"])
    ap.add_argument("--distill_lr", type=float, default=config.DISTILL["lr"])
    for old, new in cfgfile.RENAMED_SETTINGS.items():
        if any(a == f"--{old}" or a.startswith(f"--{old}=") for a in _EFFECTIVE_ARGV):
            raise SystemExit(f"--{old} is replaced by --{new}" + (
                ", the final learning rate of the schedule (e.g. --last_lr 1e-5)"
                if old == "lr_min_factor" else ""))
    args = ap.parse_args(_EFFECTIVE_ARGV)
    args.disable_pretrain_micro_batch = stage == "pretrain"
    if args.disable_pretrain_micro_batch and args.micro_batch != 0:
        ap.error("micro-batching is disabled for pretraining")
    import importlib
    importlib.import_module("model.mira_mixer").use_mira_cuda(args.mira_cuda_bin)
    # OLD YAML FILES ARE INPUT ALIASES, NOT OUTPUT NAMES. Canonicalise once before any path,
    # corpus, batch or run tag is derived, so resumed weights load and new artefacts say MIRA.
    args.model_scheme = config.canonical_scheme(args.model_scheme)
    lattice = config.scheme_lattice(args.model_scheme)
    config.MODEL.update({k: v for k, v in lattice.items() if k != "arch"})
    if not (_was_given("--arch", "arch") or _was_given("--pe", "arch")):
        args.arch = config.mira_map_arch(lattice)
    args.arch = config.canonical_arch(args.arch)
    config.MODEL["arch"] = args.arch
    args.config = cfg_path
    args.model_dir = config.MODEL_DIR
    args.data_dir = config.DOWNLOAD_DIR
    args.max_len = model_cfg_from_args(args)["block_size"]

    # THE CORPUS DIRECTORIES, resolved once here rather than read from the config module at
    # each use. --instruct_dir moves a whole tree, so it is applied through config's own setter
    # to keep the derived directories (the fine-tuning data, alpaca, hh and the RLHF
    # prompt bank) consistent; repointing the root while one of them still pointed elsewhere is
    # how a run ends up fine-tuning on one dataset and scoring preferences from another.
    if args.instruct_dir:
        config.set_instruct_root(os.path.abspath(args.instruct_dir))
    args.instruct_dir = config.INSTRUCT_DIR
    args.sft_dir = os.path.abspath(args.sft_dir) if args.sft_dir else config.SFT_DIR
    # The pretraining corpus is per SCHEME: only mira-s ships with one, so -m and -l resolve
    # to the empty string and the stage says so rather than scanning a directory that is not
    # there and reporting "0 files" as though that were a corpus.
    # A LIST OF CORPORA, not one. Each keeps its own token stream and its own signature, so
    # adding one tokenises only the one added and every stream already built is untouched --
    # which is what lets a corpus be added to a run that is already under way. Comma-separated
    # on the command line, because that is what a shell variable can carry.
    if args.pretrain_dir:
        dirs = [d.strip() for d in str(args.pretrain_dir).split(",") if d.strip()]
    else:
        dirs = list(config.PRETRAIN_CORPUS.get(args.model_scheme, []))
    args.pretrain_dir = [os.path.abspath(d) for d in dirs]

    if not _was_given("--batch", "batch"):
        args.batch = int(config.SCHEME_BATCH.get(args.model_scheme, args.batch))
    # THE STEP BUDGET IS PER SCHEME, for the same reason the batch is. A step carries
    # batch x (ctx - 1) tokens, which differs by a factor of two between -m and -l, so ONE step
    # count cannot be 1.5 epochs for both -- it would be 1.5 for one and 3 for the other, and
    # nothing in the run would say so. Same precedence as --batch: an explicit value is the
    # person's decision and is left alone.
    if not _was_given("--pretrain_steps", "pretrain_steps"):
        args.pretrain_steps = int(config.SCHEME_STEPS.get(args.model_scheme,
                                                          args.pretrain_steps))
    # MICRO-BATCHING IS OFF UNLESS ASKED FOR HERE. It costs nothing in accuracy but it does
    # cost time, and a step that already fits should be run whole. What each scheme needs at
    # its longest window is SCHEME_MICRO_TOKENS, in TOKENS per pass -- and it is read by
    # model.lm.context_plan_for when it builds the window schedule, NOT here. Nothing in
    # parse_args sets args.micro_batch from a scheme; only a flag, the yaml, or a stage's own
    # default (e.g. SFT['micro_batch']) does.
    return args


# THE COMMAND LINE THIS RUN WAS ACTUALLY BUILT FROM: the yaml's tokens followed by the caller's
# own argv. parse_args fills it; before that it is simply what was typed, so a module that asks
# _was_given without having gone through parse_args gets the truthful answer for its own argv.
_EFFECTIVE_ARGV = None


_FLAG_ALIASES = {"--batch": ("--batch", "--batch_size")}


def _was_given(flag, dest):
    """Was this setting chosen explicitly -- in the yaml, or on the command line?

    A scheme default must never quietly replace a value somebody set. Both ways of setting one
    end up as the same flag in the same list (see parse_args), so there is one place to look."""
    argv = sys.argv[1:] if _EFFECTIVE_ARGV is None else _EFFECTIVE_ARGV
    return any(a == name or a.startswith(name + "=")
               for name in _FLAG_ALIASES.get(flag, (flag,)) for a in argv)


def frozen(model):
    """A detached, eval-mode deep copy: the frozen reference pi_ref / teacher."""
    ref = copy.deepcopy(model)
    ref.requires_grad_(False)
    ref.eval()
    return ref


N_PREVIEW = 20        # generation examples printed every --print_samples_every_steps

# FEWER, WHEN EACH ONE FILLS THE CONTEXT WINDOW. A chain-of-thought sample runs to thousands of
# tokens instead of sixty, so twenty of them is a wall of text nobody reads to the end and
# several minutes of generation between training steps. Five is what fits on a screen and is
# still enough to see whether the format holds across different prompts.
N_PREVIEW_LONG = 5

INSTRUCT_MAX_NEW = 250


def corpus_prompts(corpus, n=N_PREVIEW, n_words=12, seed=0, tok=None):
    """Preview prompts drawn from a corpus: the first `n_words` words of `n` sampled
    documents.

    Three corpus shapes, because the previews must work wherever the corpus came from: a
    memory-mapped TokenStream (sampled by document index -- only those documents are paged
    in, never the corpus), pre-tokenised records, and plain text records."""
    rng = random.Random(seed)
    if hasattr(corpus, "doc") and hasattr(corpus, "n_docs"):
        idx = rng.sample(range(corpus.n_docs), min(n, corpus.n_docs))
        texts = [decode(tok, corpus.doc(i)[:4 * n_words]) if tok is not None else ""
                 for i in idx]
    else:
        docs = rng.sample(corpus, min(n, len(corpus)))
        texts = [decode(tok, d["ids"][:4 * n_words]) if ("ids" in d and tok is not None)
                 else d.get("chosen", "") for d in docs]
    return [p for p in (" ".join(t.split()[:n_words]) for t in texts) if p]


def make_preview(tok, device, max_len, log, prompts, n=N_PREVIEW, fill_window=False,
                 max_new=None):
    """A `preview(model, stage, step)` closure that prints `n` generations from a FIXED set
    of prompts in a human-friendly format (plain print, NO timestamp prefix -- these blocks
    are meant to be read, not parsed):

        [1/20] PROMPT:   the prompt ...
               RESPONSE: what the model generated ...

    The prompts are fixed so consecutive previews are directly comparable: the reader
    watches the SAME prompts get better (or collapse) over training. Called by every
    generative stage (pretrain/instruct_sft/rlhf/dpo/distill) at each
    --print_samples_every_steps.

    `fill_window` GENERATES UNTIL THE CONTEXT WINDOW IS FULL instead of stopping at
    SAMPLING["max_new"], and prints the result whole. The chain-of-thought stages use it,
    because there the LENGTH IS THE RESULT: the behaviour those runs exist to show is a policy
    that reasons for longer, and a preview capped at a few hundred tokens would cut off the
    thing being looked for -- and cut it off at the same place every time, so the reader could
    not even tell it was growing. The budget is computed per prompt, from what the window has
    left once that prompt is in it.

    `max_new` OVERRIDES SAMPLING["max_new"] FOR THIS PREVIEW ALONE, and it is a per-stage
    number because the stages do not want the same one. SAMPLING["max_new"] is 60, which is a
    glance at whether a pretrained model has learned English; an instruction-tuned model is
    being read for whether its ANSWER is any good, and sixty tokens cuts every answer off
    mid-sentence. Passing it here rather than raising the global keeps pretrain, rlhf, dpo and
    distill previewing at the length they were tuned to -- a preview is generated between
    training steps, and thirty-four times the tokens is thirty-four times the wait."""
    s = config.SAMPLING
    budget_max_new = int(max_new if max_new else s["max_new"])
    eos = helpers.stop_token_id(tok)
    sel = [p for p in prompts if p.strip()][:n]

    def preview(model, stage, step):
        # RANK 0 GENERATES; THE OTHER RANKS RETURN. Every rank holds the same weights and would
        # print the same samples from the same prompts, so N-1 of the blocks are duplicates
        # interleaved with each other. Returning before the model is touched leaves eval/train
        # mode exactly as it was on those ranks. Off DDP this is always true.
        if not parallel.is_main():
            return
        was_training = model.training
        model.eval()
        out = print          # plain, untimestamped: the log()'s "[  123s]" prefix ruins it
        cap = "fills the context window" if fill_window else f"max_new={budget_max_new}"
        out(f"\n===== {tag(stage)} @ step {step}: {len(sel)} generation examples "
            f"(temp={s['temperature']} top_k={s['top_k']} top_p={s['top_p']}, {cap}) =====\n",
            flush=True)
        for i, pr in enumerate(sel, 1):
            ids = tok(pr, add_special_tokens=False)["input_ids"]
            budget = max(1, int(max_len) - len(ids)) if fill_window else budget_max_new
            with torch.no_grad():
                gen = generate(model, ids, device, budget, s["temperature"],
                               s["top_k"], s["top_p"], eos, max_len,
                               desc=f"{tag(stage)} sample {i}/{len(sel)}" if fill_window else "")
            text = decode(tok, gen)
            out(f"[{i}/{len(sel)}] PROMPT:", flush=True)
            out(pr, flush=True)
            out(f"{' ' * 7}RESPONSE:", flush=True)
            out(text, flush=True)
            out("", flush=True)
        out(f"===== end of {tag(stage)} examples @ step {step} =====\n", flush=True)
        if was_training:
            model.train()
    return preview


def _make_sampler(tok, device, max_len, log, prompts, eos_id):
    """A `sample(model, stage)` closure for the END-OF-STAGE samples: the same readable
    PROMPT:/RESPONSE: block as the periodic preview (no progress bar -- a bar redrawing
    itself between printed lines makes the block unreadable), labelled 'final'."""
    body = make_preview(tok, device, max_len, log, prompts, n=N_SAMPLES)
    def sample(model, stage, n=N_SAMPLES):
        body(model, stage, "final")
    return sample


def setup(args, need_pairs=True, pretokenize_pairs=False, draw_bpe=False, build_bpe=True):
    """Build the context a stage needs, and NOTHING ELSE.

    A stage declares what it consumes; common has no table of stages and no idea which one is
    running. `need_pairs` loads the instruction data (and reports its statistics);
    `pretokenize_pairs` additionally caches their token ids. A corpus stage passes
    need_pairs=False and reads its own directory itself, so pretraining never touches the
    preference file. The one exception is a tokenizer that has to be BUILT: its vocabulary is
    shared, so it is trained over every corpus regardless of who asked for it.
    """
    # plain, no elapsed-time prefix (human-readable logs), and FROM RANK 0 ALONE -- this is the
    # `log` every stage is handed, so gating it here is what stops four processes interleaving
    # four copies of every line. Nothing about the text or the format changes; see _say.
    def log(m): _say(m)

    # --set FIRST, before anything reads a configuration value. The stages hold references to
    # default_config's dictionaries rather than copies, so applying the overrides here is what
    # makes them take effect everywhere without a flag per setting.
    config.apply_overrides(getattr(args, "overrides", []), log)

    # THE PROCESS GROUP IS JOINED BEFORE ANYTHING TOUCHES A DEVICE, and that ordering is the
    # whole reason it is here rather than at the first place a rank is needed. dist_init calls
    # torch.cuda.set_device(local_rank), which is what makes the bare name "cuda" mean THIS
    # rank's card; anything allocated on "cuda" before it lands on cuda:0 on every rank at once.
    # Silent when the run was not launched by torchrun, so every stage still calls it and every
    # single-process run is unaffected.
    os.environ["GPU"] = helpers.resolve_device(args.gpu).type
    parallel.dist_init(log, device=helpers.resolve_device(args.gpu))

    # every checkpoint, history and figure this run writes is named after the configuration,
    # so an ablation never overwrites the run it is compared with
    tag = helpers.set_run_tag(model_cfg_from_args(args))
    # WHAT THIS RUN WRITES, and only that. args.data_dir is data/download -- the DOWNLOADED
    # corpora, which are input: read-only, produced by the download tool and by nothing else.
    # Creating it here wrote an empty directory into the data root on any machine that had not
    # downloaded anything, and turned "the corpus is missing" -- which is what happened, and is
    # actionable -- into "the corpus is empty", which is neither. A stage reads that tree; it
    # does not make it. Everything a run PRODUCES goes under cache/, checkpoints/ or outputs/.
    for d in (args.model_dir, CKDIR, PLOTDIR):
        os.makedirs(d, exist_ok=True)
    device = helpers.resolve_device(args.gpu)
    args._device_type = device.type
    # ASKED FOR AND AVAILABLE ARE DIFFERENT THINGS, and the difference is settled HERE, once,
    # before any stage reads the flag. A run that asked for mixed precision on a CPU gets fp32
    # and is TOLD so; the alternative is a flag that reads as on in the argument dump while
    # every forward quietly ignores it.
    if args.mixed_precision and not helpers.mixed_precision_ok(device):
        log(f"[precision] mixed precision requested but {device} has no bfloat16 path: "
            f"training in fp32")
        args.mixed_precision = False
    log(f"device={device}  run={tag}  dataset={args.dataset}  "
        f"checkpoints={CKDIR}  plots={PLOTDIR}")
    log("precision: " + ("bf16 activations, fp32 weights and optimiser state (mixed)"
                         if args.mixed_precision else "fp32 everywhere"))

    # resolve() has already printed the selected modular YAML verbatim. Dumping every field in
    # the shared argparse namespace immediately afterward mixed unrelated stages into that
    # configuration record -- a benchmark log contained reward, RLHF, DPO, distillation and
    # language-SFT defaults that the benchmark never consumes. The selected YAML plus the
    # explicit override lines emitted by apply_overrides() are the run's configuration record.

    if getattr(args, "tokenizer", ""):
        _saved, _bpe_src = train_bpe.resolve_named(args.tokenizer, log)
    else:
        _saved, _bpe_src = train_bpe.resolve(BPE_PATH, config.BPE_DOWNLOAD_PATH, log)
    need_bpe_build = _saved is None
    # A STAGE THAT INHERITS A MODEL INHERITS ITS VOCABULARY, and cannot build one. Stage 11
    # initialises from the instruct-sft checkpoint, whose embedding rows are that tokenizer's
    # ids; a vocabulary built here would be a different set of ids under the same embedding.
    if need_bpe_build and not build_bpe:
        raise SystemExit(
            f"[{helpers.stage_label(getattr(args, 'stage', '')) or 'stage'}] tokenizer "
            f"{getattr(args, 'tokenizer', '') or BPE_PATH} is not on disk, and this stage does "
            f"not build one: its model is loaded from another stage and must keep that "
            f"vocabulary. Run ./old_scripts/stage3_pretrain_bpe_tokenizer.sh")
    train_pairs, ev_pairs = ([], [])
    if need_pairs or need_bpe_build:
        train_pairs, ev_pairs = dsets.load_pairs(args.dataset, args.data_dir, args.val_frac,
                                                 args.seed, args.limit)
    texts = [p[k] for p in (train_pairs + ev_pairs) for k in ("prompt", "chosen", "rejected")]
    if need_bpe_build:
        # The tokenizer is trained on EVERY text the pipeline will ever see, so one vocabulary
        # serves every stage. That is two sources, not three:
        #
        #   the pretraining corpus   scanned as documents
        #   the instruction data     ALREADY in `texts` above -- load_pairs extracted every
        #                            prompt, chosen and rejected string from it
        #
        # There is no separate scan of --sft_dir. It points at the rlhf_hh tree, whose files
        # are preference JSON, and json is a corpus extension: scanning it would have read
        # `{"pairs": [{"id": 0, "prompt": ...` as prose and taught the vocabulary merges over
        # JSON punctuation -- while the very same prompts and responses were already in
        # `texts`, properly extracted. One source, counted once, in its parsed form.
        pre, n_pre = dsets.load_pretrain_corpus(
            args.pretrain_dir, config.PRETRAIN["max_words"],
            exclude_dirs=config.PRETRAIN["exclude_dirs"],
            text_column=config.PRETRAIN["text_column"])
        helpers.table("BPE training corpus", [
            ("pretrain corpora", ", ".join(os.path.relpath(d, config.ROOT)
                                            for d in args.pretrain_dir)
                                 or "(unset for this scheme)"),
            ("pretrain excluded", ", ".join(config.PRETRAIN["exclude_dirs"]) or "(none)"),
            ("pretrain corpus files", f"{n_pre:,}"),
            ("pretrain documents", f"{len(pre):,}"),
            ("instruction data", config.INSTRUCT_DIR),
            ("instruction texts", f"{len(texts):,}  (prompt/chosen/rejected of "
                                  f"{len(train_pairs) + len(ev_pairs):,} pairs)"),
            ("merges requested", f"{args.bpe_merges:,}")], out=log)
        texts = texts + [d["chosen"] for d in pre]
    def build_tokenizer(preloaded):
        return train_bpe.build(
            texts, _bpe_src, log,
            # ONLY THE STAGE THAT OWNS THE TOKENIZER DRAWS ITS FIGURE. `plotdir` is what makes
            # build() redraw bpe_dynamics.pdf, and it was passed unconditionally -- so every
            # stage that merely LOADED the vocabulary rewrote stage 3's figure on startup, which
            # is what put "[plot] .../bpe/bpe_dynamics.pdf created (58,000 merges)" at the top of
            # a DPO run that trained no tokenizer at all. The `draw_bpe` flag for this has been
            # in the signature and in this module's docstring since it was written.
            plotdir=plot_dir("bpe") if draw_bpe else None,
            preloaded=preloaded,              # already parsed above; don't re-read
            num_merges=args.bpe_merges,
            plot_every=args.plot_every_steps,
            checkpoint_every=args.checkpoint_every_steps,
            min_freq=config.BPE.get("min_freq", 1))

    if need_bpe_build and parallel.dist_active():
        tok, failure = None, None
        if parallel.is_main():
            try:
                tok = build_tokenizer(_saved)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
        failure = parallel.broadcast_ckpt(failure)
        if failure is not None:
            raise RuntimeError(f"distributed tokenizer build failed: {failure}")
        parallel.barrier()
        if not parallel.is_main():
            _saved, _bpe_src = train_bpe.resolve(BPE_PATH, config.BPE_DOWNLOAD_PATH, log)
            if _saved is None:
                raise RuntimeError("distributed tokenizer build completed without a tokenizer")
            tok = _saved
    else:
        tok = build_tokenizer(_saved)
    # EVERY CHECKPOINT WRITTEN FROM HERE CARRIES THIS TOKENIZER. A stage that swaps it --
    # lang_sft, whose vocabulary is extended for another language -- calls set_bpe_source again
    # with its own file, so the checkpoint and the vocabulary its embedding was built for can
    # never come apart. `_bpe_src` and NOT BPE_PATH, for the same reason: when the downloaded
    # tokenizer was taken, checkpoints/bpe/bpe.json does not exist, and naming a file that is
    # not there would embed nothing and leave the checkpoint unable to say what its ids mean.
    # `log`, so the structured tokenizer section lands in the run's log beside everything else
    # rather than on bare stdout. It says WHICH vocabulary every trainer -- pretrain, the
    # benchmark, the instruction stages, lang_sft -- has just loaded.
    helpers.set_bpe_source(_bpe_src, log, tokenizer=tok)
    if need_pairs and train_pairs:
        helpers.table("dataset", dsets.stats_rows(args.dataset, train_pairs, ev_pairs, tok,
                                                  source=config.INSTRUCT_DIR),
                      out=log)
    # Pre-tokenise the preference pairs into cache/tokens/<tokenizer>/ -- ONLY for the stages that
    # train on them, so a corpus stage never touches the instruction data.
    #
    # The cache is keyed by the directory the pairs CAME FROM, not by config.HH_DIR: --dataset
    # now takes a path, and keying two different folders to one name would let a run be served
    # the other folder's tokens whenever the two happened to agree on pair count.
    if pretokenize_pairs and train_pairs:
        try:
            src = dsets.resolve_root(getattr(args, "dataset", "hh"), args.data_dir)
            failure = None
            if parallel.is_main():
                try:
                    helpers.attach_pair_ids(train_pairs, src, tok, split="train", log=log,
                                            resume=args.resume)
                    helpers.attach_pair_ids(ev_pairs, src, tok, split="val", log=log,
                                            resume=args.resume)
                except Exception as exc:
                    failure = f"{type(exc).__name__}: {exc}"
            failure = parallel.broadcast_ckpt(failure)
            if failure is not None:
                raise RuntimeError(failure)
            parallel.barrier()
            if not parallel.is_main():
                helpers.attach_pair_ids(train_pairs, src, tok, split="train", log=log,
                                        resume=True)
                helpers.attach_pair_ids(ev_pairs, src, tok, split="val", log=log,
                                        resume=True)
        except Exception as e:                                    # noqa: BLE001
            log(f"[cache] preference pre-tokenisation skipped: {e}")
    enc = dsets.Encoder(tok, device, args.max_len)
    def new_model(): return build_model(tok, device, args=args)

    def monitor(mode, records, step=None):
        """Live/final dynamics figure for `mode`, written under outputs/plots/<mode>/. Plotting must
        never kill a training run, so failures are logged and swallowed.

        RANK 0 DRAWS IT. `records` is the same on every rank -- the loss is all-reduced before
        it is appended -- so the other ranks would render the identical figure onto the identical
        path, N writers on one file for one picture."""
        if not parallel.is_main():
            return
        try:
            viz.method_monitor(mode, records, plot_dir(mode), "")
        except Exception as e:                                    # noqa: BLE001
            log(f"[plot] {mode} failed: {e}")

    rng = random.Random(args.seed)
    # fixed prompts for the end-of-stage samples and the periodic previews. A corpus stage
    # has no preference pairs to draw them from and supplies its own (common.corpus_prompts).
    prompts, pref_prompts = [], []
    if train_pairs:
        prompts = [train_pairs[i]["prompt"]
                   for i in rng.sample(range(len(train_pairs)),
                                       min(N_SAMPLES, len(train_pairs)))]
        pref_prompts = [train_pairs[i]["prompt"]
                        for i in rng.sample(range(len(train_pairs)),
                                            min(N_PREVIEW, len(train_pairs)))]
    sample = _make_sampler(tok, device, args.max_len, log, prompts,
                           getattr(tok, "eos_token_id", None))

    return {"args": args, "enc": enc, "tok": tok, "ckdir": CKDIR, "plotdir": PLOTDIR,
            "device": device, "log": log, "monitor": monitor, "new_model": new_model,
            "sample": sample, "train_pairs": train_pairs, "ev_pairs": ev_pairs,
            "pref_prompts": pref_prompts}


def adopt_context(ctx, saved, stage):
    """Take the context length from a checkpoint's own architecture. Returns it, or 0.

    A stage continuing from a model trained at 8,192 must encode at 8,192 -- the same argument
    that makes a checkpoint's width and depth authoritative over whatever default_config.py
    currently says. There is no --max_len to override it with any more, and this is why: the
    number belongs to the model, and every place it was also written down was a place for the
    two to disagree.

    Both the arguments and the ENCODER are updated. The encoder holds its own copy and is what
    actually truncates, so moving one without the other would leave the log reporting a context
    the data never saw."""
    saved = {str(key).lower(): value for key, value in (saved or {}).items()}
    block = int(saved.get("block_size") or 0)
    args = ctx.get("args")
    if not block or args is None:
        return 0
    if int(getattr(args, "max_len", 0)) != block:
        ctx.get("log", print)(
            f"[{stage}] context {int(getattr(args, 'max_len', 0)):,} -> {block:,}, from the "
            f"checkpoint's own architecture")
    args.max_len = block
    if ctx.get("enc") is not None:
        ctx["enc"].max_len = block
    return block


def load_stage_model(ctx, stage, train_mode=False, path=None):
    """A fresh model with stage `stage`'s saved weights loaded. Exits with a clear message if
    that checkpoint does not exist yet, since the stages depend on each other's outputs.

    `path` LOADS THAT FILE INSTEAD OF DERIVING ONE, and everything after the load is unchanged:
    the architecture still comes from the checkpoint's own model_cfg, the context still comes
    with it, the weights still go through load_growing. Stage 11 uses it because the file it
    starts from is not at a path this function could compute -- stage 7 writes under
    checkpoints/instruct_sft/<scheme>_<dataset>/, whose leaf is named for a corpus stage 11's
    yaml need not mention -- and because --checkpoint lets the person name any file at all."""
    ck = helpers.read_ckpt(path) if path else helpers.load_ckpt(ctx["ckdir"], stage)
    if not ck or "model" not in ck:
        where = path or f"{ctx['ckdir']}/{stage}/"
        raise SystemExit(f"[{stage}] no checkpoint at {where} -- "
                         f"run the {stage} stage first")
    # THE CHECKPOINT'S OWN ARCHITECTURE WINS: a stage trained at a different width or depth
    # must be rebuilt as it was, or load_state_dict fails (or, worse, silently loads a
    # differently-shaped model).
    saved = ck.get("model_cfg")
    if not isinstance(saved, dict):
        raise SystemExit(f"[{stage}] checkpoint architecture metadata is missing")
    m = build_model(ctx["tok"], ctx["device"], model_cfg=translate_cfg(saved), args=ctx["args"])
    # THE CHECKPOINT'S CONTEXT COMES WITH ITS ARCHITECTURE. A stage continuing from a model
    # trained at 8,192 must encode at 8,192, whatever the configuration in force happens to
    # say -- the same argument that makes the checkpoint's width and depth authoritative.
    adopt_context(ctx, saved, stage)
    load_growing(m, ck["model"], log=ctx.get("log", print))
    m.train() if train_mode else m.eval()
    return m


def load_growing(model, state, log=print):
    """Load `state` into `model`, GROWING the vocabulary if tokens were registered since.

    The state dict is loaded at ITS OWN size and the model grown afterwards, rather than the
    checkpoint being padded: the new rows are then initialised by resize_token_embeddings,
    in one place, instead of by whatever this function happened to choose.

    A checkpoint LARGER than the current vocabulary is refused. That is not a registration --
    it means the tokenizer was rebuilt, so the ids in the checkpoint mean different words, and
    loading it would produce a model that runs and talks nonsense."""
    want = model.tok.weight.shape[0]
    have = state["tok.weight"].shape[0] if "tok.weight" in state else want
    if have == want:
        model.load_state_dict(state)
        return model
    if have > want:
        raise SystemExit(
            f"[model] checkpoint has a vocabulary of {have:,} but the tokenizer now has "
            f"{want:,}.\n"
            f"        Tokens can be REGISTERED (the vocabulary grows and old ids keep their "
            f"meaning) but not removed, so this checkpoint was trained against a DIFFERENT "
            f"vocabulary. Retrain, or restore the tokenizer it was built with.")
    model.resize_token_embeddings(have, allow_shrink=True)   # fresh model, nothing lost
    model.load_state_dict(state)
    model.resize_token_embeddings(want)          # then grow, initialising only the new rows
    log(f"[model] {model._resize_note}; the checkpoint's {have:,} rows are unchanged")
    return model
