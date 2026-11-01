"""
default_config.py -- ALL DEFAULTS for the mira_llm pipeline in one place.

These are DEFAULTS, not the settings of a run: the stage YAML steers a run without editing
Python, and a command-line flag beats both. Precedence is

    command-line flag  >  config/<stage>/*.yaml  >  default_config.py

Every stage imports this module as `config`, so the code reads `config.MODEL[...]` while the
filename says what the values actually are.

Every stage trainer (stage3..stage10, and distill/) reads its defaults from here via
common.parse_args();
any value can still be overridden on the command line (a flag present in argv wins), so
`python -m instruct_sft.run --sft_steps 5000` works without touching this file.

Layout (all paths relative to the project root, i.e. the directory of this file). The
framework is DATA-AGNOSTIC: a corpus is whatever files with a CORPUS_EXTENSIONS extension
live under the configured directory, and the instruction data is whatever preference json the
instruction tree holds. NOTHING HERE DOWNLOADS; see DATASETS below and tools/download_data.py.

    data/download/llm-pretrain-wikitext103/
                                pretraining corpus for Tiny: <split>-<NNNNN>-of-<NNNNN>.parquet,
                                one row per article in a `text` column. The held-out shards are
                                excluded by PRETRAIN["exclude_dirs"], which matches file names
                                as well as directories
    data/download/llm-pretrain-fineweb-edu-2BT/
                                pretraining corpus for S, ~2B tokens. Every file under it
                                (recursive) whose extension is in CORPUS_EXTENSIONS, minus
                                PRETRAIN["exclude_dirs"]
    data/download/llm-rlhf-instruction-following/
                                rlhf_hh/        preference pairs, by subset and split. Serves
                                                stages 7 (demonstrations), 7 (preferences),
                                                8 (prompts) and 10 (preferences)
                                the four hh-rlhf subsets, each with its own train/test
    data/download/llm-grpo-cot-gsm8k/
                                reasoning problems: {train,test}_<batch>.json
    cache/tokens/<tokenizer>/<corpus>/
                                pre-tokenised corpora as sharded .tokens files beside
                                their _index.json, one directory per DATASET -- named
                                for the dataset, never for where it sits, so the same
                                corpus is found on every machine. The parent is the
                                TOKENIZER
                                (bpe_<256+merges>_<8 chars of the merges' blob hash>), so
                                two vocabularies coexist instead of evicting each
                                other, and no stage appears in the path: tokens depend on
                                the file and the tokenizer, so every stage shares them
    checkpoints/<stage>/        <model>_<pe>_<stage-label>/ per stage, holding checkpoint.pt,
                                history.json, config.json, legender.json, tokenizer.json
                                and dynamics.jsonl
                                (bpe/ holds the tokenizer json)
    outputs/plots/<stage>/      the stage's dynamics figures, PDF only

Pipeline: bpe -> pretrain -> sft -> reward -> rlhf(PPO) ; sft -> dpo ; rlhf -> distill
          pretrain -> cot (GRPO against a verified answer)
"""
import json
import math
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))

RETIRED_MODEL_CFG_KEYS = frozenset({
    "lattice_readout_silu",       # MiraLLM's spelling, in model_cfg
    "readout_silu",               # the mixer's own spelling, in case a stage recorded that one
    # WHAT A MIRA model_cfg USED TO RECORD AND NO LONGER DOES. resume_matches drops every key
    # here from the SAVED cfg and the BUILT one alike, so a checkpoint written when these were
    # recorded still resumes against a model that does not record them.
    "qk_rmsnorm", "qk_coupling",  # attention's, and this block has no queries or keys
    "d_conv", "ssm_chunk",        # a convolution and an SSM this block does not have
    "fir_m", "iir_n", "mixer_k",  # the pre-lattice mixer's, retired with it
    "miir_silu",                  # that mixer's activation flag, retired with it
    "lattice_iir_order",          # duplicated lattice_order and disagreed with it
    "lattice_expansion_factor",   # d_e / n_embd, and d_e itself is recorded
    "coefficient_bottleneck_factor",   # n_embd / d_r, likewise
    "lattice_state", "lattice_stage",  # never constructor parameters
})

# WHAT A MIRA model_cfg RECORDED BEFORE IT WAS CUT TO config.json's ELEVEN FIELDS. Separate from
# RETIRED_MODEL_CFG_KEYS above because these names are NOT retired everywhere: `arch`, `n_head`,
# `block_size`, `dropout`, `ffn_factor` and `gated_attn` are exactly what a rope ablation or a
# benchmark baseline is, and dropping them from every comparison would let an mha checkpoint
# resume as an rwkv one. resume_matches drops them from a MIRA cfg alone, on both sides, so a
# MIRA checkpoint written when they were recorded still resumes against one that does not record
# them -- and a baseline is still held to its full description.
RETIRED_MIRA_MODEL_CFG_KEYS = frozenset({
    "experimental_arch",       # a family name under the key baselines/ and ablation/ used
    "n_head",                  # no heads: the lattice mixer is elementwise per channel
    "gated_attn",              # attention's output gate, and there is no attention
    "block_size",              # the trained window, a yaml choice that changes between runs
    "dropout",                 # a regulariser, 0.0 everywhere and creating no tensor
    "ffn_factor",              # ffn_hidden is recorded beside it
    "lattice_output_gate",     # the readout always gates
    "lattice_positive_poles",  # the poles are a conjugate pair
})

MIRA_ONLY_MODEL_CFG_KEYS = frozenset({
    "lattice_modulus_map",
    "lattice_argument_map",
    "d_conv",
    "ssm_chunk",
    "fir_m",
    "iir_n",
    "miir_silu",
    "qk_rmsnorm",
    "qk_coupling",
    "mixer_k",
    "lattice_order",
    "lattice_iir_order",
    "lattice_output_gate",
    "lattice_d_e",
    "lattice_expansion_factor",
    "coefficient_bottleneck",
    "coefficient_bottleneck_factor",
    "memory_dimension",
    "lattice_state",
    "lattice_stage",
    "lattice_positive_poles",
    "lattice_initializer",
    "lattice_init_horizon",
})

NON_ARCHITECTURE_MODEL_CFG_KEYS = frozenset({
    "dropout",                 # a training regulariser, 0.0 everywhere by standing rule. It
                               # creates no tensor and changes no shape -- at p = 0.0 all three
                               # dropout modules are the identity -- so it belongs in the stage's
                               # yaml beside the learning rate, not in the architecture.
    "benchmark_family",        # which benchmark unit a run is. benchmark/run.py compares it on
                               # resume, and it reads it out of the .pt, not out of this file.
    "benchmark_architecture",  # the benchmark plan's geometry, compared the same way and in the
                               # same place, and a duplicate of the architecture beside it here.
    "experimental_arch",       # the architecture under a second name. mirachat and the context
                               # eval both resolve `experimental_arch or arch`, so the sidecar
                               # records that resolution once, as `arch`, and never both.
    "block_size",              # the window a run trained at. MiraLLM has no position table and
                               # nothing about it changes with the window, so this configures no
                               # part of the model; it belongs to the stage that chose it.
})

# WHAT AN ARCHITECTURE RECORDS AND NEVER READS. Each of these is written into the constructor's
# cfg and used by nothing that builds the model -- mira_llm.py says so itself, twice, about its
# own list -- so a reader who takes them for a description of what was trained is misled. Two are
# demonstrably wrong in checkpoints on disk: lattice_iir_order reads 4 beside a lattice_order of
# 2, and coefficient_bottleneck_factor is recorded against d_e where model/hf_model.py has to
# recompute it against n_embd and logs that it did.
#
# THEY STAY IN THE .pt. A resume compares the saved cfg against the built one key for key, so
# dropping them there would refuse every checkpoint written before the drop. This set governs the
# sidecar alone.
#
# ONE MIRA ENTRY, BECAUSE THERE IS ONE MIRA ARCHITECTURE. canonical_arch maps mira-matrix onto
# mira-lattice, along with mira, cdsa, mira-softmax and mira-ln, so a cfg that says
# `arch: mira-lattice` and `experimental_arch: mira-matrix` is not describing two things and
# never was -- which is why the sidecar keeps the canonical name and drops the other.
# THE SIDECAR'S KEYS, NAMED ONE BY ONE, IN THE ORDER THEY ARE WRITTEN.
#
#     "we must use keywords below strictly!!"                                 -- 2026-09-21
#
# AN ALLOW-LIST, NOT A FILTER. Every earlier version of this decided what config.json holds by
# excluding things -- a non-architecture set, a per-architecture inert set, a build-key set, an
# alias table -- and each exclusion had to be complete or a key leaked. Three did: the cuda_*
# build keys and the benchmark plan's d_model/layers spellings escaped at the top level, and
# output_gate came back through the fold under the plan's name for it. A list of what the file
# HOLDS cannot leak, cannot hold a key twice and cannot hold two names for one fact.
#
# EACH ROW IS (the name written, the keys it may be read from, in order). The first source
# present in the recorded cfg wins, and the benchmark plan's block is searched after the cfg, so
# a value recorded under the plan's own spelling still reaches the file under this one.
ARCHITECTURE_CFG_KEYS = {
    "gated_deltanet_v2": tuple((key, aliases) for key, aliases in (
        ("arch", ("arch",)), ("vocab_size", ("vocab_size",)),
        ("n_layer", ("n_layer", "layers")), ("n_embd", ("n_embd", "d_model")),
        ("n_head", ("n_head",)), ("ffn_hidden", ("ffn_hidden",)),
        ("expand_v", ("expand_v",)), ("conv_size", ("conv_size",)),
        ("norm_eps", ("norm_eps",)), ("allow_neg_eigval", ("allow_neg_eigval",)))),
    "mira-lattice": (
        ("lattice_modulus_map",   ("lattice_modulus_map",)),
        ("lattice_argument_map",  ("lattice_argument_map",)),
        ("vocab_size",            ("vocab_size",)),
        ("n_layer",               ("n_layer", "layers")),
        ("n_embd",                ("n_embd", "d_model")),
        ("lattice_order",         ("lattice_order",)),
        ("lattice_d_e",           ("lattice_d_e", "d_e")),
        ("state_write_bottleneck", ("coefficient_bottleneck",
                                    "matrix_coefficient_bottleneck")),
        ("memory_dimension",      ("memory_dimension",)),
        ("ffn_hidden",            ("ffn_hidden",)),
        ("ffn_input_bias",        ("ffn_input_bias",)),
        ("lattice_initializer",   ("lattice_initializer",)),
        ("lattice_init_horizon",  ("lattice_init_horizon",)),
        ("moe_total_experts",     ("moe_total_experts",)),
        ("moe_layers",            ("moe_layers",)),
        ("moe_active_experts",    ("moe_active_experts",)),
        ("moe_expert_width",      ("moe_expert_width",)),
        ("moe_shared_experts",    ("moe_shared_experts",)),
    ),
    # THE BASELINES KEEP THEIR FAMILY NAME, because nothing else in their file says what the
    # weights are. Every other row is a value the build READS BACK: build_lm's own parameters
    # (vocab_size, n_layer, d_model, n_head, ffn_hidden), the two it takes straight out of the
    # architecture block (mha's rope and fp32_scores), and each family's _CONFIGURABLE settings,
    # which configure_family writes onto the family module before the weights are made.
    #
    # THREE MORE ARE ABSENT AND EACH FOR ITS OWN REASON. mamba_v1's n_head, because S6 has no
    # heads -- its Mixer takes the argument and its body never uses it, and the module says so.
    # rwkv_v6's n_head, because head_ok requires d_model // n_head == HEAD_SIZE, so it is
    # d_model / head_size and the file would answer the head split twice. mamba_v2's chunk,
    # because it is the SSD scan's chunk length: a compute decomposition that shapes no
    # parameter, in the way a tile size is not part of an architecture.
    #
    # gated_attn AND ffn_input_bias ARE NOT AMONG THEM. build_lm hardcodes gated_attn=False for
    # the scaffold and never consults a recorded value, and nothing on the baseline path reads
    # ffn_input_bias at all, so both were recorded and unreadable -- a key that cannot change
    # what is built does not describe the architecture.
    "mha": (
        ("arch",                     ("arch",)),
        ("vocab_size",               ("vocab_size",)),
        ("n_layer",                  ("n_layer", "layers")),
        ("n_embd",                   ("n_embd", "d_model")),
        ("n_head",                   ("n_head",)),
        ("rope",                     ("rope",)),
        ("fp32_scores",              ("fp32_scores",)),
        ("ffn_hidden",               ("ffn_hidden",)),
    ),
    "mamba_v1": (
        ("arch",                     ("arch",)),
        ("vocab_size",               ("vocab_size",)),
        ("n_layer",                  ("n_layer", "layers")),
        ("n_embd",                   ("n_embd", "d_model")),
        ("d_state",                  ("d_state",)),
        ("d_conv",                   ("d_conv",)),
        ("expand",                   ("expand",)),
        ("dt_rank",                  ("dt_rank",)),
        ("ffn_hidden",               ("ffn_hidden",)),
    ),
    "mamba_v2": (
        ("arch",                     ("arch",)),
        ("vocab_size",               ("vocab_size",)),
        ("n_layer",                  ("n_layer", "layers")),
        ("n_embd",                   ("n_embd", "d_model")),
        ("n_head",                   ("n_head",)),
        ("d_state",                  ("d_state",)),
        ("d_conv",                   ("d_conv",)),
        ("expand",                   ("expand",)),
        ("ngroups",                  ("ngroups",)),
        ("norm_eps",                 ("norm_eps",)),
        ("a_init_min",               ("a_init_min",)),
        ("a_init_max",               ("a_init_max",)),
        ("dt_min",                   ("dt_min",)),
        ("dt_max",                   ("dt_max",)),
        ("dt_init_floor",            ("dt_init_floor",)),
        ("ffn_hidden",               ("ffn_hidden",)),
    ),
    "mamba_v3": (
        ("arch",                     ("arch",)),
        ("vocab_size",               ("vocab_size",)),
        ("n_layer",                  ("n_layer", "layers")),
        ("n_embd",                   ("n_embd", "d_model")),
        ("n_head",                   ("n_head",)),
        ("d_state",                  ("d_state",)),
        ("expand",                   ("expand",)),
        ("ngroups",                  ("ngroups",)),
        ("ffn_hidden",               ("ffn_hidden",)),
        ("mimo_rank", ("mimo_rank",)),
        ("chunk_size", ("chunk_size",)),
        ("rope_fraction", ("rope_fraction",)),
        ("is_outproj_norm", ("is_outproj_norm",)),
    ),
    "rwkv_v6": (
        ("arch",                     ("arch",)),
        ("vocab_size",               ("vocab_size",)),
        ("n_layer",                  ("n_layer", "layers")),
        ("n_embd",                   ("n_embd", "d_model")),
        ("head_size",                ("head_size",)),
        ("head_size_divisor",        ("head_size_divisor",)),
        ("ffn_hidden",               ("ffn_hidden",)),
    ),
    "rwkv_v7": (
        ("arch",                     ("arch",)),
        ("vocab_size",               ("vocab_size",)),
        ("n_layer",                  ("n_layer", "layers")),
        ("n_embd",                   ("n_embd", "d_model")),
        ("n_head",                   ("n_head",)),
        ("head_size_divisor",        ("head_size_divisor",)),
        ("w_scale",                  ("w_scale",)),
        ("ffn_hidden",               ("ffn_hidden",)),
    ),
}


INERT_MODEL_CFG_KEYS = {
    "mira-lattice": frozenset({
        "n_head", "gated_attn",           # no queries, no keys, no values and no heads
        "d_conv", "ssm_chunk",            # a convolution and an SSM this block does not have
        "fir_m", "iir_n", "mixer_k",      # the pre-lattice mixer's, retired with it
        "miir_silu",                      # that mixer's activation flag, retired with it
        "qk_rmsnorm", "qk_coupling",      # attention's, and there is no attention
        "lattice_iir_order",              # duplicates lattice_order, and disagrees with it
        "lattice_expansion_factor",       # d_e / n_embd, and d_e itself is recorded
        "coefficient_bottleneck_factor",  # d_e / coefficient_bottleneck, likewise
        "expansion_factor",               # the same ratio under the benchmark block's spelling
        # the matrix-state block spells its own geometry with a prefix; the derived factor is
        # the same derived factor and goes for the same reason.
        "matrix_coefficient_bottleneck_factor",
        # THE READOUT ALWAYS GATES, so the flag has one legal value and records nothing.
        # MiraLLM.__init__ raises on anything but true, and the readout is
        # y = W_out((sum_k rho^(k) . (V^(k) nu^(k))) . gamma) with gamma = SiLU(W_gamma z + b),
        # which is the mathematics rather than an option. IT STAYS A CONSTRUCTOR PARAMETER: the
        # .pt records it and a resume compares the saved cfg key for key, so removing the
        # parameter would refuse every checkpoint already written. Only the sidecar drops it.
        "lattice_output_gate", "output_gate",
        # THE CONSTRUCTOR TAKES NEITHER. `lattice_state` chose between the vector and matrix
        # blocks and there is one block now, which `arch` already names; `lattice_readout_silu`
        # is a retired readout flag. A key MiraLLM.__init__ cannot take cannot build the model,
        # which is the whole test this file is held to.
        "lattice_state", "lattice_readout_silu",
    }),
}


MODEL_CFG_KEY_ALIASES = {
    "coefficient_bottleneck": "state_write_bottleneck",
    "matrix_coefficient_bottleneck": "state_write_bottleneck",
}


MODEL_SETTING_ALIASES = {"n_layers": "n_layer", "d_model": "n_embd", "ffn_width": "ffn_hidden",
                         "lattice_d_r": "coefficient_bottleneck",
                         "lattice_d_m": "memory_dimension", "lattice_K": "lattice_order",
                         "lattice_modulus_map": "lattice_modulus_map",
                         "lattice_argument_map": "lattice_argument_map"}


def canonical_model_cfg_key(name):
    """The architecture's own name for a recorded key. One spelling reaches the sidecar."""
    return MODEL_CFG_KEY_ALIASES.get(name, name)


# THE BENCHMARK BLOCK IS FOLDED IN, NOT COPIED AND NOT THROWN AWAY. `benchmark_architecture` is
# the plan a benchmark unit was built from, and for MIRA every field of it is already recorded
# beside it under another name -- d_model for n_embd, layers for n_layer, d_e for lattice_d_e,
# output_gate for lattice_output_gate, and a lattice_iir_order that disagrees with lattice_order.
# For a baseline it is not: d_state, expand, ngroups and norm_eps for mamba, w_scale and
# head_size_divisor for rwkv, rope and fp32_scores for mha appear nowhere else, and they are that
# model's architecture. So the sidecar keeps what the block knows, under the names the rest of the
# file uses, and the block itself disappears -- which removes the nesting, removes the duplication
# and leaves no pair of keys able to disagree.
BENCHMARK_ARCHITECTURE_ALIASES = {
    "d_model": "n_embd",
    "layers": "n_layer",
    "d_e": "lattice_d_e",
    "output_gate": "lattice_output_gate",
}

# WHERE A KERNEL WAS BUILT FROM IS NOT WHAT THE MODEL IS. These four say which Makefile and which
# cubin the benchmark used; they describe this repository, not the checkpoint, and a path recorded
# in a checkpoint is wrong as soon as the file moves.
BENCHMARK_BUILD_KEYS = frozenset({
    "cuda_bin", "cuda_build_target", "cuda_check_target", "cuda_makefile",
})


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
# ONE DATA ROOT. The repository ships no corpus at all: every stage reads from
# data/download/<dataset>/, which stage 1 fills and which is git-ignored in full. A clone is
# therefore small, nothing large is ever in `git status`, and there is exactly one answer to
# "where did this data come from" -- DATASETS below, fetched by tools/download_data.py and by
# nothing else.
DATA_DIR = os.path.join(ROOT, "data")                      # downloads only; git-ignored
DOWNLOAD_DIR = os.path.join(DATA_DIR, "download")          # data/download/<dataset>/  <- the
                                                           # data root every stage reads from

DOWNLOAD_CONFIG = os.path.join(ROOT, "config", "download", "config_download.yaml")


def _read_datasets(path=None):
    """Parse config/download/config_download.yaml without adding a PyYAML runtime dependency.

    A second copy here is the mistake this project keeps paying for: two places holding one
    fact, one of them updated. So there is no dictionary literal below, and a dataset is added
    by adding one mapping to the YAML and by doing nothing else.

    This is the same deliberately small YAML discipline as tools/config_file.py: one root
    mapping, one dataset-name level and scalar fields beneath it. Double-quoted strings are
    decoded with JSON, whose string syntax is valid YAML; unquoted repository names and integer
    split sizes remain plain scalars. Unknown fields and malformed indentation stop at import
    rather than silently changing what stage 1 downloads."""
    path = path or DOWNLOAD_CONFIG
    raw = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as e:
        raise SystemExit(f"[config] cannot read the dataset list at {path} ({e}).\n"
                         f"         It ships with the repository; restore it from git.") from e
    current = None
    saw_root = False
    allowed = {"repo", "description", "type", "allow_patterns", "category"}
    categories = ("pretrain", "instruct-sft", "instruct-rlhf")
    for i, original in enumerate(lines, 1):
        if not original.strip() or original.lstrip().startswith("#"):
            continue
        indent = len(original) - len(original.lstrip(" "))
        line = original.strip()
        if indent == 0:
            if line != "datasets:" or saw_root:
                raise SystemExit(f"[config] {path}:{i}: expected the single root key "
                                 f"'datasets:', got {line!r}")
            saw_root = True
            current = None
            continue
        if not saw_root:
            raise SystemExit(f"[config] {path}:{i}: dataset appears before 'datasets:'")
        if indent == 2 and line.endswith(":"):
            current = line[:-1].strip()
            if not current or current in raw:
                raise SystemExit(f"[config] {path}:{i}: empty or duplicate dataset {current!r}")
            raw[current] = {}
            continue
        if indent != 4 or current is None or ":" not in line:
            raise SystemExit(
                f"[config] {path}:{i}: expected four-space 'field: value' under a dataset, "
                f"got {original.rstrip()!r}")
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key not in allowed:
            raise SystemExit(f"[config] {path}:{i}: unknown dataset field {key!r}; "
                             f"expected one of {sorted(allowed)}")
        if not value:
            raise SystemExit(f"[config] {path}:{i}: {key} must have a value")
        if value.startswith('"'):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as e:
                raise SystemExit(f"[config] {path}:{i}: invalid quoted YAML string ({e})") from e
        raw[current][key] = value
    if not saw_root:
        raise SystemExit(f"[config] {path}: missing root key 'datasets:'")

    out = {}
    for name, row in raw.items():
        repo = row.get("repo")
        if not repo:
            raise SystemExit(f"[config] {path}: dataset {name!r} has no repo")
        # THE TYPE IS VALIDATED HERE, not where it is used. A typo -- "models", "Model" -- would
        # otherwise fall through download_type's default and fetch from the dataset namespace,
        # which is a 404 whose message names the repository rather than the typo.
        kind = str(row.get("type", "dataset")).strip().lower()
        if kind not in ("dataset", "model"):
            raise SystemExit(f"[config] {path}: {name!r} has type {kind!r}; "
                             f"expected 'dataset' or 'model'")
        category = str(row.get("category", "")).strip().lower()
        if category and category not in categories:
            raise SystemExit(f"[config] {path}: {name!r} has category {category!r}; "
                             f"expected one of {', '.join(categories)}")
        if category and kind != "dataset":
            raise SystemExit(f"[config] {path}: {name!r} is type {kind!r} and carries a "
                             f"category; only a dataset is tokenised")
        out[name] = {"repo": str(repo), "what": str(row.get("description", name)),
                     "type": kind}
        if category:
            out[name]["category"] = category
        if row.get("allow_patterns"):
            out[name]["allow_patterns"] = str(row["allow_patterns"])
    return out


DATASETS = _read_datasets()


def dataset_dir(name):
    """Where `tools/download_data.py` puts a dataset, and where the trainers look for it."""
    return os.path.join(DOWNLOAD_DIR, name)


def dataset_category(name):
    """The category of the dataset directory `name`: "pretrain", "instruct-sft",
    "instruct-rlhf", or "" when the entry carries none and stage 4 leaves it alone.

    Taken from the download config, which is the one place a dataset is described."""
    return str((DATASETS.get(os.path.basename(str(name).rstrip("/"))) or {}).get("category", ""))


def datasets_in(category):
    """Every dataset name in one category, in the order the download config lists them."""
    return [n for n, e in DATASETS.items() if e.get("category") == category]


def download_type(name):
    """"dataset" or "model" for a stage-1 entry; anything unlisted is a dataset."""
    return str((DATASETS.get(name) or {}).get("type", "dataset")).strip() or "dataset"


def download_dir(name):
    """Where stage 1 puts `name`. A MODEL GOES TO saved_models/, A DATASET TO data/download/.

    MODEL_DIR is where every other downloaded model already lives -- distill/ fetches gpt2-small
    into it, evals/ defaults --model_dir to it -- so a tokenizer published as a model repository
    belongs there and not among the corpora."""
    return (os.path.join(MODEL_DIR, name) if download_type(name) == "model"
            else dataset_dir(name))


# PRETRAINING CORPUS PER SCHEME, sized to the model rather than shared, and READ THE DICTIONARY
# BELOW RATHER THAN THIS PARAGRAPH IF THE TWO EVER DISAGREE. As it stands:
#
#     baby    WikiText-103 alone (~105M tokens) -- the only scheme with no FineWeb-Edu
#     tiny    the ~10 GB / ~2B-token FineWeb-Edu subset
#     S       that same 2B subset, PLUS WikiText-103
#     M, L    the 10B-token FineWeb-Edu sample, PLUS WikiText-103
#
# A 121M model does not need 10B tokens and a 609M model is not fed by 105M, which is what the
# ladder is for. BABY IS THE SCHEME TO TRY A CHANGE ON FIRST: it is the one whose corpus is
# small enough to tokenise quickly. S is the DEFAULT, which is a different job.
#
# ROUGHLY CHINCHILLA-SHAPED, which is why the ladder is not one corpus. ~20 tokens per
# parameter puts S at ~3.1B and L at ~12.2B; the 2B subset and the 10B sample are the two
# available sizes nearest those, so each scheme trains on the one it can actually consume.
#
# NOTE that the schemes therefore differ in CORPUS as well as in depth and width, so no pair of
# them is a controlled comparison of architecture alone. Point two schemes at one corpus with
# PRETRAIN_DIR (or --pretrain_dir) when that is what you mean to measure.
#
# EACH SCHEME TAKES A LIST, and each corpus in it keeps its OWN token stream, its own signature
# and its own cache directory. Nothing is merged on disk, so adding a corpus tokenises only the
# corpus added: the streams already built are untouched, still current, still keyed by the
# signature they always had. That is what makes it safe to extend the corpus of a run that is
# already half-way through its budget -- the data it has been reading does not move.
#
# Sequences are drawn in proportion to TOKENS (token_store.MultiStream), which is what
# concatenating the corpora would have given, so the mixture is a fact about what is on disk
# rather than a weight somebody chose. WikiText-103 is ~105M tokens against FineWeb-Edu's ~2B,
# so it is a few per cent of what -s sees -- clean encyclopedic prose beside filtered web text.
# WHAT THE BENCHMARK TRAINS ON, AND WITH WHICH VOCABULARY. Two settings, because they are the
# two things a benchmark run is a comparison ACROSS: change either and the numbers are not
# comparable with the previous ones, so both are named in config/benchmark/config_benchmark.yaml
# rather than inferred.
#
# BENCHMARK_CORPORA maps the corpus NAME to its directory and to the column its text lives in.
# The column is the part that cannot be left to a default: WikiText-103 keeps prose in `text`
# and the-stack-smol-xs keeps code in `content`, and a corpus read with the wrong column yields
# NO DOCUMENTS AT ALL -- not an error, an empty benchmark. Pairing them here means selecting a
# corpus selects its column, and the two cannot be set inconsistently.
# WHAT THE METHOD IS CALLED ON A FIGURE, AND NOWHERE ELSE.
#
#     "we change Lattice -> MIRA (Modal IIR Adaptive Lattice.) for all of them ... because we
#      call our method MIRA. some case we may use Mira-Lattice, but default we use Mira."
#     "don't change the checkpoint layout etc. only legend"                    -- 2026-08-24
#
# ONE CONSTANT BECAUSE THERE ARE SIX PLACES THAT DRAW IT: the comparison figure's legend, the
# context figure's legend, twelve lattice-dynamics titles, the ablation title, the profiler's
# tables and its colour key, and the name the benchmark's plan table prints. They were six
# independent string literals and had already drifted once -- "Lattice" in two of them and
# "MIRA-Lattice" in the rest -- so renaming meant finding them all again. Now it does not.
#
# THIS IS A DISPLAY NAME AND TOUCHES NOTHING ELSE. The family key stays `mira-lattice`, and with
# it every checkpoint directory, history filename, --only argument and summary.json entry
# already on disk. A label is not an identifier.
METHOD_NAME = "MIRA"
METHOD_NAME_LONG = "MIRA-Lattice"          # where the architecture, not the method, is meant
METHOD_NAME_FULL = "Modal IIR Adaptive Lattice"

BENCHMARK_CORPORA = {
    "llm-pretrain-wikitext103": {"dir": "llm-pretrain-wikitext103", "text_column": "text"},
    # FineWeb-Edu-2BT WAS THE PROSE HALF FOR ONE COMMIT ON 2026-08-22 AND WAS PUT BACK. The entry
    # stays so a run that names it by hand still resolves.
    "llm-pretrain-fineweb-edu-2BT": {"dir": "llm-pretrain-fineweb-edu-2BT",
                                     "text_column": "text"},
    "llm-pretrain-fineweb-edu-10BT": {"dir": "llm-pretrain-fineweb-edu-10BT",
                                      "text_column": "text"},
    # sample-100BT of the same repository (2026-10-09); selectable, not the default.
    "llm-pretrain-fineweb-edu-100BT": {"dir": "llm-pretrain-fineweb-edu-100BT",
                                       "text_column": "text"},
    "the-stack-smol-xs":        {"dir": "the-stack-smol-xs",        "text_column": "content"},
}

BENCHMARK = dict(
    # One of BENCHMARK_CORPORA above.
    dataset="the-stack-smol-xs",
    # A directory under checkpoints/ holding bpe.json: `bpe` (stage 3), or one stage 12 wrote,
    # such as `bpe-sft-coder` or `bpe-sft-the-stack-smol-xs`. Used AS GIVEN, never prefixed, so
    # the base vocabulary stays reachable by name.
    tokenizer="bpe-sft-coder",
    # DEPTH IS FIXED ACROSS THE FAMILIES and is the benchmark's own, not the scheme's: every
    # family is matched on parameters at this depth, so changing it changes every matched width.
    # `--layers` on the command line still wins over it.
    layers=16,
    # ONE CONTEXT WINDOW, and the benchmark's own rather than the scheme's. A comma-separated
    # list turns the run into a sweep: the step budget is split across the windows and each
    # segment's batch is rescaled so batch x context stays constant, so a second entry both
    # halves the steps at 512 and changes what BATCH means.
    context_window="512",
    checkpoint_dir="checkpoints",
    benchmark_dir="benchmark",
    optimizer="adamw",
    # THE EMA SPAN THE COMPARISON FIGURE'S CURVES AND THEIR +/-1 s.d. BAND ARE DRAWN WITH.
    # Read by benchmark/visualize_benchmark.py, both as its --smooth-window default and by the
    # in-training redraw, so the figure is the same whichever wrote it.
    smooth_window=2001,
    # "mavg", "ema" or "block". The figure picks between them:
    # mavg is helpers.visualization._moving_average with _moving_deviation, ema is
    # benchmark.visualize_benchmark.exponential_mean with exponential_deviation, block is
    # helpers.visualization._block_mean with _block_deviation (the long-run insights averaging).
    plot_smooth_method="ema",
)

# THE GEOMETRY OF EVERY BENCHMARK FAMILY, CHOSEN AT DESIGN TIME. (d_model, n_head, ffn_hidden);
# ffn_hidden None is the ordinary 4x expansion, which is what every row uses.
#
# MIRA-LATTICE'S WIDTH IS FIXED AT 512 AND IS THE TARGET. Every other family's width is chosen
# to match its parameter count at this depth.
#
# WHY A TABLE AND NOT A SEARCH. The search minimised the residual over d_model without knowing
# that d_head is not free, so it proposed geometries whose kernels do not exist -- and a family
# whose search found nothing inside tolerance was dropped as unplannable and never trained.
# What each family may actually take:
#
#   mira-newton   d_head MUST be 64. iir_retrieval and the key transform are compiled for it;
#                 any other head width silently returns None and runs the Python reference.
#                 So d_model is a multiple of 64 and nothing finer is reachable.
#   mira-sdpa     d_head is FREE. M_qk is folded with an einsum, not a cubin, and the MIIR is
#                 per channel.
#   ssm           no head split at all.
#   rwkv_v7       d_head must be one of kernel/cuda/rwkv_v7's SIZES, which is twelve values
#                 from 16 to 128 -- coarse in d_model but far from only 64.
#   mha           d_head is free.
#
# MIRA-NEWTON IS THE ONE ROW WIDTH CANNOT MATCH. Its packed projection is 13 d^2 and its mixer
# is half again the target block, so with d_model pinned to multiples of 64 the closest reach
# is 384 at -9.3%; 448 overshoots to +19.4%. Setting its ffn_hidden to 2253 would bring it to
# +0.006% at the same 384 -- the mechanism is wired and one number away -- at the cost of an
# FFN that is 5.9x rather than 4x.
#
# Recompute these when the target moves -- MIRA-Lattice's K, expand, selector width or depth --
# and check the residuals the benchmark prints against its own built count.
# THE BENCHMARK USES ONE BATCH FOR EVERY FAMILY, and it is the largest the HEAVIEST family
# tolerates -- config/benchmark/config_benchmark.yaml's BATCH.
#
# THE BATCH IS WHAT MAKES THE LOSS COMPARABLE. Every curve in the comparison is loss against
# STEP, and a step is only the same unit of work across families if it carries the same number
# of sequences. Give one family a larger batch and its step consumes more tokens, its gradient
# is less noisy, and its curve falls faster for a reason that has nothing to do with its mixer.
# The result would read as an architectural difference and would not be one.
#
# So the batch is set by whichever family runs out of memory first, and the rest run below what
# their own memory would allow. That wastes card, and wasting it is the point: the alternative
# is a benchmark whose headline number is an artefact of who got the bigger batch.
#
# PER-FAMILY BATCHES ARE COMMENTED OUT, NOT REMOVED. They were sized from MIRA-Lattice's second
# out-of-memory, the one in the backward with 47.17 GiB in use and a further 1.54 GiB requested,
# which puts its peak at no less than 665 MiB per sequence against a 1.94 GiB
# model/gradient/Adam floor. The other five were that figure scaled by their retained
# activations. Restore by undoing this block and its companion in benchmark/pretrain_benchmark.py
# (grep: PER-FAMILY BATCHES ARE COMMENTED OUT).
#     BENCHMARK_BATCH = {
#         "mira-lattice": 48,            # >= 665 MiB/sequence, from the backward
#         "mira-newton":  40,            # narrower d, but retrieval checkpoints dominate
#         "mira-sdpa":    40,
#         "ssm":          48,
#         "mha":          64,            # lightest: fused attention keeps no score matrix
#         "rwkv_v7":      40,
#     }

# THE MIRA-S MATRIX-STATE TARGET WAS 146,411,376 at V = 50,259 and 152,555,376 at
# V = 58,259: 12 layers, d = 768, d_e = 1536, lattice_order = 4,
# coefficient_bottleneck = 48, and memory_dimension = 16 (HISTORICAL; MIRA-S is 145,242,432 at
# V = 50,259 with lattice_K 2, lattice_d_m 64 since 2026-10-06, config/default/config_mira.yaml).
#
# THE RESIDUALS BESIDE EACH ROW ARE NOW THE MODULE'S COUNT, NOT THE QUADRATIC'S, and that
# distinction mattered: three rows of the previous table recorded what the thumb rule predicted
# rather than what baselines.build_lm builds.
#
#     gated_deltanet  recorded k = 4.5 and -1.699%; the true mixer is k = 6 -- the note omitted
#                     g_proj = Linear(d, 1.5d) -- so that row was really +7.04%, the largest
#                     error in the table.
#     rwkv_v6         k = 6 was a rounding of 5 d^2 + 458 d, so -7.07% was really -8.04%.
#     rwkv_v7         its ffn_hidden was zeroed to honour the fixed 4x FFN and the width was
#                     never re-solved, leaving 560 at -2.42% under a comment claiming 0.175%.
#
# The quadratic still PICKS the candidate; the residual written down is what the family builds.
#
# SOLVED AS A MINIMAX OVER BOTH BENCHMARK VOCABULARIES, because run_scripts/run_pretrain_benchmark.sh runs
# three corpora across two tokenizers -- `bpe` at 50,259 for WikiText-103 and FineWeb-Edu,
# `bpe-sft-coder` at 58,259 for the-stack -- and one table serves all of them.
#
# THE TWO-VOCABULARY SPREAD SETS A FLOOR ON EVERY ROW. The only term that differs between the two
# targets is V*d, so a family whose width is not exactly 512 cannot hit both: the irreducible
# minimax residual is about 4,000*(d-512)/target, which is 0.17% at d = 560 and 0.46% at d = 640.
# A row at 0.66% is therefore not slack in the search, it is mostly that floor.
#
# THE FFN FACTOR IS 4 FOR EVERY FAMILY AND IS NOT A KNOB.
#
#     "ffn factor is fixed to 4!"                                             -- 2026-08-23
#     "we fix layers, basically the hyperparameter is only the width!"
#
# The third slot of every row is None, which is the plain 4x FFN. RWKV-7 briefly carried an
# ffn_hidden of 2,380 to close its residual; that was a second knob and it is gone. DEPTH IS
# FIXED, THE FFN FACTOR IS FIXED, AND THE WIDTH IS THE ONLY THING THAT MOVES.
#
# SO SOME RESIDUALS ARE LARGE, AND THAT IS THE HONEST ANSWER RATHER THAN A FAILURE TO SEARCH.
# A family whose head width is compiled at a fixed size has a width grid that steps by that
# size: RWKV-6 is 64, so its reachable widths near the target are 576 (-7.07%) and 640 (+11.5%)
# and there is nothing between them. Every row below is the closed-form width, snapped to that
# family's own grid, and its residual is written down.
# A FAMILY ABSENT FROM THIS TABLE IS WIDTH-SEARCHED AT RUN TIME rather than refused.
#
#     "we must include gated deltanet ... HGRN2 ... Hyena ... mamba-v2 ... retnet"  -- 2026-08-23
#
# The six families added on 2026-08-23 have no row here because their parameter counts cannot be
# solved from this machine -- Hyena's is not even a function of d_model alone, it carries a
# trainable positional embedding of L_MAX x EMB_DIM per layer. plan_widths falls back to
# there is no fallback: a family with no row is refused by name. Every width is written down, and
# the printed line is what belongs here once a run has produced it. Nothing is guessed.
def benchmark_dirname():
    """The single directory component naming this stage, under checkpoints/ and outputs/plots/."""
    return str(BENCHMARK.get("benchmark_dir") or "benchmark").strip().strip("/")


def benchmark_checkpoint_root(dataset):
    """The directory a benchmark family's checkpoints live under, for one dataset."""
    root = str(BENCHMARK.get("checkpoint_dir") or "checkpoints").strip()
    if not os.path.isabs(root):
        root = os.path.join(ROOT, root)
    return os.path.join(root, benchmark_dirname(), str(dataset))


def _mira_checkpoint_geometry():
    s = scheme(DEFAULT_SCHEME)
    return (f"K{MODEL['lattice_order']}_d{s['n_embd']}_de{s['lattice_d_e']}"
            f"_dr{MODEL.get('coefficient_bottleneck', s['coefficient_bottleneck'])}"
            f"_dm{MODEL.get('memory_dimension', s['memory_dimension'])}")


def _mira_checkpoint_init(init=None):
    """The lattice_initializer a checkpoint name carries: `init`, else the configured one."""
    return canonical_lattice_initializer(MODEL["lattice_initializer"] if init is None else init)


def mira_checkpoint_name(arch=None, init=None):
    """The benchmark MIRA checkpoint directory, <scheme>_<geometry> (2026-10-09), e.g.
    mira-s_K2_d768_de1536_dr48_dm16. mira_checkpoint_units also finds the directories written under
    the earlier names, which carried the arch and the initializer."""
    prefix = canonical_scheme(DEFAULT_SCHEME)
    return f"{prefix}_{_mira_checkpoint_geometry()}"


def mira_checkpoint_units(arch=None, init=None):
    """Every directory name a benchmark MIRA checkpoint of `arch` and `init` may be under: the name
    above, <scheme>_<geometry>, first; then <scheme>_<arch code>_<initializer short name>_<geometry>,
    written from 2026-10-05 to 2026-10-09 (arch code from ARCH_CODES, cauchy_x_hyperbolic = rccxRHB;
    short name from LATTICE_INITIALIZER_SHORT, uniform_weight_jeffreys_horizon_bias_init =
    jeffreys); then the full-name forms written before 2026-10-05 --
    <scheme>_<arch>_<init>_<geometry>, <scheme>_<arch>_<init>_rmsnorm_<geometry> (written while the
    retired lattice_norm_module was in the name) -- then the names written before the initializer was
    part of it: <scheme>_<arch>_<geometry> and the same under the arch's former spelling."""
    arch = lattice_arch(MODEL["arch"] if arch is None else arch)
    prefix = canonical_scheme(DEFAULT_SCHEME)
    geometry = _mira_checkpoint_geometry()
    method = _mira_checkpoint_init(init)
    units = [mira_checkpoint_name(arch, init),
             f"{prefix}_{arch_code(arch)}_{lattice_initializer_short(method)}_{geometry}",
             f"{prefix}_{arch}_{legacy_lattice_initializer_name(method)}_{geometry}",
             f"{prefix}_{arch}_{legacy_lattice_initializer_name(method)}_rmsnorm_{geometry}",
             f"{prefix}_{arch}_{geometry}"]
    units += [f"{prefix}_{old}_{geometry}" for old, new in LEGACY_ARCH_ALIASES.items()
              if new == arch]
    return units


def _legacy_unit_matches(path, init, horizon=None):
    """Does the checkpoint directory `path`, written under an older name, hold a run of
    lattice_initializer `init` and, when its config.json records one, of the same
    lattice_init_horizon? config.json records the method as lattice_initializer, or as
    lattice_init_method before 2026-10-05. A directory without a readable config.json, or one that
    records no method, is taken as matching, so a checkpoint that exists is not stopped from being
    found. A recorded horizon range that differs (e.g. a run without the L0 override, found by a run
    with it) does not match."""
    try:
        with open(os.path.join(path, "config.json")) as handle:
            recorded_cfg = json.load(handle)
    except (OSError, ValueError, AttributeError):
        return True
    if not isinstance(recorded_cfg, dict):
        return True
    recorded = recorded_cfg.get("lattice_initializer", recorded_cfg.get("lattice_init_method"))
    recorded_horizon = recorded_cfg.get("lattice_init_horizon")
    if recorded_horizon is not None and horizon is not None:
        try:
            if canonical_lattice_init_horizon(recorded_horizon) != canonical_lattice_init_horizon(horizon):
                return False
        except (ValueError, TypeError):
            return False
    if recorded is None:
        return True
    try:
        return canonical_lattice_initializer(recorded) == init
    except ValueError:
        return False


def mira_checkpoint_unit(root, arch=None, init=None):
    """The one of mira_checkpoint_units that exists under `root`, else mira_checkpoint_name. The name
    and the name written from 2026-10-05 to 2026-10-09 are taken as they are; an older name is taken
    only when its config.json records the same lattice_initializer and the same lattice_init_horizon
    (when it records one)."""
    units = mira_checkpoint_units(arch, init)
    method = _mira_checkpoint_init(init)
    horizon = MODEL.get("lattice_init_horizon")
    for index, unit in enumerate(units):
        path = os.path.join(root, unit)
        if os.path.isdir(path) and (index <= 1 or _legacy_unit_matches(path, method, horizon)):
            return unit
    return units[0]


def benchmark_corpus(name):
    """(directory, text_column) for a benchmark corpus name. Raises on an unknown one.

    REFUSED RATHER THAN GUESSED. A name that is not in the table has no column this file knows,
    and defaulting it to `text` is exactly how a code corpus reads as empty."""
    entry = BENCHMARK_CORPORA.get(str(name).strip())
    if entry is None:
        raise SystemExit(
            f"[benchmark] unknown BENCHMARK_DATASET {name!r}; expected one of "
            + ", ".join(sorted(BENCHMARK_CORPORA)))
    return dataset_dir(entry["dir"]), entry["text_column"]


PRETRAIN_CORPUS = {
    "mira-s": [dataset_dir("llm-pretrain-wikitext103")],
    # "mira_llm-tiny": [dataset_dir("llm-pretrain-fineweb-edu-2BT")],
    # "mira-s": [dataset_dir("llm-pretrain-fineweb-edu-2BT"),
    #               dataset_dir("llm-pretrain-wikitext103")],
    # FineWeb-Edu sample-100BT alone since 2026-10-09 ("in pretrain, we configure mira-m as
    # default, and fineweb-edu 100BT as training dataset"); 10BT + wikitext103 before.
    "mira-m": [dataset_dir("llm-pretrain-fineweb-edu-100BT")],
    "mira-l": [dataset_dir("llm-pretrain-fineweb-edu-10BT"),
                  dataset_dir("llm-pretrain-wikitext103")],
}

# Anthropic/hh-rlhf's own layout, which puts the subsets at the root of the dataset directory
# and names the split in the FILE:
#
#     llm-rlhf-instruction-following/
#         harmless-base/{train,test}.jsonl.gz
#         helpful-base/{train,test}.jsonl.gz
#         helpful-online/{train,test}.jsonl.gz
#         helpful-rejection-sampled/{train,test}.jsonl.gz
#
# So HH_DIR is the dataset directory itself; there is no rlhf_hh/ level under it.
INSTRUCT_DIR = dataset_dir("llm-rlhf-instruction-following")
INSTRUCTION_DIR = INSTRUCT_DIR                             # no nesting; kept as a name
HH_DIR = INSTRUCT_DIR

SFT_DIR = dataset_dir("llm-ultrachat")


def set_instruct_root(path):
    """Repoint the instruction-tuning tree and everything derived from it."""
    global INSTRUCT_DIR, INSTRUCTION_DIR, HH_DIR
    INSTRUCT_DIR = INSTRUCTION_DIR = path
    HH_DIR = path
    RLHF["prompt_dir"] = HH_DIR
    return INSTRUCT_DIR


COT_DIR = dataset_dir("llm-grpo-cot-gsm8k")             # chain-of-thought / reasoning
COT_GSM8K_DIR = COT_DIR                                    # {train,test}_<batch>.json
COT_COUNTDOWN_DIR = dataset_dir("llm-cot-countdown-game-20k")
DISTILL_DIR = HH_DIR        # distillation generates from the same prompts stage 9 rolls out on
CACHE_DIR = os.path.join(ROOT, "cache")                    # cache/tokens/bpe_<256+merges>_<fp>/<corpus>/
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")         # checkpoints/<stage>/<tag>/checkpoint.pt
OUTPUT_DIR = os.path.join(ROOT, "outputs")                  # everything a run produces
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")               # outputs/plots/<stage>/<figure>.pdf
MODEL_DIR = os.path.join(ROOT, "saved_models")
MIRA_CUDA_BIN = "mira_dspv2"
MIRA_CUDA_BINS = ("mira_dspv1", "mira_dspv2")
BPE_PATH = os.path.join(CHECKPOINT_DIR, "bpe", "bpe.json") # the trained tokenizer

BPE_DOWNLOAD_NAME = "tokenizer-bpe-fineweb-edu-2BT"
BPE_DOWNLOAD_PATH = os.path.join(download_dir(BPE_DOWNLOAD_NAME), "bpe", "bpe.json")

# Corpus file types. The scanner reads EVERY file with one of these extensions, so a corpus
# can mix prose, markdown and source code; a language model has no reason to care which.
# Source is included deliberately: code is the cleanest available supervision for long-range
# structure. Extensions are matched case-insensitively.
CORPUS_EXTENSIONS = [
    # prose and markup
    "txt", "md", "markdown", "rst", "tex", "org",
    # data / config that reads as text
    "json", "jsonl", "yaml", "yml", "toml", "ini", "csv", "tsv",
    # C family
    "c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx", "cu", "cuh",
    # assembly
    "s", "asm",
    # JVM and .NET
    "java", "kt", "kts", "scala", "cs",
    # scripting and dynamic
    "py", "pyx", "rb", "pl", "pm", "php", "lua", "r", "jl", "m",
    # web
    "js", "jsx", "ts", "tsx", "html", "htm", "css", "scss", "vue", "svelte",
    # systems and functional
    "go", "rs", "swift", "zig", "hs", "ml", "mli", "ex", "exs", "erl", "clj", "lisp", "el",
    # shell and build
    "sh", "bash", "zsh", "fish", "ps1", "bat", "mk", "cmake", "gradle",
    # query and hardware description
    "sql", "v", "sv", "vhd", "vhdl",
    # columnar corpora: a parquet file is not text, so it is read through pyarrow and the
    # column named by PRETRAIN["text_column"] is treated as the document text
    "parquet",
]

# --------------------------------------------------------------------------- #
# model architecture (MiraLLM: positional-encoding-free; see model/mira_llm.py)
# --------------------------------------------------------------------------- #
SCHEME_MOE_KEYS = ("moe_total_experts", "moe_layers", "moe_active_experts", "moe_expert_width",
                   "moe_shared_experts")


def _read_mira_schemes(path=None):
    """The MIRA schemes of config/default/config_mira.yaml as {"mira-<size>": {key: value}}.

    One root key `mira-<size>:` per scheme and beneath it, two-space indented, one `field: value`
    line per field: n_layers, d_model, ffn_width, lattice_d_e, lattice_d_r, lattice_d_m and
    lattice_K are positive integers, lattice_modulus_map, lattice_argument_map and lattice_initializer are names, and
    lattice_init_horizon is a mapping of four-space `default: [lo, hi]` / `L<n>: [lo, hi]` lines.
    ffn_width, the FFN hidden width, is the one optional field: absent, it is 4 x d_model. The
    entries carry the names the model and its checkpoints use (n_layer, n_embd, ffn_hidden,
    coefficient_bottleneck, memory_dimension, lattice_order, lattice_modulus_map, lattice_argument_map), plus n_head = d_model / 64,
    lattice_expansion_factor =
    d_e / d_model and coefficient_bottleneck_factor = d_model / d_r, which every checkpoint
    records. The mixture-of-experts fields are optional: moe_total_experts (absent or 1 = no
    MoE), moe_active_experts and moe_expert_width are positive integers, moe_shared_experts is a
    non-negative integer and moe_layers is [first, last], the layers whose FFN is MoE; with
    moe_total_experts above 1 all five are required and carried under the same names. A scheme
    missing a field, an unknown field or a malformed line stops the import."""
    path = path or os.path.join(ROOT, "config", "default", "config_mira.yaml")
    integers = ("n_layers", "d_model", "ffn_width", "lattice_d_e", "lattice_d_r", "lattice_d_m",
                "lattice_K", "moe_total_experts", "moe_active_experts", "moe_expert_width",
                "moe_shared_experts")
    names = ("lattice_modulus_map", "lattice_argument_map", "lattice_initializer")
    fields = integers + names + ("lattice_init_horizon", "moe_layers")
    optional = ("ffn_width",) + SCHEME_MOE_KEYS
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError as e:
        raise SystemExit(f"[config] cannot read the MIRA schemes at {path} ({e}).\n"
                         f"         It ships with the repository; restore it from git.") from e
    raw, current, mapping = {}, None, None
    for i, original in enumerate(lines, 1):
        line = original.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            name = line[:-1].strip() if line.endswith(":") else ""
            if not re.fullmatch(r"mira-[a-z0-9]+(-[a-z0-9]+)*", name):
                raise SystemExit(f"[config] {path}:{i}: expected a scheme key 'mira-<size>:', "
                                 f"got {original.rstrip()!r}")
            current, mapping = name, None
            if current in raw:
                raise SystemExit(f"[config] {path}:{i}: duplicate scheme {name!r}")
            raw[current] = {}
            continue
        key, sep, value = line.strip().partition(":")
        key, value = key.strip(), value.strip()
        if indent == 4 and mapping is not None and sep:
            try:
                bounds = json.loads(value)
            except json.JSONDecodeError:
                bounds = None
            if (not isinstance(bounds, list) or len(bounds) != 2
                    or not all(type(v) in (int, float) and v > 0 for v in bounds)):
                raise SystemExit(f"[config] {path}:{i}: lattice_init_horizon {key} must be "
                                 f"[lo, hi], got {value!r}")
            if key in mapping:
                raise SystemExit(f"[config] {path}:{i}: duplicate lattice_init_horizon {key!r}")
            mapping[key] = bounds
            continue
        if indent != 2 or current is None or not sep:
            raise SystemExit(f"[config] {path}:{i}: expected two-space 'field: value' under a "
                             f"scheme, got {original.rstrip()!r}")
        mapping = None
        if key in ("arch", "lattice_arch"):
            raise SystemExit(f"[config] {path}:{i}: use lattice_modulus_map and lattice_argument_map")
        if key not in fields:
            raise SystemExit(f"[config] {path}:{i}: unknown scheme field {key!r}; expected one "
                             f"of {', '.join(fields)}")
        if key in raw[current]:
            raise SystemExit(f"[config] {path}:{i}: duplicate field {key!r}")
        if key == "lattice_init_horizon":
            if value:
                raise SystemExit(f"[config] {path}:{i}: lattice_init_horizon is a mapping of "
                                 f"four-space 'default: [lo, hi]' lines below it")
            mapping = raw[current][key] = {}
            continue
        if key == "moe_layers":
            try:
                span = json.loads(value)
            except json.JSONDecodeError:
                span = None
            if (not isinstance(span, list) or len(span) != 2
                    or not all(type(v) is int and v >= 0 for v in span) or span[0] > span[1]):
                raise SystemExit(f"[config] {path}:{i}: moe_layers must be [first, last] with "
                                 f"0 <= first <= last, got {value!r}")
            raw[current][key] = span
            continue
        if key in names:
            if not re.fullmatch(r"[A-Za-z0-9_\-]+", value):
                raise SystemExit(f"[config] {path}:{i}: {key} must be a name, got {value!r}")
            raw[current][key] = value
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if key == "moe_shared_experts" and type(parsed) is int and parsed == 0:
            pass
        elif not (type(parsed) is int and parsed > 0):
            raise SystemExit(f"[config] {path}:{i}: {key} must be a positive integer, "
                             f"got {value!r}")
        raw[current][key] = parsed
    if not raw:
        raise SystemExit(f"[config] {path}: no scheme defined")
    out = {}
    for name, s in raw.items():
        missing = [f for f in fields if f not in s and f not in optional]
        if missing:
            raise SystemExit(f"[config] {path}: scheme {name!r} is missing "
                             f"{', '.join(missing)}")
        if "default" not in s["lattice_init_horizon"]:
            raise SystemExit(f"[config] {path}: scheme {name!r} "
                             f"lattice_init_horizon has no default")
        if s.get("moe_total_experts", 1) > 1:
            absent = [f for f in SCHEME_MOE_KEYS if f not in s]
            if absent:
                raise SystemExit(f"[config] {path}: scheme {name!r} sets moe_total_experts "
                                 f"{s['moe_total_experts']} and is missing {', '.join(absent)}")
            if s["moe_active_experts"] > s["moe_total_experts"]:
                raise SystemExit(f"[config] {path}: scheme {name!r} moe_active_experts "
                                 f"{s['moe_active_experts']} exceeds moe_total_experts "
                                 f"{s['moe_total_experts']}")
            if s["moe_layers"][1] >= s["n_layers"]:
                raise SystemExit(f"[config] {path}: scheme {name!r} moe_layers "
                                 f"{s['moe_layers']} reaches past its {s['n_layers']} layers")
        if s["lattice_modulus_map"] not in ("cauchy", "gaussian", "expexp", "expsoftplus", "hill"):
            raise SystemExit(f"[config] {path}: invalid lattice_modulus_map in {name}")
        if s["lattice_argument_map"] not in ("mobius", "hyperbolic", "identity"):
            raise SystemExit(f"[config] {path}: invalid lattice_argument_map in {name}")
        d = s["d_model"]
        out[name] = {"n_layer": s["n_layers"], "n_head": d // 64, "n_embd": d,
                     "ffn_hidden": s.get("ffn_width", 4 * d),
                     "lattice_d_e": s["lattice_d_e"],
                     "lattice_expansion_factor": s["lattice_d_e"] // d,
                     "coefficient_bottleneck": s["lattice_d_r"],
                     "coefficient_bottleneck_factor": d // s["lattice_d_r"],
                     "memory_dimension": s["lattice_d_m"], "lattice_order": s["lattice_K"],
                     "lattice_modulus_map": s["lattice_modulus_map"],
                     "lattice_argument_map": s["lattice_argument_map"],
                     "lattice_initializer": s["lattice_initializer"],
                     "lattice_init_horizon": s["lattice_init_horizon"]}
        out[name].update({f: s[f] for f in SCHEME_MOE_KEYS if f in s})
    return out


# THE THREE CONFIGURATION SCHEMES, in one place. Depth, width and the context window each
# scheme is pretrained at; everything else is shared and lives in MODEL below.
#
# THE HEAD DIMENSION IS FIXED AT 64 in every scheme (model.mira_llm.HEAD_DIM), so n_head is not
# a free choice: it is n_embd / 64. A constant head dimension is what makes the three sizes
# comparable -- widening by adding heads of the same size changes how much the model can
# attend to, while widening the heads themselves changes what a single head IS, and the two
# are then confounded in any result that spans sizes.
#
# THE CONTEXT WINDOW IS A SCHEME PROPERTY, NOT AN ARCHITECTURAL ONE. Nothing in MiraLLM refers
# to an absolute position, so no length is forbidden and a trained model can be evaluated at
# any length; the window here is what the scheme is PRETRAINED at, chosen against the cost of
# attention, which is quadratic in it. The small scheme is the one meant to run on a single
# machine. The `context` column is the SCHEDULE each scheme trains through, shortest first;
#
#     scheme          layers  heads  d_model  d_h  context
#     MiraLLM-Baby        12      8      512   64  512..1k
#     MiraLLM-Tiny        24      8      512   64  1k..8k
#     MiraLLM-S           32      8      512   64  1k..8k
#     MiraLLM-M           48      8      512   64  1k..16k
#     MiraLLM-L           64     12      768   64  1k..32k
#
# Parameter counts are not duplicated here. They depend on the architecture, the FIR/MIIR/
# mixer orders, and the tokenizer vocabulary; every trainer prints the exact count from the
# model it actually built.
#
# BABY, TINY, S AND M share width 512 and differ in depth -- 12, 24, 32 and 48 layers. Only L
# is wider. Parameter counts depend on the selected architecture and tokenizer vocabulary and
# are printed from the built model rather than maintained as a second static table here.
# `context_window` IS A LIST: the windows this scheme trains through, in order. Pretraining
# does not sit at one sequence length -- it starts short and lengthens -- so the schedule is
# part of the scheme rather than a table kept alongside it, where the two could disagree.
#
# THE VALUES ARE ARBITRARY. Nothing anywhere assumes they are powers of two, that each is a
# multiple of the last, or that there are two of them: [768, 3000, 5000] is as valid as
# [1024, 4096]. A single number is accepted too and means one window all the way through.
# The LARGEST entry is the model's block_size, since that is the widest it is ever asked for.
SCHEMES = _read_mira_schemes()
    # THE THREE SIZES use lattice_order 4, coefficient_bottleneck_factor 16 and
    # memory_dimension 16, and d_r is d / 16 at every rung. S is narrower than M and L; M and L are
    # the same width and differ in depth alone. The quoted instruction below predates every one of
    # those moves: it says order 8 and bottleneck 32, and it names four rungs plus a `baby`.
    #
    # WHAT THE LADDER WAS AT lattice_order 4, memory_dimension 16 (HISTORICAL; the current sizes, at
    # lattice_K 2 and lattice_d_m 64, are the size blocks in config/default/config_mira.yaml), at V = 50,259:
    #
    #     mira-s    12 layers  d=768   d_e 1,536  d_r 48  ->   150,153,216
    #     mira-m    28 layers  d=1,024 d_e 2,048  d_r 64  ->   513,664,960
    #     mira-l    64 layers  d=1,024 d_e 2,048  d_r 64  -> 1,107,917,824
    #
    # The block is 16,507,024 parameters at d = 1,024 and d_e = 2,048, so M and L differ only in
    # depth. State-to-parameter is 0.00786 / 0.00714 / 0.00757 -- flat across the ladder, because
    # d and d_e no longer climb with it.
    #
    # `mira-s` IS THE GEOMETRY THE BENCHMARK, THE ABLATION AND THE PROFILER MEASURE.
    #
    #     "baby:   lattice_order=8, layers=16, d=512, d_e=1536, coefficient_bottleneck=32
    #      small:  lattice_order=8, layers=24, d=512, d_e=2048, coefficient_bottleneck=32
    #      medium: lattice_order=8, layers=48, d=512, d_e=3072, coefficient_bottleneck=32
    #      large:  lattice_order=8, layers=96, d=512, d_e=4096, coefficient_bottleneck=32"            -- 2026-08-22
    #
    # S TO M GROWS BOTH AXES, M TO L GROWS DEPTH ALONE -- 2026-09-04, at the user's sizing. The
    # width is pinned at d = 1,024 and d_e = 2,048 from M upward, so the top of the ladder is a
    # depth series and the pair (M, L) reads as a controlled comparison on layers; the S-to-M step
    # does not, since it moves d, d_e and d_r at once. The counts are in the table above.
    #
    # d_e IS PART OF THE SCHEME rather than an expansion factor, because it has not always been an
    # integer multiple of d_model. It reaches the model as MODEL["lattice_d_e"] and MiraLLM
    # forwards it to every MiraBlock.
    #
    # n_head FOLLOWS THE WIDTH: 768 / HEAD_DIM 64 = 12 exactly for MIRA-S. The split reaches
    # nothing -- the mixer is elementwise per channel -- and survives only because MiraLLM builds
    # a disposable RoPE scaffold before replacing it.

# Public scheme names are canonical MiraLLM names. Older checkpoints are converted explicitly.
LEGACY_SCHEME_ALIASES = {"mira_llm-" + name[len("mira-"):]: name for name in SCHEMES}


def canonical_scheme(name):
    """Canonical MIRA scheme name, mira-<size>. The spelling mira_llm-<size> (recorded cfgs, older
    checkpoints and command lines) is read as mira-<size>."""
    return LEGACY_SCHEME_ALIASES.get(name, name)


LEGACY_ARCH_ALIASES = {"mira": "mira-lattice", "cdsa": "mira-lattice",
                       "mira-softmax": "mira-lattice", "mira-ln": "mira-lattice",
                       "mira-matrix": "mira-lattice",
                       # cayley_parameterization's names before they were canonicalised
                       "mobius": "cauchy_x_mobius"}


# SHORT CODES OF THE LATTICE ARCHES (2026-10-05), used in checkpoint names and accepted wherever an
# arch is read: <radius code>x<rotation code>, e.g. cauchy_x_hyperbolic = rccxRHB.
ARCH_RADIUS_CODES = {"cauchy": "rcc", "gaussian": "rgs", "expexp": "ree", "expsoftplus": "res",
                     "hill": "rhl"}
ARCH_ROTATION_CODES = {"mobius": "RMB", "identity": "RID", "hyperbolic": "RHB"}
ARCH_CODES = {f"{radius}_x_{rotation}": f"{rcode}x{qcode}"
              for radius, rcode in ARCH_RADIUS_CODES.items()
              for rotation, qcode in ARCH_ROTATION_CODES.items()}
_ARCH_FROM_CODE = {code.lower(): arch for arch, code in ARCH_CODES.items()}


def arch_code(arch):
    """The short code of a lattice arch (cauchy_x_hyperbolic -> rccxRHB)."""
    return ARCH_CODES[lattice_arch(arch)]


def canonical_arch(name):
    """Canonical architecture name. Every arch string in the pipeline passes through here --
    `common.parse_args`, `common.build_model` and `helpers.arch_of` -- so an alias is resolved
    once, before anything branches on it or writes it into a filename. A short code (rccxRHB,
    any case) is its full name."""
    if isinstance(name, str) and name.strip().lower() in _ARCH_FROM_CODE:
        return _ARCH_FROM_CODE[name.strip().lower()]
    return LEGACY_ARCH_ALIASES.get(name, name)


# THE MIRA cayley_parameterization, <radius>_x_<rotation>: the map from beta to r and from alpha
# to (sin, cos) of the Cayley stage matrix.
#     radius    cauchy     r = 1/(1 + beta^2)
#     rotation  mobius     sin = (1 - alpha^2)/(1 + alpha^2), cos = 2 alpha/(1 + alpha^2)
#               hyperbolic sin = tanh(alpha), cos = sech(alpha)
#               identity   theta = alpha: sin = sin(alpha), cos = cos(alpha)
#     radius    gaussian   r = exp(-beta^2)
#     radius    expexp     r = exp(-exp(beta))
#     radius    expsoftplus r = exp(-exp(s_r^(c)) softplus(beta)), s_r^(c) per channel
#     radius    hill       r = beta^2/(1 + beta^2)
# All are the lattice family "mira-lattice". "mira-lattice", its aliases and a missing arch mean
# cauchy_x_mobius, the name used before (mobius) resolves through LEGACY_ARCH_ALIASES.
LATTICE_ARCHES = ("cauchy_x_mobius", "cauchy_x_hyperbolic", "cauchy_x_identity",
                  "gaussian_x_mobius", "gaussian_x_hyperbolic", "gaussian_x_identity",
                  "expexp_x_mobius", "expexp_x_hyperbolic", "expexp_x_identity",
                  "expsoftplus_x_mobius", "expsoftplus_x_hyperbolic", "expsoftplus_x_identity",
                  "hill_x_mobius", "hill_x_hyperbolic", "hill_x_identity")


def lattice_arch(name):
    """The MIRA architecture `name` selects, or None when it is not a lattice name at all.

    None, the empty string, "mira-lattice" and every alias of it are cauchy_x_mobius."""
    if name is None or name == "":
        return "cauchy_x_mobius"
    name = canonical_arch(name)
    if name in LATTICE_ARCHES:
        return name
    return "cauchy_x_mobius" if name == "mira-lattice" else None


def mira_map_arch(cfg):
    if "lattice_modulus_map" not in cfg or "lattice_argument_map" not in cfg:
        raise ValueError("MIRA requires lattice_modulus_map and lattice_argument_map; "
                         "a checkpoint saved before those keys cannot be loaded")
    modulus = cfg["lattice_modulus_map"]
    argument = cfg["lattice_argument_map"]
    if modulus not in ("cauchy", "gaussian", "expexp", "expsoftplus", "hill"):
        raise ValueError(f"invalid lattice_modulus_map: {modulus!r}")
    if argument not in ("mobius", "hyperbolic", "identity"):
        raise ValueError(f"invalid lattice_argument_map: {argument!r}")
    return f"{modulus}_x_{argument}"


def is_lattice_arch(name):
    """Does `name` select the lattice family -- one of LATTICE_ARCHES or a legacy name?"""
    return name not in (None, "") and lattice_arch(name) is not None


# THE INITIALIZERS BY THEIR CANONICAL NAMES (2026-10-13), the names config files and checkpoints
# use; a former name is read as an alias of its canonical name (canonical_lattice_initializer).
LATTICE_INITIALIZERS = (
    "W_b_uniform",
    "W_b_normal",
    "W_jeffreys_b_uniform",
    "W_jeffreys_b_normal",
    "W_uniform_b_jeffreys",
    "W_normal_b_jeffreys",
    "W_uniform_b_nested_jeffreys",
    "uniform_horizon_fixed_angles",
    "uniform_horizon_fixed_angles_right",
    "alpha_beta_uniform",
)
# THE FORMER FULL NAMES, accepted wherever an initializer is read and kept in the directory names
# written before 2026-10-13.
LEGACY_LATTICE_INITIALIZER_NAMES = {
    "W_uniform_b_jeffreys": "uniform_weight_jeffreys_horizon_bias_init",
    "W_uniform_b_nested_jeffreys": "uniform_weight_nested_jeffreys_horizon_bias_init",
    "uniform_horizon_fixed_angles": "schur_init_both_side",
    "uniform_horizon_fixed_angles_right": "schur_init_right_side",
    "alpha_beta_uniform": "uniform_init",
}
# THE FIGURE LEGEND OF EACH INITIALIZER (2026-10-13).
LATTICE_INITIALIZER_LEGEND = {
    "W_b_uniform": "$W,b$ pytorch-uniform",
    "W_b_normal": "$W,b$ pytorch-normal",
    "W_jeffreys_b_uniform": "$W$ jeffreys w/ $b$ uniform",
    "W_jeffreys_b_normal": "$W$ jeffreys w/ $b$ normal",
    "W_uniform_b_jeffreys": "$W$ uniform w/ $b$ jeffreys",
    "W_normal_b_jeffreys": "$W$ normal w/ $b$ jeffreys",
    "W_uniform_b_nested_jeffreys": "$W$ uniform w/ $b$ nested jeffreys",
    "uniform_horizon_fixed_angles": "uniform horizon w/ fixed angles",
    "uniform_horizon_fixed_angles_right": "uniform horizon w/ fixed angles, right half-plane",
    "alpha_beta_uniform": r"$\alpha,\beta$ uniform",
}
# SHORT NAMES (2026-10-05), used in checkpoint names written from 2026-10-05 to 2026-10-09 and
# accepted wherever an initializer is read.
LATTICE_INITIALIZER_SHORT = {
    "W_uniform_b_jeffreys": "jeffreys",
    "W_uniform_b_nested_jeffreys": "nested_jeffreys",
    "uniform_horizon_fixed_angles": "schurboth",
    "uniform_horizon_fixed_angles_right": "schurright",
    "alpha_beta_uniform": "uniform",
}
_LATTICE_INITIALIZER_ALIASES = {
    **{name.lower(): name for name in LATTICE_INITIALIZERS},
    **{old.lower(): name for name, old in LEGACY_LATTICE_INITIALIZER_NAMES.items()},
    **{short.lower(): name for name, short in LATTICE_INITIALIZER_SHORT.items()},
    "schur_init": "uniform_horizon_fixed_angles",
}
# DROPPED 2026-10-05. A new model refuses them; a RECORDED cfg (a checkpoint's) that names one is
# rebuilt with W_uniform_b_jeffreys, which only decides the random draw that the checkpoint's
# weights then replace.
DROPPED_LATTICE_INITIALIZERS = (
    "phase_bank_init",
    "phase_bank_horizon_init",
    "horizon_bias_uniform_init",
    "horizon_weight_uniform_init",
    "jeffreys_horizon_bias_init",
    "uniform_weight_capacity_jeffreys_horizon_bias_init",
)


def canonical_lattice_initializer(name, recorded=False):
    """The canonical lattice_initializer name of `name` (a canonical name in any case, a former full
    name, a short name or the old "schur_init"). recorded=True is for a checkpoint's cfg: a dropped
    name maps to W_uniform_b_jeffreys instead of being refused."""
    method = str(name).strip()
    canonical = _LATTICE_INITIALIZER_ALIASES.get(method.lower())
    if canonical is not None:
        return canonical
    if method in DROPPED_LATTICE_INITIALIZERS:
        if recorded:
            return "W_uniform_b_jeffreys"
        raise ValueError(f"lattice_initializer {method!r} was dropped on 2026-10-05; use one of "
                         + ", ".join(LATTICE_INITIALIZERS))
    raise ValueError("lattice_initializer must be one of " + ", ".join(LATTICE_INITIALIZERS))


def lattice_initializer_short(name):
    """The short name of a lattice_initializer (W_uniform_b_jeffreys -> jeffreys)."""
    method = canonical_lattice_initializer(name)
    return LATTICE_INITIALIZER_SHORT.get(method, method)


def legacy_lattice_initializer_name(name):
    """The former full name of a lattice_initializer (W_uniform_b_jeffreys ->
    uniform_weight_jeffreys_horizon_bias_init), or its canonical name when it has none."""
    method = canonical_lattice_initializer(name)
    return LEGACY_LATTICE_INITIALIZER_NAMES.get(method, method)


def lattice_initializer_keyword(name):
    """The canonical name of a lattice_initializer (uniform_init -> alpha_beta_uniform)."""
    return canonical_lattice_initializer(name)


def lattice_initializer_legend(name):
    """The figure legend of a lattice_initializer (W_b_uniform -> $W,b$ pytorch-uniform)."""
    return LATTICE_INITIALIZER_LEGEND[canonical_lattice_initializer(name)]


# A SCHEME'S lattice_initializer IS HELD BY ITS CANONICAL NAME, whichever keyword the yaml uses.
for _scheme in SCHEMES.values():
    _scheme["lattice_initializer"] = canonical_lattice_initializer(_scheme["lattice_initializer"])


# PER-LAYER HORIZON RANGES (2026-10-05). lattice_init_horizon is either one range for every layer,
#     lattice_init_horizon:   10,2000
# or a mapping with a `default` range and per-layer overrides L<n> (n = 0 is the first MIRA block) or
# L<n>-<m> (layers n to m inclusive, e.g. L1-3; expanded to L1, L2, L3; a layer given twice is refused):
#     lattice_init_horizon:
#       default:  [10,2000]
#       L0:       [1,128]
# canonical_lattice_init_horizon returns (min, max) for the first form and
# {"default": (min, max), "L0": (min, max), ...} for the second; every place that builds a model
# reads the range of layer n through layer_lattice_init_horizon(value, n).
_HORIZON_LAYER_KEY = re.compile(r"^l(\d+)(?:-(\d+))?$")


def _horizon_range(value):
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "[(" and text[-1:] in "])":
            text = text[1:-1]
        items = [item.strip() for item in text.split(",") if item.strip()]
    elif isinstance(value, (tuple, list)):
        items = list(value)
    else:
        raise ValueError("lattice_init_horizon must be '<minimum>,<maximum>' or a mapping of them")
    if len(items) != 2:
        raise ValueError("lattice_init_horizon must contain exactly minimum,maximum")
    minimum, maximum = (float(item) for item in items)
    if not (math.isfinite(minimum) and math.isfinite(maximum)):
        raise ValueError("lattice_init_horizon values must be finite")
    if minimum <= 0.0 or maximum < minimum:
        raise ValueError("lattice_init_horizon must satisfy 0 < minimum <= maximum")
    return minimum, maximum


def canonical_lattice_init_horizon(value):
    """(min, max) for one range; {"default": (min, max), "L<n>": (min, max), ...} for per-layer
    ranges. Accepts a tuple/list, '10,2000', '[10,2000]', a dict, or that dict as JSON (the yaml
    mapping arrives as JSON). A mapping must name `default`; keys are default and L0, L1, ..."""
    if isinstance(value, str) and value.strip().startswith("{"):
        value = json.loads(value)
    if not isinstance(value, dict):
        return _horizon_range(value)
    out = {}
    for key, entry in value.items():
        name = str(key).strip().lower()
        if name == "default":
            out["default"] = _horizon_range(entry)
            continue
        match = _HORIZON_LAYER_KEY.match(name)
        if not match:
            raise ValueError(f"lattice_init_horizon: {key!r} is not 'default', L<n> (one layer, e.g. L0) "
                             f"or L<n>-<m> (layers n to m, e.g. L1-3)")
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) is not None else first
        if last < first:
            raise ValueError(f"lattice_init_horizon: {key!r} names layers {first} to {last}; "
                             f"write the lower layer first (L{last}-{first})")
        horizon = _horizon_range(entry)
        for layer in range(first, last + 1):
            if f"L{layer}" in out:
                raise ValueError(f"lattice_init_horizon: layer {layer} is given twice ({key!r})")
            out[f"L{layer}"] = horizon
    if "default" not in out:
        raise ValueError("lattice_init_horizon: a per-layer mapping must name a `default` range")
    if len(out) == 1:
        return out["default"]
    return {"default": out["default"],
            **{key: out[key] for key in sorted((k for k in out if k != "default"),
                                                key=lambda k: int(k[1:]))}}


def layer_lattice_init_horizon(value, layer):
    """The (min, max) horizon range of MIRA layer `layer` (0-based) under lattice_init_horizon `value`."""
    spec = canonical_lattice_init_horizon(value)
    if isinstance(spec, dict):
        return spec.get(f"L{int(layer)}", spec["default"])
    return spec


def lattice_init_horizon_text(value):
    """'10,2000' for one range, else the mapping as compact JSON with [min, max] lists."""
    spec = canonical_lattice_init_horizon(value)
    if isinstance(spec, dict):
        return json.dumps({key: [float(f"{a:g}"), float(f"{b:g}")] for key, (a, b) in spec.items()},
                          separators=(",", ":"))
    return ",".join(f"{float(item):g}" for item in spec)


# THE DEFAULT IS THE SIZE THE WORK IS REPORTED AT. mira-s is 32 layers at d_model 512
# through a 1,024 -> 8,192 window, and it needs a datacentre card and the ~10 GB FineWeb-Edu
# 2BT subset, with WikiText-103 mixed in beside it. THAT IS NOT A LAPTOP CONFIGURATION and
# nothing here should be read as if it were: the default is the reported size, not the
# convenient one. THIS VALUE AND config.sh's MIRA_CONFIG MUST AGREE: config.sh forwards
# --mira_config on every stage, so a disagreement here is not overridden, it is a second
# answer to one question that only shows up in whichever artefact is named from the other.
#
# mira_llm-baby is the small path -- 12 layers at d_model 512 through 512 then 1,024 tokens, on
# WikiText-103 alone. It is the scheme the benchmark runs at. One variable away:
#   MIRA_CONFIG=mira_llm-baby ./old_scripts/stage6_pretrain.sh
DEFAULT_SCHEME = "mira-s"

# ONE PATH, FOR THE FEW READERS THAT WANT ONE. The FIRST corpus of the DEFAULT scheme.
#
# DEFINED HERE AND NOT BESIDE PRETRAIN_CORPUS, because it reads DEFAULT_SCHEME and that is
# bound on the line above -- at the dictionary, 150 lines earlier, the name does not exist yet
# and the module raises NameError on import. It did: every stage died at
# `import default_config` until this moved. py_compile passes a forward reference at module
# scope; only an actual import catches it.
PRETRAIN_DIR = PRETRAIN_CORPUS[DEFAULT_SCHEME][0]

# PER-GPU BATCH AT THE LONGEST WINDOW, and the shorter windows scale UP from it. The batch is
# sized where memory is tightest -- the last window -- and every shorter window then runs at
# batch x (longest / this window), so TOKENS PER GPU PER STEP stays constant. The global batch
# is this value multiplied by the number of GPUs.
#
# That constancy is the point. A step is the unit the learning-rate schedule, the step budget
# and every history record are counted in, and if a step carried 8x fewer tokens at the start
# than at the end, none of those three would mean the same thing twice. Keeping tokens per step
# fixed makes the schedule invisible to everything except the memory it was introduced to save.
SCHEME_BATCH = {
    "mira-s": 50,          # per GPU; old global 200 / 4 GPUs. Was mira_llm-baby's.
    # "mira_llm-tiny": 15,     # per GPU; old global 60 / 4 GPUs
    # "mira-s": 14,        # the 24-layer scheme's; dropped with it
    "mira-m": 6,           # per GPU; old global 24 / 4 GPUs
    "mira-l": 1,           # per GPU; old global 4 / 4 GPUs
}

# PRETRAINING STEPS PER SCHEME: 1.5 EPOCHS over the packed corpus.
#
#     steps = round(1.5 x corpus_tokens / (batch x (ctx_longest - 1)))
#
# over the MEASURED 2,177,432,967-token stream = 3,266,149,450 target tokens. Recompute
# whenever SCHEME_BATCH, the longest window, or the corpus changes -- all three are in that
# one line and none of them is stable across a change of scheme.
#
# THE THREE ARE NOT EQUAL AND CANNOT BE MADE SO. -s and -m land together at ~199,400 steps
# because their tokens/step happen to coincide (16,382 against 16,383); -l is half that, at
# 99,678, because one sequence at 32,768 is twice the step and one sequence is the floor. Equal
# EPOCHS is the invariant being held, not equal steps -- the three see the same data the same
# number of times, in differently sized bites.
SCHEME_STEPS = {
    "mira-s": 60000,      # corpus read many times over; see the note above. Was baby's.
    # "mira_llm-tiny": 250000,  # at batch 60 this was 56.4 epochs
    # "mira-s": 7121,       # the 24-layer scheme's 1.5 epochs; dropped with it
    "mira-m": 8307,       # 1.5 epochs
    "mira-l": 24920,      # 1.5 epochs
}

# MICRO-BATCH BY TOKENS PER PASS, not by sequences. Activations are linear in TOKENS -- one
# sequence of 8,192 costs exactly what eight of 1,024 cost -- so the quantity that has to be
# held below what the card can take is tokens per forward pass, and the number of sequences
# that amounts to is whatever the window makes it. Sizing it in sequences instead ties the
# micro-batch to the window in the wrong direction: it grows as the window shrinks, and at the
# shortest window it grows back to the full batch, which is where the saving was supposed to be.
#
# WHAT A TOKEN COSTS, per block: roughly 29 tensors the width of d_model are kept for the
# backward pass -- the state space module's 2d input projection, its convolution, decay and
# gate, attention's 3d qkv and its gate, and the feed-forward's 4d hidden twice over. That is
# 29 x d_model x 4 bytes x n_layer, at the depths and widths SCHEMES carries today:
#
#     baby 16x512  0.91 MiB      tiny 24x512  1.36 MiB      -S 32x512  1.81 MiB
#     -M   48x512  2.72 MiB      -L   64x768  5.44 MiB
#
# Multiply by the budget below to see what each scheme holds. THE COUNT 29 IS ITSELF PRE-
# 2026-08-14: it still counts the state gate that was removed and an in_proj of width 2d that is
# now d, so every figure in that row is an UPPER bound rather than a fit.
# THE ANCHOR IS A MEASUREMENT, not arithmetic: mira-s at 8,192 tokens per pass was 18.1 GiB
# on the card, which puts a token at 1.88 MB for the 24 layers of width 512 that -s was WHEN
# THIS WAS MEASURED -- a third more than the estimate that preceded it. -s is 32 layers now, and
# 24 layers of 512 is mira_llm-tiny's shape today rather than nobody's. The other schemes scale
# from that anchor by layers x width. Confirm with nvidia-smi on the first hundred steps before
# trusting any of it.
#
# RE-MEASURED UNDER MIXED PRECISION. 12,288 tokens per pass was 29.6 GiB in fp32 and is 25.4 GiB
# with bf16 activations -- a saving of 4.2 GiB rather than half, because the part that did NOT
# change is large: fp32 weights, their fp32 masters, fp32 gradients and Adam's two fp32 moments
# come to ~7.0 GiB before a single activation exists, and that floor is flat in tokens. Only the
# 22.6 GiB of activations moved, to 18.4 GiB, which is 1.53 MiB per token.
#
# SO THE BUDGET WAS RAISED RATHER THAN BANKED: 12,288 -> 19,456 tokens, ~36 GiB, since the point
# of the saving is a bigger pass and not an idle card. Two figures come from that one:
#
#     peak     ~36.1 GiB, at the SHORTEST window, where 19 x 1,023 packs the budget most tightly
#     8,192    2 sequences per pass instead of 1 -- the old setting could not fit two, so the
#              longest window ran at 8,191 tokens per pass and left a third of the card unused
#
# SAFE TO CHANGE MID-RUN. The micro-batch splits a step into passes whose gradients accumulate;
# the optimiser sees the same step either way, so a resumed run continues on the same curve.
SCHEME_MICRO_TOKENS = {
    # TOKENS PER FORWARD PASS PER GPU. Re-solved 2026-08-14 against 36 GiB PER GPU,
    # CHECKPOINTING OFF ("we also use micro batch! we never use checkpointing!"), and OUR CUDA
    # SCAN KERNEL LOADED, which is the default on CUDA and is what makes the long windows reach.
    #
    # WHY THESE ARE BIG AGAIN. pairwise_scan keeps 2*log2(T) tensors per scan; kernel/cuda's
    # mira_scan keeps (a, h) and nothing else. So with the cubin the cost per token is FLAT in
    # the window instead of growing with it -- 1.15 MiB/token for -tiny at 8,192 against 4.20
    # -- which is the profile the first commit had with blocked_scan, at pairwise_scan's launch
    # count. Sizing a run as if the reference scan were running charges it for work the GPU is
    # not doing, and is what put 8,192 out of reach.
    #
    # PER-GPU PEAK AT THESE VALUES (limit 36 GiB, chunked loss 8,192):
    #
    #     scheme    L    d   1,024    2,048    4,096    8,192   16,384+
    #     tiny     20  512   35.2G    33.9G    31.3G    26.1G
    #     s        32  512   35.5G    35.5G    31.3G    23.0G
    #     m        48  512   35.9G    32.8G    32.8G    32.8G   does not fit
    #     l        64  768   33.4G    26.7G    does not fit
    #
    # -m STOPS AT 8,192 AND -l AT 2,048, and no micro-batch reaches further: a pass is at
    # minimum ONE sequence. That is reported rather than bought with checkpointing, exactly as
    # 31074a5 did -- "-L trains at the windows it fits".
    #
    # The value is the MINIMUM over the windows that fit, since one constant has to hold at all
    # of them. Now that the cost is flat in T it barely varies, which is the point.
    # THIS MOVES WITH SCHEME_BATCH OR THE BATCH DOES NOTHING, and that is the trap the first
    # setting fell into. `micro = min(batch, micro_tokens // ctx)`, then the pass is `micro`
    # sequences per GPU. At 16,384 and a window of 1,024 that is 16 per GPU, 64 across four --
    # which already covered the whole batch of 60 in ONE pass. Raising the batch alone would
    # therefore have added gradient-accumulation passes at unchanged memory: same 6 GiB, same
    # occupancy, four times the step time. Occupancy is set HERE; SCHEME_BATCH only decides how
    # many tokens an optimiser step is worth.
    #
    # 73,728 = 72 sequences x 1,024, and 72 x 4 GPUs = 288 = the batch, so a step is still one
    # pass at the longest window. It is also one pass at 512, where the batch doubles to 576 and
    # the sequences halve in length -- which is the invariant the whole schedule is built on:
    # tokens per pass, and so memory, stay flat as the window changes.
    "mira-s": 51200,          # 50 seq x 1,024 x 4 GPUs = 200 = the batch, still ONE pass
    # "mira_llm-tiny": 16384,
    # "mira-s": 8192,         # the 24-layer scheme's; dropped with it
    "mira-m": 8192,
    "mira-l": 2048,
}

def windows(value):
    """A `context_window` entry as a list of ints, whatever shape it was written in.

    A list stays a list, a bare number becomes a list of one, and a string of numbers -- which
    is what a shell variable can carry -- is split on commas. Everything downstream then has
    one shape to handle, and CONTEXT_WINDOW=1024,4096,8192 works from config.sh exactly as the
    Python list does."""
    if isinstance(value, str):
        value = [p for p in value.replace(" ", "").split(",") if p]
    elif not isinstance(value, (list, tuple)):
        value = [value]
    out = [int(v) for v in value if int(v) > 0]
    return out or [0]


def _scheme_entry(name):
    """SCHEMES[name]; a scheme config/default/config_mira.yaml does not define is a KeyError."""
    key = canonical_scheme(name)
    if key not in SCHEMES:
        raise KeyError(f"MIRA scheme {key!r} is not defined in config/default/config_mira.yaml, "
                       f"which defines {', '.join(SCHEMES)}")
    return SCHEMES[key]


def context_windows(name=DEFAULT_SCHEME):
    """The windows a scheme trains through, in the order it trains through them: the
    `context_window` of the one pretraining yaml, config/pretrain/config_pretrain.yaml, for every
    scheme (2026-10-09; there is no config/pretrain/config_<scheme>.yaml any more). A missing file
    or a blank or missing value stops with the file named."""
    key = canonical_scheme(name)
    _scheme_entry(key)
    path = os.path.join(ROOT, "config", "pretrain", "config_pretrain.yaml")
    if not os.path.isfile(path):
        raise SystemExit(f"[config] no pretraining yaml for {key!r}: "
                         f"{os.path.relpath(path, ROOT)} does not exist")
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            match = re.match(r"\s+context_window:(.*)$", line)
            if match:
                value = match.group(1).split("#", 1)[0].strip().strip("[]")
                if value:
                    return windows(value)
                break
    raise SystemExit(f"[config] {os.path.relpath(path, ROOT)}: no context_window value")


def context_window(name=DEFAULT_SCHEME):
    """The LONGEST window a scheme trains at, which is the model's block_size."""
    return max(context_windows(name))


def context_plan(windows, steps, batch_at_longest, micro_tokens=0):
    """[(start, stop, context, batch, micro_batch), ...] covering EXACTLY `steps` steps.

    Equal steps per window; the remainder goes to the last one, so the segments tile the budget
    with no step unaccounted for and none counted twice. The batch of each segment is scaled so
    that batch x context matches the longest window's, which is what keeps a step the same unit
    of work from the first to the last."""
    windows = list(windows) or [1]
    longest = max(windows)
    n, per = len(windows), int(steps) // len(windows)
    out, at = [], 0
    for i, ctx in enumerate(windows):
        take = (int(steps) - at) if i == n - 1 else per
        batch = max(1, round(batch_at_longest * longest / int(ctx)))
        # tokens per pass, turned into sequences by this window; never more than the batch
        # (there is nothing to split) and never less than one (there is nothing smaller).
        micro = min(batch, max(1, int(micro_tokens) // int(ctx))) if micro_tokens else 0
        out.append((at, at + take, int(ctx), batch, micro))
        at += take
    return out


def scheme(name=DEFAULT_SCHEME):
    """A scheme's architecture as MiraLLM constructor arguments, read from
    config/default/config_mira.yaml alone. `block_size` is not among them: it is the LONGEST
    pretraining `context_window`, context_window(name), and helpers.common.model_cfg_from_args
    sets it. A scheme with moe_total_experts above 1 stops here: MiraLLM has no
    mixture-of-experts FFN."""
    entry = _scheme_entry(name)
    if entry.get("moe_total_experts", 1) > 1:
        raise SystemExit(f"[config] scheme {canonical_scheme(name)!r} sets moe_total_experts "
                         f"{entry['moe_total_experts']}; MiraLLM has no mixture-of-experts FFN to "
                         f"build it with")
    return {k: v for k, v in entry.items()
            if k not in SCHEME_LATTICE_KEYS and k not in SCHEME_MOE_KEYS}


SCHEME_LATTICE_KEYS = ("lattice_order", "lattice_modulus_map", "lattice_argument_map",
                       "lattice_initializer", "lattice_init_horizon")


def scheme_lattice(name=DEFAULT_SCHEME):
    """A scheme's lattice settings under MODEL's keys: lattice_order (lattice_K), lattice_modulus_map, lattice_argument_map,
    lattice_initializer and lattice_init_horizon, each in its canonical form."""
    s = _scheme_entry(name)
    return {"lattice_order": s["lattice_order"],
            "lattice_modulus_map": s["lattice_modulus_map"],
            "lattice_argument_map": s["lattice_argument_map"],
            "lattice_initializer": canonical_lattice_initializer(s["lattice_initializer"]),
            "lattice_init_horizon": canonical_lattice_init_horizon(s["lattice_init_horizon"])}


# INFERENCE-ONLY SETTINGS, AND THEY ARE DELIBERATELY NOT IN MODEL. helpers.common
# model_cfg_from_args starts from `dict(MODEL)`, so every key there is written into a
# checkpoint's model_cfg and compared exactly on resume -- adding a knob to MODEL would make
# every checkpoint already on disk mismatch. These describe how a trained model is RUN, not what
# is built, so they live in their own section and never reach model_cfg.
#
# prefix_cache        1 enables the prefix cache in inference paths that ask for it -- evaluation
#                     and chat. Training never passes a cache and is unaffected by this value.
# prefix_cache_size   the ceiling on what the cache may hold, in GiB. The MIRA state is constant
#                     in sequence length -- K * d_e * d_m floats per layer per sequence -- so this
#                     bounds the batch a cached run may hold rather than its context.
INFERENCE = dict(
    prefix_cache=1,
    prefix_cache_size=2,
    # THE CONTEXT LENGTH REPORTED TO HUGGINGFACE, read by whichever stage builds an HF model and
    # PASSED to model/hf_model.py, which reads no configuration of its own. A ceiling rather than
    # a property of the model: MIRA has no positional encoding and no attention, so nothing in it
    # refers to an absolute position. The field cannot be omitted from an HF config, because a
    # consumer that finds none substitutes its own default -- 2048 in lm-evaluation-harness -- so
    # saying nothing imposes a SMALLER limit than saying a large number.
    max_context=262144,
)


MODEL = dict(
    **scheme(DEFAULT_SCHEME),   # n_layer, n_head, n_embd, ffn_hidden (ffn_width)
    # MIRA HAS NO FFN FACTOR, NO DROPOUT AND NO SSM CHUNK SETTING. The FFN width is the scheme's
    # ffn_width (4 x d_model when absent), dropout is MiraLLM's own 0.0, and the scan chunk is
    # MiraLLM's own "optimal".
    # THE ATTENTION OUTPUT GATE IS NOT A SETTING, and `gated_attn` is deliberately absent here.
    #
    #     "drop MODEL_GATED_ATTN is a arch decision. drop it. we always with gated"
    #                                                                          -- 2026-08-16
    # y = W_o(head_out * sigmoid(W_g x)) is part of the architecture, not a knob, so the value
    # lives in ONE place -- MiraLLM's own `gated_attn=True` default -- and every model this
    # dictionary builds carries the gate. A key here would be a second place for it to be
    # decided from, reachable through `--set MODEL.gated_attn=0`, and a run that took that path
    # would share its name and its checkpoint layout with every reported number while not being
    # the same model. THE ONE UNGATED MODEL IN THIS REPOSITORY is the MHA benchmark control, and
    # baselines/build_lm builds it by passing gated_attn=False to the constructor in code.
    # THE ARCHITECTURE, AND WITH IT HOW POSITION ENTERS.
    #
    # "mira-newton" IS THE PROPOSED ARCHITECTURE AND THE DEFAULT (model/mira_newton.py); it is
    # the default for EVERY stage. "mira-sdpa" is the quadratic form it approximates, kept
    # as the reference the approximation is measured against (model/mira.py). The queries
    # and the keys are themselves multi-timescale IIR filter states -- H_Q = MIIR_Q(X),
    # H_K = MIIR_K(X), V = X W_V -- so the attention score compares two summaries of sequence
    # history rather than two tokens. Every channel carries its own input-selective pole and the
    # starting horizons are spread continuously across the width, so the states entering the
    # score span four orders of magnitude of history from the first step. Nothing
    # position-dependent is written to the residual stream; order enters only through the score
    # matrix. The block is MIRA -> FFN.
    #
    # "mira-newton" IS THE SECOND MIRA (model/mira_newton.py), and the difference is the MIXER,
    # not the filters: positive query/key features update a normalized recurrent key--value
    # memory with learned per-channel mixer-b softmax coefficients. No T-by-T score matrix is ever formed, so work
    # stays linear in sequence length. The compatibility name remains while experiments compare
    # this replacement directly against the exact MIRA-SDPA control.
    #
    # MIRA-HIGH IS COMMENTED OUT, NOT REMOVED. arch="mira-high" is refused by MiraLLM and is not
    # a benchmark family, so nothing in this file may name it as a choice. model/mira_high.py and
    # kernel/cuda/mira-high/ are untouched on disk. Restore by undoing this block and its
    # companions (grep: MIRA-HIGH IS COMMENTED OUT).
    #     # "mira-high" IS THE HIGHER-ORDER MIRA (model/mira_high.py), and it is not a mixer variant:
    #     # it removes the mixer. There are no queries, no keys, no values and no metric. Each channel
    #     # carries ONE recurrence of order iir_n whose feedback coefficients are produced PER TOKEN
    #     # from the input through a low-rank bottleneck and passed through a softmax over the lag
    #     # axis, so the state is a convex combination of its own iir_n most recent values and the
    #     # convolved local write -- selective at higher order, and unconditionally stable for that
    #     # reason. Because it has no Q/K/V, `miir_silu`, `qk_rmsnorm`, `qk_coupling` and `mixer_k`
    #     # are recorded but do not reach it. IT REQUIRES iir_n IN [2, 8] AND IS REPORTED AT 8, which
    #     # the default of 4 below no longer supplies: model/mira_high.py refuses only order 1 -- there
    #     # the softmax over a single lag is the constant 1 and the architecture collapses to the
    #     # ordinary first-order recurrence -- so 4 is now ACCEPTED rather than refused, and a run at
    #     # arch="mira-high" that wants order 8 must set MIRA_IIR_N (MODEL.iir_n) explicitly. The
    #     # benchmark does not depend on that: baselines.high_order pins its own order (see
    #     # baselines/__init__.py).
    #
    # THE OTHER TWO ARE ABLATION CONTROLS, kept so the contribution of the design can be
    # measured rather than asserted. "ssm" makes every block SSM -> Attention -> FFN: a
    # recurrence in front of attention, also encoding-free, which isolates what moving the
    # recurrence INTO the queries and keys is worth. "rope" removes the recurrence entirely and
    # applies rotary positions inside attention, giving the conventional transformer, which
    # isolates what the recurrence is worth at all.
    #
    # The model always has exactly one source of position, and the choice is recorded in the
    # checkpoint and in every filename the run writes, so an ablation never overwrites the run
    # it is being compared with.
    #
    # THERE IS NO BANK AXIS AND NO `n_timescales`. One IIR-b coefficient simplex per CHANNEL,
    # initialized over horizons spread continuously
    # from a horizon of ~2 tokens to ~10,000 by `mira.spread_timescales`; the discrete R banks
    # this key used to configure were removed when the spread became continuous, and a
    # configuration that still sets it is setting nothing.
    # MIRA-HIGH IS COMMENTED OUT, NOT REMOVED. The enumeration below is the set of names
    # MiraLLM accepts, and "mira-high" is not one of them any more. Restore by undoing this
    # block and its companions (grep: MIRA-HIGH IS COMMENTED OUT).
    #     arch="mira-newton",          # "mira-newton" (default) | "mira-sdpa" | "mira-high" | "ssm"
    #                                 # | "rope"
    lattice_modulus_map=scheme_lattice(DEFAULT_SCHEME)["lattice_modulus_map"],
    lattice_argument_map=scheme_lattice(DEFAULT_SCHEME)["lattice_argument_map"],
    arch=(scheme_lattice(DEFAULT_SCHEME)["lattice_modulus_map"] + "_x_"
          + scheme_lattice(DEFAULT_SCHEME)["lattice_argument_map"]),  # a LATTICE_ARCHES name (this is the default) | "rope"
    fir_m=4,                    # Conv1D_a: causal FIR tap count
    iir_n=2,                    # MIIR denominator: recurrent-state lag count. TWO by
    mixer_k=2,
    # SHARED MIRA SETTINGS. The MIRA-S/M/L schemes above supply their width, expansion,
    # coefficient bottleneck, and memory dimension; these values supply the common order and
    # activation used by every matrix-state implementation.
    lattice_order=scheme_lattice(DEFAULT_SCHEME)["lattice_order"],
    lattice_initializer=scheme_lattice(DEFAULT_SCHEME)["lattice_initializer"],
    lattice_init_horizon=scheme_lattice(DEFAULT_SCHEME)["lattice_init_horizon"],
    lattice_iir_order=scheme_lattice(DEFAULT_SCHEME)["lattice_order"],
    lattice_output_gate=True,
    # BOTH OFF. Disabling them is not an ablation any more, it is the architecture: with the
    # value path linear and the states unnormalised, h_t = a_t*h_{t-1} + (1-a_t)*v_t is AFFINE
    # in the state, which is what admits an associative scan and takes the recurrence from O(T)
    # sequential depth to O(log T). RMSNorm was also the only cross-channel term, and so the
    # only reason for a block-wide barrier per token. M_QK STAYS -- it is the coupling, not a
    # normalisation, and nothing about it blocks the scan.
    miir_silu=False,            # legacy field; active MIRA fixes local/readout SiLU on
    qk_rmsnorm=False,           # legacy field; active MIRA fixes Q/K RMSNorm off
    qk_coupling=True,           # learned per-head M_QK; False compares states directly
    d_conv=4,                   # standalone SSM ablation's local convolution
)

# --------------------------------------------------------------------------- #
# shared trainer knobs (defaults of common.parse_args; CLI flags override)
# --------------------------------------------------------------------------- #
TRAIN = dict(
    dataset="hh",               # "hh" = the downloaded rlhf_hh tree, or a PATH to your
                                # own folder of records (the layout is detected)
    gpu="auto",                 # auto | cuda | mps | cpu
    seed=0,
    batch=32,                   # examples (or preference pairs) per step. ~28.9 GiB at
                                # mira-s / context 512: sized for a ~44 GB card budgeted
                                # to ~35 GB, because fragmentation, allocator rounding and
                                # kernel workspaces all come out of the nameplate figure.
                                # 56 was an estimate and ran out of memory in the loss.
                                # `python -m tools.vram --sweep` measures it for real.
    micro_batch=0,              # gradient-accumulation micro-batch; 0 = OFF, the
                                # default. SCHEME_MICRO_TOKENS says what each scheme
                                # needs at its longest window, in TOKENS per pass, and
                                # is read by model.lm, not by parse_args
    # LAYERS ACROSS EVERY VISIBLE GPU, OFF BY DEFAULT. It is a MEMORY measure and a SPEED
    # COST: the devices take turns, so N cards give the memory of N and the speed of about one,
    # minus a host-to-host copy of the hidden state at every shard boundary.
    #
    # IT WAS ON BY DEFAULT AND THAT WAS WRONG FOR THESE MODELS. mira-s peaks at 1.9 GiB of a
    # 40 GiB card -- 5% -- so sharding it four ways bought nothing and cost three device
    # transfers per pass, sixteen passes per step. Measured on the cluster: 17.94 s/step at
    # context 1,024, against an attention term of 1.07 GFLOP. The cards were idle.
    #
    # Turn it on -- TENSOR_PARALLEL=1, or --tensor_parallel -- when a model or a window does
    # NOT FIT ONE CARD AT ALL, which for these schemes means -l at its longest windows and
    # nothing else. Batch, micro-batch and checkpointing remain explicit user configuration.
    distributed_parallelism="ddp",
    loss_chunk=0,            # TOKENS per slice of the vocabulary projection. 3 x 8192 x
                                # 50,259 x 4 = 4.6 GiB, which is worth spending: a slice small
                                # enough to be free is also small enough that the step becomes
                                # dozens of tiny matmuls, each recomputed in the backward pass
    # THE ADAMW RECIPE, SHARED BY EVERY TRAINER. beta2 is 0.95 rather than torch's 0.999 and
    # the decoupled weight decay is 0.1, which is what GPT-2 and Qwen train with; the second
    # moment then forgets over roughly 20 steps instead of 1,000, which is what keeps the
    # update scale honest when the gradient distribution moves.
    #
    # THE LEARNING RATE IS NOT PART OF THIS AND IS NOT DERIVED FROM IT.
    #
    # grad_clip IS THE GLOBAL GRADIENT NORM EVERY TRAINER CLIPS TO BEFORE opt.step(),
    # carried here rather than as a literal in each trainer so a yaml can set it. It is
    # the norm over ALL of the model's parameters taken together, after the gradients
    # have been all-reduced; 0 or a negative value disables clipping, and the step then
    # takes the gradient as it is.
    adam_beta1=0.9,
    adam_beta2=0.95,
    weight_decay=0.1,
    grad_clip=1.0,
    adam_eps=1e-8,
    beta=0.1,                   # implicit-reward beta, shared by DPO / evaluation
    val_frac=0.05,              # validation fraction of the preference file
    limit=0,                    # cap #preference pairs loaded (0 = all)
    resume=True,                # resume from this stage's checkpoint, if one is there
    # learning-rate schedule, shared by EVERY trainer: cosine decay from the stage's lr
    # down to last_lr over that stage's own step budget.
    #     lr(t) = lr_min + 0.5 (lr - lr_min) (1 + cos(pi t / T)),  lr_min = last_lr
    # The schedule is a function of the ABSOLUTE step, so a resumed run continues on the
    # same curve rather than restarting it.
    lr_schedule="cosine",       # "cosine" | "constant"
    last_lr=2e-5,               # the final lr the cosine decays to, an absolute value
    # THE RAMP IN FRONT OF THE COSINE, in steps, and the decay spans what is left (T - W).
    # 0 turns it off and restores the curve exactly. See helpers.utils.CosineLR for why the
    # opening steps need a smaller rate than the rest.
    warmup_steps=1000,          # linear warmup from lr/W to lr over the first W steps
    checkpoint_every_steps=200,
    grad_checkpoint=0,
    plot_every_steps=200,
    profile_every_steps=200,
    # GENERATION EXAMPLES: 20 of them, every this many steps, for every stage that generates.
    # SEPARATE FROM THE FIGURE CADENCE, because the two are watched differently -- a figure is
    # glanced at, twenty generations are READ, and wanting them at a different rhythm from the
    # redraw is the normal case rather than the exception. 0 = off.
    print_samples_every_steps=200,
    # State space diagnostics: memory horizon, selectivity, residual write ratio and the rest
    # held-out probe during DPO (0 = use plot_every_steps)
    eval_every=0, eval_pairs=32,
    # end-of-stage evaluation
    n_hist=256,
)

# --------------------------------------------------------------------------- #
# per-stage config
# --------------------------------------------------------------------------- #
# EXTRA SPECIAL TOKENS, on top of the predefined <|endoftext|>, <|pad|> and <|unk|>.
#
# A registered token is an ATOM: the tokenizer emits ONE id for it and never breaks it into
# subwords, so a later model can attach a meaning to it -- a control marker, a role tag, a
# modality boundary.
#
#     EXTRA_SPECIAL_TOKENS = ["<vp>", "<|im_start|>", "<|im_end|>"]
#     tok.encode("a <vp> b")     ->  [.., <vp>, ..]   rather than the subwords of "<vp>"
#
# They are appended AFTER the merges and after the predefined specials, so adding one never
# moves an existing id: a checkpoint trained before the change only needs its embedding row
# count raised, not retraining.
#
# Note the consequence: text containing those exact characters now produces the token. Corpus
# text you do not control should be encoded with tok.encode_ordinary, which treats every
# special as plain text.
# THE CHAT TURN MARKERS ARE NOT HERE, AND THAT IS DELIBERATE. This list is registered on EVERY
# tokenizer this project loads, which would grow the vocabulary for pretraining, the benchmark
# and the ablation as well -- none of which has a turn to mark.
#
#     "so instruct-sft bpe has special tokens.
#      <|startofmessage|>,<|endofmessage|>,<startofthink>,<endofthink>"        -- 2026-09-06
#
# Stage 7 registers its four on the tokenizer it loads and saves that tokenizer inside its
# checkpoint, so every reader of that checkpoint -- mirachat, stages 8 to 11 -- gets the ids
# with the weights. See helpers.utils.CHAT_SPECIALS.
EXTRA_SPECIAL_TOKENS = []

# THE TOKEN CACHE. Shards exist because a corpus can be 10 TB: one file that size cannot be
# resumed after an interruption, cannot be copied incrementally, and loses everything to a
# single bad byte. ~100 MB is the usual compromise -- small enough that a shard is cheap to
# re-fetch and to verify, large enough that per-file overhead does not matter. The manifest is
# rewritten after every completed shard, so an interrupted build restarts from the last one
# rather than from nothing.
TOKENS = dict(
    shard_mb=100,
    # HOW A TOKEN STREAM IS CHECKED BEFORE IT IS MAPPED. mmap does not read a file, so a
    # corrupt shard is not an error -- it is wrong tokens, surfacing thousands of steps later
    # as a loss curve that will not come down.
    #   "md5"   read every shard and compare the digest recorded when it was sealed. Seconds
    #           per corpus against a run of hours, and the only setting that catches a shard
    #           whose MIDDLE changed rather than whose tail was lost.
    #   "size"  compare lengths only: one stat per shard, catches the truncated tail a killed
    #           writer leaves, misses everything else.
    #   "off"   trust the manifest.
    verify="md5",
)

BPE = dict(
    num_merges=50000,
    # A word type must occur at least this often to be counted when merges are learned.
    # In a corpus of this size half the distinct words appear EXACTLY ONCE -- proper nouns,
    # typos, one-off quotations -- and they are under 2% of the running text. The merge loop's
    # cost scales with the number of distinct words a pair occurs in, so the tail is most of
    # the work and almost none of the evidence. Nothing is removed from the vocabulary: a
    # byte-level tokenizer can always encode a word it never counted, using the merges the
    # frequent words justified. Set to 1 to count every word.
    min_freq=2,
)                                    # vocab = 256 bytes + num_merges + specials
                                     # = 256 + 50000 + 3 = 50259
                                     # (<|endoftext|>, <|pad|>, <|unk|>, plus any
                                     #  EXTRA_SPECIAL_TOKENS)

# Step budgets are EPOCH BUDGETS in disguise: steps = ceil(epochs * examples / batch) at the
# default batch of 16. Recompute them if the corpus or the batch size changes.
#   pretrain  3 epochs over 345,811 documents  -> 64,840 steps
#   sft       2 epochs over  18,729 documents  ->  2,342 steps
#   reward    1 epoch  over  40,000 pairs      ->  2,500 steps
#   rlhf      1 epoch  over  52,002 prompts    ->  3,251 steps
#   dpo       1 epoch  over   9,437 pairs      ->    590 steps
#   distill   1 epoch  over  52,002 prompts    ->  3,251 steps
# THE STEP BUDGET IS TWO EPOCHS OVER THE ~2B-TOKEN CORPUS, FOR THE DEFAULT SCHEME.
#
#     tokens/step = batch x context = 16 x 512 = 8,192
#     2 epochs    = 2 x 2e9 = 4e9 tokens
#     steps       = 4e9 / 8,192 = 488,281
#
# That arithmetic is only true if a step really carries batch x context tokens, which is why
# max_words moved from 200 to 344: at ~1.49 tokens per word the old packing gave ~298-token
# documents in a 512-token window, so a step actually carried 4,768 tokens and "two epochs"
# would have needed 838,926 steps. Packing at 344 words fills the window, and the number below
# means what it says. Recompute both if the batch, the context or the corpus changes.
PRETRAIN = dict(
    # THE FALLBACK ONLY. helpers.common.parse_args replaces this with SCHEME_STEPS[scheme]
    # unless --pretrain_steps was given, because a step carries batch x (ctx - 1) tokens and
    # that differs by a factor of two between -m and -l: one number cannot be 1.5 epochs for
    # both. See SCHEME_STEPS for the arithmetic and the corpus it was taken over.
    steps=168317, lr=2e-4,       # the fallback; SCHEME_STEPS[scheme] replaces it
    # WHICH SCHEME IS PRETRAINED, and at what context window. Both are exposed on the command
    # line (--mira_config / --context_window) and in config.sh, so a size can be changed for
    # one run without editing Python. context_window=0 takes the scheme's own value from
    # SCHEMES above (512 for -s, 1024 for -m and -l); any other value overrides it, which is
    # what an ablation on context length needs.
    model_scheme=DEFAULT_SCHEME,
    context_window=0,
    max_words=344,              # ~344 words ~ 512 tokens: one document fills the context
    # The FineWeb-Edu subset ships as parquet, one document per row. This names the column
    # holding the text; the loader lists the file's actual columns if it is wrong, rather than
    # failing with a KeyError several thousand files in.
    text_column="text",
    # Subdirectories EXCLUDED from the corpus scan, matched against any path component so
    # nested directories with these names are skipped too.
    #
    # "test" and "valid" are here because the scan is RECURSIVE and the WikiText corpus ships
    # split into train/, valid/ and test/. Without this a pretraining run over the corpus root
    # would train on the held-out splits -- silently, since more data simply looks like more
    # data. The names are harmless for a corpus that has no such directories, which is why they
    # are excluded by default rather than set per corpus.
    # Held-out splits, excluded from pretraining. Each entry skips a DIRECTORY of that
    # name AND any file whose leading name token matches it, so the same corpus is
    # handled whether its splits are valid/ and test/ subdirectories or flat parquet
    # shards named validation-00000-of-00001.parquet. "validation" is listed beside
    # "valid" because the two layouts spell it differently and only one of them is the
    # Hugging Face convention.
    exclude_dirs=["test", "valid", "validation"],
)

# SFT is INSTRUCTION fine-tuning: the same length-normalized LM objective as pretraining, with
# the loss masked to the RESPONSE, on UltraChat (SFT_DIR).
#
# THE BATCH IS THIS STAGE'S OWN, 32, and not SCHEME_BATCH's 3. SCHEME_BATCH sizes a PRETRAINING
# step, where every sequence fills the whole 8,192-token window; an instruction demonstration is
# a few hundred tokens, so 3 would leave the card almost empty and make one epoch 313,115 steps.
# config.sh passes --batch for this stage, which is also what stops SCHEME_BATCH being applied
# (helpers.common._was_given). A batch is padded to its LONGEST record and this corpus is
# ragged, so the micro-batch is ONE SEQUENCE PER PASS: a fixed number of sequences does not
# bound the tokens per pass, and one sequence does -- at most the context window, 8,192, inside
# the 19,456 SCHEME_MICRO_TOKENS measured for this scheme.
#
# THE STEP BUDGET IS ONE EPOCH. UltraChat holds about 1.47 M conversations, which flatten to
# more demonstrations than that -- one per assistant turn, so a multi-turn conversation
# contributes several -- and at batch 32 that is on the order of 46,000 steps.
#
# The true figure is measured at load time, after deduplication, and PRINTED beside this one
# ("one epoch = N steps ... this run does M"), because the two are allowed to differ and a
# budget that turns out to be a third of an epoch looks exactly like one that is a whole epoch
# from the loss curve alone. Raise SFT.limit's twin SFT_LIMIT in config.sh to shorten the run
# instead of the budget: half the corpus for half the steps is a different experiment from the
# whole corpus seen half as often.
#
# ONE epoch and not two, at 1e-6. Instruction tuning is teaching a format the pretrained model
# already has the language for, and a second pass over a corpus this size buys memorisation of
# individual answers rather than more instruction following -- this is a 153M model, mira-s,
# the default, with little capacity to spend on remembering.
SFT = dict(
    steps=10000, lr=1e-6,       # calibrated at global batch 32 on four GPUs
    batch=8, micro_batch=1,     # per GPU; global batch is 32 on four GPUs
    # WHERE THIS STAGE WRITES, above the leaf that names the run:
    # <ckpt_dir>/<model scheme>_<sft dataset>/. instruct_sft/run.py composes the leaf and
    # helpers.STAGE_DIRS maps instruct_sft to nothing, so no further level is appended.
    ckpt_dir="checkpoints/instruct_sft",
    # THE CHECKPOINT THIS STAGE INITIALISES FROM. Empty means the stage resolves it: the
    # pretrain checkpoint of stage 6 first, then a MIRA checkpoint under checkpoints/benchmark/.
    checkpoint="",
    # WHICH rlhf_hh SUBSETS the demonstrations come from, IF this stage is pointed back at an
    # hh tree (--sft_dir). The TRAIN splits only: the test splits are what stages 8 and 11 hold
    # themselves out on, and fine-tuning on them would make every later evaluation a measurement
    # of memorisation instead of generalisation. Ignored by the chat layout, which has no
    # subset directories -- its subsets are a COLUMN, and it is read whole.
    subsets=["helpful_train", "harmless_train"],
    limit=0,                    # cap the demonstrations loaded (0 = all)
)

# reward model: MiraLLM trunk + scalar head; binary classification with sigmoid + BCE,
# chosen -> label 1, rejected -> label 0.
#
# WHICH PREFERENCES. The reward model must speak the domain the policy will be rolled out in,
# or its scores are meaningless there -- which is why stages 7, 8, 9 and 11 all read rlhf_hh:
# the model is fine-tuned on those demonstrations, scored by a reward model fitted to those
# preferences, and rolled out on those prompts. The three agreeing is the assumption PPO's
# objective is written under.
REWARD = dict(
    # WHERE STAGE 8 WRITES, and stage 9 reads its scorer from:
    # <ckpt_dir>/<model scheme>_<preference dataset>/, composed by helpers.utils.reward_dir.
    ckpt_dir="checkpoints/instruct_reward",
    steps=2500, lr=1e-5,        # 1 epoch over the hh pairs
    # Split directories to read when the data is an hh-layout tree. They are used WHERE THEY
    # EXIST and discovered otherwise, so a tree that names its splits differently still loads;
    # for a plain local folder they are ignored and --val_frac cuts the validation set.
    # The four preference subsets of Anthropic/hh-rlhf. Each holds its own train and test
    # file, so the same names serve both halves and the split is chosen by the reader.
    hh_subsets=["harmless-base", "helpful-base", "helpful-online",
                "helpful-rejection-sampled"],
    hh_val_subsets=["harmless-base", "helpful-base", "helpful-online",
                    "helpful-rejection-sampled"],
    hh_limit=20000,             # pairs per subset, 0 = all (~86k train pairs in total)
)

# RLHF = PPO against the stage-7 reward model, starting FROM THE SFT MODEL, with a per-token
# KL penalty to the frozen SFT reference (the standard InstructGPT recipe).
RLHF = dict(
    steps=3251, lr=1e-6,        # 1 epoch over the instruction prompts; KL-anchored
    max_new_tokens=48,          # response length of each rollout
    gen_temperature=1.0,        # rollouts are SAMPLED (a greedy rollout is not a draw from pi)
    kl_coef=0.05,               # per-token KL penalty to the SFT reference
    ppo_epochs=4,               # optimisation epochs per batch of rollouts
    clip_eps=0.2,               # PPO ratio clip
    gamma=1.0, lam=0.95,        # discount / GAE lambda
    vf_coef=0.5,                # value-loss weight
    ent_coef=0.0,               # entropy bonus weight
    whiten_adv=True,            # normalise advantages per batch
    # ROLLOUT PROMPTS. PPO needs prompts to generate from, not preference pairs: they come
    # from the instruction batches in <instruct_dir>/alpaca_gpt4/
    # (alpaca_gpt4_<batch>.json, each record carrying `prompt`). Set prompt_dir=None to fall
    # back to the preference file's prompts.
    prompt_dir=HH_DIR,          # the prompts of the same rlhf_hh records stages 7/7/10 use
    prompt_limit=0,             # cap the prompt bank (0 = all ~52k)
    prompts_per_file=1000,      # what alpaca_to_json.py writes
)

DPO = dict(steps=20000, lr=1e-6)     # starts from SFT, ref = frozen SFT


# LANGUAGE ADAPTATION (stage 12): extend the vocabulary for a new language, then continue
# pretraining the English model on that language's corpus.
#
# TWO STEPS, AND THE FIRST IS THE VOCABULARY. A byte-level BPE never fails on an unseen
# language -- it falls back to raw bytes -- so training Irish on the English vocabulary would
# run, and would quietly spend four or five tokens on words an adapted vocabulary spells in
# one. That tax falls on the context window and on the step budget together. Extending first
# is what makes the same budget cover several times as much Irish.
#
# THE EXTENSION KEEPS EVERY EXISTING ID. The English merges stay, in order, and the new ones
# are appended, so the pretrained embedding still means what it learned and the model needs
# only more rows. A FRESH vocabulary would tokenise Irish better and throw the pretrained model
# away with it. See tokenizer_sft/run.py, including the one thing extension is not free of: the id
# layout puts the specials LAST, so appending merges moves them, and a checkpoint has to be
# rearranged for it.
LANG_SFT = dict(
    # WHICH CORPUS. A name under data/download/, or a path to one anywhere.
    dataset="llm-pretrain-gaelic-uccix-irish-textual-corpus",
    # THE LABEL IN EVERY FILENAME -- `gaelic`, not the whole dataset name, which would make
    # checkpoint_mira_llm-s_ssm_lang-sft-llm-pretrain-gaelic-uccix-irish-textual-corpus.pt.
    short="gaelic",
    # HOW MANY MERGES TO ADD. Enough to cover the new language's morphology without doubling
    # the embedding: 8,000 on top of 50,000 grows the vocabulary by 16% and the parameter count
    # by ~4M at d_model 512.
    extra_merges=8000,
    steps=20000, lr=1e-5,
)

# CHAIN OF THOUGHT: SUPERVISED FINE-TUNING, THEN GROUP RELATIVE POLICY OPTIMISATION (GRPO).
#
# TWO TRAINING RUNS, ONE STAGE, and they read the same problems:
#
#   1. SFT     the dataset's own reference trace, rewritten into this pipeline's
#              <think>...</think> <answer>...</answer> form, trained by maximum likelihood.
#              -> checkpoints/cot/<run>_cot-sft/checkpoint.pt
#   2. GRPO    reinforcement learning from that checkpoint against a VERIFIED answer.
#              -> checkpoints/cot/<run>_cot-grpo/checkpoint.pt
#
# WHY THE SFT COMES FIRST. GRPO can only amplify behaviour the policy ALREADY SAMPLES. A
# reward for a format no completion in the group produces is a reward with no gradient: every
# completion scores the same, every advantage is zero, and the run trains on nothing while
# every curve looks healthy. A base model has never seen <think> tags, so the format reward
# has nothing to grip until something demonstrates the container. The SFT demonstrates it, and
# only it -- the reasoning CONTENT is still whatever the optimisation makes of it.
#
# THIS IS R1, NOT R1-ZERO, and the difference is worth naming. DeepSeek-R1-Zero is RL directly
# on the base model with no supervised reasoning anywhere, which is a claim about what RL alone
# can produce; R1 itself uses a small supervised cold start first, for exactly the reason above.
# Set sft=False (--no-cot_sft, COT_SFT=0) for the zero setting.
#
# The reward is not a learned model but an ARITHMETIC CHECK -- the final answer is extracted and
# verified against the gold -- plus terms for obeying the format.
#
# GRPO replaces PPO's value head with a GROUP BASELINE: the policy samples `group_size`
# completions for the same prompt and each completion's advantage is its reward standardised
# WITHIN that group. No critic is trained, which is what makes the recipe cheap enough to run
# here. The KL to the frozen reference is added to the loss directly (not shaped into the
# per-token reward as PPO does), using the unbiased non-negative k3 estimator.
#
# WHAT TO WATCH. The reported "aha moment" is a length phenomenon: mean think length grows as
# the policy discovers that more deliberation earns more reward, and self-correcting phrases
# ("wait", "let me re-check") start to appear. Both are tracked every step and drawn in
# outputs/plots/cot/cot_dynamics.pdf, alongside accuracy against the verifier.
COT = dict(
    # 1e-6 (10^-6) for BOTH sub-stages. This is the rate the GRPO literature uses at this
    # scale -- DeepSeek's own GRPO runs sit at 1e-6 -- and it is not interchangeable with 1e-5:
    # a policy-gradient step is taken against an advantage that is already standardised within
    # its group, so the update is large before the rate is applied, and ten times too much
    # learning rate does not train ten times faster, it collapses the policy.
    steps=10000, lr=1e-6,       # GRPO: the second of the stage's two runs
    # THE REASONING SFT, run first. `sft_steps` and `sft_lr` are its own budget: the GRPO
    # numbers above do not apply to it, because a likelihood objective over demonstrations and
    # a policy-gradient objective over rollouts are not the same optimisation and never want
    # the same schedule.
    sft=True,                   # False (--no-cot_sft) = the R1-Zero setting, GRPO alone
    # 7,000 steps at batch 3 is 21,000 demonstrations seen, which was a little over one epoch of
    # the countdown bank's 20,071 problems. The count on the current default task, gsm8k, is
    # measured at load time and printed. The format is what this sub-stage teaches, and a format
    # is learned in roughly one pass; more epochs mostly memorise the particular traces.
    sft_steps=7000,
    sft_lr=1e-6,
    # The demonstrations are FILTERED before they are trained on: a reference trace whose own
    # final answer the verifier rejects would teach the model to be confidently wrong in the
    # right format, and a stage that checks answers arithmetically has no excuse for training
    # on unchecked ones.
    sft_verify=True,
    init_stage="pretrain",      # WHICH CHECKPOINT THE REASONING SFT STARTS FROM -- and, with
                                # sft=False, which one GRPO itself starts from.
                                #   "pretrain" -- the base model (R1's cold start)
                                #   "sft"      -- after the general instruction tuning
                                #   "rlhf" / "dpo" -- continue an aligned policy
                                # Overridable per run: COT_INIT=sft ./old_scripts/stage10_cot_aha_moment.sh
    # PROMPTS PER STEP, this stage's own. It cannot borrow SCHEME_BATCH: that is sized for a
    # pretraining step of `batch` sequences, and a GRPO step is `batch x group_size` sequences
    # of a length the policy chooses -- up to the whole context window. 2 x 4 = 8 rollouts is
    # what fits a 44 GiB card at an 8,192 window; raise it on a larger one.
    batch=2,
    group_size=4,               # completions sampled per prompt; the group IS the baseline.
                                # MUST be >= 2 -- with 1 every advantage is exactly zero. 4 is
                                # the smallest group with usable spread; 8 is less noisy and
                                # costs twice the rollout memory.
    # RESPONSE LENGTH: 0 = WHATEVER THE CONTEXT WINDOW HAS LEFT once the prompt is in it,
    # computed per batch from the longest prompt actually in it. A fixed number cannot be right
    # across the schemes -- their windows run from 1,024 to 32,768 -- and this stage exists to
    # see whether a policy learns to think for longer, which a cap decides in advance. Set a
    # number to pin it, for a controlled comparison or to bound the cost of a rollout.
    max_new_tokens=0,
    gen_temperature=1.0,        # rollouts must be SAMPLED for the group to have spread
    kl_coef=0.001,              # KL to the frozen reference, added to the loss (k3 estimator)
    clip_eps=0.2,               # GRPO ratio clip
    grpo_epochs=2,              # optimisation epochs per batch of rollouts
    ent_coef=0.0,               # entropy bonus (0 = off)
    # REWARD. Correctness is verified arithmetically; the format term pays only for producing
    # the think/answer structure at all, and is kept small so that formatting cannot be traded
    # against being right.
    # THE REWARD IS A STAIRCASE, not a gate. Each tag is paid for INDEPENDENTLY, so a policy
    # that has learned to close <think> but not yet <answer> is rewarded for the half it has --
    # a single both-or-nothing format term leaves that state worth exactly as much as emitting
    # neither, and nothing pulls the model the rest of the way. For a base model that has never
    # seen this format, that gradient is the difference between learning it and not.
    #
    # THE WEIGHTS DO NOT INVITE TAG-FARMING, because GRPO's advantage is relative to the GROUP.
    # A term every completion earns cancels out of the advantage entirely: the format terms
    # teach while they still vary and go quiet once the whole group has the format, after which
    # only correctness separates one completion from another.
    correct_reward=2.0,         # the answer is right   -- worth both tags together
    think_format_reward=1.0,    # a closed <think>...</think>
    answer_format_reward=1.0,   # a closed <answer>...</answer>
    # A THIRD TERM, between format and correctness. Format pays for the tags; correctness pays
    # for the answer; a policy can satisfy the first with tags around nothing. This pays only
    # when the reasoning is long enough to be reasoning AND mentions the numbers the problem
    # gave it. It is a floor against the emptiest reward hacking, NOT a judge of whether the
    # reasoning is right -- that cannot be checked by rule, which is why correctness exists.
    think_reward=0.3,
    think_min_words=8,          # below this, a "think" block is not thinking
    # THE SYSTEM PROMPT. R1-Zero's only intervention on the prompt side: name the container for
    # the reasoning, never demonstrate its content. Empty = verifier.SYSTEM_PROMPT.
    #
    # IT MUST ASK FOR WHAT THE REWARD PAYS FOR. The countdown questions carry their own
    # instruction to answer inside \boxed{}; the extractor now reads \boxed{} first, so the two
    # agree. When they did not, a completion that obeyed the prompt exactly scored zero.
    system_prompt="",
    # THE QUESTION IS REWRITTEN INTO OUR CONVENTION before it is asked. A dataset carries its
    # own instruction about where the answer goes -- the countdown questions say \boxed{}, in
    # the worked example and again in the closing line -- and a system prompt asking for
    # <answer> tags on top of that leaves the model with two conventions and only one of them
    # rewarded. normalise_question() turns every \boxed{X} into <answer> X </answer>, so the
    # question, the instruction and the reward all name the same thing. Set False to ask the
    # dataset's questions exactly as they are written.
    normalise_question=True,
    length_penalty=0.0,         # per-token penalty on the response (0 = let it grow freely)
    task="gsm8k",
    # INCREMENTAL DECODING. Rollouts are the dominant cost of this stage: without a cache,
    # generating n tokens redoes the whole prefix n times, which is O(n^2) work. With one, each
    # step extends the keys, values, convolution window and recurrence state that are already
    # there. --no-kv_cache turns it off, for a comparison or when something is suspected.
    # THERE IS NO bf16 KNOB HERE ANY MORE. This stage used to cast its live WEIGHTS to
    # bfloat16 to make a long-completion step fit; mixed precision (MIXED_PRECISION, on by
    # default) now gets the same activation saving across every stage without touching a
    # weight -- which is the whole point, since an update below bf16's ~3 significant digits
    # rounds away and fp32 masters are what prevent it. One precision policy, shared.
    kv_cache=True,
    # WHAT THE CACHE MAY HOLD, in GiB. Attention's part grows with every token generated and a
    # step decodes batch x group_size sequences at once -- 128 at the defaults -- so left
    # unbounded it is tens of gigabytes on a long window. The rollout is split into groups that
    # fit this and decoded one group at a time; sequences are independent, so the completions
    # are the same and only the peak changes.
    #
    # GiB, not bytes: this is a number a person sets against a card they can see, and 2 is
    # readable where 2147483648 is a thing to be trusted. A float is accepted, so 0.5 works.
    kv_cache_size=2.0,          # GiB
    # HOW OFTEN THE ROLLOUT SAMPLES ARE PRINTED, for the GRPO half only. Tighter than the
    # pipeline's --print_samples_every_steps (200) because a GRPO step is not comparable to an
    # LM step: it costs a whole group of rollouts, there are far fewer of them, and what this
    # stage is watching -- the format appearing, the reasoning lengthening -- shows up in the
    # TEXT rather than in the loss. 0 = follow --print_samples_every_steps.
    samples_every=50,
    # DATA. Every .json under the task's directory; a gsm8k record carries a question, an
    # answer and a reference trace, and a countdown record carries a prompt, an answer of
    # {target, numbers}, and a trace. The SFT reads the trace; GRPO does not -- it sees only the
    # question and, through the verifier, the answer.
    #
    # BOTH DIRECTORIES STAY NAMED HERE and cot.run.task_dir picks between them, so switching the
    # task still switches the directory. data_dir is countdown's because that is the key
    # task_dir falls back to; the default task reads data_dir_gsm8k.
    data_dir=COT_COUNTDOWN_DIR,      # data/download/llm-cot-countdown-game-20k
    data_dir_gsm8k=COT_GSM8K_DIR,    # used when task="gsm8k", which is the default
    question_field="prompt",         # countdown's field names; gsm8k uses question/answer
    answer_field="answer",
    # THE REFERENCE TRACE, read by the SFT only, and named PER TASK because no two datasets
    # agree. Countdown keeps a "response" holding a <think>...</think> span and a closing
    # \boxed{}; gsm8k keeps bare prose in "reasoning" with the answer in a separate field.
    # verifier.demonstration() converts either into <think>...</think> <answer>...</answer>,
    # so the only thing that has to be configured is WHERE the trace is.
    trace_field="response",
    # "reasoning" IS FOR A DATASET THAT SHIPS THE REASONING SEPARATELY, and openai/gsm8k does
    # not: its `answer` is the whole worked solution ending in "#### N". No record carries
    # "reasoning", so load_problems finds no trace, sees the "####" line in the answer text, and
    # takes the whole text as the trace with the gold read off that line. The name is kept
    # because it is right for any converted copy that does split the two.
    trace_field_gsm8k="reasoning",
    val_frac=0.02,                   # held out from the END when the files carry no split
    train_prefix="train", test_prefix="test",
    records_per_file=1000,      # what cot_to_json.py writes
    limit=0,                    # cap the problem bank (0 = all)
    eval_problems=200,          # held-out problems scored at the end of the stage
)

# distillation: SEQUENCE-LEVEL KD. The teacher is our ALIGNED MiraLLM (the RLHF-ed model by
# default; set teacher_stage="dpo" for the DPO-ed model); the student is gpt2-small. Teacher
# and student use different tokenizers, so the transfer is through TEXT: each step the
# teacher generates responses to a batch of prompts read from the prompt bankprompt_<batch>.txt
# (one prompt per line, prompts_per_file per file) and the student is trained by CE on the
# teacher's generations.
DISTILL = dict(
    steps=3251, lr=1e-6,        # 1 epoch over the distillation prompts
    teacher_stage="rlhf",       # which aligned checkpoint teaches: "rlhf" (default) or "dpo"
    student="gpt2",             # HF model id of the student (gpt2-small, 124M)
    student_max_len=256,        # student-side truncation of prompt+response
    prompts_per_file=100,       # prompts per the prompt bankprompt_<batch>.txt
    max_new_tokens=60,          # teacher generation length per prompt
    gen_temperature=0.8,        # teacher sampling temperature
    # KL CONSTRAINT. The student is a PRETRAINED gpt2-small, so fitting it to the teacher's
    # text can wash out what it already knows. A per-token KL to a FROZEN COPY OF ITS OWN
    # INITIAL WEIGHTS holds it near that prior, exactly as the RLHF stage anchors its policy
    # to the frozen SFT model:
    #     L = CE(student, teacher text) + kl_coef * KL( p_student || p_student_init )
    # (full-vocabulary KL at every response position; the reference shares the student's
    # tokenizer, so this is exact -- unlike a teacher/student KL, which is undefined here.)
    kl_coef=0.05,               # 0 disables the constraint (and its extra forward pass)
)

SCALING = dict(
    models=["zs/8", "zs/6", "zs/4", "zs/2", "zs"],       # 1.1M .. 53.6M non-embedding
    budgets=[2_000_000, 8_000_000, 32_000_000, 100_000_000],
    context=512,                # FIXED across the grid; a third moving axis would be fitted
                                # as if it were one of the two under study
    batch=16,
    # Peak rate AT lr_ref_width; every other width follows lr_rule. A single rate across a 4x
    # width range is a confound rather than a control -- too small for the narrow models or too
    # large for the wide ones -- and the N-axis then carries a bend that is not capacity.
    lr=6e-4, lr_ref_width=512, lr_rule="sqrt_width",     # sqrt_width | fixed
    eval_every=0,               # 0 = ten held-out probes per point
    val_windows=64,             # fixed windows, IDENTICAL for every point
)

# --------------------------------------------------------------------------- #
# end-of-stage sample generation (qualitative check, logged after each stage)
# --------------------------------------------------------------------------- #
N_SAMPLES = 10
SAMPLING = dict(max_new=60, temperature=0.8, top_k=50, top_p=0.95)


# --------------------------------------------------------------------------- #
# what the shell falls back to
# --------------------------------------------------------------------------- #
# config.sh prints the configuration in force at the head of every run. When one of its
# variables is left blank the flag is simply not passed and the trainer uses the value below,
# so the shell has to be told what that value IS -- otherwise the summary can only print a
# dash, which is precisely the case where the reader most needs the number.
#
#     python default_config.py --shell-defaults
#
# emits `MIRALLM_DEF_<VAR>='<value>'` lines for config.sh to eval. Keys are the shell variable
# names; a variable with no Python default (PY, NO_RESUME, the assembled *_FLAGS) is absent.
def apply_overrides(pairs, log=print):
    """Apply `SECTION.key=value` overrides to the configuration dictionaries, in place.

    Forty-odd settings live inside these dictionaries -- BPE["min_freq"], RLHF["kl_coef"],
    PRETRAIN["max_words"] and so on -- and giving each its own command-line flag would mean
    forty flags, forty forwarding lines in config.sh, and a new one to add every time a
    setting appears. One override reaches all of them, and reaches settings added later
    without any further work.

    Applied by common.setup BEFORE any stage reads its configuration, and by the stages that
    do not go through setup. The stages hold references to these dicts rather than copies of
    their values, so mutating them here is what makes the override take effect.

    The value is coerced to the TYPE ALREADY THERE, so `--set BPE.min_freq=3` yields an int
    and `--set RLHF.kl_coef=0.02` a float; a bool accepts the spellings a shell produces
    (1/0, true/false, yes/no). An unknown section or key is refused rather than ignored --
    a silently-dropped override is a run that reports settings it did not use.
    """
    import ast
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"[config] --set expects SECTION.key=value, got {item!r}")
        path, _, raw = item.partition("=")
        section, _, key = path.strip().partition(".")
        d = globals().get(section.strip().upper())
        if not isinstance(d, dict):
            raise SystemExit(f"[config] --set: no configuration section {section!r}. "
                             f"Sections: BPE, PRETRAIN, SFT, REWARD, RLHF, COT, DPO, "
                             f"DISTILL, TRAIN, SCALING, MODEL")
        if d is MODEL:
            key = MODEL_SETTING_ALIASES.get(key, key)
        if d is PRETRAIN and key == "mira_config":
            key = "model_scheme"
        if key not in d:
            raise SystemExit(f"[config] --set: {section.upper()} has no key {key!r}. "
                             f"Keys: {', '.join(sorted(d))}")
        cur = d[key]
        try:
            if isinstance(cur, bool):
                val = str(raw).strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, int) and not isinstance(cur, bool):
                val = int(raw)
            elif isinstance(cur, float):
                val = float(raw)
            elif section.strip().upper() == "MODEL" and key == "lattice_init_horizon":
                val = canonical_lattice_init_horizon(raw)       # one range, or a per-layer mapping
            elif isinstance(cur, (list, tuple)):
                val = ast.literal_eval(raw) if raw.strip().startswith(("[", "("))         \
                      else [x for x in raw.split(",") if x]
            else:
                val = raw
        except (ValueError, SyntaxError) as e:
            raise SystemExit(f"[config] --set {path}: {raw!r} is not a "
                             f"{type(cur).__name__} ({e})") from e
        if val != cur:
            log(f"[config] {section.upper()}[{key!r}] {cur!r} -> {val!r}")
        d[key] = val


SHELL_DEFAULTS = {
    "GPU": TRAIN["gpu"],
    "SEED": TRAIN["seed"],
    "BATCH": TRAIN["batch"],
    "MICRO_BATCH": TRAIN["micro_batch"],
    "DISTRIBUTED_PARALLELISM": TRAIN["distributed_parallelism"],
    "DATASET": TRAIN["dataset"],
    "DATA_LIMIT": TRAIN["limit"],
    "VAL_FRAC": TRAIN["val_frac"],
    "BETA": TRAIN["beta"],
    "MIRA_CONFIG": PRETRAIN["model_scheme"],
    # a comma-separated list, which is what config.sh's PRETRAIN_DIR now carries
    "PRETRAIN_DIR": ",".join(os.path.relpath(d, ROOT)
                             for d in PRETRAIN_CORPUS[PRETRAIN["model_scheme"]])
                    or "(unset for this scheme)",
    "INSTRUCT_DIR": os.path.relpath(INSTRUCT_DIR, ROOT),
    "SFT_DIR": os.path.relpath(SFT_DIR, ROOT),
    # 0 means "the scheme's own", so report the number that ends up in force
    "CONTEXT_WINDOW": PRETRAIN["context_window"],
    "LR_SCHEDULE": TRAIN["lr_schedule"],
    "LAST_LR": TRAIN["last_lr"],
    "WARMUP_STEPS": TRAIN["warmup_steps"],
    "GRAD_CHECKPOINT": TRAIN["grad_checkpoint"],
    "PLOT_EVERY": TRAIN["plot_every_steps"],
    "PROFILE_EVERY_STEPS": TRAIN["profile_every_steps"],
    "CKPT_EVERY": TRAIN["checkpoint_every_steps"],
    "CAYLEY_PARAMETERIZATION": MODEL["arch"],
    # NO_RESUME is the shell's inverted spelling of resume: 0 means "resume from checkpoints"
    "NO_RESUME": 0 if TRAIN["resume"] else 1,
    "EVAL_EVERY": TRAIN["eval_every"],
    "EVAL_PAIRS": TRAIN["eval_pairs"],
    "N_HIST": TRAIN["n_hist"],
    "BPE_MERGES": BPE["num_merges"],
    "PRETRAIN_STEPS": PRETRAIN["steps"],
    "PEAK_LR": PRETRAIN["lr"],
    "SFT_STEPS": SFT["steps"],
    "SFT_LR": SFT["lr"],
    "REWARD_STEPS": REWARD["steps"],
    "REWARD_LR": REWARD["lr"],
    "RLHF_STEPS": RLHF["steps"],
    "RLHF_LR": RLHF["lr"],
    "COT_STEPS": COT["steps"],
    "COT_LR": COT["lr"],
    "COT_INIT": COT["init_stage"],
    "COT_GROUP": COT["group_size"],
    "DPO_STEPS": DPO["steps"],
    "DPO_LR": DPO["lr"],
    "DISTILL_STEPS": DISTILL["steps"],
    "DISTILL_LR": DISTILL["lr"],
    "SCALING_MODELS": ",".join(SCALING["models"]),
    "SCALING_BUDGETS": ",".join(str(d) for d in SCALING["budgets"]),
    "SCALING_CONTEXT": SCALING["context"],
    "SCALING_BATCH": SCALING["batch"],
    "SCALING_LR": SCALING["lr"],
    "SCALING_LR_RULE": SCALING["lr_rule"],
}


if __name__ == "__main__":
    import sys
    if not SHELL_DEFAULTS["CONTEXT_WINDOW"]:
        SHELL_DEFAULTS["CONTEXT_WINDOW"] = ",".join(
            str(w) for w in context_windows(PRETRAIN["model_scheme"]))
    if "--shell-defaults" in sys.argv:
        for _k, _v in SHELL_DEFAULTS.items():
            print(f"MIRALLM_DEF_{_k}='{_v}'")
    else:
        for _k, _v in sorted(SHELL_DEFAULTS.items()):
            print(f"{_k:16s} {_v}")
