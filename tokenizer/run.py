"""
tokenizer/run.py -- stage 3: the byte-level BPE tokenizer every later stage shares.

Trains (or loads, if the checkpoint already exists) the corpus BPE that every later stage
shares, and saves it WITH its full vocabulary and per-merge build history under

    checkpoints/bpe/bpe.json

A BUILD IS THE LAST RESORT, NOT THE FIRST STEP. `resolve()` takes the tokenizer at
checkpoints/bpe/bpe.json whenever that file parses; failing that, the DOWNLOADED one under
saved_models/tokenizer-bpe-fineweb-edu-2BT/ -- bpe.json at the root or under bpe/ -- which is
this project's own tokenizer and arrives with stage 1, AND WHICH IS THEN COPIED INTO
checkpoints/bpe/ so the next run finds it there; and only when neither exists does it train,
which is the one path that needs the corpus. The log names the file that was taken.

The training corpus is the pretraining text (every corpus file under the pretraining
corpus -- prose, markdown, code or parquet -- recursive,
minus config.PRETRAIN["exclude_dirs"]) plus the preference texts, so the single tokenizer
covers both. build() is the one place the tokenizer is constructed, so every stage gets a
byte-for-byte identical vocabulary.

    python -m tokenizer.run [--bpe_merges N]
    ./old_scripts/stage3_pretrain_bpe_tokenizer.sh
"""
import os

from . import BPETokenizer
from helpers import visualization as viz


def with_extra_specials(tok):
    """Register default_config.EXTRA_SPECIAL_TOKENS on a tokenizer, and return it.

    Applied on every load and every build, so a token added to the configuration is present
    everywhere without editing the tokenizer. They append AFTER the predefined ones, so adding
    to the list never moves an existing id -- but it does grow the vocabulary, and a model
    trained before the change needs its embedding extended, so the log says what was added."""
    import default_config as config
    extra = list(getattr(config, "EXTRA_SPECIAL_TOKENS", []) or [])
    added = [t for t in extra if t not in tok.special2id]
    for t in added:
        tok.register_special(t)
    if added:
        print(f"[bpe] registered {len(added)} extra special token(s): {', '.join(added)} "
              f"-> vocabulary {len(tok):,}", flush=True)
    return tok


def saved_candidates(name):
    """Where a published vocabulary sits under saved_models/, most likely place first.

    config.MODEL_DIR DIRECTLY, WHICH download_candidates REACHES ONLY IF THE VOCABULARY IS
    CONFIGURED. download_dir routes by `type`, so it answers saved_models/ for an entry declared
    `type: model` in config/download/config_download.yaml and data/download/ for one that is not there at
    all -- and a vocabulary that has been MADE but not yet published has no entry. This looks in
    the right place either way, and both lists are tried, so nothing else changes."""
    import default_config as config
    directory = os.path.join(config.MODEL_DIR, download_name(name))
    return [os.path.join(directory, "bpe.json"),
            os.path.join(directory, "bpe", "bpe.json")]


def populate_from_saved(name, source, log=print):
    """Copy the published vocabulary at `source` into checkpoints/<name>/. Returns the new path,
    or `source` when nothing could be written.

    WHY COPY AT ALL, when reading it where it lies works. checkpoints/<name>/ is the PRIMARY
    path: it is what resolve_named looks at first, what helpers.bpe_subpath reports, and what a
    person deletes to force a rebuild. A run that reads out of saved_models/ every time leaves
    that path permanently empty, so every later run pays the fallback and the log says
    "not found at ..." forever about a vocabulary the project has.

    THE WHOLE DIRECTORY, NOT JUST bpe.json. `special_tokens.json` is what says which specials
    are appended and at which ids -- bpe-instruct is bpe-sft-coder's bpe.json byte for byte plus
    that sidecar -- so copying the vocabulary without it installs a file that reads as the base
    vocabulary with nothing to say so. That is the exact failure this function exists to end,
    not one to reproduce one directory over.

    NOTHING AT THE DESTINATION IS EVER REPLACED. A file already in checkpoints/<name>/ was put
    there by a build or by an earlier copy and is what this run was going to use anyway.

    THE COPY IS STAGED AND RENAMED, so an interrupted run cannot leave a half-written bpe.json
    at a path everything else treats as authoritative.

    A FAILURE HERE IS NOT A FAILURE OF THE RUN. The tokenizer was already found and is already
    correct; the copy is a convenience for the next run, and a read-only or full filesystem is
    reported and stepped over."""
    import shutil

    import default_config as config

    src_dir = os.path.dirname(os.path.abspath(source))
    dst_dir = os.path.join(config.CHECKPOINT_DIR, str(name).strip())
    dst = os.path.join(dst_dir, os.path.basename(source))
    try:
        os.makedirs(dst_dir, exist_ok=True)
        copied = []
        for fn in sorted(os.listdir(src_dir)):
            if fn.startswith(".") or not fn.endswith(".json"):
                continue
            target = os.path.join(dst_dir, fn)
            if os.path.exists(target):
                continue
            staged = target + ".incoming"
            shutil.copyfile(os.path.join(src_dir, fn), staged)
            os.replace(staged, target)
            copied.append(fn)
        if copied:
            log(f"[tokenizer] populated {_subpath(dst_dir)} from {_subpath(src_dir)}: "
                + ", ".join(copied))
    except OSError as e:
        log(f"[tokenizer] could not populate {_subpath(dst_dir)} from {_subpath(src_dir)} "
            f"({e}); reading the published copy where it lies")
        return source
    return dst if os.path.isfile(dst) else source


def resolve_named(name, log=print):
    """(tokenizer, path) for `--tokenizer NAME`, falling back down a fixed chain.

    The chain, in order, stopping at the first file that parses:

        checkpoints/<NAME>/bpe.json     the vocabulary that was asked for, as a build left it
        saved_models/tokenizer-<NAME>/  THE SAME VOCABULARY, as stage 1 fetched it -- bpe.json
                                        at the root and under bpe/. Taking it also INSTALLS it
                                        at the path above, so the next run stops at step one
        checkpoints/bpe/bpe.json        stage 3's
        saved_models/tokenizer-bpe-fineweb-edu-2BT/   the BASE vocabulary, in both shapes

    THE PUBLISHED COPY OF THE REQUESTED NAME COMES BEFORE THE FALLBACK, and that ordering is the
    point of it: a published `tokenizer-<NAME>` IS the vocabulary that was asked for, so taking
    it is not a substitution, while dropping to `bpe` is. A run that wanted bpe-sft-coder and
    silently got the base vocabulary tokenises code correctly but expensively, and nothing
    downstream says so. Only the first kind is installed into checkpoints/<NAME>/.

    Returns (None, wanted) when the chain is exhausted, so the caller reports the name that was
    asked for rather than the last one tried. Never builds: a token stream is tied to the
    vocabulary that produced it.
    """
    import default_config as config

    wanted = os.path.join(config.CHECKPOINT_DIR, str(name).strip(), "bpe.json")
    base_dir = config.download_dir(config.BPE_DOWNLOAD_NAME)
    base_published = [os.path.join(base_dir, "bpe.json"),
                      os.path.join(base_dir, "bpe", "bpe.json"),
                      config.BPE_DOWNLOAD_PATH]
    # WHERE THE BASE'S PUBLISHED COPY SITS IN THE CHAIN DEPENDS ON WHETHER IT WAS ASKED FOR.
    # For `--tokenizer bpe` it IS the vocabulary requested, so it is a saved_models candidate and
    # goes before checkpoints/ like every other name's. For any OTHER name it is the last-resort
    # fallback it has always been, and dropping to it is a substitution rather than a find.
    is_base = os.path.abspath(wanted) == os.path.abspath(config.BPE_PATH)
    chain = [wanted] + saved_candidates(name) + download_candidates(name)
    if is_base:
        chain += base_published
    else:
        chain.append(config.BPE_PATH)
        chain += base_published

    # WHICH CANDIDATES ARE THE VOCABULARY THAT WAS ASKED FOR, as opposed to a substitution. Only
    # these are copied into checkpoints/<name>/: installing the BASE under the name of some other
    # vocabulary would put the wrong ids at the right path, and every later run would find it
    # first and never look further.
    same_name = set(saved_candidates(name)) | set(download_candidates(name))
    if is_base:
        same_name |= set(base_published)
    tried = []
    for candidate in chain:
        if candidate in tried:
            continue
        tried.append(candidate)
        tok = peek(candidate)
        if tok is not None:
            if candidate == wanted:
                return tok, candidate
            if candidate in same_name:
                log(f"[tokenizer] {name} is not built at {_subpath(wanted)}; reading the "
                    f"published copy of the same vocabulary at {_subpath(candidate)}")
                return tok, populate_from_saved(name, candidate, log)
            log(f"[tokenizer] {name} not found at {_subpath(wanted)}; using "
                f"{_subpath(candidate)} instead")
            return tok, candidate
    log(f"[tokenizer] no vocabulary called {name}. Tried: "
        + ", ".join(_subpath(p) for p in tried))
    log(f"[tokenizer] fetch it with `./run_scripts/run_download_data.sh --only "
        f"{download_name(name)}`, or build it with `./old_scripts/stage12_tokenizer_sft.sh --tokenizer "
        f"{name}`")
    return None, wanted


def download_name(name):
    """The stage-1 dataset that publishes the vocabulary called `name`.

    One rule, so that adding a vocabulary to config/download/config_download.yaml is the whole of the
    work: `bpe-sft-coder` is published as `tokenizer-bpe-sft-coder`. The base vocabulary is the
    one exception -- it predates the convention and keeps its own name in config.
    """
    return f"tokenizer-{str(name).strip()}"


def download_candidates(name):
    """Where stage 1 may have left the vocabulary called `name`, most likely place first.

    BOTH LAYOUTS ARE TRIED because both exist in the wild: a repository may keep bpe.json at
    its root, or under bpe/ as the base tokenizer's does. Which one a repository used is not
    worth a configuration key, and guessing wrong costs a run.
    """
    import default_config as config
    # download_dir, NOT dataset_dir: a vocabulary is published as a MODEL repository and stage 1
    # puts it under saved_models/. dataset_dir would look in data/download/ and report every
    # vocabulary missing however many times it had been fetched.
    directory = config.download_dir(download_name(name))
    return [os.path.join(directory, "bpe.json"),
            os.path.join(directory, "bpe", "bpe.json")]


def _subpath(path):
    import helpers
    return helpers.bpe_subpath(path)


def peek(bpe_path):
    """The saved tokenizer, or None. Lets a caller ask how many merges are already on disk
    WITHOUT assembling a training corpus -- which is how the corpus stages avoid reading
    another stage's data just to find out whether the vocabulary needs building."""
    if not os.path.isfile(bpe_path):
        return None
    try:
        return with_extra_specials(BPETokenizer.load(bpe_path))
    except Exception:                                         # noqa: BLE001
        return None


def expected_vocab():
    """V AS THIS REPOSITORY COMPUTES IT EVERYWHERE ELSE: 256 raw bytes + the merges + the three
    predefined specials + any extras. 50,259 today.

    The same expression tools/vram.py:71 and the fused-loss unit test use, so the number checked
    here is the number the memory tables, the kernel timings and the embedding are sized by."""
    import default_config as config
    return 256 + int(config.BPE["num_merges"]) + len(config.EXTRA_SPECIAL_TOKENS) + 3


def resolve(bpe_path, download_path, log):
    """(tokenizer, path) for the vocabulary this run will use, or (None, bpe_path) to BUILD one.

    checkpoints/ FIRST, THEN saved_models/, AND THE TWO ARE NOT NAMED THE SAME.

    THIS REVERSES THE ORDER OF 2026-09-05 AND OF EARLIER ON 2026-09-06, and the populate step is
    why it is not a reversal of the intent. saved_models/ was put first so that a published
    vocabulary, which arrives whole with its sidecars, would win over a possibly partial local
    copy. Installing it into checkpoints/ on the way through gets the same guarantee: the local
    copy IS the published one after the first run that needed it, so looking there first can no
    longer read something partial.

    THE NAMES DIFFER AND download_name IS WHAT BRIDGES THEM. The working copy is
    `checkpoints/bpe/`, named for the vocabulary; the published one is
    `saved_models/tokenizer-bpe-fineweb-edu-2BT/`, named for the repository it was fetched from.
    Neither name can be derived by string surgery on the other, so the published name comes from
    config.BPE_DOWNLOAD_NAME and every other vocabulary's from download_name().

    THE ORDER, AND ONLY THE FIRST THREE STEPS CAN AVOID A BUILD:

        1. bpe_path        checkpoints/bpe/bpe.json, whatever a build made here
        2. saved_models/tokenizer-bpe-fineweb-edu-2BT/bpe.json       the published vocabulary,
        3. saved_models/tokenizer-bpe-fineweb-edu-2BT/bpe/bpe.json   in either of its two shapes
        4. none of them    return None, and the caller builds -- which needs the pretraining
                           corpus and takes hours. That path is unchanged.

    REACHING 2 OR 3 INSTALLS THE FILE AT 1, so the next run stops at the first step.

    BOTH SHAPES ARE TRIED because both exist in the wild -- a repository may keep bpe.json at its
    root or under bpe/ -- and `download_path` alone named only the second. That is not
    hypothetical: the base vocabulary is published with bpe.json at the ROOT, so a tree that had
    fetched it was told there was no tokenizer and offered to spend hours training one.

    A LOCAL BUILD IS PREFERRED TO THE PUBLISHED COPY, and the published copy is INSTALLED
    rather than merely read. Where both are the same vocabulary nothing moves either way -- the
    token cache is keyed by a fingerprint of the MERGES, not by the file's location, so either
    file reaches the same shards.

    WHICH ONE WAS TAKEN IS LOGGED, from here, naming the file. A run must never be trained
    against a different vocabulary than its operator believes it is using, and the files are a
    directory apart with the same name.

    THE DOWNLOADED FILE IS READ WHERE IT LIES. It is not copied into checkpoints/: that
    directory is written by builds, and a copy there would be indistinguishable from one. The
    caller passes the returned path on to build() and to helpers.set_bpe_source, so the
    checkpoint embeds the bytes actually in use.

    ITS `.partial` NEIGHBOUR IS IGNORED, DELIBERATELY. `<path>.partial` is BPETokenizer.build's
    own resume file -- merges so far, keyed by a signature of the word table they were counted
    over -- and it is read by a build, from beside bpe_path, and never anywhere else. Adopting
    a finished bpe.json runs no build, so nothing consults it; and a build that does run is on
    THIS machine's corpus, whose signature the downloaded partial cannot match.

    A DOWNLOADED VOCABULARY OF THE WRONG SIZE STOPS THE RUN. The embedding is tied to the output
    head, every VRAM table and the fused-loss kernel are sized at 256 + merges + specials, and
    the dataset's own name asserts 50,259 -- so a file that parses to some other size is not the
    artefact it claims to be. It is refused here rather than absorbed, since absorbing it means
    days of training against a vocabulary nobody chose. checkpoints/bpe/bpe.json is NOT held to
    this: it is whatever a build made, and checkpoints already depend on it."""
    import default_config as config
    # THE PUBLISHED COPIES, IN BOTH SHAPES, AND `download_path` AMONG THEM.
    #
    # config.BPE_DOWNLOAD_NAME DIRECTLY, NOT THROUGH download_name(). The base vocabulary is the
    # one repository that predates the tokenizer-<name> convention and carries its own name in
    # the configuration -- download_name's own docstring says so -- so deriving it from "bpe"
    # would name a repository that does not exist. Every OTHER vocabulary goes through
    # download_name, in resolve_named.
    #
    # The caller passes config.BPE_DOWNLOAD_PATH, which names one shape; the directory is asked
    # for the other too, and duplicates are dropped so a file is never peeked at twice.
    directory = config.download_dir(config.BPE_DOWNLOAD_NAME)
    published, seen = [], set()
    for candidate in ([os.path.join(directory, "bpe.json"),
                       os.path.join(directory, "bpe", "bpe.json")]
                      + ([download_path] if download_path else [])):
        real = os.path.abspath(candidate)
        if real not in seen:
            seen.add(real)
            published.append(candidate)

    # THE LOCAL BUILD FIRST. It is NOT size-checked: it is whatever stage 3 produced, and every
    # checkpoint already depends on it.
    tok = peek(bpe_path)
    if tok is not None:
        return tok, bpe_path

    # THEN THE PUBLISHED COPY, AND IT IS INSTALLED INTO checkpoints/ ON THE WAY THROUGH, so the
    # primary path stops being empty after the first run that needs it.
    for candidate in published:
        if not os.path.isfile(candidate):
            continue
        tok = peek(candidate)
        if tok is None:
            log(f"BPE: {candidate} exists but does not parse as a tokenizer; ignoring it")
            continue
        _check_published_size(tok, candidate)
        log(f"BPE: no vocabulary at {_subpath(bpe_path)}; taking the published one at "
            f"{_subpath(candidate)}")
        return tok, populate_from_saved(os.path.basename(os.path.dirname(bpe_path)),
                                        candidate, log)
    return None, bpe_path


def base_path(log=None):
    """WHERE THE BASE VOCABULARY IS, saved_models/ first: a path, or config.BPE_PATH if nowhere.

    Returns config.BPE_PATH when nothing is found, so a caller's own "missing" message still
    names the file a build would write."""
    import default_config as config
    tok, path = resolve(config.BPE_PATH, config.BPE_DOWNLOAD_PATH, log or (lambda _m: None))
    return path if tok is not None else config.BPE_PATH


def _check_published_size(tok, path):
    """A published vocabulary of the wrong size stops the run."""
    want = expected_vocab()
    if len(tok) != want:
        raise SystemExit(
            f"[bpe] VOCABULARY SIZE MISMATCH in the downloaded tokenizer -- REFUSED.\n"
            f"      file      {path}\n"
            f"      it has    {len(tok):,} tokens ({len(tok.merges):,} merges)\n"
            f"      expected  {want:,} = 256 bytes + merges + specials, which is what the\n"
            f"                embedding, the tied output head, the VRAM tables and the fused\n"
            f"                loss kernel are all sized at\n"
            f"      The dataset's own name asserts 50,259. Training against a vocabulary that\n"
            f"      is not the one this configuration describes is a silently wrong model, so\n"
            f"      it is not done by accident: either train your own (move the download aside\n"
            f"      and run stage 3) or make the configuration match the file.")


def build(texts, bpe_path, log, plotdir=None, num_merges=8000, plot_every=200,
          checkpoint_every=250, min_freq=1, preloaded=None):
    """Return the BPE tokenizer, REUSING the checkpoint at `bpe_path` whenever there is one.

    Only a MISSING or UNREADABLE file builds. A build is then RESUMABLE and WATCHED: merges are
    checkpointed to `<bpe_path>.partial` every `checkpoint_every` merges, so an interrupted
    build continues where it stopped instead of restarting, and bpe_dynamics.pdf is redrawn
    every `plot_every` merges -- the same cadence every trainer plots at.

    `preloaded` is the tokenizer the caller already parsed with peek(); passing it avoids a
    second 20 MB json parse and a second reconstruction of the vocabulary on every run.

    WHICH FILE `bpe_path` IS, WHEN THE CALLER CAME THROUGH resolve(): the trained tokenizer if
    there is one, otherwise the DOWNLOADED one, and in that second case `preloaded` is already
    the tokenizer read from it -- so nothing is built, nothing is written, and the line below
    names the file that is actually in use rather than the one a build would have made."""
    os.makedirs(os.path.dirname(bpe_path), exist_ok=True)
    tok = preloaded
    if tok is None and os.path.isfile(bpe_path):
        try:
            tok = BPETokenizer.load(bpe_path)
        except Exception:                                     # noqa: BLE001
            tok = None                                        # stale/old format -> rebuild
    if tok is None:
        def monitor(history, merge):
            if plotdir and history:
                viz.bpe_monitor(history, plotdir, "")
        tok = BPETokenizer.build(texts, num_merges=num_merges, checkpoint_path=bpe_path,
                                 checkpoint_every=checkpoint_every, log=log,
                                 min_freq=min_freq,
                                 monitor=monitor if plotdir else None,
                                 plot_every=plot_every,
                                 init_merges=tok.merges if tok else None,
                                 init_history=tok.build_history if tok else None,
                                 init_sig=tok.corpus_sig if tok else None)
        tok.save(bpe_path)
    hist = getattr(tok, "build_history", None)
    if plotdir and hist:
        out = viz.bpe_monitor(hist, plotdir, "")
        if out:
            log(f"[plot] {out} created ({len(hist):,} merges)")
    return tok


def main():
    """Standalone stage 3: build (or reuse) the BPE tokenizer and draw outputs/plots/bpe/bpe_dynamics.pdf."""
    from helpers import common
    args = common.parse_args(stage="bpe")
    # need_pairs=False, DELIBERATELY. The preference data is an input to a BUILD, not to a
    # load, and setup() already loads it whenever a build is actually needed. Asking for it
    # unconditionally made stage 3 read the corpus even when the answer was "the tokenizer is
    # already on disk" -- which is slow when the data is there and fatal when it is not.
    common.setup(args, need_pairs=False, draw_bpe=True)    # builds/loads the tokenizer and draws bpe_dynamics.pdf


if __name__ == "__main__":
    main()
