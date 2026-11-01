"""
tools/download_data.py -- fetch the datasets into data/download/. The ONLY code in this
repository that downloads.

    python -m tools.download_data                     every dataset
    python -m tools.download_data --list              what would be fetched, and where
    python -m tools.download_data --only llm-grpo-cot-gsm8k
    python -m tools.download_data --force             re-fetch even if present

    data/download/llm-pretrain-wikitext103/         Tiny's corpus
    data/download/llm-pretrain-fineweb-edu-2BT/   ~10 GB, ~2B tokens
    data/download/llm-rlhf-instruction-following/               instruction-tuning data
    data/download/llm-grpo-cot-gsm8k/                           chain-of-thought problems

WHY THE TRAINERS DO NOT DO THIS. A stage that fetches its own corpus on first use is a stage
whose corpus is not pinned: the run cannot be reproduced offline, a network hiccup surfaces as
a training failure hours in, and nobody can tell from a log whether the data on disk is the
data that was intended. Fetching is therefore a separate, deliberate act with its own command,
and every trainer reads a local directory and stops if it is not there.

WHAT IS DOWNLOADED is listed in default_config.DATASETS, one entry per dataset, each naming
the source repository and what it is for. The two larger schemes ship with NO pretraining
corpus and nothing here provides one; point PRETRAIN_DIR at your own.

HOW THE FETCH RESUMES, and why it is done the long way round. `snapshot_download` resumes
through THE SHARED CACHE and only through it: a blob is keyed by its hash, a partial transfer
is an `.incomplete` beside it, and a second run continues that same file. Handed `local_dir=`
instead, recent versions write each attempt to a UNIQUELY NAMED temporary file, so nothing is
ever continued -- an interrupted 17 GB fetch starts again from zero and leaves its partial
behind. That is not a hypothesis: this project accumulated 24 orphaned `.incomplete` files,
19 GB of them, three per shard, one per interruption, for a dataset that had not landed a
single visible file.

So the download goes to `data/.hf_cache/` and the snapshot is then HARD-LINKED into
data/download/<name>/. A hard link costs no space -- the cache and the dataset directory are
deliberately on one filesystem so that it cannot silently become a copy -- and it leaves real
files rather than symlinks for the corpus scanner to follow. Deleting the cache afterwards is
safe: the links keep the data alive.

THE REPOSITORY'S OWN LAYOUT IS PRESERVED VERBATIM, with no exception. A downloaded corpus is
read-only input from the moment it lands: nothing here rearranges it, splits it, renames it or
adds to it afterwards. The loaders walk recursively, so no local layout policy is needed.
"""
import argparse
import json
import os
import shutil

import default_config as config


def target(name):
    """Where this entry lands: saved_models/ for a model repository, data/download/ for a
    corpus. One resolver, so the listing, the fetch and the completion marker cannot disagree
    about where a thing is."""
    return config.download_dir(name)


# THE SHARED CACHE, ON THE SAME FILESYSTEM AS THE DATASETS. Beside data/download/ rather than in
# ~/.cache, for one reason: the snapshot is hard-linked into the dataset directory afterwards,
# and a hard link cannot cross a filesystem. Under $HOME it would silently degrade to a copy on
# any machine whose data lives on a separate mount -- which is every cluster -- and a 17 GB
# corpus would occupy 34 GB with nothing in the log to say why.
HF_CACHE = os.path.join(os.path.dirname(os.path.normpath(config.DOWNLOAD_DIR)), ".hf_cache")


def cache_root(repo):
    """Where huggingface_hub keeps this repo inside HF_CACHE."""
    return os.path.join(HF_CACHE, "datasets--" + repo.replace("/", "--"))


def have_locally(repo, dest):
    """{relative path: bytes} of every file of `repo` ALREADY ON THIS DISK, and the bytes
    sitting in partial transfers. No network.

    TWO PLACES COUNT AS HAVING IT. A file is present if it is in the cache snapshot -- where
    the resume happens -- or already hard-linked into the dataset directory by a previous run
    that got as far as materialising. Checking only one of them under-reports after an
    interruption at the wrong moment and tells the reader to expect a download that will not
    happen.

    os.stat FOLLOWS the snapshot's symlink into the blob, so a dangling link left by a deleted
    blob raises and the file is correctly counted as missing rather than as present at zero
    bytes."""
    have = {}
    snaps = os.path.join(cache_root(repo), "snapshots")
    if os.path.isdir(snaps):
        for rev in sorted(os.listdir(snaps)):
            base = os.path.join(snaps, rev)
            for dirpath, dirs, names in os.walk(base):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for n in names:
                    if n.startswith("."):
                        continue
                    p = os.path.join(dirpath, n)
                    try:
                        have[os.path.relpath(p, base)] = os.stat(p).st_size
                    except OSError:
                        pass
    if os.path.isdir(dest):
        for dirpath, dirs, names in os.walk(dest):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for n in names:
                if n.startswith("."):
                    continue
                p = os.path.join(dirpath, n)
                try:
                    have.setdefault(os.path.relpath(p, dest), os.stat(p).st_size)
                except OSError:
                    pass
    part_n = part_b = 0
    blobs = os.path.join(cache_root(repo), "blobs")
    if os.path.isdir(blobs):
        for n in os.listdir(blobs):
            if n.endswith(".incomplete"):
                part_n += 1
                try:
                    part_b += os.path.getsize(os.path.join(blobs, n))
                except OSError:
                    pass
    return have, part_n, part_b


def selected(paths, patterns):
    """The subset of `paths` a dataset's allow_patterns keeps; all of them when it has none.

    THE DENOMINATOR HAS TO MATCH WHAT IS FETCHED. A dataset that takes five of MMLU's fifty-seven
    subjects would otherwise report "12 of 400 files" forever and never look complete, because
    the hub lists the whole repository while snapshot_download brings a fraction of it. The
    matching is fnmatch on the repository-relative path, which is what huggingface_hub itself
    applies to allow_patterns, so this cannot disagree with the transfer."""
    if not patterns:
        return list(paths)
    import fnmatch
    globs = [g.strip() for g in patterns.split(",") if g.strip()]
    return [p for p in paths if any(fnmatch.fnmatch(p, g) for g in globs)]


def remote_manifest(repo, repo_type="dataset", patterns=None):
    """{relative path: bytes} for the repo, from the hub, narrowed to `patterns`. None if it
    cannot be asked.

    ONE SMALL METADATA CALL, and it is what makes the count honest: without the repo's own file
    list there is no denominator, and "8 files present" is not "8 of 15". Failure is not fatal
    -- a hub that will not answer a metadata request will fail the download itself a moment
    later with a better message -- so this returns None and the caller reports what it has."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().repo_info(repo_id=repo, repo_type=repo_type, files_metadata=True)
        whole = {s.rfilename: (s.size or 0) for s in (info.siblings or [])}
    except Exception:                                              # noqa: BLE001
        return None
    return {name: whole[name] for name in selected(whole, patterns)}


# How many filenames to print before summarising. Enough to recognise the shape of a shard set,
# short enough that a 500-file corpus does not bury the line that matters.
INVENTORY_LINES = 8


def report_progress(name, repo, dest, log, repo_type="dataset", patterns=None):
    """What is already here, what is missing, and how far through the fetch is -- BEFORE it
    starts.

    WHY THIS EXISTS. huggingface_hub prints two bars and neither answers the question a person
    resuming actually has. "Fetching 15 files: 0/15" counts what THIS session has linked, so it
    reads 0 when 8 files are already on disk; the byte bar counts only the files this session
    decided to transfer, so its total shrinks as more is already present. Both are honest about
    what they measure and neither is what "how far along am I" means. This is."""
    remote = remote_manifest(repo, repo_type, patterns)
    have, part_n, part_b = have_locally(repo, dest)
    if remote is None:
        log(f"[download]   already here: {len(have):,} files ({_size(sum(have.values()))}); "
            f"the hub would not list the repo, so the total is unknown")
        return
    done = {k: v for k, v in remote.items() if k in have and have[k] >= v > 0}
    todo = {k: v for k, v in remote.items() if k not in done}
    dn, tn = len(done), len(remote)
    db, tb = sum(done.values()), sum(remote.values())
    pct = (100.0 * db / tb) if tb else 100.0
    log(f"[download]   PROGRESS  {dn:,}/{tn:,} files  "
        f"{_size(db)} of {_size(tb)}  ({pct:.1f}%)")
    if part_n:
        log(f"[download]   partial   {part_n:,} file(s) mid-transfer, {_size(part_b)} of them "
            f"already on disk and resumable")
    if dn:
        for f in sorted(done)[:INVENTORY_LINES]:
            log(f"[download]     have  {f}  ({_size(done[f])})")
        if dn > INVENTORY_LINES:
            log(f"[download]     have  ... and {dn - INVENTORY_LINES:,} more")
    if todo:
        log(f"[download]   REMAINING {len(todo):,} files, {_size(sum(todo.values()))}")
        for f in sorted(todo)[:INVENTORY_LINES]:
            log(f"[download]     need  {f}  ({_size(todo[f])})")
        if len(todo) > INVENTORY_LINES:
            log(f"[download]     need  ... and {len(todo) - INVENTORY_LINES:,} more")
    else:
        log(f"[download]   REMAINING nothing -- every file is already on this disk")


def stale_staging(name):
    """(files, bytes) of the DEAD `.incomplete` staging a local_dir download left behind."""
    root = os.path.join(target(name), ".cache", "huggingface", "download")
    n = total = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if not f.endswith(".incomplete"):
                continue
            n += 1
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return n, total


def _link(src, dst):
    """One file from the snapshot into the dataset tree: hard link, else symlink, else copy.

    HARD LINK FIRST because it is free and indistinguishable from a real file afterwards --
    which matters, since the corpus scanner opens these and a broken symlink is a corpus that
    reads as empty. os.link follows the snapshot's own symlink into the blob, so what is linked
    is the data and not the pointer."""
    if os.path.exists(dst):
        return "kept"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(os.path.realpath(src), dst)
        return "linked"
    except OSError:
        pass
    try:
        os.symlink(os.path.realpath(src), dst)
        return "symlinked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def materialise(snapshot, dest, log):
    """The cache snapshot -> the dataset directory. Returns how many files arrived, by method."""
    try:
        from tqdm import tqdm
    except ImportError:                                            # noqa: BLE001
        tqdm = None
    files = []
    for dirpath, dirs, names in os.walk(snapshot):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        files += [os.path.join(dirpath, f) for f in names if not f.startswith(".")]
    how = {}
    it = tqdm(files, desc="[download] linking into place", unit="file") if tqdm else files
    for src in it:
        rel = os.path.relpath(src, snapshot)
        k = _link(src, os.path.join(dest, rel))
        how[k] = how.get(k, 0) + 1
    if tqdm and hasattr(it, "close"):
        it.close()
    log(f"[download]   {len(files):,} files into place ("
        + ", ".join(f"{v:,} {k}" for k, v in sorted(how.items())) + ")")
    return len(files)


def _size(n):
    """Bytes at a scale a reader can check against `du`, whatever the scale."""
    for unit, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n:,} B"


MARKER = ".mira_llm_download.json"
MAGIC = "mira_llm-download"
LEGACY_MARKER = ".nullgpt_download.json"
LEGACY_MAGIC = "nullgpt-download"


def state(name):
    """(file count, total bytes) of what is on disk now. Dotfiles are ignored -- `._*` stubs
    and `.cache/` are written by macOS and by huggingface_hub, belong to neither the repo nor
    the corpus, and would make an otherwise identical tree count differently."""
    n = total = 0
    for root, dirs, files in os.walk(target(name)):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.startswith("."):
                continue
            n += 1
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return n, total


def read_marker(name):
    """The completion record written after a successful fetch, or None."""
    for marker, magic in ((MARKER, MAGIC), (LEGACY_MARKER, LEGACY_MAGIC)):
        p = os.path.join(target(name), marker)
        try:
            with open(p, "r", encoding="utf-8") as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            continue
        if m.get("magic") == magic:
            return m
    return None


def wanted_patterns(name):
    """The configured allow_patterns for one dataset, normalised, or None for the whole repository.

    A list rather than the raw string so the comparison in is_complete does not turn on spacing,
    and sorted so reordering the field is not a change.
    """
    raw = (config.DATASETS.get(name) or {}).get("allow_patterns")
    if not raw:
        return None
    parts = sorted(p.strip() for p in str(raw).split(",") if p.strip())
    return parts or None


def write_marker(name, sharded):
    """Record that this dataset is COMPLETE, and what complete looks like.

    A download is finished when the thing that downloads it says so -- not when a directory
    exists, and not when it holds a file. Those are true a second after the first byte lands.
    So completion is recorded explicitly, once snapshot_download has returned, and the file
    count and byte total go in with it so that a later truncation, a half-deleted tree or an
    interrupted copy between machines is caught rather than assumed away.

    NOTHING MACHINE-DEPENDENT GOES IN, for the same reason it does not go in a corpus
    signature: no path, no host, no timestamp. The repo it came from and the shape of what
    arrived are properties of the data.

    WHAT WAS ASKED FOR GOES IN TOO, NOT ONLY WHAT ARRIVED. allow_patterns decides which files a
    fetch requests, so it is part of what "complete" means: a tree holding every validation shard
    is complete under one pattern and incomplete under another. Recording only the file count and
    the byte total could not see that, and changing the pattern from validation to test left the
    marker valid, the fetch skipped, and the evaluation reporting "no test shard" against a full
    directory."""
    n, total = state(name)
    p = os.path.join(target(name), MARKER)
    tmp = p + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"magic": MAGIC, "version": 1,
                   "repo": config.DATASETS[name]["repo"],
                   "allow_patterns": wanted_patterns(name),
                   "n_files": n, "bytes": total, "sharded": bool(sharded)}, fh, indent=1)
    os.replace(tmp, p)          # atomic: a half-written marker must never look like completion
    return n, total


def is_complete(name):
    """(complete?, why not). COMPLETION IS A RECORD, NOT AN OBSERVATION."""
    d = target(name)
    if not os.path.isdir(d):
        return False, "not downloaded"
    m = read_marker(name)
    n, total = state(name)
    if m is None:
        return False, ("no completion marker" if n else "empty directory")
    # THE PATTERN IS PART OF WHAT COMPLETE MEANS. A tree fetched under one allow_patterns is not
    # complete under another, however many files it holds, so a changed pattern re-fetches.
    asked, had = wanted_patterns(name), m.get("allow_patterns")
    if asked != had:
        return False, (f"allow_patterns changed since it was fetched: "
                       f"{','.join(asked) if asked else 'the whole repository'} now, "
                       f"{','.join(had) if had else 'the whole repository'} when it completed")
    if n != m.get("n_files") or total != m.get("bytes"):
        return False, (f"changed since it was fetched: "
                       f"{n:,} files / {_size(total)} now, "
                       f"{m.get('n_files', 0):,} files / {_size(m.get('bytes', 0))} "
                       f"when it completed")
    return True, ""


def looks_sharded(name):
    """Has this tree already been split into part_NNNN/ directories?

    It matters because sharding MOVES files out of the layout the hub published, so re-running
    snapshot_download over a sharded tree would fetch every file again into the flat layout and
    leave two copies. A tree that is sharded but has no marker therefore cannot be verified by
    re-fetching, and this is what lets that case be refused rather than blundered into."""
    for _root, dirs, _files in os.walk(target(name)):
        if any(d.startswith("part_") for d in dirs):
            return True
    return False


def is_present(name):
    """Kept for the summary: is there anything here at all? Completion is is_complete()."""
    n, _ = state(name)
    return n > 0


def size_on_disk(name):
    total = 0
    for root, _dirs, files in os.walk(target(name)):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def describe(log=print):
    log("")
    log("datasets")
    log("-" * 96)
    log(f"  {'name':<52} {'status':<10} {'destination'}")
    for name, spec in config.DATASETS.items():
        done, why = is_complete(name)
        if done:
            status = f"{size_on_disk(name) / 2**30:.1f} GiB"
        elif is_present(name):
            status = "PARTIAL"          # present is not complete; say which
        else:
            status = "missing"
        log(f"  {name:<52} {status:<10} {os.path.relpath(target(name), config.ROOT)}")
        if status == "PARTIAL":
            log(f"  {'':<52} {'':<10} {why} -- re-run to resume")
        log(f"  {'':<52} {'':<10} from {spec['repo']}")
        log(f"  {'':<52} {'':<10} {spec['what']}")
    log("")
    log("  pretraining corpus per scheme")
    for scheme, paths in config.PRETRAIN_CORPUS.items():
        # A scheme reads SEVERAL corpora, each one standalone in the cache, so list them one
        # per line rather than joining them into a path-shaped string nobody can parse.
        paths = [p for p in (paths if isinstance(paths, (list, tuple)) else [paths]) if p]
        if not paths:
            log(f"    {scheme:<12} (unset -- configure PRETRAIN_DIR for your own corpus)")
            continue
        for i, path in enumerate(paths):
            log(f"    {scheme if i == 0 else '':<12} {os.path.relpath(path, config.ROOT)}")
    log("-" * 96)


def free_space(path):
    d = path
    while d and not os.path.isdir(d):
        d = os.path.dirname(d)
    return shutil.disk_usage(d or ".").free


# =============================================================================== #
# verification: the downloader owns whether a file is good
# =============================================================================== #
# A transfer that is interrupted, resumed against a mismatched partial, or copied over a flaky
# link leaves files that open but do not read. Stage 4 then reports "CORRUPT DATA SKIPPED" per
# row group, hours in, and the corpus it writes is quietly short. The check belongs here, where
# the file arrived and where the remedy -- fetch it again -- is available.
#
# WHAT IS CHECKED IS THE ENVELOPE, NOT EVERY BYTE. A parquet footer records where each column
# chunk starts and how long it is, so the highest byte the file claims is known from metadata
# alone: if that is past the end of the file, the file is short and every chunk after the missing
# bytes reads at the wrong offset. That is one metadata read per shard rather than a decode of
# the corpus, and it is what catches a truncated or badly resumed transfer.
PARQUET_EXT = (".parquet",)
LINE_EXT = (".json", ".jsonl")

# HOW MUCH OF A ONE-DOCUMENT JSON IS ACTUALLY PARSED. Below this the file is parsed outright,
# which is the definitive answer; above it only the envelope is checked, because a corpus stored
# as one enormous json array must not cost a full parse to verify. A published vocabulary is
# ~20 MB, so it is parsed.
JSON_PARSE_MAX = 64 << 20


def verify_parquet(path):
    """(ok, why) for one parquet file, from its magic and its footer."""
    size = os.path.getsize(path)
    if size < 12:
        return False, f"{size} bytes: too small to be parquet"
    with open(path, "rb") as handle:
        head = handle.read(4)
        handle.seek(-8, os.SEEK_END)
        footer_len = int.from_bytes(handle.read(4), "little")
        tail = handle.read(4)
    if head != b"PAR1" or tail != b"PAR1":
        return False, "the PAR1 magic is missing at one or both ends"
    # A CHECKER THAT IS NOT INSTALLED IS NOT A DAMAGED FILE. Without pyarrow the envelope above
    # is the whole check, and reporting damage on that basis would re-fetch a corpus that is
    # sound because a package is missing.
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return True, ""
    try:
        metadata = pq.ParquetFile(path).metadata
    except Exception as e:                                         # noqa: BLE001
        return False, f"the footer does not parse ({type(e).__name__}: {e})"
    highest = 0
    for g in range(metadata.num_row_groups):
        group = metadata.row_group(g)
        for c in range(group.num_columns):
            chunk = group.column(c)
            start = chunk.dictionary_page_offset or chunk.data_page_offset
            highest = max(highest, int(start) + int(chunk.total_compressed_size))
    if highest > size:
        return False, (f"the footer reaches byte {highest:,} and the file holds {size:,}: "
                       f"{highest - size:,} bytes are missing")
    return True, ""


def verify_lines(path):
    """(ok, why) for a json or jsonl file, from its first and last records.

    A truncated transfer cuts the file mid-record, so the LAST line is where it shows. Reading
    every line of an 8 GiB corpus to find that would cost more than the fetch did.
    """
    size = os.path.getsize(path)
    if size == 0:
        return False, "empty"
    import json as _json
    with open(path, "rb") as handle:
        first = handle.readline()
        handle.seek(max(size - (1 << 20), 0))
        tail = handle.read()
    last = tail.rsplit(b"\n", 2)[-1] or (tail.rsplit(b"\n", 2)[-2] if b"\n" in tail else b"")
    for label, raw in (("first", first), ("last", last)):
        text = raw.strip()
        if not text:
            continue
        try:
            _json.loads(text.decode("utf-8"))
        except Exception:                                          # noqa: BLE001
            return False, f"the {label} line is not valid json; the file is cut short"
    return True, ""


def verify_json_document(path):
    """(ok, why) for a json file holding ONE document rather than one record per line.

    A TOKENIZER IS THIS, AND IT IS WHAT verify_lines GETS WRONG.

    saved_models/tokenizer-bpe-fineweb-edu-2BT/bpe.json is a single json object spanning 20.5 MB,
    pretty-printed, so its FIRST LINE is `{` -- which is not valid json on its own. verify_lines
    read that line, could not parse it, and reported "the first line is not valid json; the file
    is cut short" about a file that had just been downloaded intact and whose size matched the
    published byte count exactly. Stage 1 then re-fetched it, reached the same verdict, refused
    to write the completion marker, and told the user to delete a file that was never damaged.

    THE ENVELOPE IS WHAT A TRUNCATED TRANSFER BREAKS. A complete json document opens with `{` or
    `[` and closes with the matching `}` or `]`; a transfer cut short has the opener and not the
    closer. That is two seeks whatever the file's size.

    AND UNDER JSON_PARSE_MAX IT IS PARSED OUTRIGHT, which is definitive rather than indicative.
    A vocabulary is ~20 MB and parsing it costs a moment; a corpus published as one enormous
    array is not parsed, because verifying it would then cost more than fetching it did."""
    import json as _json
    size = os.path.getsize(path)
    if size == 0:
        return False, "empty"
    with open(path, "rb") as handle:
        head = handle.read(64).lstrip()
        handle.seek(max(size - 64, 0))
        tail = handle.read().rstrip()
    if not head or not tail:
        return False, "no json content"
    opener, closer = head[:1], tail[-1:]
    want = {b"{": b"}", b"[": b"]"}.get(opener)
    if want is None:
        return False, f"does not open as a json document (starts {opener!r})"
    if closer != want:
        return False, (f"opens {opener.decode()} and ends {closer.decode()!r} rather than "
                       f"{want.decode()}; the file is cut short")
    if size <= JSON_PARSE_MAX:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                _json.load(handle)
        except Exception as e:                                     # noqa: BLE001
            return False, f"does not parse as json ({type(e).__name__})"
    return True, ""


def verify_file(path, document=False):
    """(ok, why) for one downloaded file, by extension.

    `document` SAYS THE .json FILES HERE ARE WHOLE DOCUMENTS, and it is set for a repository
    declared `type: model` -- a published tokenizer, which is what saved_models/ holds. Those are
    never line-delimited, so checking them as records is checking the wrong thing.

    A DATASET'S .json IS TRIED BOTH WAYS BEFORE IT IS CALLED DAMAGED. Line-delimited first, since
    that is what a corpus is; a whole document second. Reporting damage is expensive -- it
    re-fetches gigabytes and it stops the stage -- so the benefit of the doubt costs less than a
    false positive does. When BOTH fail the file is genuinely broken, and the LINE reason is the
    one reported: a .json in a dataset repository is a corpus far more often than it is a
    document, so that is the reading a person should be given first."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in PARQUET_EXT:
            return verify_parquet(path)
        if ext == ".json" and document:
            return verify_json_document(path)
        if ext in LINE_EXT:
            ok, why = verify_lines(path)
            if ok or ext != ".json":
                return ok, why
            doc_ok, doc_why = verify_json_document(path)
            return (True, "") if doc_ok else (False, why)
        return (True, "") if os.path.getsize(path) else (False, "empty")
    except OSError as e:
        return False, f"cannot be read ({e})"


def verify_dataset(name, log):
    """Every damaged file under one dataset, as paths relative to its directory.

    THE PUBLISHED SIZE IS THE FIRST AND STRONGEST CHECK. The hub lists the byte count of every
    file in the repository, so a short transfer is one comparison rather than an inference: it
    catches a file whose own footer happens to be self-consistent, which the envelope checks
    below cannot. It needs the network, so a run that cannot reach the hub falls through to
    them and says so.
    """
    dest = target(name)
    spec = config.DATASETS[name]
    files = []
    for dirpath, dirs, names in os.walk(dest):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        files += [os.path.join(dirpath, f) for f in sorted(names) if not f.startswith(".")]
    if not files:
        return []
    published = remote_manifest(spec["repo"], spec.get("type", "dataset"),
                                spec.get("allow_patterns"))
    if published is None:
        log(f"[verify] {name}: the hub did not answer; checking what is on disk only")
    else:
        short = []
        for rel, want in sorted(published.items()):
            path = os.path.join(dest, rel)
            if not os.path.exists(path):
                short.append((rel, "not present"))
            elif os.path.getsize(path) != want:
                have = os.path.getsize(path)
                short.append((rel, f"{have:,} bytes on disk against {want:,} published"
                                   f" ({want - have:+,})"))
        if short:
            log(f"[verify] {name}: {len(short):,} file(s) do not match the published size")
            for rel, why in short:
                log(f"[verify]   {rel}: {why}")
            return short
    try:
        from tqdm import tqdm
    except ImportError:                                            # noqa: BLE001
        tqdm = None
    damaged = []
    # A MODEL REPOSITORY'S .json IS A DOCUMENT, NOT A CORPUS. saved_models/ holds published
    # tokenizers, whose bpe.json is one pretty-printed object; the download config already says
    # which entries are model repositories, so the check does not have to guess.
    document = spec.get("type", "dataset") == "model"
    it = (tqdm(files, desc=f"[verify] {name}", unit="file", total=len(files))
          if tqdm else files)
    for path in it:
        ok, why = verify_file(path, document=document)
        if not ok:
            damaged.append((os.path.relpath(path, dest), why))
            if tqdm:
                it.set_postfix_str(f"{len(damaged):,} damaged")
    if tqdm and hasattr(it, "close"):
        it.close()
    log(f"[verify] {name}: {len(files):,} file(s) checked, {len(damaged):,} damaged")
    for rel, why in damaged:
        log(f"[verify]   {rel}: {why}")
    return damaged


def refetch(name, damaged, log):
    """Fetch each damaged file again and put it in place. Returns how many were replaced."""
    if not damaged:
        return 0
    spec = config.DATASETS[name]
    dest = target(name)
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:                                         # noqa: BLE001
        raise SystemExit(
            f"[verify] {name}: {len(damaged)} damaged file(s) and huggingface_hub is not "
            f"importable ({e}); cannot re-fetch") from e
    replaced = 0
    for rel, _why in damaged:
        log(f"[verify] {name}: re-fetching {rel}")
        try:
            blob = hf_hub_download(repo_id=spec["repo"], filename=rel.replace(os.sep, "/"),
                                   repo_type=spec.get("type", "dataset"), cache_dir=HF_CACHE,
                                   force_download=True)
        except Exception as e:                                     # noqa: BLE001
            log(f"[verify] {name}: {rel} could not be fetched ({type(e).__name__}: {e})")
            continue
        dst = os.path.join(dest, rel)
        # THE DAMAGED FILE IS REMOVED FIRST, because _link keeps whatever is already there.
        if os.path.exists(dst):
            os.remove(dst)
        _link(blob, dst)
        replaced += 1
    log(f"[verify] {name}: {replaced:,} of {len(damaged):,} file(s) replaced")
    return replaced


def verify_and_repair(name, log, attempts=2):
    """Check the dataset, re-fetch what is damaged, check again. True when it ends clean."""
    for attempt in range(1, attempts + 1):
        damaged = verify_dataset(name, log)
        if not damaged:
            return True
        if attempt == attempts:
            log(f"[verify] {name}: STILL DAMAGED after {attempts} attempt(s); "
                f"{len(damaged):,} file(s) are not usable")
            return False
        refetch(name, damaged, log)
    return False


def fetch(name, force, log):
    spec = config.DATASETS[name]
    dest = target(name)
    rel = os.path.relpath(dest, config.ROOT)
    done, why = is_complete(name)

    if done and not force:
        log(f"[download] {name}: complete at {rel} "
            f"({size_on_disk(name) / 2**30:.1f} GiB); --force to re-fetch")
        # COMPLETE MEANS THE FILES ARRIVED, NOT THAT THEY READ. The marker records what the
        # transfer returned; damage from an interrupted or badly resumed fetch is only visible
        # by opening them, so a run over an already-complete dataset checks it and re-fetches
        # whatever fails.
        if not verify_and_repair(name, log):
            raise SystemExit(
                f"[download] {name}: damaged files remain after re-fetching them.\n"
                f"           Delete the file(s) named above under {rel} and run this again.")
        return dest

    if is_present(name) and not force and looks_sharded(name):
        # A SHARDED TREE CANNOT BE RE-FETCHED, so it is recorded rather than re-downloaded.
        # Sharding MOVES files out of the layout the hub published, so snapshot_download would
        # fetch every one of them again into the flat layout and leave two copies -- a corpus
        # that scans at twice its size. There is no way to verify it against the hub, and the
        # tree in front of us is the only evidence there is, so it is taken as the answer and
        # written down. A dataset from before completion was recorded lands here exactly once.
        if read_marker(name) is None:
            n, total = write_marker(name, sharded=True)
            log(f"[download] {name}: already split into part_NNNN/ and predates the completion "
                f"record -- taking it as complete ({n:,} files, {_size(total)})")
            return dest
        # It HAD a marker and no longer matches it: something was removed or truncated after
        # the fetch. Re-fetching is still unsafe for the reason above, so say so plainly.
        raise SystemExit(
            f"[download] {name}: {why}\n"
            f"           This dataset is split into part_NNNN/ directories, so it cannot be\n"
            f"           re-fetched in place -- the hub's flat layout would land beside the\n"
            f"           split one and leave two copies of everything.\n"
            f"           Delete {rel} and run this again.")

    if is_present(name) and not force:
        log(f"[download] {name}: incomplete -- {why}")
        log(f"[download] {name}: resuming from {os.path.relpath(HF_CACHE, config.ROOT)}")
    # DEAD STAGING FROM THE OLD local_dir FETCH. Reported here, once, with its size and the
    # command that removes it -- it is never continued by anything and never will be.
    n_dead, b_dead = stale_staging(name)
    if n_dead:
        log(f"[download] {name}: {n_dead:,} orphaned .incomplete files ({_size(b_dead)}) from "
            f"the old local_dir fetch,")
        log(f"[download]   which nothing resumes. Remove with:")
        log(f"[download]   rm -rf {os.path.relpath(target(name), config.ROOT)}"
            f"/.cache/huggingface/download")
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:                                         # noqa: BLE001
        raise SystemExit(
            f"[download] huggingface_hub is not importable ({e}).\n"
            f"           pip install huggingface_hub, or fetch {spec['repo']} by hand into\n"
            f"           {os.path.relpath(dest, config.ROOT)}") from e

    os.makedirs(dest, exist_ok=True)
    os.makedirs(HF_CACHE, exist_ok=True)
    log(f"[download] {name}")
    log(f"[download]   from {spec['repo']}")
    log(f"[download]   via  {os.path.relpath(HF_CACHE, config.ROOT)}  (resumable; hard-linked "
        f"into place after)")
    log(f"[download]   into {os.path.relpath(dest, config.ROOT)}")
    log(f"[download]   {free_space(dest) / 2**30:.1f} GiB free on that filesystem")
    report_progress(name, spec["repo"], dest, log, spec.get("type", "dataset"),
                    spec.get("allow_patterns"))
    # THE TWO BARS BELOW ARE THE LIBRARY'S AND MEAN SOMETHING NARROWER THAN THEY LOOK. Said
    # here, once, because the alternative is a reader concluding a resumed fetch has restarted:
    # "Fetching N files" counts what THIS session links, so it opens at 0 with 8 files already
    # on disk, and the byte bar totals only what this session decided to transfer.
    log(f"[download]   the bars below are huggingface_hub's and count THIS SESSION only;")
    log(f"[download]   the PROGRESS line above is the whole dataset")
    try:
        # NO local_dir=. That is the whole fix: with local_dir the library writes each attempt
        # to a uniquely named temporary file and resumes nothing, so an interrupted transfer
        # restarts from zero and leaves its partial behind. Into the cache, a partial file is
        # an .incomplete keyed by the blob's hash and the next run CONTINUES IT.
        # THE NAMESPACE COMES FROM THE ENTRY. A tokenizer is published as a MODEL repository,
        # and asking /api/datasets/ for one is a 404 that names the repository rather than the
        # mistake -- which is exactly how tokenizer-bpe-sft-coder read before this.
        # allow_patterns NARROWS THE FETCH TO PART OF A REPOSITORY, and it is the only way some
        # datasets are usable at all: cais/mmlu publishes all fifty-seven subjects, and this
        # pipeline scores five of them. Absent, the argument is None and snapshot_download takes
        # the whole repository exactly as before, so every other entry is unchanged.
        _patterns = spec.get("allow_patterns")
        snapshot = snapshot_download(repo_id=spec["repo"], repo_type=spec.get("type", "dataset"),
                                     cache_dir=HF_CACHE,
                                     allow_patterns=([g.strip() for g in _patterns.split(",")
                                                      if g.strip()] if _patterns else None))
    except Exception as e:                                         # noqa: BLE001
        # Left partially fetched ON PURPOSE: the cache resumes, so the right response to a
        # failure is to run this again, and deleting the partial would throw away everything
        # already transferred.
        raise SystemExit(
            f"[download] {name} failed: {type(e).__name__}: {e}\n"
            f"           Whatever transferred is kept in "
            f"{os.path.relpath(HF_CACHE, config.ROOT)};\n"
            f"           re-run this command to resume from there.") from e
    materialise(snapshot, dest, log)
    log(f"[download] {name}: transferred, {size_on_disk(name) / 2**30:.1f} GiB")
    # CHECKED BEFORE THE MARKER IS WRITTEN, so a dataset is never recorded complete while
    # holding a file that does not read.
    if not verify_and_repair(name, log):
        raise SystemExit(
            f"[download] {name}: damaged files remain after re-fetching them; the completion "
            f"marker was NOT written.\n"
            f"           Delete the file(s) named above under {rel} and run this again.")
    # THE MARKER IS WRITTEN THE MOMENT THE FETCH RETURNS. Nothing rearranges the corpus after
    # this point, so the tree the marker describes is the tree that stays on disk.
    write_marker(name, sharded=False)
    n, total = state(name)
    log(f"[download] {name}: COMPLETE -- {n:,} files, {_size(total)}, recorded in "
        f"{MARKER}")
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description="fetch the datasets into data/download/")
    ap.add_argument("--only", action="append", choices=list(config.DATASETS),
                    help="fetch just this dataset; repeatable")
    ap.add_argument("--list", action="store_true", help="show what would be fetched, then stop")
    ap.add_argument("--force", action="store_true", help="re-fetch even if already complete")
    # ACCEPTED AND DELIBERATELY UNUSED. Every stage launcher ends
    # `exec $PY -m <module> "$@" --config "$CONFIG"`, because `run_scripts/stage_config.sh`
    # resolves one settings file and hands it to the stage. This stage is the one that must not
    # read it: WHAT gets fetched is `config/download/config_download.yaml`, one mapping per dataset, and
    # it is the same list whatever scheme or stage the caller happens to be configured for. A
    # download stage that narrowed its list from the current scheme would make a fresh checkout
    # unreproducible -- the standing rule is that stage 1 fetches everything in that file, full
    # stop -- so the flag is swallowed here rather than acted on.
    #
    # WITHOUT THIS, ARGPARSE REJECTED IT AND STAGE 1 COULD NOT RUN AT ALL. Not `--list`, not a
    # plain fetch, not `--force`: `run_scripts/run_download_data.sh` appends `--config` unconditionally,
    # so every invocation died on "unrecognized arguments" before a single byte moved.
    ap.add_argument("--config", default=None,
                    help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    def log(m): print(m, flush=True)

    describe(log)
    if a.list:
        return
    for name in (a.only or list(config.DATASETS)):
        fetch(name, a.force, log)
    log("")
    describe(log)
    log("[download] the trainers read these directories; nothing else fetches.")


if __name__ == "__main__":
    main()
