"""
tools/txt_to_parquet.py -- a directory of per-document .txt files -> sharded parquet.

    python -m tools.txt_to_parquet data/download/llm-pretrain-wikitext103
    python -m tools.txt_to_parquet <src> --out <dir> --max_mb 100 --dry_run

WHY PARQUET. A corpus of tens of thousands of small text files is slow to list, slow to
open (one syscall pair per document) and unpleasant to move between machines. The same
corpus as a handful of compressed columnar files is read in a few large sequential reads,
travels as four files instead of thirty thousand, and is what `helpers.load_token_corpus`
already prefers -- `_parquet_lines` feeds parquet through the same packer as text, so
nothing downstream changes.

WHY A SIZE CAP. Shards exist so the corpus can be read in parallel, resumed after a failed
transfer, and held in memory one piece at a time. A cap of ~100 MB is the usual compromise:
small enough that a single shard is cheap to re-fetch and to buffer, large enough that the
per-file overhead and the loss of compression across a small block do not matter. Files are
rolled over by MEASURING the file on disk after each row group rather than by predicting
its compressed size, because the compression ratio of prose varies enough (here ~2.6x) that
a prediction would be wrong in one direction or the other.

NAMING. The output follows the convention the Hugging Face Hub infers splits from:

    <split>-<NNNNN>-of-<NNNNN>.parquet        e.g. train-00000-of-00004.parquet

so `load_dataset(<repo>)` and the dataset viewer both work with no configuration file. The
shard count is in every name, which is what makes a missing shard visible: `train-00002-of-
00004` alone in a directory is obviously incomplete, `train_2` is not. Split directories
named `val` or `valid` are written as `validation`, which is the name the Hub expects.

COLUMNS. `text` is the document (config.PRETRAIN["text_column"]), `source` its original
file name, `id` its ordinal within the split. The source name is kept rather than parsed
into a title: these files are named `000123_Kiss-You-One-Direction-song.txt`, and turning
that back into "Kiss You (One Direction song)" is guesswork that would silently corrupt
every title containing a real hyphen.
"""

# THE REPOSITORY ROOT ON sys.path, BECAUSE THIS IS RUN AS A SCRIPT. `python tools/<name>.py`
# puts tools/ on sys.path[0], not the root, so a first-party import raises ModuleNotFoundError.
# Found by searching upward for the directory that contains model/ rather than by counting
# parents: counting breaks silently when a file moves.
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "model").is_dir():
    _ROOT = _ROOT.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import os
import sys

# HF split names. A directory called val/ or valid/ becomes "validation": the Hub, datasets
# and every published corpus use that spelling, and a split it does not recognise is simply
# not offered to the reader.
SPLIT_NAMES = {"train": "train", "valid": "validation", "val": "validation",
               "validation": "validation", "test": "test", "dev": "validation"}


def _pq():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        return pa, pq
    except ImportError as e:                                   # noqa: BLE001
        raise SystemExit(f"[parquet] pyarrow is required ({e}).\n"
                         f"          pip install pyarrow") from e


def text_files(split_dir, ext=".txt"):
    """Every .txt under a split directory, recursively, sorted -- so a re-run shards the
    corpus identically instead of reshuffling it into different files."""
    out = []
    for dirpath, dirnames, filenames in os.walk(split_dir):
        dirnames.sort()
        for fn in sorted(filenames):
            if fn.lower().endswith(ext):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def convert_split(files, out_dir, split, max_bytes, rows_per_group=200,
                  compression="zstd", root=None, log=print, dry_run=False):
    """Write `files` as <split>-<i>-of-<n>.parquet shards of at most `max_bytes` each.

    Shards are written under their FINAL-ish name and given the `-of-<n>` suffix at the end,
    because that suffix cannot be known until the last document is written. What is not done
    is holding them under a .part name meanwhile: a conversion killed after nine shards would
    then leave nine files nothing can read, and at this scale an interrupted run must leave
    every byte it produced usable. A shard without the suffix is a complete, readable parquet
    file; only its name is provisional."""
    pa, pq = _pq()
    schema = pa.schema([("id", pa.int32()), ("source", pa.string()), ("text", pa.string())])
    parts, writer, path = [], None, None
    n_rows = n_chars = 0
    buf_id, buf_src, buf_txt = [], [], []

    def flush():
        """One row group, then check whether this shard has reached the cap."""
        nonlocal writer, path, buf_id, buf_src, buf_txt
        if not buf_id:
            return
        writer.write_table(pa.Table.from_arrays(
            [pa.array(buf_id, pa.int32()), pa.array(buf_src, pa.string()),
             pa.array(buf_txt, pa.string())], schema=schema))
        buf_id, buf_src, buf_txt = [], [], []

    def close():
        nonlocal writer, path
        if writer is not None:
            writer.close(); writer = None
            log(f"    {os.path.basename(path)}  {os.path.getsize(path) / 1048576:.1f} MB")

    from helpers import progress
    for i, fp in enumerate(progress(files, desc=f"[parquet] {split}", total=len(files))):
        try:
            text = open(fp, "r", encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if not text.strip():
            continue
        if dry_run:
            n_rows += 1; n_chars += len(text)
            continue
        if writer is None:
            path = os.path.join(out_dir, f"{split}-{len(parts):05d}.parquet")
            parts.append(path)
            writer = pq.ParquetWriter(path, schema, compression=compression)
        buf_id.append(n_rows); buf_src.append(os.path.relpath(fp, root or os.path.dirname(fp)))
        buf_txt.append(text)
        n_rows += 1; n_chars += len(text)
        if len(buf_id) >= rows_per_group:
            flush()
            # measured, not predicted: the footer is still to come, so the check leaves the
            # last row group's worth of headroom rather than trusting an estimate
            if os.path.getsize(path) >= max_bytes * 0.97:
                close()
    if not dry_run:
        flush(); close()
        n = len(parts)
        final = []
        for i, p in enumerate(parts):
            dst = os.path.join(out_dir, f"{split}-{i:05d}-of-{n:05d}.parquet")
            os.replace(p, dst); final.append(dst)
        parts = final
    return parts, n_rows, n_chars


def main(argv=None):
    ap = argparse.ArgumentParser(description="per-document .txt -> sharded parquet")
    ap.add_argument("src", help="corpus root holding train/ valid/ test/ subdirectories")
    ap.add_argument("--out", default="", help="output directory (default: the source root)")
    ap.add_argument("--max_mb", type=float, default=100.0, help="cap per shard, MB")
    ap.add_argument("--rows_per_group", type=int, default=200)
    ap.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip"])
    ap.add_argument("--ext", default=".txt")
    ap.add_argument("--dry_run", action="store_true", help="count only; write nothing")
    a = ap.parse_args(argv)

    src = os.path.abspath(a.src)
    out_dir = os.path.abspath(a.out) if a.out else src
    if not os.path.isdir(src):
        raise SystemExit(f"[parquet] not a directory: {src}")
    splits = [d for d in sorted(os.listdir(src))
              if os.path.isdir(os.path.join(src, d)) and d.lower() in SPLIT_NAMES]
    if not splits:
        raise SystemExit(f"[parquet] no split subdirectories under {src}\n"
                         f"          expected one or more of: "
                         f"{', '.join(sorted(set(SPLIT_NAMES)))}")
    os.makedirs(out_dir, exist_ok=True)

    import helpers
    rows = []
    for d in splits:
        split = SPLIT_NAMES[d.lower()]
        files = text_files(os.path.join(src, d), a.ext)
        print(f"[parquet] {d}/ -> {split}: {len(files):,} files", flush=True)
        parts, n, chars = convert_split(
            files, out_dir, split, int(a.max_mb * 1048576), a.rows_per_group,
            a.compression, root=src, dry_run=a.dry_run)
        size = sum(os.path.getsize(p) for p in parts)
        rows += [(f"{split} documents", helpers.count(n)),
                 (f"{split} characters", helpers.count(chars)),
                 (f"{split} shards", f"{len(parts)}  ({size / 1048576:.1f} MB)"
                  if parts else "(dry run)")]
    helpers.table("txt -> parquet", rows + [("output", out_dir),
                                            ("cap per shard", f"{a.max_mb:g} MB"),
                                            ("compression", a.compression)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
