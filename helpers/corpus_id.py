"""
corpus_id.py -- what identifies a corpus, and NOTHING THAT NEEDS A GPU.

    corpus_files(root, exclude_dirs)          every corpus file under a directory
    corpus_name(root)                         the dataset name: the directory's last component
    corpus_signature(tok, root, files, ...)   the portable identity of a corpus + tokenizer
    _sig_tag(sig)                             the eight hex characters that go in a filename

SPLIT OUT OF helpers/utils.py SO IT CAN BE IMPORTED WITHOUT TORCH. These functions are pure
path and hash arithmetic -- they were always torch-free -- but they lived in a module that
imports torch, model and the DPO stage at the top, so anything wanting a corpus signature had
to load a deep-learning stack to get one. tools/rename_dataset.py wants exactly that and
nothing else: it renames directories and rewrites a manifest, work for a login node, and it
should not need the GPU virtualenv to do it.

helpers/utils.py imports these names, so THERE IS STILL ONE IMPLEMENTATION. That matters more
here than anywhere: a signature computed two ways is two answers to "are these the tokens of
this corpus?", and the day they disagree is the day a finished corpus is silently tokenised
again.
"""
import hashlib
import os
import re

import default_config as config

def _split_token(name):
    """The leading name token of a file: `validation-00000-of-00001.parquet` -> "validation",
    `test_0003.txt` -> "test", `doc.0.parquet` -> "doc"."""
    stem = os.path.splitext(name)[0]
    for sep in ("-", "_", "."):
        stem = stem.split(sep)[0]
    return stem.lower()


def corpus_files(root, exclude_dirs=(), extensions=None):
    """Every corpus file under `root`, recursively: any file whose extension is in
    `extensions` (config.CORPUS_EXTENSIONS by default -- prose, markdown AND source code),
    skipping anything excluded by `exclude_dirs`. Sorted, so a run is reproducible.

    An entry in `exclude_dirs` excludes a DIRECTORY of that name and also any FILE whose
    leading name token matches it. Both, because a corpus keeps its held-out split in whichever
    of the two shapes its format favours: thirty thousand text files are laid out as valid/ and
    test/ subdirectories, while the same corpus as parquet is a handful of files named
    validation-00000-of-00001.parquet in one flat directory -- the layout the Hugging Face Hub
    infers splits from. A directories-only rule silently pretrains on the test split the moment
    a corpus is converted, and a model evaluated on text it was trained on reports a perplexity
    that means nothing."""
    exts = {("." + e.lower().lstrip(".")) for e in (extensions or config.CORPUS_EXTENSIONS)}
    excl = {d.lower() for d in (exclude_dirs or ())}
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        # DOTFILES AND DOT-DIRECTORIES ARE NOT CORPUS, whatever they are named. Three things
        # land in a corpus directory that end in a corpus extension and are not text:
        #
        #   ._corpus_00001.parquet     an AppleDouble stub, written beside every file when the
        #                              tree is copied from or through a Mac -- binary, and a
        #                              duplicate of a name already counted
        #   .mira_llm_download.json     this pipeline's own completion marker
        #   .cache/huggingface/...     what snapshot_download keeps to make resume work
        #
        # Counting any of them changes the file count and the byte total, and those two ARE the
        # corpus signature -- so a marker written beside a corpus would have silently
        # invalidated every token stream built from it. A cache that took hours, discarded by a
        # bookkeeping file.
        dirnames[:] = [d for d in dirnames if d.lower() not in excl and not d.startswith(".")]
        for fn in filenames:
            if fn.startswith("."):
                continue
            if os.path.splitext(fn)[1].lower() in exts and _split_token(fn) not in excl:
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def corpus_signature(tok, root, files, max_words, text_column):
    """Identifies a whole corpus AND the tokenizer and packing applied to it.

      MODIFICATION TIMES. An mtime records when a file last arrived on THIS filesystem, not
      what is in it. It moves when a corpus is rsynced, re-downloaded, restored from a backup,
      copied between nodes or checked out again -- none of which changes a byte -- so the same
      corpus on two servers signed differently, and a finished token stream became invisible to
      the machine that was about to use it.

      PATHS, absolute or relative. A dataset under a scratch filesystem, the same dataset in
      a home directory and the same dataset on a workstation are one corpus. What identifies
      it is its NAME -- the dataset directory,
      llm-pretrain-fineweb-edu-2BT -- which is the name it was published
      under and travels with it. Where a given machine chose to put it is that machine's
      business and belongs nowhere near an identity.

    So: the dataset name, the packing, and two numbers that describe the corpus rather than the
    host -- how many files it has and how many bytes they hold. Those cost one stat each, are
    identical wherever the corpus is copied, and still move when a file is added, removed,
    truncated or rewritten. Content hashing would be stricter, and reading 10 TB to decide
    whether 10 TB may be skipped defeats the purpose.

    The result is HASHED rather than stored, so a corpus of thirty thousand files does not put
    a megabyte of file listing in the header of every stream."""
    import hashlib
    total = 0
    for fp in files:
        try:
            total += os.path.getsize(fp)
        except OSError:
            pass
    h = hashlib.sha256()
    h.update(f"{corpus_name(root)}|{_tok_signature(tok, max_words, text_column)}"
             f"|files={len(files)}|bytes={total}".encode("utf-8"))
    return f"corpus|v2|files={len(files)}|{h.hexdigest()[:32]}"


def corpus_name(root):
    """THE DATASET NAME, and nothing else: the last component of the corpus directory.

    `llm-pretrain-fineweb-edu-2BT`. That is what the dataset is called, it is
    the name it was published under, and it is the same on every machine that has a copy. No
    part of the path above it is used: a scratch filesystem, a home directory and a
    workstation may each hold the same corpus, and all three must reach the same cache."""
    return os.path.basename(os.path.normpath(os.path.abspath(root))) or "corpus"


def _sig_tag(sig):
    """Eight hex characters of the signature, for the file name."""
    import hashlib
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:8]


def _tok_signature(tok, max_words, text_column=""):
    """Identifies the tokenizer and the packing; any change invalidates every entry. The
    parquet text column is part of it: reading a different column is a different corpus, and a
    cache that survived that change would train on tokens nobody asked for.

    THE SIZE COUNTED IS THE ENCODING, 256 + merges, NOT len(tok). len(tok) includes the
    special tokens, so it rises the moment <|im_start|> is registered -- and a corpus that has
    not changed by a single token would then sign differently and be tokenised all over again.
    Registering a special after pretraining must leave every cached stream valid; that is the
    entire point of being able to register one, it is why tokenizer_tag counts the same way,
    and encoding_blob explains why the ids do not move."""
    n_merges = len(getattr(tok, "merges", []) or [])
    n_base = 256 + n_merges if hasattr(tok, "merges") else len(tok)
    return (f"v2|encoding={n_base}|merges={n_merges}|max_words={int(max_words)}"
            f"|col={text_column}")
