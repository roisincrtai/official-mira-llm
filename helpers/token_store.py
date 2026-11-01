"""
token_store.py -- the pre-tokenised corpus as a set of memory-mapped SHARD FILES.

    build(stem, sig, documents, eos, vocab, log)   write it, appending as it goes
    TokenStream(stem)                              open it; nothing is read into memory
    stream.batch(B, T, generator, device)          (ids, attn, rmask); attn is None -- see batch()

RESUME IS THE DEFAULT, AND IT COSTS NOTHING. A build finds the manifest, truncates the last
shard back to the last recorded position (the bytes after it were written but not yet
accounted for), and continues from the document after the last one counted. This is sound
because the document order is deterministic -- the corpus files are sorted, each is packed in
order -- so document N is always the same document. Pass resume=False to start again.

What makes it cost nothing is the CURSOR. The manifest carries an opaque position handed back
to the producer, which uses it to seek: files already consumed are never opened, and documents
already stored are never tokenised again. Resuming by counting instead -- reading the corpus
from the top and discarding the first N documents -- runs the tokenizer over work already on
disk, which at 10 TB is days of a GPU-free machine spent producing bytes it already has. The
rule is that the cost of resuming is proportional to what was LOST, not to what was done.

The cursor is written by the producer and never interpreted here; the store only persists it
next to the counts it belongs with, and only at the moment those counts are written, so the
position and the shard contents can never disagree. A manifest from before the cursor existed
resumes by count instead, which is slower but still correct.

    <corpus dir>/<name>_<sig>_00000.tokens     shard 0: RAW tokens, nothing else
    <corpus dir>/<name>_<sig>_00001.tokens     shard 1
    <corpus dir>/<name>_<sig>_index.json       the manifest: dtype, eos, per-shard counts

A shard is raw tokens with no header, which is what makes appending to it trivial and makes a
partial file a valid prefix of a whole one rather than a corrupt file. Everything a reader
needs is in the manifest; document boundaries are recovered from the EOS separators, lazily,
per shard, and only for the shards a caller actually asks about.

WHY SHARDS. A corpus can be 10 TB, and one file that size cannot be resumed, cannot be copied
incrementally, and loses everything to a single bad byte. ~100 MB pieces are re-fetchable,
verifiable one at a time, and readable in parallel.

WHY MEMORY-MAPPED. The corpus was once a Python list of lists of ints: a 28-byte object per
token plus 8 bytes of list slot, so 2 billion tokens would want ~70 GB of RAM to hold 4 GB of
data. Flat arrays paged in by the OS make resident memory a function of the batch, not of the
corpus -- the same choice as GPT-2's and nanoGPT's .bin, Megatron's indexed dataset and the
mmap'd shards of GPT-NeoX.

uint16 WHEN THE VOCABULARY ALLOWS IT. 50,259 tokens fit in 16 bits, so each costs 2 bytes
rather than 4 -- half the file, half the page cache, half the bytes per step. The dtype is
chosen from the vocabulary size and recorded, so a larger vocabulary widens to uint32 instead
of wrapping around into a silent corruption.

CONTIGUOUS, EOS-SEPARATED sampling, as in GPT-2, nanoGPT and Megatron: documents concatenated
with an end-of-sequence token between them, a sample being T+1 tokens from a random offset.
No padding, so every position carries a gradient. A window that begins near the end of a shard
is completed from the head of the next, so the shard size is invisible to training -- drawing
a shard first and then an offset inside it would mean the last T positions of every shard could
never begin a window, a bias nothing downstream could detect.

DATA PARALLELISM IS THE ONE EXCEPTION TO THAT LAST SENTENCE, and it is deliberate. With
`regions` > 1 the shard FILES are shuffled once and dealt out, one set per GPU, and a row is
drawn by choosing one of that device's files and then an offset inside it -- exactly the
shard-first order the paragraph above warns about, and with exactly the bias it names: the last
T tokens of each shard never BEGIN a window. It is accepted because the alternative is worse.
Completing a window from the head of the next shard would read a file belonging to a different
device, and the whole point of the deal is that the sets are disjoint.

The cost is bounded and small: T of ~52M tokens per shard, 0.016% at T = 8,192, and NO TOKEN
BECOMES UNREACHABLE -- every one of them is still reached from an earlier start inside the same
shard. Only the set of starting positions loses its last T entries. See batch() and
shard_regions().
"""
import glob
import hashlib
import json
import os
import time

import default_config as config

# THE MARKER IS THE PROJECT'S OWN NAME AND STAYS THAT WAY. It says which tree wrote this manifest.
#
# EVERY VALUE EVER WRITTEN IS STILL ACCEPTED, which is the part that was missing. The marker has
# been "nullgpt-tokens" and "miragpt-tokens" before this one, and each rename left every manifest
# already on disk unreadable by the tree that had just been renamed:
#
#     ValueError: ..._index.json is not a mira_llm-tokens manifest
#
# on a complete 2,068,028,808-token stream in 40 shards that had not changed by a byte. LEGACY_MAGIC
# is now a TUPLE of every historical value rather than a single one, so a rename costs nothing
# retroactively: the reader takes any of them, the writer emits the current one, and
# tools/convert_tokens.py brings an old manifest forward when uniformity is wanted.
MAGIC = "mira_llm-tokens"
LEGACY_MAGIC = ("nullgpt-tokens", "miragpt-tokens", "mirallm-tokens")
VERSION = 4                              # 4 adds "cursor"; a v3 manifest still resumes, by count
SHARD_BYTES = 100 * 1024 * 1024          # ~100 MB of tokens per shard
FLUSH_SECONDS = 5.0                      # how often the manifest catches up with the file


def dtype_for(vocab_size):
    """The narrowest integer type that can hold every id. numpy names, stored in the manifest."""
    return "uint16" if int(vocab_size) <= 65535 else "uint32"


def _itemsize(dtype):
    return 2 if dtype == "uint16" else 4


def _np():
    try:
        import numpy as np
        return np
    except ImportError as e:                                   # noqa: BLE001
        raise SystemExit(f"[tokens] numpy is required for the token store ({e}).\n"
                         f"         pip install numpy") from e


# `stem` throughout is a FILE STEM, not a directory: <corpus dir>/<name>_<sig>
def index_path(stem):
    return f"{stem}_index.json"


def shard_path(stem, i):
    return f"{stem}_{i:05d}.tokens"


MD5_CHUNK = 64 << 20            # 64 MB: the unit a digest covers, and the unit repair keeps


def chunk_md5s(path, chunk=MD5_CHUNK, on_bytes=None):
    """A LIST of md5s, one per `chunk` bytes, the last covering whatever is left.

    Per chunk, not per file, because one digest over a whole 100 MB shard can only say
    "something in here is wrong". Per chunk it says WHERE: every chunk before the first bad one
    is known good, so a repair keeps them and re-tokenises from that boundary instead of
    discarding the shard. The final chunk is short whenever the file does not divide evenly and
    is hashed at whatever length it has."""
    out = []
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            out.append(hashlib.md5(block).hexdigest())
            if on_bytes is not None:
                on_bytes(len(block))          # a 20 GB corpus must not verify in silence
    return out


def _as_list(md5):
    """Tolerate a bare string: the one-digest-per-file form is a single chunk."""
    if not md5:
        return []
    return [md5] if isinstance(md5, str) else list(md5)


def first_damaged(stem, shards, dtype, check_md5=False, chunk=MD5_CHUNK, on_bytes=None):
    """(shard index, what is wrong, bytes known good) for the first damaged shard, else None.

    A manifest can get ahead of its data: the bytes were buffered, the counts were written, and
    the process died before the buffer reached the disk. Nothing downstream detects that --
    numpy just refuses to map the shard, naming neither the file nor the shortfall -- so it is
    checked in one place and acted on in two: readers refuse the stream, and build() repairs it.

    Only SHORT counts as damage by length. A file LONGER than its count is an uncounted tail,
    the normal state of an interrupted build, and is trimmed on resume.

    `check_md5` also verifies CONTENT, chunk by chunk. Length catches the tail a killed writer
    loses and nothing else: a shard whose middle was rewritten, or truncated and refilled to the
    same size, passes a length check and then trains on wrong tokens.

    THE THIRD ELEMENT IS WHAT MAKES REPAIR CHEAP -- the offset up to which the shard is known
    good: the start of the first bad chunk, or the length on disk when only the tail was lost.
    Shards written before digests were recorded carry none and are length-checked only."""
    isz = _itemsize(dtype)
    for i, sh in enumerate(shards):
        p = shard_path(stem, i)
        want = int(sh["n_tokens"]) * isz
        have = os.path.getsize(p) if os.path.exists(p) else 0
        if have < want:
            return i, (f"{have:,} bytes on disk against {want:,} counted "
                       f"({(want - have) / 1048576:.1f} MB missing)"), have
        recorded = _as_list(sh.get("md5"))
        if check_md5 and recorded:
            got = chunk_md5s(p, chunk, on_bytes)
            for c in range(min(len(got), len(recorded))):
                if got[c] != recorded[c]:
                    return i, (f"chunk {c} (bytes {c * chunk:,}..{min((c + 1) * chunk, have):,}) "
                               f"md5 {got[c]} != recorded {recorded[c]}"), c * chunk
            if len(got) != len(recorded):
                return i, (f"{len(got)} chunk(s) on disk against {len(recorded)} recorded"), \
                       min(len(got), len(recorded)) * chunk
    return None


def _read_index(stem):
    try:
        with open(index_path(stem), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_index(stem, sig, dtype, eos, vocab_size, shards, complete, cursor=None, packing=""):
    """Rewrite the manifest, through a .part rename so a reader never sees half of it.

    `cursor` is the producer's position AT THE MOMENT these counts were taken. The two are
    written together, in one file, by one rename: a cursor that could land in the manifest
    without the shard counts it belongs to would resume from the wrong document, and the
    corpus would silently gain or lose a stretch of text that nothing downstream could see."""
    from .utils import input_only
    input_only(stem, "write a token manifest")
    tmp = index_path(stem) + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"magic": MAGIC, "version": VERSION, "sig": sig, "dtype": dtype,
                   "eos": int(eos), "vocab_size": int(vocab_size), "complete": bool(complete),
                   "n_tokens": sum(s["n_tokens"] for s in shards),
                   "n_docs": sum(s["n_docs"] for s in shards),
                   "cursor": cursor,
                   # WHAT PACKING PRODUCED THESE TOKENS, in words rather than as a hash. `sig`
                   # decides whether a stream may be reused; this says WHY when it may not, so
                   # a stage about to retokenise a corpus it already tokenised can name the
                   # setting that differs instead of starting silently from zero.
                   "packing": packing,
                   "shards": shards}, f, indent=1)
    os.replace(tmp, index_path(stem))


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
def masks_path(stem, i):
    """The response-mask sidecar of shard `i`: `<stem>_00011.masks` beside `<stem>_00011.tokens`.

    A SIDECAR, BECAUSE THE BINARY LAYOUT IS SHARED. Pretraining, the benchmark and the ablation
    all read `.tokens`, so a new requirement may not change what is in one. The mask is a
    separate file, written only for a conversation corpus, and a reader that does not know about
    it is unaffected by its existence."""
    return f"{stem}_{i:05d}.masks"


def read_masks(stem, i):
    """Shard `i`'s mask as a list of span lists, one entry per document, in document order.

    Each entry is [[offset, length], ...] in tokens RELATIVE TO THE DOCUMENT'S FIRST TOKEN, so
    the file says nothing about where the document landed and stays valid however the shards are
    read. `[]` means the document has no response in it. Returns None when the sidecar is
    absent, which is what every non-conversation corpus looks like."""
    path = masks_path(stem, i)
    if not os.path.exists(path):
        return None
    out = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line).get("spans") or [])
    return out


def _cut_masks(stem, i, keep, log=print):
    """Cut shard `i`'s sidecar back to its first `keep` documents, in step with the shard.

    THE TWO FILES ARE TRUNCATED TOGETHER OR NOT AT ALL. A `.masks` longer than its shard would
    give document j the spans of some other document from the moment of the first repair, which
    is a mask that is wrong rather than missing -- the loss would land on the user's words and
    nothing would say so."""
    path = masks_path(stem, i)
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    if len(lines) <= keep:
        return
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.writelines(lines[:keep])
    os.replace(tmp, path)
    log(f"[tokens] {os.path.basename(path)} cut to {keep:,} document(s) with its shard")


def build(stem, sig, documents, eos, vocab_size, log=print, shard_bytes=SHARD_BYTES,
          resume=True, flush_seconds=FLUSH_SECONDS, packing="", masks=False):
    """Append documents to the shards at `stem`. Returns (n_tokens, n_docs).

    `documents` is a PRODUCER FACTORY: `documents(cursor, skip_docs)` returns an iterator of
    `(ids, cursor)` or `(ids, cursor, spans)`, having already positioned itself. It is called
    once, here, with whatever the manifest recorded:

        cursor is not None    seek to it; nothing before it is read or tokenised
        cursor is None,       an older manifest: read from the top but do not TOKENISE the
        skip_docs > 0         first `skip_docs` documents -- still the expensive half saved
        neither               a fresh build

    The producer yields a pair for EVERY document it packs, including one that tokenises to
    nothing. Empty ids are written as nothing but still advance the cursor, so the recorded
    position counts packed documents while the manifest's n_docs counts written ones, and the
    two are never required to agree -- an equality that held by luck would be a desynchronised
    resume the first time it did not.

    Every document is written as it arrives; the manifest, and the cursor with it, catch up
    every `flush_seconds`. A plain iterable is still accepted, and resumes by count."""
    import array
    from .utils import input_only
    # BEFORE A DIRECTORY IS CREATED OR A BYTE WRITTEN. Everything below writes, truncates and
    # deletes relative to `stem`, so one check here covers the shards, the manifest, the
    # trimming of an uncounted tail and the discarding of stale shards. The corpora in data/
    # are the input; a token stream that tried to build itself among them would be writing
    # into the one tree that cannot be regenerated.
    input_only(stem, "build a token stream")
    if os.environ.get("MIRALLM_STAGE") != "tokenize_data":
        raise PermissionError("Only run_scripts/run_tokenize_data.sh may write token streams")
    dtype = dtype_for(vocab_size)
    isz = _itemsize(dtype)
    typecode = "H" if isz == 2 else "I"
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)

    idx = _read_index(stem)
    stale = idx is not None and idx.get("sig") != sig
    if stale or not resume:
        if idx is not None:
            why = "a different corpus or packing" if stale else "--no-resume"
            log(f"[tokens] discarding the existing shards ({why})")
        # THE SIDECARS GO WITH THE SHARDS THEY DESCRIBE. A `.masks` left behind from a discarded
        # build would be appended to by the new one, and line j would then describe a document
        # from the previous corpus.
        for f in (glob.glob(f"{stem}_*.tokens") + glob.glob(f"{stem}_*.part")
                  + glob.glob(f"{stem}_*.masks")):
            try:
                os.remove(f)
            except OSError:
                pass
        idx = None

    shards = list(idx["shards"]) if idx else []
    # THE SAME DEPTH OF CHECK THE READER WILL APPLY. Checking length only here while
    # TokenStream checks content meant a corrupt chunk was reported and never repaired:
    # build returned "already complete", the reader refused the stream, and the two
    # disagreed forever. TOKENS["verify"] decides once, for both.
    _mode = str(getattr(config, "TOKENS", {}).get("verify", "md5")).lower()
    bad = (first_damaged(stem, shards, dtype, check_md5=(_mode == "md5"))
           if shards and _mode != "off" else None)
    if idx and idx.get("complete") and bad is None:
        log(f"[tokens] already complete: {idx['n_tokens']:,} tokens in {len(shards)} shard(s)")
        return idx["n_tokens"], idx["n_docs"]

    # DAMAGE IS REPAIRED FROM THE DAMAGED SHARD ONWARD, never from zero. Everything before it is
    # intact and already tokenised, and re-reading a 10 GB corpus to reproduce the same bytes
    # would cost hours. The damaged shard is cut back to its last COMPLETE document -- documents
    # end in EOS, so that boundary is recoverable from the data itself -- every shard after it is
    # emptied, and tokenising resumes from there.
    #
    # `complete` IS NOT TRUSTED WHEN A SHARD IS SHORT. It was written by a build that believed
    # the bytes were on disk; honouring it would return here immediately and leave the reader
    # refusing the same stream forever.
    if bad is not None:
        i, why, good = bad
        npmod = _np()
        p = shard_path(stem, i)
        # EVERYTHING BEFORE THE FIRST BAD CHUNK IS KNOWN GOOD, so the scan for the last complete
        # document runs over that prefix alone. On a 100 MB shard whose last chunk went bad that
        # is the difference between keeping 64 MB of tokens and discarding the lot.
        ends = []
        if good:
            got = npmod.memmap(p, dtype=dtype, mode="r", shape=(good // isz,))
            ends = npmod.flatnonzero(npmod.asarray(got) == eos)
            del got
        keep = int(ends[-1]) + 1 if len(ends) else 0
        log(f"[tokens] DAMAGED {os.path.basename(p)}: {why}. Good to byte {good:,}; cutting to "
            f"the last complete document within it ({keep:,} tokens, {len(ends):,} documents) "
            f"and re-tokenising from there. The {i} shard(s) before it are kept.")
        with open(p, "r+b") as f:
            f.truncate(keep * isz)
        _cut_masks(stem, i, int(len(ends)), log)
        shards[i]["n_tokens"], shards[i]["n_docs"] = keep, int(len(ends))
        shards[i].pop("md5", None)      # truncated: the recorded digest describes the old bytes
        # Shards after the damaged one describe documents that will be produced again, so they
        # leave the manifest and their files are emptied -- open_shard() APPENDS, and appending
        # to a stale shard would splice two runs together. Their sidecars are emptied with them,
        # for the same reason and in the same step.
        for j in range(i + 1, len(shards)):
            q = shard_path(stem, j)
            if os.path.exists(q):
                with open(q, "r+b") as f:
                    f.truncate(0)
            _cut_masks(stem, j, 0, log)
        del shards[i + 1:]
        idx = dict(idx or {}, complete=False, cursor=None)

    # RESUME. The last shard may hold bytes written after the manifest was last updated; they
    # belong to documents the manifest does not count, so they are cut off. Truncating to a
    # recorded boundary is what makes "written" and "counted" agree again, and it is what lets
    # the cursor beside those counts be trusted.
    skip_docs = sum(s["n_docs"] for s in shards)
    cursor = idx.get("cursor") if idx else None
    if shards:
        last = shard_path(stem, len(shards) - 1)
        want = shards[-1]["n_tokens"] * isz
        have = os.path.getsize(last) if os.path.exists(last) else 0
        if have > want:
            with open(last, "r+b") as f:
                f.truncate(want)
            log(f"[tokens] trimmed {(have - want) / 1048576:.1f} MB of uncounted tail from "
                f"{os.path.basename(last)}")
        # THE SIDECAR IS TRIMMED TO THE SAME COUNT, ALWAYS, not only when the shard was. A
        # manifest flush and a `.masks` write are not one atomic act, so the sidecar can carry
        # lines for documents the manifest never counted; the shard's own recorded n_docs is the
        # one number both files are made to agree with.
        _cut_masks(stem, len(shards) - 1, int(shards[-1]["n_docs"]), log)
        # A SHORT file cannot reach here: first_damaged() repaired it above.
        where = "seeking to the recorded position" if cursor else \
                "no cursor in this manifest: re-reading the corpus, but NOT re-tokenising it"
        log(f"[tokens] resuming after {skip_docs:,} documents in {len(shards)} shard(s); "
            f"{where}")

    # The producer positions itself ONCE, here. Everything after this point is new work.
    stream = documents(cursor, skip_docs) if callable(documents) else documents

    per_shard = max(shard_bytes // isz, 1)
    fh = mh = None
    cur = shards[-1] if shards and shards[-1]["n_tokens"] < per_shard else None
    if cur is None and shards:
        pass                                     # the last shard is full; a new one starts below
    last_flush = time.time()

    def open_shard():
        nonlocal fh, mh, cur
        if cur is None:
            cur = {"file": os.path.basename(shard_path(stem, len(shards))),
                   "n_tokens": 0, "n_docs": 0}
            shards.append(cur)
        fh = open(os.path.join(os.path.dirname(stem) or ".", cur["file"]), "ab")
        # THE SIDECAR IS APPENDED IN LOCKSTEP WITH THE SHARD, one line per document written, so
        # line j of the file describes document j of the shard. Opened with the shard rather
        # than on the first span, because a conversation with no assistant turn writes `[]` and
        # a file that skipped it would renumber everything after it.
        if masks:
            mh = open(masks_path(stem, len(shards) - 1), "a", encoding="utf-8")

    def close_shard():
        """Close the open shard and SEAL it with its digest.

        A shard is hashed exactly once, here, because this is the moment it stops changing --
        hashing on every manifest flush would re-read the whole shard every few seconds, and
        hashing at open would make a reader pay for a write-side guarantee."""
        nonlocal fh, mh, cur
        if fh is not None:
            fh.flush(); os.fsync(fh.fileno()); fh.close(); fh = None
        if mh is not None:
            mh.flush(); os.fsync(mh.fileno()); mh.close(); mh = None
        if cur is not None:
            cur["md5"] = chunk_md5s(os.path.join(os.path.dirname(stem) or ".", cur["file"]))
        cur = None

    try:
        for item in stream:
            # (ids, cursor) from a producer that can seek, (ids, cursor, spans) from one that
            # also knows which tokens were the assistant's; a bare list from a producer that can
            # do neither, in which case the position stays as it was and resume falls back to
            # count. Unpacked tolerantly so a two-item producer keeps working untouched.
            if isinstance(item, tuple):
                ids, cursor, spans = (tuple(item) + (None,))[:3]
            else:
                ids, spans = item, None
            if not ids:                          # tokenised to nothing: the cursor still moved
                continue
            if fh is None:
                open_shard()
                log(f"[tokens] shard {len(shards) - 1:05d} open")
            # WRITTEN NOW, not when the shard fills: a run killed at any moment leaves every
            # document it read on disk, and the manifest at most `flush_seconds` behind.
            array.array(typecode, ids + [eos]).tofile(fh)
            if mh is not None:
                # ONE LINE PER DOCUMENT, SPARSE, RELATIVE TO THE DOCUMENT. `n` is the document's
                # token count without its EOS, which is what lets a reader check that the line
                # it is holding belongs to the document it thinks it does.
                mh.write(json.dumps({"doc": cur["n_docs"], "n": len(ids),
                                     "spans": spans or []}, separators=(",", ":")) + "\n")
            cur["n_tokens"] += len(ids) + 1
            cur["n_docs"] += 1
            now = time.time()
            if now - last_flush >= flush_seconds:
                fh.flush()
                if mh is not None:
                    mh.flush()
                _write_index(stem, sig, dtype, eos, vocab_size, shards, False, cursor, packing)
                last_flush = now
            if cur["n_tokens"] >= per_shard:
                n = len(shards) - 1
                log(f"[tokens] shard {n:05d} done: {cur['n_tokens']:,} tokens, "
                    f"{cur['n_tokens'] * isz / 1048576:.1f} MB")
                close_shard()
                _write_index(stem, sig, dtype, eos, vocab_size, shards, False, cursor, packing)
                last_flush = time.time()
    finally:
        # The same sealing as close_shard(), for the shard the loop ended on.
        if mh is not None:
            mh.flush(); os.fsync(mh.fileno()); mh.close(); mh = None
        if fh is not None:
            fh.flush(); os.fsync(fh.fileno()); fh.close(); fh = None
            if cur is not None:
                cur["md5"] = chunk_md5s(os.path.join(os.path.dirname(stem) or ".", cur["file"]))
        # the manifest is brought up to date even when the loop raised, so an interrupted
        # build resumes from where it truly stopped rather than from the last periodic flush
        if shards:
            _write_index(stem, sig, dtype, eos, vocab_size, shards, False, cursor, packing)

    _write_index(stem, sig, dtype, eos, vocab_size, shards, True, cursor, packing)
    n_tok = sum(s["n_tokens"] for s in shards)
    n_doc = sum(s["n_docs"] for s in shards)
    log(f"[tokens] wrote {n_tok:,} tokens ({dtype}) in {len(shards)} shard(s) from "
        f"{n_doc:,} documents -> {stem}_*.tokens")
    if masks:
        log(f"[tokens] response mask: {n_doc:,} line(s) of (offset, length) spans -> "
            f"{stem}_*.masks")
    return n_tok, n_doc


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
# PUBLIC NAMES FOR THE TWO THINGS A TOOL LEGITIMATELY NEEDS. `--fsck` and any future inspector
# reads a manifest and sizes a dtype; reaching through an underscore for that would make every
# such tool depend on this file's private surface staying still.
read_index = _read_index
itemsize = _itemsize


class _Shard:
    """One shard, memory-mapped lazily: a corpus of ten thousand shards must not open ten
    thousand files to be described, so the mapping is created on first touch."""

    def __init__(self, path, dtype, n_tokens, n_docs, eos, np):
        self.path, self.dtype, self.eos, self._np = path, dtype, eos, np
        self.n_tokens, self.n_docs = n_tokens, n_docs
        self._tokens = self._starts = None

    @property
    def tokens(self):
        if self._tokens is None:
            # TokenStream refuses a damaged stream at open, so by here the file is long enough.
            # READ ONLY, AND IT MATTERS MORE THAN IT LOOKS. Under data parallelism N processes
            # map THE SAME FILE at once and each reads its own contiguous region of it. mode="r"
            # maps the pages shared and clean: one copy in the page cache serves every replica,
            # and no process can dirty a page another is reading from. mode="r+" or "c" would
            # give each process private, copy-on-write pages the moment anything touched them --
            # N times the resident memory for a file nothing is supposed to modify, and a stream
            # that could silently differ between replicas.
            self._tokens = self._np.memmap(self.path, dtype=self.dtype, mode="r",
                                           shape=(self.n_tokens,))
        return self._tokens

    @property
    def starts(self):
        """Where each document begins, RECOVERED from the EOS separators.

        Not stored: a manifest carrying an offset per document would be gigabytes for a corpus
        of this size, and the information is already in the stream -- every document ends with
        exactly one EOS. Derived per shard on first use, so previews touch a handful of shards
        and training, which never asks, touches none."""
        if self._starts is None:
            np = self._np
            ends = np.flatnonzero(np.asarray(self.tokens) == self.eos)
            self._starts = np.concatenate(([0], ends[:-1] + 1)) if len(ends) else np.zeros(0, "int64")
            self._ends = ends
        return self._starts


class TokenStream:
    """The shards at a stem, read as one contiguous stream.

    `len(stream)` is the DOCUMENT count, so stage logs and tables read as they did when the
    corpus was a list of documents; `stream.n_tokens` is the token count."""

    def __init__(self, path):
        np = _np()
        self._np = np
        self.path = path
        idx = _read_index(path)
        # (MAGIC,) + LEGACY_MAGIC, NOT (MAGIC, LEGACY_MAGIC). The second is now a TUPLE of every
        # marker this format has ever been written under, so nesting it inside another tuple
        # would compare a string against a tuple and reject every historical manifest -- the
        # exact failure the name-free marker above exists to end.
        if not idx or idx.get("magic") not in (MAGIC,) + LEGACY_MAGIC:
            raise ValueError(f"{index_path(path)} is not a {MAGIC} manifest "
                             f"(found {idx.get('magic')!r})" if idx else
                             f"{index_path(path)} is not a {MAGIC} manifest")
        if not idx.get("complete"):
            raise ValueError(f"{path} is an unfinished build "
                             f"({idx.get('n_tokens', 0):,} tokens so far)")
        # VERIFIED BEFORE IT IS MAPPED. mmap does not read a file, it maps it, so a corrupt
        # shard surfaces thousands of steps later as loss that will not come down -- or not at
        # all. The check is here, once, at open.
        #
        # HOW MUCH is TOKENS["verify"]: "md5" reads every shard and compares the digest recorded
        # when it was sealed; "size" compares lengths only, which catches the lost tail a killed
        # writer leaves and costs one stat per shard; "off" trusts the manifest. md5 is the
        # default because reading a corpus once at ~1 GB/s is seconds against a run of hours,
        # and because the failure it catches is silent.
        mode = "md5"
        missing = [i for i, sh in enumerate(idx["shards"]) if not sh.get("md5")]
        if missing:
            raise ValueError(f"{shard_path(path, missing[0])} has no recorded content digest; "
                             "the consumer cannot verify it and will not modify it")
        if mode != "off":
            t0 = time.time()
            # A BAR, BECAUSE THIS READS THE WHOLE CORPUS. At md5 the trainer reads every shard
            # before it maps one, which on a 20 GB stream is minutes of a process that has
            # printed nothing and looks hung -- exactly the thing this pipeline refuses to do
            # anywhere else. Counted in bytes, since that is the work; `size` mode reads nothing
            # and gets no bar.
            from .utils import bar
            name = os.path.basename(path)
            if mode == "md5":
                on_disk = sum(os.path.getsize(shard_path(path, i))
                              for i in range(len(idx["shards"]))
                              if os.path.exists(shard_path(path, i)))
                with bar(f"[tokens] verifying {name}", unit="B", unit_scale=True,
                         total=on_disk) as vb:
                    bad = first_damaged(path, idx["shards"], idx["dtype"],
                                        check_md5=True, on_bytes=vb.update)
            else:
                bad = first_damaged(path, idx["shards"], idx["dtype"], check_md5=False)
            if bad is not None:
                i, why, _ = bad
                raise ValueError(
                    f"{shard_path(path, i)} is DAMAGED: {why}. Re-run stage 4 -- it cuts this "
                    f"shard back to its last complete document and re-tokenises from there, "
                    f"keeping the {i} intact shard(s) before it.")
            missing = sum(1 for sh in idx["shards"] if not sh.get("md5"))
            if mode == "md5" and missing:
                print(f"[tokens] {os.path.basename(path)}: {missing} of {len(idx['shards'])} "
                      f"shard(s) predate content digests and were length-checked only; they "
                      f"gain one when stage 4 next rewrites them", flush=True)
            elif mode == "md5":
                print(f"[tokens] verified {len(idx['shards'])} shard(s) in "
                      f"{time.time() - t0:.1f}s", flush=True)
        self.head, self.sig, self.dtype = idx, idx.get("sig", ""), idx["dtype"]
        self.eos, self.n_tokens, self.n_docs = idx["eos"], idx["n_tokens"], idx["n_docs"]
        # HOW THIS STREAM WAS PACKED, as the manifest recorded it: the tokenizer signature, the
        # word budget, the text column and the file count. It is what `build` compares to decide
        # whether shards on disk belong to this run or are stale.
        self.packing = idx.get("packing", "") or ""
        base = os.path.dirname(path) or "."
        self.shards = [_Shard(os.path.join(base, s["file"]), self.dtype, s["n_tokens"],
                              s["n_docs"], self.eos, np) for s in idx["shards"]]
        self._tok_start, self._doc_start, t, d = [], [], 0, 0
        for s in self.shards:
            self._tok_start.append(t); self._doc_start.append(d)
            t += s.n_tokens; d += s.n_docs
        self._tok_start.append(t); self._doc_start.append(d)

    def __len__(self):
        return self.n_docs

    @property
    def nbytes(self):
        return sum(os.path.getsize(s.path) for s in self.shards)

    def _locate(self, starts, i):
        import bisect
        return max(0, min(bisect.bisect_right(starts, i) - 1, len(self.shards) - 1))

    def read(self, start, n):
        """`n` tokens from global offset `start`, ACROSS SHARD BOUNDARIES."""
        np = self._np
        out, k = [], self._locate(self._tok_start, start)
        off = start - self._tok_start[k]
        while n > 0 and k < len(self.shards):
            s = self.shards[k]
            take = min(n, s.n_tokens - off)
            if take > 0:
                out.append(np.asarray(s.tokens[off:off + take], dtype="int64"))
                n -= take
            k += 1
            off = 0
        return out[0] if len(out) == 1 else np.concatenate(out)

    def doc(self, i):
        """Document `i` as a list of ids, WITHOUT its trailing eos."""
        k = self._locate(self._doc_start, i)
        s = self.shards[k]
        j = i - self._doc_start[k]
        starts = s.starts
        a = int(starts[j])
        b = int(s._ends[j])
        return s.tokens[a:b].tolist()


    # SHARD SEED. Fixed, and deliberately NOT the training seed: the file-to-GPU map must be the
    # same on every resume of the same corpus, or a restarted run would hand each device a
    # different quarter and the "no device sees another's tokens" property would hold only
    # within a single process lifetime. It must also not draw from the training generator,
    # whose state is checkpointed and whose consumption sets the data order.
    SHARD_SEED = 0x5EED

    def shard_regions(self, regions):
        """[[shard indices], ...] -- the SHUFFLED shard files dealt out, one list per GPU.

        WHY FILES AND NOT BYTE RANGES. Cutting the stream into N contiguous byte ranges gives
        device 0 the first N-th of the corpus -- and the shards were written in corpus order,
        which is sorted file order, so that quarter is not a sample of the corpus but its
        beginning: one domain, one crawl segment, one language, whatever happened to sort first.
        Dealing out whole FILES after a shuffle makes each device's share a random sample of the
        corpus instead of a prefix of it.

        SHUFFLED ONCE, DETERMINISTICALLY, so a resumed run deals the same hand.

        Cached: the deal is a property of (corpus, regions) and not of the batch."""
        r = max(1, int(regions))
        got = getattr(self, "_regions_cache", None)
        if got is not None and got[0] == r:
            return got[1]
        import torch
        n = len(self.shards)
        order = torch.randperm(n, generator=torch.Generator().manual_seed(self.SHARD_SEED))
        out = [order[i * n // r:(i + 1) * n // r].tolist() for i in range(r)]
        self._regions_cache = (r, out)
        return out

    def _shard_span(self, k):
        """(first global token offset, count) of shard `k`."""
        return int(self._tok_start[k]), int(self.shards[k].n_tokens)

    def batch(self, batch, seq_len, generator=None, device=None, regions=1,
              region=None, mask=None):
        """One training batch: (ids, attn, rmask), each (batch, seq_len + 1).

        seq_len + 1 tokens are drawn so the loop's `logits[:, :-1]` against `ids[:, 1:]` yields
        exactly seq_len predictions. attn and rmask are all ones: a packed stream has no
        padding to hide and no prompt to exclude, so every position is a target.

        Offsets come from the SAME torch generator the training loop checkpoints, so a resumed
        run continues the data order it would have had."""
        import torch
        np = self._np
        n = seq_len + 1
        hi = max(self.n_tokens - n, 1)
        r_ = int(regions or 1)
        if r_ > 1 and len(self.shards) >= r_:
            # FILES DEALT TO DEVICES. Row i belongs to device i*r//batch, which is the row
            # nn.DataParallel will scatter there; that device owns a shuffled set of whole
            # shard files, and the row is drawn from one of them, chosen in proportion to its
            # token count so a short shard is not over-sampled.
            #
            # THE WINDOW STAYS INSIDE ITS SHARD. Globally, read() completes a window that runs
            # off the end of a shard from the head of the NEXT one -- which after the deal
            # belongs to a different device. Keeping the start below (shard end - n) is what
            # makes the file sets actually disjoint. It costs the last n tokens of each shard
            # as a starting point: ~8k of ~52M, 0.016%, and no token is unreachable -- they are
            # all still reached from earlier starts within the same shard.
            deal = self.shard_regions(r_)
            starts = []
            for i in range(batch):
                d = min(i * r_ // batch, r_ - 1) if region is None else int(region)
                mine = [k for k in deal[min(d, r_ - 1)] if self.shards[k].n_tokens > n]
                if not mine:
                    starts.append(int(torch.randint(0, hi, (1,), generator=generator)))
                    continue
                w = torch.tensor([float(self.shards[k].n_tokens) for k in mine],
                                 dtype=torch.double)
                k = mine[int(torch.multinomial(w / w.sum(), 1, generator=generator))]
                lo, cnt = self._shard_span(k)
                starts.append(lo + int(torch.randint(0, max(cnt - n, 1), (1,),
                                                     generator=generator)))
        elif region is not None and int(regions or 1) > 1:
            # ALL ROWS FROM ONE REGION, which is what MultiStream needs to keep the row order
            # it assembles from matching the device the row will be scattered to.
            r = int(regions)
            size = max(self.n_tokens // r, n + 1)
            base = min(int(region), r - 1) * size
            span = max(min(size, self.n_tokens - base) - n, 1)
            starts = [base + int(torch.randint(0, span, (1,), generator=generator))
                      for _ in range(batch)]
        elif int(regions or 1) > 1:
            r = int(regions)
            size = max(self.n_tokens // r, n + 1)
            starts = []
            for i in range(batch):
                base = min(i * r // batch, r - 1) * size
                span = max(min(size, self.n_tokens - base) - n, 1)
                starts.append(base + int(torch.randint(0, span, (1,), generator=generator)))
        else:
            starts = torch.randint(0, hi, (batch,), generator=generator).tolist()
        ids = torch.from_numpy(np.stack([self.read(s, n) for s in starts]))
        if device is not None:
            ids = ids.to(device)
        # THE ATTENTION MASK IS None, NOT A TENSOR OF ONES, and the difference is a gigabyte.
        # A packed stream has no padding, so an all-ones mask says exactly what `None` says --
        # but it says it in a way the model cannot see through. Given a mask, attention must
        # build the (B,1,T,S) boolean it would otherwise have got for free from `is_causal`:
        # 64 MiB at T = 8,192 and about 1 GiB at 32,768, allocated every forward pass, plus
        # the loss of whichever fused kernel only accepts the causal flag. `rmask` stays a
        # tensor because the loss really does weight by it.
        #
        # `mask` IS THE ONE THING SUPERVISED FINE-TUNING ADDS. The ids, the packing, the window
        # draw and the region deal are pretraining's and are untouched; a stage that trains on
        # only part of each window passes a callable and gets its own rmask instead of ones.
        # Absent it -- pretraining, the benchmark, the ablation -- this is the same line it was.
        if mask is not None:
            return ids, None, mask(starts, n).to(ids.device)
        return ids, None, torch.ones_like(ids)

    def describe(self):
        return (f"{self.n_tokens:,} tokens ({self.dtype}, {self.nbytes / 1048576:.1f} MB in "
                f"{len(self.shards)} shard(s), memory-mapped) in {self.n_docs:,} documents")


class MultiStream:
    """Several TokenStreams read as ONE corpus, without being merged on disk.

    A CORPUS KEEPS ITS OWN STREAM, ITS OWN SIGNATURE AND ITS OWN CACHE. That is the whole point:
    adding a second corpus to a training run must not touch the first one's tokens. Merging them
    into a single stream would change the signature of the result, discard the shards already
    built, and re-tokenise hours of text to arrive at the same tokens in a different file --
    which for a run already half-way through its budget is not a data change but a restart.

    Instead each corpus is built and cached exactly as if it were alone, and this reads across
    them. Add a corpus tomorrow and only the new one is tokenised; remove it and the other's
    cache is still there, still current, still keyed by the same signature it always had.

    SEQUENCES ARE DRAWN IN PROPORTION TO TOKENS, which is what concatenating the corpora would
    have given: a corpus of 4 GiB contributes twenty times as often as one of 0.2 GiB. The
    mixture is therefore a fact about the data on disk rather than a weighting somebody chose,
    and it is reproducible -- the draw uses the caller's generator, so a resumed run continues
    the same sequence of batches.

    The batch is shuffled after assembly so it is not ordered by corpus: gradients are averaged
    over a batch, and a batch whose first half is one corpus and second half another is fine for
    the mean but reads as a bug in any per-example diagnostic."""

    def __init__(self, streams, names=()):
        self.streams = list(streams)
        self.names = list(names) or [f"corpus {i + 1}" for i in range(len(self.streams))]
        self.n_tokens = sum(s.n_tokens for s in self.streams)
        self.n_docs = sum(s.n_docs for s in self.streams)
        self.eos = self.streams[0].eos
        self.dtype = self.streams[0].dtype
        self.nbytes = sum(s.nbytes for s in self.streams)

    def __len__(self):
        return self.n_docs

    def doc(self, i):
        """Document `i` as a list of ids, the corpora laid end to end in their listed order.

        A TokenStream has this and the sample previews use it (helpers.common.corpus_prompts
        tests for `doc` and `n_docs` together), so a corpus that had `n_docs` but no `doc`
        would take the branch meant for a list of records and fail on a type that is not a
        sequence. Nothing here reads a corpus by index during training -- batches are drawn,
        not indexed -- so this exists for the previews and for anything else that reasonably
        expects a corpus to be enumerable."""
        j = i
        for s in self.streams:
            if j < s.n_docs:
                return s.doc(j)
            j -= s.n_docs
        raise IndexError(f"document {i:,} past the end of {self.n_docs:,} documents")

    def batch(self, batch, seq_len, generator=None, device=None, regions=1, region=None):
        """One training batch, drawn across the corpora. Same contract as TokenStream.batch.

        `region` PICKS ONE, AND THAT IS THE DDP CASE. Under torchrun each rank draws its own
        batch, so it wants every row from ITS region and no row order constraint -- the
        shuffle is safe and wanted, because nothing downstream scatters. Without this
        parameter the signature simply did not accept what TokenStream.batch already did, and
        any scheme with two corpora -- mira-s, -m and -l all mix FineWeb-Edu with
        WikiText-103 -- would have raised TypeError the moment DDP asked for a region.

        WITH `regions` > 1 AND `region` None THE ROW ORDER IS LOAD-BEARING and the shuffle
        below is skipped: row i must belong to region i*regions//batch, because that is the
        row nn.DataParallel will scatter to device i*regions//batch. The rows are assembled
        REGION BY REGION, each still drawn across the corpora in proportion to their token
        counts, so the mixture is what it was and only the ordering is constrained."""
        import torch
        w = torch.tensor([float(s.n_tokens) for s in self.streams], dtype=torch.double)
        r = max(1, int(regions or 1))
        if r > 1 and region is not None:
            # ONE REGION, WHOLE BATCH, SHUFFLED. The mixture across corpora is drawn exactly
            # as the single-region path draws it; only the slice of the stream differs.
            pick = torch.multinomial(w / w.sum(), batch, replacement=True, generator=generator)
            parts = []
            for k, st in enumerate(self.streams):
                n = int((pick == k).sum())
                if n:
                    parts.append(st.batch(n, seq_len, generator=generator, device=None,
                                          regions=r, region=int(region))[0])
            ids = parts[0] if len(parts) == 1 else torch.cat(parts)
            ids = ids[torch.randperm(ids.shape[0], generator=generator)]
            if device is not None:
                ids = ids.to(device)
            return ids, None, torch.ones_like(ids)
        if r > 1:
            parts = []
            for i in range(r):
                nrow = batch * (i + 1) // r - batch * i // r
                if not nrow:
                    continue
                pick = torch.multinomial(w / w.sum(), nrow, replacement=True,
                                         generator=generator)
                for k, s in enumerate(self.streams):
                    n = int((pick == k).sum())
                    if n:
                        parts.append(s.batch(n, seq_len, generator=generator, device=None,
                                             regions=r, region=i)[0])
            ids = parts[0] if len(parts) == 1 else torch.cat(parts)
            if device is not None:
                ids = ids.to(device)
            return ids, None, torch.ones_like(ids)
        pick = torch.multinomial(w / w.sum(), batch, replacement=True, generator=generator)
        parts = []
        for k, s in enumerate(self.streams):
            n = int((pick == k).sum())
            if n:
                parts.append(s.batch(n, seq_len, generator=generator, device=None)[0])
        ids = parts[0] if len(parts) == 1 else torch.cat(parts)
        ids = ids[torch.randperm(ids.shape[0], generator=generator)]
        if device is not None:
            ids = ids.to(device)
        # None, as TokenStream.batch returns: a packed stream has no padding to describe.
        return ids, None, torch.ones_like(ids)

    def describe(self):
        share = [f"{n} {s.n_tokens / max(self.n_tokens, 1):.1%}"
                 for n, s in zip(self.names, self.streams)]
        return (f"{self.n_tokens:,} tokens across {len(self.streams)} corpora "
                f"({', '.join(share)}) in {self.n_docs:,} documents, each memory-mapped from "
                f"its own cache")


def open_if_current(path, sig):
    """The stream at the stem `path` when it is complete and its signature matches."""
    idx = _read_index(path)
    if not idx or idx.get("sig") != sig or not idx.get("complete"):
        return None
    try:
        return TokenStream(path)
    except Exception:                                          # noqa: BLE001
        return None
