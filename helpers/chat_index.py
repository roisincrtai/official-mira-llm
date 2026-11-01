"""chat_index.py -- the response mask for a conversation corpus, READ from stage 4's sidecar.

    stream = mask_stream(token_stream, log)
    stream.batch(B, T, generator, device)      (ids, None, rmask) -- rmask from the mask

SUPERVISED FINE-TUNING IS PRETRAINING PLUS A MASK. The tokens are the ones stage 4 wrote and
every other trainer reads; the packing, the window draw, the region deal and the batching are
pretraining's and are untouched. The single difference is that `rmask` marks the ASSISTANT
tokens instead of being all ones, so the loss lands on what the model is being taught to say and
not on what it was told.

THE MASK IS STAGE 4'S OUTPUT, NOT THIS STAGE'S. Stage 4 encodes a conversation line by line, so
it knows the token offset of every assistant line exactly, and it writes those offsets beside
each shard as `<stem>_NNNNN.masks`. Nothing here re-reads the corpus, re-tokenises anything, or
reconstructs a boundary: this module opens the sidecar, checks it against the shard it belongs
to, and turns it into arrays. That is what makes the mask exact rather than inferred, and it is
what keeps every consumer read-only.

ONE CONVERSATION IS ONE DOCUMENT, which is what makes the sidecar a flat file: line j of shard
i's `.masks` describes document j of shard i, and there are exactly as many lines as the
manifest says that shard has documents. A count that disagrees is a stream and a mask built by
different runs, and it stops the stage rather than being reconciled.

THE MASK IS SPARSE. It is (offset, length) pairs -- 8 bytes an assistant run, held as two uint32
arrays with a per-document pointer -- and never a per-token array. The one dense object anywhere
here is `rmask` itself, (batch, seq_len + 1), which is the weight the loss multiplies by: 4 KB at
batch 8 and 512 tokens.

THE MEMORY FOOTPRINT at UltraChat's 1,468,352 conversations:

    doc_start    uint64 per document        11.8 MB   where the document begins, in stream tokens
    ptr          uint64 per document + 1    11.8 MB   where its spans begin in the two arrays
    off, length  uint32 per span            ~46 MB    the spans themselves, ~4 per conversation

Under 70 MB, none of it growing with the corpus's token count, and no tokenizer call at any
point in the run.
"""
import os

import numpy as np

from helpers import token_store


def document_starts(stream, log=print):
    """The token offset of every document in `stream`, from the EOS separators it stores.

    NO TOKENISING AND NO MANIFEST CHANGE. token_store writes eos between documents and recovers
    boundaries from it lazily, per shard; this does the same across the whole stream once, with
    one vectorised scan per memory-mapped shard. A document begins at 0 and after every eos."""
    from helpers import bar
    starts = [np.zeros(1, dtype=np.uint64)]
    base = 0
    # SCANNED IN BLOCKS SO NOTHING DENSE IS MATERIALISED. `tokens == eos` over a whole shard is
    # a boolean the size of the shard -- 52 MB per shard, for an answer that is a few thousand
    # positions. A block at a time holds 8 M booleans and produces the same offsets.
    #
    # AND THE BLOCK IS WHAT THE BAR COUNTS. This reads the whole stream, which is billions of
    # tokens at UltraChat's size; without a bar it is a silent minute between two bars that do
    # have one, and a silent minute is indistinguishable from a hang.
    block = 8 << 20
    found = 0
    with bar("[instruct-sft] scanning documents", unit="tok", unit_scale=True,
             total=int(stream.n_tokens)) as pb:
        for shard in stream.shards:
            tokens = np.asarray(shard.tokens)
            count = int(tokens.shape[0])
            for lo in range(0, count, block):
                hit = np.flatnonzero(tokens[lo:lo + block] == stream.eos)
                if hit.size:
                    starts.append((hit + lo + 1 + base).astype(np.uint64))
                    found += int(hit.size)
                pb.update(min(block, count - lo))
                pb.set_postfix_str(f"{found:,} documents")
            base += count
    out = np.concatenate(starts)
    # THE LAST eos ENDS THE LAST DOCUMENT and opens nothing; a start at the end of the stream
    # would name a document with no tokens in it.
    if out.size and int(out[-1]) >= int(stream.n_tokens):
        out = out[:-1]
    return out


def read_spans(stream, log=print):
    """(ptr, offsets, lengths) -- every shard's `.masks`, read in shard order.

    `ptr` has one entry per document plus one, so document d's spans are
    `offsets[ptr[d]:ptr[d + 1]]` against `lengths[ptr[d]:ptr[d + 1]]`, with offsets counted from
    that document's own first token. A document with no response has ptr[d] == ptr[d + 1].

    EVERY SHARD IS CHECKED AGAINST THE MANIFEST BEFORE ITS LINES ARE USED. A sidecar with fewer
    or more lines than its shard has documents would silently shift every document after the
    difference onto another conversation's spans -- a mask that is wrong rather than missing, so
    the loss would land on the user's words and nothing would say so."""
    from helpers import bar
    ptr, offs, lens = [0], [], []
    with bar("[instruct-sft] reading the response mask", unit="doc", unit_scale=True,
             total=int(stream.n_docs)) as pb:
        for i, shard in enumerate(stream.shards):
            got = token_store.read_masks(stream.path, i)
            if got is None:
                raise SystemExit(
                    f"[instruct-sft] no response mask beside shard {i:05d} at\n"
                    f"      {token_store.masks_path(stream.path, i)}\n"
                    f"      The mask is written by stage 4 when it tokenises a conversation "
                    f"corpus, one line of (offset, length) spans per conversation. This stream "
                    f"predates that, or was built from a corpus stage 4 did not read as "
                    f"conversations. It has to be tokenised again:\n"
                    f"          ./run_scripts/run_tokenize_data.sh --force --only "
                    f"{os.path.basename(os.path.dirname(stream.path))}")
            if len(got) != int(shard.n_docs):
                raise SystemExit(
                    f"[instruct-sft] shard {i:05d} holds {int(shard.n_docs):,} documents but "
                    f"its mask has {len(got):,} line(s).\n"
                    f"      {token_store.masks_path(stream.path, i)}\n"
                    f"      The two were written by different runs, so line j does not describe "
                    f"document j and no mask over this stream can be trusted. Tokenise the "
                    f"corpus again:\n"
                    f"          ./run_scripts/run_tokenize_data.sh --force --only "
                    f"{os.path.basename(os.path.dirname(stream.path))}")
            for spans in got:
                for a, length in spans:
                    offs.append(a); lens.append(length)
                ptr.append(len(offs))
            pb.update(len(got))
            pb.set_postfix_str(f"shard {i:05d}, {len(offs):,} spans")
    # uint32 HOLDS BOTH, and they are DOCUMENT-relative: an offset is inside its own document
    # and a length is inside that. Eight bytes a span rather than sixteen.
    return (np.asarray(ptr, dtype=np.uint64),
            np.asarray(offs, dtype=np.uint32),
            np.asarray(lens, dtype=np.uint32))


class ResponseMask:
    """The assistant spans of the corpus, as three arrays, with no work left to do at run time.

    NOTHING IS COMPUTED LAZILY HERE ANY MORE, because there is nothing left to compute: stage 4
    measured the offsets while it was encoding, so a step is an array lookup rather than a line
    read and a tokenizer call."""

    def __init__(self, doc_start, ptr, offs, lens):
        self.doc_start, self.ptr, self.off, self.len = doc_start, ptr, offs, lens

    def nbytes(self):
        return int(self.doc_start.nbytes + self.ptr.nbytes + self.off.nbytes + self.len.nbytes)

    def n_spans(self):
        return int(self.off.size)

    def document(self, doc):
        """The (offset, length) pairs of one document, counted from its own first token.

        THIS IS THE FORM THE SIDECAR IS WRITTEN IN, handed back unchanged. A batch of whole
        conversations needs no clipping and no global arithmetic: the spans of document d are
        already relative to the start of document d."""
        lo, hi = int(self.ptr[doc]), int(self.ptr[doc + 1])
        return self.off[lo:hi], self.len[lo:hi]

    def __call__(self, starts, n):
        """The (batch, n) response mask for windows beginning at `starts`.

        ONE IS AN ASSISTANT TOKEN, ZERO IS CONTEXT. A window may cover several documents and may
        begin or end inside one, so every span is clipped to the window rather than assumed to
        fit.

        THIS TENSOR IS THE ONLY DENSE THING IN THE MODULE, and it is `rmask` itself -- the
        weight the loss multiplies by, (batch, seq_len + 1), 4 KB at batch 8 and 512 tokens. The
        MASK is the (offset, length) arrays; this is one window's worth of it, expanded because
        the loss takes a weight per position."""
        import torch
        out = torch.zeros(len(starts), int(n), dtype=torch.long)
        n_docs = int(self.doc_start.size)
        for row, start in enumerate(starts):
            start = int(start)
            stop = start + int(n)
            lo = int(np.searchsorted(self.doc_start, start, side="right")) - 1
            for doc in range(max(lo, 0), n_docs):
                base = int(self.doc_start[doc])
                if base >= stop:
                    break
                for k in range(int(self.ptr[doc]), int(self.ptr[doc + 1])):
                    a = base + int(self.off[k])
                    b = a + int(self.len[k])
                    if b <= start:
                        continue
                    if a >= stop:
                        break
                    out[row, max(a, start) - start:min(b, stop) - start] = 1
        return out


TRUNCATE = 65536


class MaskedStream:
    """`stream`, drawn ONE WHOLE CONVERSATION AT A TIME, with `rmask` from `mask`.

    THIS IS THE ONE PLACE STAGE 7 STOPS BEING PRETRAINING. Pretraining draws a fixed window from
    anywhere in the packed stream, because an article has no natural length and every position is
    a target. A conversation does: it is a question and the answer to it, and a window that
    begins in the middle of one trains a reply against a prompt the model was never shown. So a
    batch here is `batch` whole documents, and its width is the longest of THOSE documents --
    not a configured window, and not the truncation ceiling.

    PADDING IS TO THE BATCH'S OWN LONGEST SEQUENCE. A batch of eight 300-token conversations is
    300 wide; one that happens to draw a 4,000-token conversation is 4,000 wide for that step
    alone. Nothing is padded to 65,536, which would make every step the cost of the worst
    document in the corpus.

    THE PAD IS EOS AND IT CARRIES NO ATTENTION MASK, WHICH IS DELIBERATE. Passing an attn_mask
    would force the mixer onto the PyTorch reference -- it takes the kernel only when `valid is
    None` (model/mira_mixer.py) -- and MIRA never falls back to PyTorch. It is not
    needed: padding sits at the END of a sequence, so no real token is ever after it, and `rmask`
    is zero across it, so it contributes nothing to the loss. The recurrence absorbs the pad
    tokens and nothing reads what it becomes.

    model.lm.train recognises a packed corpus by `batch` and `n_tokens` (model/lm.py:520),
    and reads `describe`, `n_docs` and `doc` for its report and its previews, so those are
    forwarded rather than reimplemented -- there is one stream and this is a view of it."""

    def __init__(self, stream, mask, tok=None):
        self._stream, self._mask = stream, mask
        # THE MARKER IDS, OR None, AND None IS A SUPPORTED STATE. This is CHECKPOINT
        # compatibility and it stays: a tokenizer that predates the instruct markers -- any
        # checkpoint written before stage 7 registered them -- has no id for them, and this
        # stage still has to run against it. The markers are then simply not inserted and eos
        # closes the document, which is exactly what it did before.
        from helpers.utils import (CHAT_ASSISTANT_PREFIX, CHAT_END, CHAT_START,
                                   CHAT_USER_PREFIX)
        marks = getattr(tok, "special2id", None) or {}
        self._start_id = marks.get(CHAT_START)
        self._end_id = marks.get(CHAT_END)
        # THE ROLE PREFIXES ARE TEXT, SO THEY ARE ENCODED ONCE AND KEPT. `User:\n` is ordinary
        # text with no special id of its own, and encoding it per turn would run the BPE merge
        # loop millions of times over two constant strings. encode_ordinary, not encode: these
        # hold no specials, and a text prefix must never be able to produce a marker id.
        self._user_ids, self._assistant_ids = (), ()
        if self._start_id is not None and hasattr(tok, "encode_ordinary"):
            self._user_ids = tuple(tok.encode_ordinary(CHAT_USER_PREFIX))
            self._assistant_ids = tuple(tok.encode_ordinary(CHAT_ASSISTANT_PREFIX))

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __len__(self):
        return len(self._stream)

    def spans(self, doc):
        """The (offset, length) arrays of document `doc`, relative to its own first token.

        THE MASK READ OUT, NOT THE MASK APPLIED. `batch` needs (offset, length) expanded into
        rmask; the PREVIEW needs the offsets themselves, because the first of them is the exact
        token at which the conversation stops being the prompt and starts being the answer --
        which is where a preview prompt must end if it is to be the whole question and none of
        the reply. Exposed here rather than read off `_mask` from another module: what the mask
        is made of is this file's business, and where a document's prompt ends is a fact the
        sidecar already knows."""
        return self._mask.document(int(doc))

    def batch(self, batch, seq_len, generator=None, device=None, regions=1, region=None):
        """(ids, None, rmask) for `batch` whole conversations, padded to the longest of them.

        `seq_len` IS IGNORED, AND THAT IS THE POINT. lm.train passes the configured window; this
        stage has none, and the width of a step is decided by the documents it drew.

        ONE REGION PER RANK, as the packed path does it: the generator is seeded identically on
        every rank so a resumed run reproduces its data order, and without a slice every rank
        would train on the same conversations."""
        import torch
        del seq_len
        n_docs = len(self._stream)
        r = max(int(regions or 1), 1)
        if r > 1 and region is not None:
            # THE RANK'S OWN SLICE OF THE DOCUMENT INDEX, contiguous and disjoint.
            size = max(n_docs // r, 1)
            base = min(int(region), r - 1) * size
            span = min(size, n_docs - base) or 1
            idx = [base + int(torch.randint(0, span, (1,), generator=generator))
                   for _ in range(batch)]
        else:
            idx = torch.randint(0, n_docs, (batch,), generator=generator).tolist()
        whole = [self._stream.doc(int(i)) for i in idx]
        rows = [r[:TRUNCATE] for r in whole]
        built = [self._turns(int(d), t, len(w) <= TRUNCATE)
                 for d, t, w in zip(idx, rows, whole)]
        width = max((len(t) for t, _ in built), default=1) or 1
        pad = int(self._stream.eos)
        ids = torch.full((batch, width), pad, dtype=torch.long)
        rmask = torch.zeros((batch, width), dtype=torch.long)
        for row, (tokens, weights) in enumerate(built):
            if tokens:
                ids[row, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                rmask[row, :len(tokens)] = torch.tensor(weights, dtype=torch.long)
        if device is not None:
            ids, rmask = ids.to(device), rmask.to(device)
        return ids, None, rmask

    def _turns(self, doc, tokens, complete):
        """One document -> (ids, weights), with the role markers INSERTED HERE.

        THE STREAM IS NOT TOUCHED AND IS NOT RE-TOKENISED. Stage 4 wrote conversations and the
        (offset, length) spans of their assistant runs, and that is all it wrote. The spans are
        already a complete description of who is speaking: [0, s0) is the user, [s0, s0+l0) is
        the assistant, the gap to the next span is the user again. So the markers can be put in
        at batch time, from the spans, without a byte on disk changing.

        WHAT IS INSERTED. Every turn becomes a delimited MESSAGE:

            <|startofmessage|> User:\\n      ...the question...   <|endofmessage|>
            <|startofmessage|> Assistant:\\n ...the answer...     <|endofmessage|>

        one after another with nothing between them. There is no per-role token: the role is the
        ordinary text `User:\\n` or `Assistant:\\n`, encoded once in __init__ and spliced in here.

        WHAT IS SUPERVISED IS THE WHOLE ASSISTANT MESSAGE.

        So the opening delimiter, the `Assistant:\\n` cue, the answer and the closing delimiter
        all carry weight 1. That is a change from the previous scheme, where the cue was the
        harness speaking and carried 0: the model now has to produce the whole reply block,
        including the decision to start replying and the token that says it has finished.
        Without <|endofmessage|> in the targets there is no stop to learn, which is why replies
        ran on. The user's message carries weight 0 throughout -- delimiters, cue and text.

        AN EMPTY USER MESSAGE IS NOT WRITTEN. A document whose first span starts at token 0
        opens with the assistant, and a `User:` block with nothing in it would teach the model
        that an empty question is answered.

        A TRUNCATED DOCUMENT GETS NO FINAL <|endofmessage|>. It does not end where it was cut,
        and marking the cut as a stop would teach the model to fall silent mid-answer.
        """
        s, e = self._start_id, self._end_id
        up, ap = self._user_ids, self._assistant_ids
        offs, lens = self._mask.document(doc)
        out, w = [], []

        def message(prefix, body, weight, close):
            """One message block at one weight; the body alone when there are no markers."""
            if s is None:
                out.extend(body); w.extend([weight] * len(body))
                return
            out.append(s); w.append(weight)
            out.extend(prefix); w.extend([weight] * len(prefix))
            out.extend(body); w.extend([weight] * len(body))
            if close and e is not None:
                out.append(e); w.append(weight)

        at = 0
        for a, length in zip(offs.tolist(), lens.tolist()):
            if a >= len(tokens):
                break                           # the span is past a truncated document
            stop = min(a + length, len(tokens))
            if a > at:
                message(up, tokens[at:a], 0, True)            # the user's turn
            # A MESSAGE IS CLOSED WHEN SOMETHING FOLLOWS IT, whatever `complete` says: the cut
            # can only fall in the LAST message of the row.
            message(ap, tokens[a:stop], 1, complete or stop < len(tokens))
            at = stop
        if at < len(tokens):
            # a trailing user turn with no answer: written, closed, and not supervised
            message(up, tokens[at:], 0, complete)
        # NO MARKERS AT ALL: exactly what this did before they existed -- the document's tokens
        # with the response spans weighted, and eos as the thing that ends it.
        if s is None and complete and out:
            out.append(int(self._stream.eos)); w.append(1)
        return out, w


def mask_stream(stream, log=print, tok=None):
    """`stream` with its response mask over it, read from stage 4's sidecars and held in memory."""
    ptr, offs, lens = read_spans(stream, log)
    doc_start = document_starts(stream, log)
    if doc_start.size != ptr.size - 1:
        raise SystemExit(
            f"[instruct-sft] the stream's EOS separators mark {doc_start.size:,} documents but "
            f"the mask describes {ptr.size - 1:,}.\n"
            f"      The manifest and the shards disagree about where documents end, so the "
            f"spans cannot be anchored. Tokenise the corpus again:\n"
            f"          ./run_scripts/run_tokenize_data.sh --force --only "
            f"{os.path.basename(os.path.dirname(stream.path))}")
    mask = ResponseMask(doc_start, ptr, offs, lens)
    # `tok` IS OPTIONAL AND ITS ABSENCE IS NOT AN ERROR. Without it the markers are not
    # inserted and the stage trains exactly as it did before them.
    covered = int(lens.sum())
    log(f"[instruct-sft] response mask: {doc_start.size:,} conversations, "
        f"{mask.n_spans():,} (offset, length) spans, {covered:,} response tokens of "
        f"{int(stream.n_tokens):,} ({100 * covered / max(int(stream.n_tokens), 1):.1f}%), "
        f"{mask.nbytes() / 1024 ** 2:.0f} MiB in memory")
    return MaskedStream(stream, mask, tok)
