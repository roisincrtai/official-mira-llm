"""
bpe_tokenizer.py -- a small, dependency-free byte-level BPE tokenizer for the from-scratch
MiraLLM.

The token-bag synthetic datasets used a word-level vocabulary (models.WordTokenizer). The readable
pref_<theme>.json datasets are grammatical English sentences, so MiraLLM needs sub-word units: a
word-level table would be huge and share nothing across inflections. This BPE is byte-level (like
GPT-2), so EVERY string encodes without ever needing an unknown token -- <|unk|> exists as a
reserved id but nothing can produce it -- and it is trained on the dataset corpus.

Interface parity with models.WordTokenizer so it is a drop-in for build_tokenizer/build_model:
    tok(text, add_special_tokens=False) -> {"input_ids": [...]}
    tok.pad_token_id / eos_token_id / unk_token_id / bos_token_id
    len(tok)                      vocabulary size (256 bytes + #merges + #specials)
    tok.decode(ids)               bytes back to text
    tok.save(path) / BPETokenizer.load(path)
    BPETokenizer.build(texts, num_merges=...)

ID LAYOUT, GPT-2's: the 256 raw bytes [0..255], then the merged symbols in merge order, then
the SPECIALS LAST. A symbol is a tuple of byte values; encoding applies the learned merges
greedily by rank, exactly as classic BPE.

The specials sit at the end because that is the only arrangement in which REGISTERING A NEW
ONE IS CHEAP. Put them first and every byte and every merge shifts by one the moment a token
is added, which invalidates every checkpoint and every cached token stream ever produced with
that vocabulary. Put them last and a new special is simply the next id: nothing already
trained changes meaning, and a model only needs its embedding matrix extended by a row --
which is exactly what `resize_token_embeddings` does elsewhere in the ecosystem.

The predefined set follows GPT-2's naming:

    <|endoftext|>   end of a document, and the bos/eos the pipeline generates against
    <|pad|>         padding, kept DISTINCT from eos
    <|unk|>         reserved; see below

GPT-2 itself uses one token and pads with it. A separate pad costs one id and removes a real
ambiguity: with pad == eos, a padded batch and a batch of empty completions have identical
ids, and only the attention mask tells them apart -- so a mask bug becomes a silent training
error rather than a loud one.

<|unk|> IS RESERVED RATHER THAN USED. A byte-level vocabulary never needs an unknown token:
every string decomposes into the 256 byte tokens at worst, so encode() cannot produce it and
no corpus can make it appear. It is kept as a real id because interfaces expect an
unk_token_id to exist, and because a reserved slot costs one embedding row and is there if a
later use wants it.

REGISTERING A TOKEN makes it an ATOM of the vocabulary:

    tok.register_special("<vp>")
    tok.encode("a <vp> b")        -> [.., <vp>, ..]   one id, never split into < v p >

which is what lets a later model attach a meaning to it -- a control marker, a role tag, a
modality boundary. Matching happens on the raw string before the byte-level pre-tokenizer, so
the token is never split and never merged with its neighbours. `encode_ordinary` is the
escape hatch that treats specials as ordinary text, and is the right call for untrusted input:
otherwise a scraped document containing the characters "<|endoftext|>" could insert a document
boundary of its own.
"""
import json
import os
import re
from collections import Counter

# PRE-TOKENIZER. Text is cut into chunks before any merge is applied, and merges never span a
# chunk, so this rule decides what the vocabulary is even able to learn. Three alternatives,
# tried in order:
#
#     \s*\d           one DIGIT, with any whitespace in front of it
#     \s*[^\s\d]+     a run of non-space non-digit, with its leading whitespace
#     \s+             whitespace on its own
#
# Every character is whitespace, a digit, or neither, so the three cover the string exactly:
# the chunks concatenate back to the input, and nothing can be dropped.
#
# DIGITS ARE SPLIT ONE BY ONE, as LLaMA and DeepSeek do. Left alone, BPE learns whichever
# number strings happen to be frequent -- "2015", " 100", "42" -- and the model must then do
# arithmetic over units that vary with the corpus rather than with the maths: 1000 is one
# token, 1001 is two, and the relationship between them is invisible. Per-digit tokens make
# place value explicit and uniform, which is what the chain-of-thought stage needs, since it
# is graded on whether the arithmetic is right. The cost is a few more tokens per number.
#
# Whitespace still attaches to what FOLLOWS it, which is how a run of indentation can become a
# single token -- better than GPT-2 on code, at the price of the token-boundary effect
# DeepSeek-V3 documents for prompts that end mid-line.
_PAT = re.compile(r"\s*\d|\s*[^\s\d]+|\s+")

# Bumped whenever _PAT changes. It goes into the tokenizer's fingerprint, because the same
# merges under a different pre-tokenizer produce different ids: without it, editing the rule
# above would leave every cached token stream keyed as though nothing had happened.
PRETOK_VERSION = 2

# HOW MANY PRE-TOKEN CHUNKS THE ENCODER REMEMBERS. `_merge_word` rescans every adjacent pair on
# each merge, so it is quadratic in the symbols of a chunk, and without this it redid that work
# in full for every occurrence of every word. Pre-token chunks are Zipfian: the same few
# thousand carry most of the corpus, so a chunk that has been merged once is almost always about
# to be merged again.
#
# THE CAP IS A CEILING ON MEMORY, AND ENTRIES ARE NOT EVICTED. The first chunks a corpus
# presents are the frequent ones, so a cache that stops accepting new entries when it is full
# keeps the ones that matter; an eviction policy in Python would cost more per lookup than the
# merges it saves. At roughly 260 bytes an entry this is about 130 MB.
CHUNK_CACHE_MAX = 500_000


class BPETokenizer:
    # The predefined set. EOS first so that adding a special later never moves it -- the one
    # id the generation loops, the packer and every checkpoint agree on.
    EOS_TOKEN = "<|end-of-session|>"
    PAD_TOKEN = "<|pad|>"
    UNK_TOKEN = "<|unknown|>"
    SPECIALS = [EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]
    SPECIAL_ALIASES = {"<|end|>": "<|endofmessage|>", "<|endoftext|>": EOS_TOKEN,
                       "<|unk|>": UNK_TOKEN}
    LAYOUT = "specials-last"          # written into the file; see load()

    def __init__(self, merges, build_history=None, specials=None):
        # merges: ordered list of (sym_a, sym_b), each sym a tuple of byte ints (priority = index)
        self.merges = [(tuple(a), tuple(b)) for a, b in merges]
        # per-merge training dynamics (empty unless this tokenizer was just build()-ed or loaded
        # from a file that carried them); see build() for the recorded fields.
        self.build_history = list(build_history or [])
        # signature of the corpus this vocabulary was trained on, when known: it is what
        # lets a later run confirm the corpus is unchanged and CONTINUE adding merges.
        self.corpus_sig = None
        self.specials = [self.SPECIAL_ALIASES.get(s, s) for s in (specials or self.SPECIALS)]
        if len(set(self.specials)) != len(self.specials):
            raise ValueError("special-token aliases must not occupy different token IDs")
        self._rebuild()

    def _rebuild(self):
        """(Re)derive every id table. Called on construction and after a special is registered,
        so there is ONE description of the layout rather than two that can disagree."""
        self.sym2id, nxt = {}, 0
        for b in range(256):                                 # 0..255: the raw bytes
            self.sym2id[(b,)] = nxt; nxt += 1
        for a, b in self.merges:                             # then the merges, in rank order
            self.sym2id[a + b] = nxt; nxt += 1
        self.n_base = nxt                                    # first special id
        self.special2id = {s: self.n_base + i for i, s in enumerate(self.specials)}
        self.id2special = {i: s for s, i in self.special2id.items()}
        self.n_special = len(self.specials)
        self.id2sym = {i: s for s, i in self.sym2id.items()}
        self.ranks = {(a, b): i for i, (a, b) in enumerate(self.merges)}
        # THE SAME RANKS, KEYED IN TWO LEVELS, for the encoder's inner loop. Probing
        # `ranks[(a, b)]` builds a tuple per probe, and the loop probes every adjacent pair of
        # every chunk it merges; `ranks2[a][b]` asks the same question and allocates nothing.
        self.ranks2 = {}
        for (a, b), i in self.ranks.items():
            self.ranks2.setdefault(a, {})[b] = i
        # CLEARED HERE AND NOWHERE ELSE. A chunk's ids depend only on the merges, their ranks
        # and sym2id, and all three are derived in this method -- so every path that can change
        # what a chunk encodes to passes through here, and none can leave a stale entry behind.
        self._chunk_cache = {}
        self._size = self.n_base + self.n_special
        self.eos_token_id = self.special2id[self.EOS_TOKEN]
        self.pad_token_id = self.special2id.get(self.PAD_TOKEN, self.eos_token_id)
        self.bos_token_id = self.eos_token_id                # no separate BOS; reuse EOS
        # <|unk|> is RESERVED, not reachable: a byte-level vocabulary never needs it, since
        # every string decomposes into the 256 byte tokens at worst, and encode() cannot
        # produce it. It is kept as a real id anyway -- interfaces expect an unk_token_id, and
        # a reserved slot costs one row and is there if a later use wants it. Falls back to
        # eos if a caller supplies a specials list without it.
        self.unk_token_id = self.special2id.get(self.UNK_TOKEN, self.eos_token_id)
        # Recogniser for the specials, LONGEST FIRST so that when one token is a prefix of
        # another the longer one wins -- with <|im|> and <|im_end|> both registered, the
        # shorter must not claim the first four characters of the longer.
        self._special_re = (re.compile("|".join(re.escape(t) for t in
                                                sorted(self.specials + [alias for alias in
                                                       ("<|endoftext|>", "<|unk|>")
                                                       if self.SPECIAL_ALIASES[alias] in self.special2id],
                                                       key=len, reverse=True)))
                            if self.specials else None)

    def register_special(self, token):
        """Add a special token and return its id; registering an existing one returns it
        unchanged, so this is safe to call repeatedly.

        The new id is the next one after the current vocabulary, so NOTHING already trained
        changes meaning. A model built before the call has an embedding matrix one row short:
        extend it (and the tied output head) before using the new id, exactly as
        resize_token_embeddings does. This is why the specials are last."""
        token = self.SPECIAL_ALIASES.get(token, token)
        if token in self.special2id:
            return self.special2id[token]
        if not isinstance(token, str) or not token:
            raise ValueError("a special token must be a non-empty string")
        # From here on encode() RECOGNISES the token: "a <vp> b" yields one id for <vp>
        # instead of the subwords of "<", "vp", ">". That is the whole point -- an atom the
        # model can attach a meaning to, which it cannot do with pieces that also occur in
        # ordinary text. The cost is that any text containing the literal characters now
        # produces the token, so untrusted input should go through encode_ordinary.
        self.specials = list(self.specials) + [token]
        self._rebuild()
        return self.special2id[token]

    # ----- construction ----------------------------------------------------- #
    @classmethod
    def _pretok(cls, text):
        return _PAT.findall(text)

    @classmethod
    def build(cls, texts, num_merges=10000, checkpoint_path=None, checkpoint_every=250,
              log=print, monitor=None, plot_every=200, init_merges=None, init_history=None,
              init_sig=None, min_freq=1, init_across_corpora=False):
        """Train byte-level BPE on `texts` (an iterable of strings).

        INCREMENTAL. The textbook loop recounts every adjacent pair in the whole corpus
        before each merge, which is O(corpus) per merge and quadratic overall -- hours on a
        corpus of this size. Here the pair counts are built ONCE and then maintained: a merge
        only touches the words that actually contain the merged pair, found through a
        pair -> word-index inverted index, and the best pair comes off a lazy max-heap
        instead of a full scan. Same greedy vocabulary, very different cost.

        Two further things make the inner loop cheap:

        * SYMBOLS ARE INTERNED TO INTEGERS. The natural representation -- a symbol is a
          tuple of byte values, a pair a tuple of two of those -- makes every dictionary
          operation hash a nested tuple, and this loop is nothing but dictionary operations.
          Symbols are therefore integer ids into a table and a pair is packed into ONE
          integer, `a << 20 | b`, so the hot dicts are keyed by small ints.

        * THE HEAP KEEPS AN INVARIANT INSTEAD OF CHURNING. Every pair whose count changed
          during a merge is pushed ONCE, at its final count, after the merge is applied. A
          correct entry therefore exists for every live pair, which means a stale entry can
          be discarded on sight -- no re-pushing, and no repeated pushes of one pair inside
          a single merge. (Discarding stale entries WITHOUT that invariant is the classic
          bug: it silently removes a still-competitive pair from consideration.)

        RESUMABLE, IN TWO SENSES.

        * An INTERRUPTED build continues: `checkpoint_path` writes the merges learned so far,
          with their history, to `<path>.partial` every `checkpoint_every` merges and on
          exit, and a later call reloads and REPLAYS them onto the freshly counted corpus --
          seconds -- before carrying on where it stopped.

        * A FINISHED but SHORTER vocabulary is EXTENDED: pass its merges as `init_merges`
          (with `init_history`, and `init_sig` if it recorded one) and raising `num_merges`
          continues from merge N rather than restarting at zero. This is what happens when
          the configured merge budget is increased.

        Both are guarded by a signature over the word table, so a changed corpus can never
        silently continue someone else's vocabulary; whichever source carries more merges is
        the one replayed.

        `monitor(history, merge)` is called every `plot_every` merges, the same live-figure
        cadence every trainer uses, so the merge dynamics are watchable while the vocabulary
        is still being learned rather than only at the end.
        """
        import heapq
        from tqdm import tqdm

        SHIFT = 20                                   # pair key = a << SHIFT | b
        MASK = (1 << SHIFT) - 1

        words = Counter()
        for t in tqdm(list(texts), desc="BPE: counting words", unit="seg", dynamic_ncols=True):
            for chunk in cls._pretok(t):
                words[chunk] += 1
        items = [(w, c) for w, c in words.items() if w]

        # MINIMUM FREQUENCY. A word seen once costs as much to maintain as one seen a million
        # times -- the merge loop's cost is driven by how many distinct words each pair occurs
        # in, not by how often those words appear -- while contributing a millionth of the
        # evidence for any merge. Dropping the tail therefore buys speed almost for free.
        #
        # Dropped words are dropped from the COUNTING ONLY. Nothing leaves the vocabulary,
        # because a byte-level vocabulary has nothing to leave: every string still encodes,
        # using whatever merges the frequent words justified. The only effect is that a merge
        # which ONLY rare words would have argued for is not learned, so those words cost a
        # token or two more.
        if min_freq > 1:
            kept = [(w, c) for w, c in items if c >= min_freq]
            occ, occ_kept = sum(c for _, c in items), sum(c for _, c in kept)
            if kept:
                log(f"BPE: min_freq={min_freq} keeps {len(kept):,} of {len(items):,} word "
                    f"types ({100 * len(kept) / max(len(items), 1):.1f}%), which are "
                    f"{100 * occ_kept / max(occ, 1):.2f}% of all occurrences")
                items = kept
            else:
                log(f"BPE: min_freq={min_freq} would discard every word type; ignored")

        # the signature covers min_freq through the word table it produced, so a partial build
        # started at one threshold is never resumed at another
        sig = cls._corpus_signature(items)

        # ---- intern symbols: id <-> bytes ---- #
        sym = [bytes([b]) for b in range(256)]       # ids 0..255 are the raw bytes
        wordsyms, freqs = [], []
        for w, c in items:
            enc = w.encode("utf-8")
            if enc:
                wordsyms.append(list(enc))           # byte values ARE their symbol ids
                freqs.append(c)

        n_special = len(cls.SPECIALS)
        total_bytes = sum(sum(len(sym[s]) for s in ws) * c for ws, c in zip(wordsyms, freqs))
        tok_count = sum(len(ws) * c for ws, c in zip(wordsyms, freqs))

        # ---- pair counts + inverted index, built once ---- #
        counts, where = {}, {}
        for i in tqdm(range(len(wordsyms)), desc="BPE: indexing pairs", unit="word",
                      dynamic_ncols=True):
            s, c = wordsyms[i], freqs[i]
            for j in range(len(s) - 1):
                k = s[j] << SHIFT | s[j + 1]
                counts[k] = counts.get(k, 0) + c
                w = where.get(k)
                if w is None:
                    where[k] = {i}
                else:
                    w.add(i)
        heap = [(-c, k) for k, c in counts.items() if c > 0]
        heapq.heapify(heap)

        def apply_merge(a, b, ab):
            """Merge symbol pair (a,b) into `ab` everywhere; returns its corpus frequency."""
            nonlocal tok_count
            key = a << SHIFT | b
            hit, touched = 0, set()
            for i in where.pop(key, ()):
                s = wordsyms[i]
                if len(s) < 2:
                    continue
                out, j, n, ln = [], 0, 0, len(s)
                while j < ln:
                    if j < ln - 1 and s[j] == a and s[j + 1] == b:
                        out.append(ab); j += 2; n += 1
                    else:
                        out.append(s[j]); j += 1
                if not n:
                    continue
                c = freqs[i]
                for j in range(ln - 1):              # retract the word's old pairs
                    k = s[j] << SHIFT | s[j + 1]
                    counts[k] = counts.get(k, 0) - c
                    touched.add(k)
                    w = where.get(k)
                    if w is not None:
                        w.discard(i)
                wordsyms[i] = out
                for j in range(len(out) - 1):        # and post its new ones
                    k = out[j] << SHIFT | out[j + 1]
                    counts[k] = counts.get(k, 0) + c
                    touched.add(k)
                    w = where.get(k)
                    if w is None:
                        where[k] = {i}
                    else:
                        w.add(i)
                hit += n * c
                tok_count -= n * c
            # ONE push per changed pair, at its final count: this is the heap invariant
            for k in touched:
                v = counts.get(k, 0)
                if v > 0:
                    heapq.heappush(heap, (-v, k))
            return hit

        def intern(a, b):
            """Symbol id for the concatenation of `a` and `b`, creating it if new."""
            s = sym[a] + sym[b]
            sym.append(s)
            return len(sym) - 1

        merges, history = [], []
        # per-merge learning dynamics, recorded so bpe_dynamics can show how the vocabulary is
        # learned over the merge "steps" (bytes are conserved, so total_bytes is a constant
        # that turns the corpus token count into a compression ratio):
        #   merge          1-based merge index (the "step")
        #   pair_freq      corpus frequency of the pair just merged (the greedy objective)
        #   distinct_pairs number of distinct adjacent pairs still available before this merge
        #   vocab_size     3 specials + 256 bytes + merges so far
        #   tokens         total corpus length in tokens AFTER this merge
        #   new_token_len  #bytes in the symbol this merge created
        #   bytes_per_token total_bytes / tokens (compression, rises toward longer tokens)

        # ---- resume: replay whichever source carries the most merges ---- #
        done = cls._load_partial(checkpoint_path, sig)
        if init_merges:
            # A DIFFERENT CORPUS IS USUALLY A MISTAKE AND SOMETIMES THE POINT. Extending a
            # vocabulary onto new text is exactly what language adaptation does, and there
            # the signatures MUST differ -- so the caller says which case this is rather
            # than the signature deciding for it. Without `init_across_corpora`, stage 12
            # asked to extend, was silently told 'different corpus, starting from scratch',
            # learned a fresh vocabulary over hours, and only then failed its own check that
            # the base ids had survived.
            if init_sig is not None and init_sig != sig and not init_across_corpora:
                log("BPE: the existing vocabulary was trained on a DIFFERENT corpus; "
                    "starting from scratch (pass init_across_corpora=True to extend it "
                    "anyway, which is what language adaptation wants)")
            else:
                if init_sig is None:
                    log("BPE: the existing vocabulary carries no corpus signature; "
                        "assuming it belongs to this corpus and extending it")
                elif init_sig != sig:
                    log("BPE: extending a vocabulary trained on a DIFFERENT corpus, as "
                        "asked. The base merges are kept in order, so every id already "
                        "learned keeps its meaning and only new symbols are appended.")
                if not done or len(init_merges) > len(done[0]):
                    done = (list(init_merges), list(init_history or []))
        if done:
            m_done, history = done
            m_done = m_done[:num_merges]             # a shorter budget: keep the prefix
            history = history[:num_merges]
            sid = {bytes(sym[i]): i for i in range(256)}
            for a_t, b_t in tqdm(m_done, desc="BPE: replaying checkpoint", unit="merge",
                                 dynamic_ncols=True):
                a, b = sid[bytes(a_t)], sid[bytes(b_t)]
                ab = intern(a, b)
                sid[sym[ab]] = ab
                apply_merge(a, b, ab)
                merges.append((a_t, b_t))
            log(f"BPE: resumed at merge {len(merges):,}/{num_merges:,}")
            if len(merges) >= num_merges:
                tok = cls(merges[:num_merges], build_history=history[:num_merges])
                tok.corpus_sig = sig
                return tok

        bar = tqdm(range(len(merges), num_merges), desc="BPE: learning merges", unit="merge",
                   initial=len(merges), total=num_merges, dynamic_ncols=True)
        for _ in bar:
            best = None
            while heap:                              # lazy heap: the invariant makes it safe
                nc, k = heapq.heappop(heap)          # to DISCARD a stale entry
                if counts.get(k, 0) == -nc and -nc > 0:
                    best = (k, -nc)
                    break
            if best is None:
                break
            k, freq = best
            if freq < 2:                             # no pair repeats; nothing left to gain
                break
            a, b = k >> SHIFT, k & MASK
            n_pairs = len(heap)
            ab = intern(a, b)
            if not apply_merge(a, b, ab):
                sym.pop()                            # the pair vanished under earlier merges
                continue
            merges.append((tuple(sym[a]), tuple(sym[b])))
            history.append({"merge": len(merges), "pair_freq": int(freq),
                            "distinct_pairs": n_pairs,
                            "vocab_size": n_special + 256 + len(merges),
                            "tokens": int(tok_count), "new_token_len": len(sym[ab]),
                            "bytes_per_token": total_bytes / max(tok_count, 1)})
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(pair_freq=int(freq), vocab=n_special + 256 + len(merges),
                                bytes_per_tok=f"{total_bytes / max(tok_count, 1):.2f}")
            if (checkpoint_path and checkpoint_every > 0
                    and len(merges) % checkpoint_every == 0):
                cls._save_partial(checkpoint_path, sig, merges, history)
            if monitor and plot_every > 0 and len(merges) % plot_every == 0:
                try:
                    monitor(history, len(merges))    # live bpe_dynamics.pdf
                except Exception as e:               # plotting must never kill a long build
                    log(f"[plot] bpe: {e}")
        tok = cls(merges, build_history=history)
        tok.corpus_sig = sig                         # lets a later run verify and EXTEND it
        if checkpoint_path:                          # final state, marked complete
            cls._save_partial(checkpoint_path, sig, merges, history, done=True)
        return tok

    # ----- resumable-build checkpoint --------------------------------------- #
    @staticmethod
    def _corpus_signature(items):
        """Identifies the word table, so a partial build is only resumed on ITS corpus."""
        import hashlib
        h = hashlib.sha1()
        h.update(str(len(items)).encode())
        for w, c in sorted(items)[:20000]:           # a large, deterministic sample
            h.update(w.encode("utf-8", "replace")); h.update(str(c).encode())
        h.update(str(sum(c for _, c in items)).encode())
        return h.hexdigest()

    @staticmethod
    def _partial_path(path):
        return (path or "") + ".partial"

    @classmethod
    def _save_partial(cls, path, sig, merges, history, done=False):
        p = cls._partial_path(path)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"sig": sig, "done": done, "n_merges": len(merges),
                       "merges": [[list(a), list(b)] for a, b in merges],
                       "build_history": history}, f)
        os.replace(tmp, p)                           # atomic: never a half-written resume file

    @classmethod
    def _load_partial(cls, path, sig):
        """(merges, history) from a matching partial build, or None."""
        p = cls._partial_path(path)
        if not path or not os.path.isfile(p):
            return None
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            return None
        if d.get("sig") != sig or not d.get("merges"):
            return None
        return ([(tuple(a), tuple(b)) for a, b in d["merges"]], d.get("build_history", []))

    # ----- encode / decode -------------------------------------------------- #
    def _merge_word(self, syms):
        """The byte symbols of one chunk, merged in rank order. `syms` is modified in place.

        THE PAIR RANKS ARE CARRIED, NOT RECOMPUTED. Merging at position i changes which pairs
        exist only at i-1 and i, so the ranks of every other adjacent pair are still what they
        were. This kept a parallel list of them and re-derived all of them on every pass: on
        source text that was 148 dictionary probes per chunk, each allocating a tuple for the
        key, and it was four fifths of the tokenisation stage's time.

        The result is unchanged. The loop still takes the lowest-ranked adjacent pair, leftmost
        on a tie, and stops when no adjacent pair is a merge -- the same rule, asked of the same
        table.
        """
        n = len(syms)
        if n < 2:
            return syms
        ranks2 = self.ranks2
        pair = [None] * (n - 1)                    # rank of the pair starting at i
        for i in range(n - 1):
            row = ranks2.get(syms[i])
            if row is not None:
                pair[i] = row.get(syms[i + 1])
        while True:
            best, bi = None, -1
            for i, r in enumerate(pair):
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi < 0:
                return syms
            syms[bi:bi + 2] = [syms[bi] + syms[bi + 1]]
            del pair[bi]
            for j in (bi - 1, bi):                 # only these two pairs changed
                if 0 <= j < len(syms) - 1:
                    row = ranks2.get(syms[j])
                    pair[j] = row.get(syms[j + 1]) if row is not None else None

    def _chunk_ids(self, chunk):
        """One pre-token chunk -> its ids, remembered. See CHUNK_CACHE_MAX.

        A REGISTERED SPECIAL CANNOT COME OUT OF THIS. `_merge_word` produces base bytes and
        merges, and their ids are below n_base; the specials sit above it and are inserted by
        `encode` around the spans this is called on. So registering one does not change any
        answer here, and the cache survives it -- `_rebuild` clears it anyway."""
        got = self._chunk_cache.get(chunk)
        if got is None:
            syms = self._merge_word([(b,) for b in chunk.encode("utf-8")])
            got = tuple(self.sym2id[s] for s in syms)        # base bytes + merges always present
            if len(self._chunk_cache) < CHUNK_CACHE_MAX:
                self._chunk_cache[chunk] = got
        return got

    def encode_ordinary(self, text):
        """Text -> ids, treating EVERY character as text. A registered special appearing in
        the string encodes as its literal bytes and cannot be produced. tiktoken's name for
        the same idea, and the right call for untrusted input -- a scraped document containing
        the characters "<|endoftext|>" should not be able to insert a document boundary.

        THE CACHE IS READ HERE AND `_chunk_ids` IS CALLED ONLY ON A MISS. A dict lookup that
        hits costs one bytecode; reaching it through a method call costs a frame, and on a
        corpus this size the hits are the overwhelming majority of the chunks. `_chunk_ids`
        remains the one place a chunk is merged and remembered.
        """
        ids = []
        cache = self._chunk_cache
        chunk_ids = self._chunk_ids
        for chunk in self._pretok(text):
            got = cache.get(chunk)
            ids.extend(got if got is not None else chunk_ids(chunk))
        return ids

    def encode(self, text, split_specials=True):
        """Text -> ids, with registered specials RECOGNISED as single tokens.

        This is what registering a token is for: after `tok.register_special("<vp>")`, the
        string "a <vp> b" encodes with one id for <vp> rather than splitting it into "<", "vp"
        and ">". The token becomes an atom the model can learn a meaning for -- a control
        marker, a role tag, a modality boundary -- which it cannot if it arrives as four
        subwords that also occur in ordinary text.

        The specials are matched on the RAW STRING before the byte-level pre-tokenizer runs,
        so they are never split and never merged with their neighbours. Pass
        split_specials=False, or call encode_ordinary, to treat them as plain text."""
        if not split_specials or self._special_re is None:
            return self.encode_ordinary(text)
        ids, pos = [], 0
        for m in self._special_re.finditer(text):
            if m.start() > pos:
                ids.extend(self.encode_ordinary(text[pos:m.start()]))
            ids.append(self.special2id[self.SPECIAL_ALIASES.get(m.group(0), m.group(0))])
            pos = m.end()
        if pos < len(text):
            ids.extend(self.encode_ordinary(text[pos:]))
        return ids

    def __call__(self, text, add_special_tokens=False, split_specials=True):
        ids = self.encode(text, split_specials=split_specials)
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        """Ids back to text. Specials carry no bytes; with skip_special_tokens=False they are
        rendered in their written form, which is what makes a transcript readable when the
        question is where the model actually emitted an end of text."""
        buf, out = bytearray(), []
        for i in ids:
            i = int(i)
            if i >= self.n_base:                             # a special: no bytes to add
                if not skip_special_tokens:
                    tokn = self.id2special.get(i)
                    if tokn:
                        out.append(buf.decode("utf-8", errors="replace")); buf = bytearray()
                        out.append(tokn)
                continue
            s = self.id2sym.get(i)
            if s:
                buf.extend(s)
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    # ----- parity helpers --------------------------------------------------- #
    def __len__(self):
        return self._size

    def get_vocab(self):
        v = {bytes(s).decode("utf-8", "replace"): i for s, i in self.sym2id.items()}
        v.update(self.special2id)
        return v

    def convert_tokens_to_ids(self, t):
        if t in self.special2id:
            return self.special2id[t]
        ids = self.encode(t)
        return ids[0] if ids else self.unk_token_id

    # ----- persistence ------------------------------------------------------ #
    def vocab_records(self):
        """Full vocabulary as an id-ordered list of records, one per token:
            {"id": int, "token": readable-string, "bytes": [byte ints] | null}
        `token` is the UTF-8 decoding (invalid bytes shown as U+FFFD) for reading; `bytes` is the
        exact, lossless byte sequence (null for the specials, which carry no bytes)."""
        out = []
        for i in range(self._size):
            if i >= self.n_base:                             # the specials, at the end
                out.append({"id": i, "token": self.id2special[i], "bytes": None})
            else:
                sym = self.id2sym[i]
                out.append({"id": i, "token": bytes(sym).decode("utf-8", "replace"),
                            "bytes": list(sym)})
        return out

    def dumps(self):
        """The tokenizer as the JSON text `save` writes and `loads` reads."""
        return json.dumps({"type": "byte_bpe",
                           "vocab_size": self._size,
                           "corpus_sig": self.corpus_sig,
                           "layout": self.LAYOUT,   # see load(): an older file means other ids
                           "specials": self.specials,
                           "vocab": self.vocab_records(),
                           "merges": [[list(a), list(b)] for a, b in self.merges],
                           "build_history": self.build_history},
                          ensure_ascii=False, indent=1)

    def save(self, path):
        """Write the tokenizer as human-readable JSON. Keeps FULL information: the ordered `merges`
        (needed to encode) AND the complete `vocab` (readable token + exact bytes for every id) so
        the file can be inspected/printed/debugged without re-deriving anything."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.dumps())

    SPECIAL_TOKENS_FILE = "special_tokens.json"
    ADDED_TOKENS_FILE = "added_tokens.json"

    @classmethod
    def load(cls, path):
        """The tokenizer written at `path`, PLUS whatever its sidecars append to it."""
        with open(path, "r", encoding="utf-8") as fh:
            tok = cls.loads(fh.read(), where=path)
        directory = os.path.dirname(os.path.abspath(path))
        for name in (cls.SPECIAL_TOKENS_FILE, cls.ADDED_TOKENS_FILE):
            tok.append_from(os.path.join(directory, name))
        return tok

    def append_from(self, path):
        """Register the tokens named by the sidecar at `path`, in order. Returns their ids.

        TWO SHAPES ARE ACCEPTED, and one of them is a promise. A LIST is registration order and
        nothing more:

            ["<|user|>", "<|assistant|>", "<|end|>"]

        A MAPPING pins the id each token must land on:

            {"<|user|>": 58259, "<|assistant|>": 58260, "<|end|>": 58261}

        and is registered in id order, then CHECKED. A pinned id that does not come out where it
        was pinned means this sidecar belongs to a different base vocabulary -- the file it is
        beside has a different number of merges -- and every id in it would mean something else.
        That is refused rather than loaded, because a tokenizer whose ids are off by a constant
        does not fail, it talks nonsense.

        A missing sidecar is not an error: most vocabularies have none."""
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
        if isinstance(spec, dict):
            order = [t for t, _ in sorted(spec.items(), key=lambda kv: int(kv[1]))]
            pinned = {t: int(i) for t, i in spec.items()}
        elif isinstance(spec, list):
            order, pinned = [str(t) for t in spec], {}
        else:
            raise SystemExit(f"[bpe] {path}: expected a list of tokens or a {{token: id}} map")
        got = {t: self.register_special(t) for t in order}
        wrong = {t: (pinned[t], got[t]) for t in got if t in pinned and pinned[t] != got[t]}
        if wrong:
            token, (want, have) = sorted(wrong.items())[0]
            raise SystemExit(
                f"[bpe] {path}: {token} is pinned to id {want} but lands on {have}.\n"
                f"      This sidecar belongs to a different base vocabulary than the bpe.json "
                f"beside it, so every id it names would mean something else.")
        return got

    @classmethod
    def loads(cls, text, where="<string>"):
        """The tokenizer in a JSON STRING, without going through a file."""
        # `merges` is the source of truth for encoding; `vocab` in the file is informational and
        # is rebuilt deterministically from the merges here.
        d = json.loads(text)
        path = where
        # A FILE WITHOUT `layout` PREDATES specials-last, and its ids mean something else: the
        # specials were 0..2 and every byte and merge sat three higher. Loading it under the
        # current layout would not fail -- it would silently return a tokenizer that decodes
        # every checkpoint's output shifted by three, which is far worse than refusing.
        layout = d.get("layout")
        if layout != cls.LAYOUT:
            raise SystemExit(
                f"[bpe] {path} was written with the old id layout "
                f"({layout or 'specials-first, pre-<|endoftext|>'}), which is not compatible: "
                f"its ids are offset by the specials.\n"
                f"      Delete it and re-run ./old_scripts/stage3_pretrain_bpe_tokenizer.sh. Any checkpoint "
                f"or token stream built with it must be rebuilt too -- the cache directory is "
                f"named after the tokenizer's fingerprint, so that happens by itself.")
        tok = cls([(tuple(a), tuple(b)) for a, b in d["merges"]],
                  build_history=d.get("build_history", []),
                  specials=d.get("specials"))
        tok.corpus_sig = d.get("corpus_sig")
        return tok
