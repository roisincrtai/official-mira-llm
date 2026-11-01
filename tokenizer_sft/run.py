"""
tokenizer_sft/run.py -- vocabulary adaptation: EXTEND a trained BPE with merges for a new corpus.

    extend(base_path, out_path, texts, extra_merges, log)  -> the extended tokenizer
    extended_path(dataset)                                  where it is written
    remap_specials(state, old_tok, new_tok, log)            a checkpoint's rows, rearranged

    python -m tokenizer_sft.run --lang_sft_dataset <name> [--lang_sft_merges N]

WHY EXTEND RATHER THAN RETRAIN. A vocabulary trained on English tokenises Irish badly -- not
wrongly, a byte-level BPE can always fall back to raw bytes, but expensively, splitting common
Irish words into four or five tokens where an Irish vocabulary would use one. The fix is more
merges, not different ones. Training a FRESH vocabulary would tokenise Irish beautifully and
throw away the pretrained model with it: new ids mean the embedding rows learned in stage 6
now sit against different tokens, so the checkpoint could not be used and the model would have
to be trained from scratch.

So the English merges are kept, in their order, and Irish merges are appended after them. The
ids already learned keep their meaning and the model only needs more rows -- which is exactly
what `resize_token_embeddings` is for.

------------------------------------------------------------------------------------------
THE SPECIALS MOVE, AND A CHECKPOINT MUST BE REARRANGED FOR IT. This is the one thing about
this that is not free, and it is silent when it is wrong.

The id layout is bytes, then merges, then THE SPECIALS LAST (tokenizer/bpe.py). Appending
merges therefore pushes the specials up by exactly the number appended:

    before:  [0..255 bytes][256..50,255 merges][50,256 eos][50,257 pad][50,258 unk]
    after:   [0..255 bytes][256..50,255 merges][50,256..N new merges][N+1 eos][N+2 pad][N+3 unk]

Every byte and every original merge keeps its id -- verified, not assumed. The three specials
do not. Growing the embedding naively would leave the trained <|endoftext|> row sitting at an
id that is now an Irish merge, and the new <|endoftext|> randomly initialised: a model that
runs, trains, and has lost the token every document ends with. `remap_specials` moves those
rows to where they now belong, and reports the move.
------------------------------------------------------------------------------------------
"""
import os

import default_config as config
import helpers
from helpers import visualization as viz
from tokenizer import BPETokenizer
from tokenizer.run import with_extra_specials

STAGE = "tokenizer_sft"


def tokenizer_name(dataset, tokenizer=""):
    """The directory under checkpoints/ that holds this vocabulary: `bpe-sft-<dataset>`.

    NAMED FOR THE CORPUS BY DEFAULT, because a vocabulary is a property of the corpus it was
    extended for: two languages adapted from the same base must not overwrite each other, and a
    name that carries the dataset makes that impossible rather than merely unlikely.

    `--tokenizer` overrides it outright, for the case where one vocabulary is meant to serve
    several corpora -- `bpe-sft-coder` over a handful of code datasets -- and the dataset name
    would therefore be the wrong label. Whatever is chosen here, stage 13 must be given the
    same name, which is why both stages take the same flag."""
    # HYPHENS, not underscores: checkpoints/bpe-sft-<dataset>/, beside checkpoints/bpe/.
    return str(tokenizer).strip() or f"bpe-sft-{dataset}"


def extended_path(dataset, ckdir=None, tokenizer=""):
    """checkpoints/bpe-sft-<dataset>/bpe.json -- the adapted vocabulary's home.

    Or checkpoints/<--tokenizer>/bpe.json when a name was given. One file, at the top of the
    checkpoint tree beside the base vocabulary rather than buried under a training stage's
    output, because it is an INPUT to stage 13 and not an artefact of it."""
    root = ckdir or config.CHECKPOINT_DIR
    return os.path.join(root, tokenizer_name(dataset, tokenizer), "bpe.json")


def dataset_name(args):
    """The corpus to extend onto: --only wins, then --lang_sft_dataset, then the yaml."""
    return (getattr(args, "only", "") or getattr(args, "lang_sft_dataset", "")
            or config.LANG_SFT["dataset"])


def corpus_dir(args):
    """Where that corpus is. A bare name resolves under data/download/; a path is used as
    given, so a corpus outside the download tree needs no configuration to reach."""
    name = dataset_name(args)
    return name if os.path.isdir(name) else config.dataset_dir(name)


def target_path(ctx):
    """Where this run will write, from the run's own arguments."""
    args = ctx["args"]
    return extended_path(dataset_name(args), ctx["ckdir"],
                         getattr(args, "tokenizer", ""))


def extend(base_path, out_path, texts, extra_merges, log=print, plotdir=None,
           plot_every=200, checkpoint_every=250, min_freq=1):
    """The base tokenizer plus `extra_merges` merges learned on `texts`. Returns it, saved.

    RESUMABLE AND REUSED, on the same rule stage 3 uses: a saved extension already carrying at
    least the requested number of merges is loaded as it is, and a shorter one is CONTINUED
    rather than restarted. The merge budget is part of "done".

    `texts` is only consulted when merges actually have to be learned, so a resumed run does
    not pay to assemble the corpus it is not going to read.

    The base is loaded first and its merges are handed to the builder as the starting point,
    which is what makes the result an extension rather than a new vocabulary: builder ranks
    0..len(base.merges)-1 are the base's own, in order, and everything after them is new."""
    base = BPETokenizer.load(base_path)
    target = len(base.merges) + int(extra_merges)

    have = None
    if os.path.isfile(out_path):
        try:
            have = BPETokenizer.load(out_path)
        except Exception:                                      # noqa: BLE001
            have = None                                        # stale/old layout -> rebuild
    if have is not None and have.merges[:len(base.merges)] != base.merges:
        raise SystemExit(
            f"[bpe-sft] {out_path} does not start with the base vocabulary at {base_path}.\n"
            f"          It was extended from a DIFFERENT base, so its ids do not mean what\n"
            f"          this run's checkpoints expect. Delete it, or point --lang_sft_dataset\n"
            f"          somewhere else.")
    if have is not None and len(have.merges) >= target:
        log(f"[bpe-sft] reusing {out_path}: {len(have.merges):,} merges "
            f"({len(have.merges) - len(base.merges):,} beyond the base), vocab {len(have):,}")
        return with_extra_specials(have)

    start = have if have is not None else base
    log(f"[bpe-sft] extending {base_path} ({len(base.merges):,} merges) by "
        f"{extra_merges:,} -> {target:,}, from {len(start.merges):,}")

    def monitor(history, merge):
        if plotdir and history:
            viz.bpe_monitor(history, plotdir, "")

    tok = BPETokenizer.build(texts, num_merges=target, checkpoint_path=out_path,
                             checkpoint_every=checkpoint_every, log=log, min_freq=min_freq,
                             monitor=monitor if plotdir else None, plot_every=plot_every,
                             init_merges=start.merges,
                             init_history=start.build_history,
                             init_sig=start.corpus_sig,
                             # THE CORPUS IS SUPPOSED TO BE DIFFERENT. That is the whole of
                             # stage 12: the base vocabulary was trained on English and is
                             # being extended onto another language, so the two signatures
                             # cannot match and a mismatch must not discard the base merges.
                             # The invariant below still checks that they survived.
                             init_across_corpora=True)
    tok = with_extra_specials(tok)
    tok.save(out_path)

    # THE INVARIANT, CHECKED RATHER THAN TRUSTED. Everything downstream -- the pretrained
    # embedding, every cached token stream, every checkpoint -- assumes the base ids survived.
    # It costs one pass over the merge list to know instead of assume.
    if tok.merges[:len(base.merges)] != base.merges:
        raise SystemExit(
            "[bpe-sft] the extension did not preserve the base merges. Refusing to save a "
            "vocabulary whose ids do not mean what the pretrained checkpoint learned.")
    helpers.table("extended vocabulary", [
        ("base", base_path),
        ("base merges", helpers.count(len(base.merges))),
        ("merges added", helpers.count(len(tok.merges) - len(base.merges))),
        ("vocabulary", f"{len(base):,} -> {len(tok):,}"),
        ("base ids preserved", "yes: every byte and every base merge keeps its id"),
        ("specials moved", f"{base.eos_token_id} -> {tok.eos_token_id} (eos), and pad/unk "
                           f"with it -- the layout puts specials last"),
        ("written to", out_path)], out=log)
    return tok


def remap_specials(state, old_tok, new_tok, log=print):
    """A state dict trained on `old_tok`, rearranged for `new_tok`'s id layout.

    Returns (state, n_new_rows). Only the embedding is touched -- the output head is TIED to
    it, so moving a row moves both -- and only the special rows move, because everything below
    them keeps its id.

    Called BEFORE the model is resized: the rows are placed at their new indices in a matrix
    that is about to grow, so `resize_token_embeddings` then initialises exactly the rows
    nothing was moved into (the new merges).

    A no-op when the specials did not move, which is the case for every ordinary stage."""
    import torch
    shift = new_tok.n_base - old_tok.n_base
    if shift <= 0 or "tok.weight" not in state:
        return state, 0
    w = state["tok.weight"]
    n_special = min(old_tok.n_special, new_tok.n_special)
    grown = torch.empty(len(new_tok), w.shape[1], dtype=w.dtype)
    # the new rows are filled with the mean of the old ones, the same place
    # resize_token_embeddings puts a token of no particular meaning yet
    grown[:] = w.mean(0, keepdim=True)
    grown[:old_tok.n_base] = w[:old_tok.n_base]                       # bytes + base merges
    grown[new_tok.n_base:new_tok.n_base + n_special] = \
        w[old_tok.n_base:old_tok.n_base + n_special]                  # the specials, moved
    state = dict(state)
    state["tok.weight"] = grown
    if "head.weight" in state:            # tied, but written out; keep the two consistent
        state["head.weight"] = grown
    log(f"[bpe-sft] embedding remapped for the extended vocabulary: "
        f"{old_tok.n_base:,} base rows kept in place, {n_special} special rows moved "
        f"{old_tok.n_base:,} -> {new_tok.n_base:,}, {shift:,} new rows initialised at the mean")
    return state, shift


class LazyTexts:
    """An iterable that materialises only when it is actually iterated.

    `extend` consults its corpus ONLY when merges have to be learned, so a resumed or
    already-finished extension must not pay to assemble a multi-gigabyte corpus it will not
    read. Handing it a list would read the corpus before that decision was made."""

    def __init__(self, make):
        self._make = make
        self._items = None

    def __iter__(self):
        if self._items is None:
            self._items = self._make()
        return iter(self._items)

    def __len__(self):
        if self._items is None:
            self._items = self._make()
        return len(self._items)


def build(ctx):
    """Stage 12 itself: the extended vocabulary for this corpus. Returns (tokenizer, path).

    THE CORPUS IS READ WITH THE BASE TOKENIZER'S OWN PRE-TOKENIZER, which is what
    BPETokenizer.build does with any text: the pre-token pattern and the byte alphabet are
    properties of the class, not of a trained vocabulary, so the words counted here are split
    exactly as stage 3 split its own. Only the merges learned from them are new.

    The corpus text is a GENERATOR of documents, not a list: this is a multi-gigabyte corpus
    and the merge counter consumes it once."""
    # helpers.dataset_helpers, NOT the top-level `datasets`. `import datasets as dsets`
    # resolves to Hugging Face's package, which is installed and has no load_pretrain_corpus --
    # so it imported cleanly and failed with AttributeError only once the corpus was actually
    # read, several minutes into the stage. lang_sft/run.py:53 spells it the same way.
    from helpers import dataset_helpers as dsets

    args, log = ctx["args"], ctx["log"]
    d = corpus_dir(args)
    if not os.path.isdir(d):
        raise SystemExit(
            f"[bpe-sft] corpus directory does not exist: {d}\n"
            f"          No stage downloads. Fetch it first:\n"
            f"              ./run_scripts/run_download_data.sh --only {dataset_name(args)}")
    out = target_path(ctx)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def texts():
        docs, n_files = dsets.load_pretrain_corpus(
            d, config.PRETRAIN["max_words"],
            exclude_dirs=config.PRETRAIN["exclude_dirs"],
            text_column=config.PRETRAIN["text_column"])
        log(f"[bpe-sft] vocabulary corpus: {n_files:,} files, {len(docs):,} documents")
        return [r["chosen"] for r in docs]

    plotdir = os.path.join(config.PLOT_DIR, "bpe-sft", dataset_name(args))
    # saved_models/ FIRST -- the base this vocabulary EXTENDS is the published one when there
    # is no local build; tokenizer.run.base_path decides, not config.BPE_PATH.
    from tokenizer.run import base_path
    tok = extend(base_path(log), out, LazyTexts(texts),
                 getattr(args, "lang_sft_merges", 0) or config.LANG_SFT["extra_merges"],
                 log=log, plotdir=plotdir,
                 plot_every=args.plot_every_steps,
                 checkpoint_every=args.checkpoint_every_steps,
                 min_freq=config.BPE.get("min_freq", 1))
    return tok, out


def main():
    """Standalone: extend the vocabulary and stop, without training anything on it."""
    from helpers import common
    args = common.parse_args(stage="tokenizer_sft")
    ctx = common.setup(args, need_pairs=False)
    log = ctx["log"]
    log("")
    log("=" * 78)
    log(f"STAGE 12: extend the vocabulary for {dataset_name(args)!r}")
    log(f"          -> {target_path(ctx)}")
    log("=" * 78)
    log("")
    tok, out = build(ctx)
    log("")
    log(f"[bpe-sft] done. Stage 13 reads this with:")
    log(f"          ./old_scripts/stage13_lang_sft.sh --only {dataset_name(args)} "
        f"--tokenizer {tokenizer_name(dataset_name(args), getattr(args, 'tokenizer', ''))}")
    return tok, out


if __name__ == "__main__":
    main()
