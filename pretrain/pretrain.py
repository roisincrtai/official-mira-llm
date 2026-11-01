"""
pretrain.py -- stage 6's logic: language-model pretraining.

Owns everything specific to pretraining and nothing else: which corpus is read, how it is
reported, which prompts the generation previews use, and which budget and learning rate the
shared LM loop (lm.train) is run with. run.py is only the command-line wrapper.

    load_corpus(tok, corpus_dir, log)  -> (docs, n_files, n_tokens), pre-tokenised and cached
    run(ctx, corpus)        -> the pretrained model

State space diagnostics (memory horizon, selectivity, residual write ratio) are recorded every
--plot_every_steps steps and drawn in the second row of outputs/plots/pretrain/.

The corpus is every file with a CORPUS_EXTENSIONS extension under the pretraining
corpus (prose, markdown, source code, parquet), scanned recursively minus
config.PRETRAIN["exclude_dirs"], packed into ~max_words-word documents that never cross a
file boundary. Nothing outside the pretraining corpus is read.
"""
import os

import default_config as config
import helpers
from model import lm

STAGE = "pretrain"


def load_corpus(tok, corpus_dir, log=print, ctx_batch=0, ctx_len=0,
                progress_stage=STAGE, epochs=0):
    """Pre-tokenised pretraining documents (cached under cache/tokens/bpe_<256+merges>_<fp>/), reported
    as a table: the file count and the total token count the scan must announce.

    `corpus_dir` is passed in rather than read from the configuration, because it depends on
    the scheme being trained: tiny and -s ship with one, -m and -l do not. An empty path is
    refused outright instead of scanned -- a scan of nothing reports "0 files" and reads like
    an empty corpus rather than like a missing setting.

    IT IS A LIST. Several corpora train one model, and they stay SEPARATE all the way down:
    one token stream each, one signature each, one cache directory each. Nothing is
    concatenated on disk. Only the batch sampler sees them together, drawing each sequence
    from one corpus with probability proportional to its token count, so the mixture is by
    tokens and a document is never spliced across corpora. The consequence that matters: a
    corpus can be ADDED to a scheme whose other corpora are already tokenised, and those
    streams are found by their unchanged signatures and reused byte for byte. A run reading
    them does not restart and does not re-tokenise; it simply has more to read."""
    roots = [d for d in (corpus_dir if isinstance(corpus_dir, (list, tuple)) else [corpus_dir])
             if d]
    if not roots:
        raise SystemExit(
            "[pretrain] no pretraining corpus configured for this scheme.\n"
            "           Every scheme has one in default_config.PRETRAIN_CORPUS; fetch it with\n"
            "               ./run_scripts/run_download_data.sh\n"
            "           or set PRETRAIN_DIR in config/pretrain/config_user.yaml (--pretrain_dir).")
    missing = [d for d in roots if not os.path.isdir(d)]
    if missing and epochs == 1:
        raise ValueError("One-epoch corpus directories are missing: " + ", ".join(missing))
    if missing and len(missing) == len(roots):
        raise SystemExit(
            f"[pretrain] corpus director{'y does' if len(missing) == 1 else 'ies do'} not "
            f"exist:\n" + "".join(f"             {d}\n" for d in missing) +
            f"           No stage downloads. Run:  python -m tools.download_data")
    for d in missing:
        # SOME of them present is not a failure: a scheme may list a corpus this machine has
        # not fetched. Say so loudly and train on what is here, rather than dying on a corpus
        # that is optional -- but never pass over it in silence.
        log(f"[pretrain] corpus directory missing, SKIPPED: {d}")
    roots = [d for d in roots if os.path.isdir(d)]
    corpus, n_files, n_tok = helpers.load_token_corpora(
        progress_stage, roots, tok, config.PRETRAIN["max_words"],
        exclude_dirs=config.PRETRAIN["exclude_dirs"], log=log,
        text_column=config.PRETRAIN["text_column"])
    if corpus:
        # WHAT WAS LOADED, stated plainly and in full. A corpus is the one input to a run that
        # the checkpoint does not record, so if the log does not say which directory was read
        # and how many tokens came out of it, nothing later can establish what the model was
        # trained on. The epoch figure is derived rather than asserted: it is what the step
        # budget actually buys over THIS corpus, which is how a mismatch between the two gets
        # noticed at the start of a run rather than at the end.
        seen = config.PRETRAIN["steps"] * ctx_batch * ctx_len if ctx_batch and epochs != 1 else 0
        # EVERY CORPUS NAMED, with its own stream and its own share of the mixture. A single
        # "dataset: x" line would hide the fact that two more were read, and the share is the
        # number that decides what the model actually sees: sampling is proportional to
        # tokens, so a 2 BT corpus beside a 100 MT one is a 95/5 mixture, not a 50/50 one.
        # The names come from the STREAMS, not from `roots`: a listed corpus that had nothing
        # to read was skipped and reported, and a table zipped against the requested list
        # would then label every stream with the wrong corpus.
        parts = getattr(corpus, "streams", [corpus])
        names = getattr(corpus, "names", None) or [helpers.corpus_name(roots[0])]
        rows = [("corpora", f"{len(parts)} standalone stream"
                            f"{'' if len(parts) == 1 else 's'}")]
        for name, s in zip(names, parts):
            share = 100.0 * s.n_tokens / max(n_tok, 1)
            rows += [(name,
                      f"{helpers.count(s.n_tokens)} tokens, {share:.1f}% of the mixture"),
                     ("  token stream", s.path),
                     ("  stream size", f"{s.nbytes / 1048576:.1f} MB {s.dtype}, "
                                       f"memory-mapped (not loaded)"),
                     ("  documents", helpers.count(len(s)))]
        rows += [
            ("excluded subdirs", ", ".join(config.PRETRAIN["exclude_dirs"]) or "(none)"),
            ("corpus files", helpers.count(n_files)),
            ("documents", helpers.count(len(corpus))),
            ("words per document", f"~{config.PRETRAIN['max_words']}"),
            ("TOTAL TOKENS", helpers.count(n_tok)),
            ("mean tokens/document", f"{n_tok / max(len(corpus), 1):.1f}")]
        if seen:
            rows += [("tokens per step", helpers.count(ctx_batch * ctx_len)),
                     ("budget", f"{config.PRETRAIN['steps']:,} steps = "
                                f"{helpers.human(seen)} tokens"),
                     ("epochs over this corpus", f"{seen / max(n_tok, 1):.2f}")]
        if epochs == 1:
            rows += [("training policy", "exactly one pass over within-stream next-token targets")]
        helpers.table("pretrain corpus scan", rows, out=log)
    return corpus, n_files, n_tok


def run(ctx, corpus):
    """LM-pretrain a fresh mira_llm on `corpus`, preview, and return it. The periodic
    generation previews are prompted from the corpus itself."""
    from helpers import common
    args, log = ctx["args"], ctx["log"]
    if args.micro_batch != 0:
        raise ValueError("Micro-batching is disabled for pretraining")
    args.disable_pretrain_micro_batch = True
    epoch_metadata = None
    epoch_model = None
    if getattr(args, "pretrain_epochs", 0) == 1:
        from .epoch import EpochCorpus
        from model import parallel
        _, world = parallel.data_env(args.distributed_parallelism)
        epoch_model = ctx["new_model"]()
        regions = max(world, len(getattr(epoch_model, "device_ids", []) or []), 1)
        checkpoint = helpers.load_ckpt(ctx["ckdir"], STAGE) if args.resume else None
        start = int(checkpoint["step"]) if checkpoint else 0
        windows = config.windows(args.context_window) or config.context_windows(args.model_scheme)
        corpus = EpochCorpus(corpus, windows, args.batch, regions=regions, start=start)
        epoch_metadata = corpus.metadata
        if checkpoint and checkpoint.get("extra") != epoch_metadata:
            raise RuntimeError("Checkpoint does not match the one-epoch corpus and batching plan")
        args.pretrain_steps = corpus.steps
        log(f"pretrain: exactly one corpus pass, {corpus.steps:,} optimizer steps; "
            "each within-stream next-token target is used once")

    log(f"=== PRETRAIN (LM on {len(corpus):,} documents) ===")
    preview = common.make_preview(ctx["tok"], ctx["device"], args.max_len, log,
                                  common.corpus_prompts(corpus, seed=args.seed,
                                                        tok=ctx["tok"]))
    # the state space module's diagnostics are collected during PRETRAINING specifically:
    # this is the stage long enough for the memory horizon to move, and the one where a
    # degenerate module (selectivity collapsing, or the residual stream routing around it)
    # would otherwise go unnoticed until much later
    model = lm.train(epoch_model if epoch_model is not None else ctx["new_model"](), ctx["enc"], corpus, ctx["ckdir"], args, log,
                     ctx["monitor"], stage=STAGE, steps=args.pretrain_steps,
                     lr=args.peak_lr, preview=preview, checkpoint_extra=epoch_metadata)
    preview(model, STAGE, "final")
    return model
