"""
instruct_sft.py -- stage 7's logic: domain-adaptive supervised fine-tuning on the fine-tuning data.

The objective is the SAME length-normalized LM likelihood as pretraining (lm.train); what
changes is the corpus. Stage 5 buys general language from a large heterogeneous corpus,
this stage moves the model onto the distribution the alignment stages operate in. Sharing
one loop is deliberate: the two stages cannot drift apart in their treatment of masking,
normalisation or padding.

    load_corpus(tok, corpus_dir, log)  -> (docs, n_files, n_tokens), pre-tokenised and cached
    run(ctx, init_model, corpus) -> the fine-tuned model

Its output anchors everything downstream: it initialises the reward trunk, it is the
starting policy for both alignment stages, and it is the frozen reference their divergence
terms are measured against. Nothing outside the fine-tuning data is read.
"""
import glob
import os
import random

import default_config as config
import helpers
from helpers import dataset_helpers as dsets
from model import lm

STAGE = "sft"


def load_corpus(tok, corpus_dir, log=print):
    """The fine-tuning set: every demonstration under `corpus_dir`, conditioned on its prompt.

    THREE LAYOUTS, ONE RESULT. A CONVERSATION MIXTURE (the Tulu 3 SFT mixture: parquet shards
    of `messages` lists) is flattened to one demonstration per assistant turn. A PREFERENCE
    TREE is read for its chosen continuation and nothing else -- the rejected one is what
    stages 8 and 11 need, and feeding it here would teach the model the very responses the rest
    of the pipeline then spends its time pushing down. A LOCAL FOLDER of json/jsonl records is
    read for whatever prompt and response fields it uses. All three arrive in the same
    {prompt, chosen, rejected} shape the encoder already takes, so the loss lands on the
    response tokens with the prompt as context: the prompt is conditioning, not something to be
    modelled.

    A demonstration whose prompt and response are identical to another's is dropped. rlhf_hh
    repeats prompts across its helpful and harmless subsets, and a mixture assembled from
    several sources repeats whole examples; a duplicate is simply that example weighted twice.

    The layout is detected and REPORTED, not declared -- so pointing this stage at your own
    folder works, and what it decided your folder was is in the log rather than in someone's
    head. Only a prompt and a response are needed here, so a mixture with no rejected field
    anywhere in it is enough for this stage even though stages 8 and 11 would refuse it."""
    layout = dsets.detect_layout(corpus_dir)
    if layout == "missing":
        raise SystemExit(
            f"[instruct-sft] fine-tuning data not found: {corpus_dir or '(unset)'}\n"
            f"      Expected a conversation mixture (parquet/jsonl with a `messages` list),\n"
            f"      an hh tree (<dir>/*_train/*.json), or a folder of json/jsonl records\n"
            f"      carrying a prompt and a response.\n"
            f"      No stage downloads. Run:  ./run_scripts/run_download_data.sh\n"
            f"      or set SFT_DIR in config/instruct_sft/config_user.yaml (--sft_dir) to your own.")
    if layout == "chat":
        # SUPERVISED FINE-TUNING IS PRETRAINING PLUS A MASK. The tokens are the ones stage 4
        # wrote and every other trainer reads, the packing and the window draw are lm.train's
        # packed path, and the one thing this stage adds is the response mask -- which stage 4
        # measured while it was encoding and wrote beside each shard as (offset, length) spans.
        # This stage READS it. It re-reads no corpus and calls no tokenizer.
        from helpers import chat_index
        # THE STREAM IS OPENED, NEVER BUILT. This stage is a CONSUMER: only
        # run_scripts/run_tokenize_data.sh writes tokens, so a missing or stale stream stops the stage
        # and names the command that makes one. helpers.load_token_corpus would have built it,
        # which is what put a tokenising bar in front of a run that should have been reading.
        #
        # THE SIGNATURE MUST BE STAGE 4'S OWN, or the stem does not match what is on disk and
        # nothing is found. The three inputs that decide it -- max_words, exclude_dirs and
        # text_column -- are taken from config.PRETRAIN exactly as tokenize_data/run.py takes
        # them, so this computes the same stem stage 4 wrote to. A conversation corpus is
        # recognised by stage 4 from its RECORDS rather than from the column name, so the
        # signature is the same one it always was and the stream keeps its name.
        from helpers import token_store
        corpus_only = helpers.corpus_files(corpus_dir, config.PRETRAIN["exclude_dirs"])
        sig = helpers.corpus_signature(tok, corpus_dir, corpus_only,
                                       config.PRETRAIN["max_words"],
                                       config.PRETRAIN["text_column"])
        stem = helpers.corpus_stream_path(tok, corpus_dir, sig)
        stream = token_store.open_if_current(stem, sig)
        if stream is None:
            raise SystemExit(
                f"[instruct-sft] no tokenised stream for {os.path.basename(corpus_dir)} at\n"
                f"      {stem}_index.json\n"
                f"      This stage reads tokens and never writes them. Build the stream with\n"
                f"          ./run_scripts/run_tokenize_data.sh --only "
                f"{os.path.basename(corpus_dir.rstrip('/'))}\n"
                f"      and check an existing one with --check.")
        corpus = chat_index.mask_stream(stream, log, tok=tok)
        log(f"[instruct-sft] {corpus.describe()}")
        return corpus, len(corpus_only), corpus.n_tokens
    else:
        if layout == "hh":
            pairs = dsets.load_hh_pairs(subsets=dsets.subsets_for(corpus_dir,
                                                                  config.SFT["subsets"]),
                                        hh_dir=corpus_dir, limit=config.SFT["limit"], seed=0)
        else:
            pairs = dsets.load_local_pairs(corpus_dir, limit=config.SFT["limit"], seed=0)
        seen, corpus = set(), []
        for p in pairs:
            key = (p["prompt"], p["chosen"])
            if key in seen:
                continue
            seen.add(key)
            corpus.append({"prompt": p["prompt"], "chosen": p["chosen"], "rejected": ""})
        n_files = len(glob.glob(os.path.join(corpus_dir, "**", "*.json*"), recursive=True))
    n_tok = 0
    if corpus:
        # measured on a sample: tokenising every demonstration to print one number would cost
        # minutes, and this is a report rather than something the training reads
        smp = corpus if len(corpus) <= 2000 else random.Random(0).sample(corpus, 2000)
        per = sum(len(tok(d["prompt"] + " " + d["chosen"],
                          add_special_tokens=False)["input_ids"]) for d in smp) / len(smp)
        n_tok = int(per * len(corpus))
        helpers.table("sft data", [
            ("dataset", os.path.basename(corpus_dir.rstrip("/")) or corpus_dir),
            ("directory", corpus_dir),
            ("layout", dsets.describe(corpus_dir)),
            ("subsets", ", ".join(dsets.subsets_for(corpus_dir, config.SFT["subsets"]))
                        if layout == "hh" else
                        "(local folder: read recursively)"),
            ("batch files", helpers.count(n_files)),
            ("demonstrations read", helpers.count(len(pairs))),
            ("demonstrations (deduplicated)", helpers.count(len(corpus))),
            ("mean tokens/demonstration", f"{per:.1f}"),
            ("TOTAL TOKENS (estimated)", f"~{helpers.human(n_tok)}"),
            ("trained on", "the CHOSEN response, conditioned on the prompt")], out=log)
    return corpus, n_files, n_tok


# HOW MUCH THE PREVIEW GENERATES IS common.INSTRUCT_MAX_NEW, and it is read there rather than
# copied here: stage 11 was given the same number -- "in instruct-sft and instruct-dpo ... max
# generation is 250" -- and a constant written down in two stages is two numbers that agree
# until one of them is edited. It is read inside run(), which already imports helpers.common;
# a module-level import of it would close the cycle mirachat <- helpers.common <- this file.
#
# THE TRUNCATION IS THE SAMPLE, NOT THE PROMPT. "for sampling, truncate by 512" and "truncate
# to 256" were a ceiling on what the model GENERATES; I read both as a ceiling on the prompt,
# which the line before them had already settled -- "for the prompt, don't truncate".

# A FALLBACK CEILING, REACHED ONLY BY A CORPUS WITH NO RESPONSE MASK. The prompt itself is NOT
# truncated -- "for the prompt, don't truncate" -- and it does not need to be: a masked
# conversation's prompt ends where its first assistant span begins, which is an exact boundary
# and is one user turn wide. A packed stream with NO sidecar has no boundary to find, and
# printing an entire 65,536-token document as "the prompt" is not a preview.
PREVIEW_MAX_PROMPT = 2048


def _prompt_end(corpus, doc, n_tokens):
    """Where document `doc` stops being the prompt: the offset of its FIRST assistant span.

    THE SIDECAR ALREADY KNOWS THIS. Stage 4 measured every assistant run while it was encoding,
    so the first (offset, length) pair of a conversation is exactly the token at which the user
    stops talking -- no re-tokenising, no marker search, no guess. Everything before it is the
    prompt in full: the system turn if there is one, the user's question whole, and none of the
    reply the model is about to be asked to produce.

    THE CEILING APPLIES ONLY WHEN THERE IS NO SPAN. A found boundary is returned as it is, even
    at 3,000 tokens -- that is one user turn, it is what the model would be given at inference,
    and "for the prompt, don't truncate" is the instruction. PREVIEW_MAX_PROMPT bounds the case
    with nothing to find, and nothing else."""
    spans = getattr(corpus, "spans", None)
    if spans is not None:
        offs, _ = spans(doc)
        if len(offs):
            return int(offs[0])
    return min(int(n_tokens), PREVIEW_MAX_PROMPT)


def preview_prompts(corpus, tok, n=None, seed=0):
    """Prompts for the periodic previews, drawn from this stage's own corpus.

    THE WHOLE PROMPT, NOT TRUNCATED:

    TWO CORPUS SHAPES, AS lm.train HAS. A packed stream is a memory map with no records in it:
    it answers `doc(i)` with the token ids of document i and is not subscriptable, so indexing
    it raises TypeError before the first step is ever reached. Its documents are decoded back to
    text, up to that boundary; the record branch takes its `prompt` field, which is already the
    prompt and already whole.

    HERE AND NOT IN common.corpus_prompts, WHICH IS SHARED. Which field this stage's records
    carry is this stage's business, so the code that reads them is this stage's too, and every
    other caller of corpus_prompts keeps the two shapes it always had."""
    from helpers import common
    rng = random.Random(seed)
    want = common.N_PREVIEW if n is None else n
    picked = rng.sample(range(len(corpus)), min(want, len(corpus)))
    texts = []
    packed = hasattr(corpus, "doc") and hasattr(corpus, "n_tokens")
    for i in picked:
        if packed:
            # THE PROMPT IS CUT BEFORE THE DECODE, not after: a conversation runs to thousands
            # of tokens and only its opening turn is wanted, so the answer is never decoded.
            ids = corpus.doc(i)
            texts.append(tok.decode(ids[:_prompt_end(corpus, i, len(ids))],
                                    skip_special_tokens=True))
        else:
            record = corpus[i]
            texts.append(record.get("prompt", "") or record.get("chosen", ""))
    # THE PROMPT IS NOT RESHAPED EITHER. It carried `" ".join(t.split())`, which collapses every
    # run of whitespace and every newline in the text the model is about to be PROMPTED with --
    # not merely in what is printed. A prompt whose formatting has been flattened is not the
    # prompt the corpus holds. `if p.strip()` still drops an empty one, which tests the string
    # without altering it.
    return [t for t in texts if t.strip()]


def run(ctx, init_model, corpus):
    """LM-fine-tune `init_model` (the pretrained mira_llm) on `corpus`, preview, and return
    it. The previews are prompted from the SFT corpus, so they show the domain being
    adapted to."""
    from helpers import common
    args, log = ctx["args"], ctx["log"]
    log(f"=== instruct-sft (domain-adaptive LM on {len(corpus):,} documents, from pretrain) ===")
    # WHAT AN EPOCH ACTUALLY IS, beside what was asked for. The step budget is a number in
    # the yaml, chosen against an expected corpus size; the corpus is whatever is on disk,
    # after deduplication and after a conversation mixture has been expanded per assistant
    # turn. Those two are allowed to differ -- but not silently, because a budget that is
    # half an epoch and a budget that is three looks identical from a loss curve.
    from model import parallel
    local_batch, _, global_batch, _ = parallel.batch_layout(init_model, args.batch)
    epoch = -(-len(corpus) // max(global_batch, 1))           # ceil
    mb = int(getattr(args, "micro_batch", 0) or 0)
    log(f"[instruct-sft] one epoch = {epoch:,} steps at batch/GPU {local_batch} "
        f"(global batch {global_batch}; "
        f"{len(corpus):,} demonstrations); this run does {args.sft_steps:,} "
        f"({args.sft_steps / max(epoch, 1):.2f} epochs)")
    log(f"[instruct-sft] {mb} sequence{'s' if mb != 1 else ''} per forward pass, "
        f"{-(-local_batch // max(mb, 1))} passes per step/GPU"
        if mb else
        f"[instruct-sft] per-GPU batch {local_batch} in one forward pass")
    prompts = preview_prompts(corpus, ctx["tok"], seed=args.seed)
    room = max((len(ctx["tok"](p, add_special_tokens=False)["input_ids"]) for p in prompts),
               default=0)
    max_new = common.INSTRUCT_MAX_NEW
    window = max(int(getattr(args, "max_len", 0) or 0), room + max_new)
    log(f"[instruct-sft] previews: {len(prompts)} whole prompts (longest {room:,} tokens), "
        f"{max_new:,} generated tokens each at most, in a {window:,}-token window")
    preview = common.make_preview(ctx["tok"], ctx["device"], window, log, prompts,
                                  max_new=max_new)
    # THIS STAGE'S OWN DIRECTORY, PER THE RULE: checkpoints/instruct_sft/<scheme>_<dataset>/ and
    # outputs/plots/instruct_sft/<scheme>_<dataset>/. helpers.STAGE_DIRS maps "sft" to nothing,
    # so nothing further is appended to either.
    ckdir = helpers.sft_dir(corpus_dir=args.sft_dir)
    plotdir = helpers.sft_plot_dir(corpus_dir=args.sft_dir)
    log(f"[instruct-sft] checkpoint: {helpers.ckpt_path(ckdir, STAGE)}")
    log(f"[instruct-sft] figures:    {plotdir}")

    def monitor(mode, records, step=None):
        """The dynamics figure, into this stage's own directory rather than outputs/plots/sft/.

        RANK 0 DRAWS IT, and a failure to plot never stops a run -- the same two properties
        common.setup's monitor has, which this replaces only to change where the file lands.

        THE IMPORT IS INSIDE THE `try`, AND THAT IS THE POINT OF THE `try`. It sat above one,
        misspelt -- `from helpers import viz`, a name that has never existed; the module is
        helpers.visualization and every other call site in the project spells it
        `from helpers import visualization as viz`. An ImportError raised OUTSIDE the guard is
        not a figure that failed to draw, it is a training run that stops, and this one stopped
        a stage-7 run at its first plot boundary after eleven minutes of stepping. The promise
        one line above was written and not implemented: nothing in this function may now escape
        it."""
        from model import parallel
        if not parallel.is_main():
            return
        try:
            from helpers import visualization as viz
            viz.method_monitor(mode, records, plotdir, "")
        except Exception as e:                                    # noqa: BLE001
            log(f"[instruct-sft] {mode} figure failed: {e}")

    # WHAT THE WIDTH OF A STEP IS, SAID BEFORE THE FIRST ONE. lm.train announces a context
    # schedule because pretraining has one; this stage has no fixed window, so the number in
    # that banner describes nothing it does and a reader would take it for the width being
    # trained at.
    from helpers import chat_index
    log(f"[instruct-sft] no fixed context window; batch {args.batch} whole conversation(s), "
        f"padded to the longest")
    log(f"[instruct-sft] truncation {chat_index.TRUNCATE:,} tokens")
    # NO MIRA STATISTICS IN THIS STAGE. The per-layer pole radius, horizon, angle and gate are
    # measurements of what PRETRAINING built; fine-tuning does not re-derive them, and collecting
    # them costs a summarising pass over alpha, beta, rho and gamma on the profiled step. The
    # dynamics figures this stage draws are the loss, perplexity, accuracy and gradient norm from
    # its own history, which `monitor` above still writes.
    model = lm.train(init_model, ctx["enc"], corpus, ckdir, args, log,
                     monitor, stage=STAGE, steps=args.sft_steps, lr=args.sft_lr,
                     preview=preview, profile=False, schedule=False)
    preview(model, STAGE, "final")
    return model
