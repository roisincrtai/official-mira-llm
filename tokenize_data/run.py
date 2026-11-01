"""
tokenize_data/run.py -- stage 4: turn this run's corpus into a token stream.

    python -m tokenize_data.run [--force] [--list] [--all] [--only tiny]
    ./run_scripts/run_tokenize_data.sh

WHY THIS IS ITS OWN STAGE. Tokenising a corpus is the one expensive, entirely deterministic
step in the pipeline: it depends on the corpus and the tokenizer and on nothing else, it
takes hours on a 2B-token corpus, and its result is reused by every run that follows. Folding
it into the start of pretraining made those hours look like part of training -- a run that
appeared to hang before its first step, a cost paid again on every machine, and a failure
(an unreadable file, a wrong text column, a full disk) discovered only after a GPU had been
allocated. As a stage of its own it is visible, resumable, runnable on a machine with no GPU
at all, and shared: the same streams serve pretraining, evaluation and the scaling-law study.

This is the standard shape of a large-scale pipeline -- GPT-2's encoder pass, nanoGPT's
prepare.py, Megatron's preprocess_data.py all exist for the same reason.

WHAT IT TOKENISES, AND THE `category` FIELD IS WHAT SAYS SO. Every dataset in
config/download/config_download.yaml whose category is `pretrain` or `instruct-sft` is tokenised, plus
the corpora THIS RUN will train on -- PRETRAIN_DIR from the yaml named by --config (a
comma-separated list), or the list belonging to MIRA_CONFIG. A corpus is added to this stage
by giving it a category in the download config and nowhere else.

WHAT IT DOES NOT TOKENISE, AND THE MECHANISM IS THE CATEGORY. `instruct-rlhf` is left alone
because its trainers encode their pairs on the fly, and an entry with NO category is left alone
because it is not a training corpus -- which is what the evaluation sets are. llm-hellaswag,
llm-lambada and mmlu carry none, so neither the default nor --all reaches them, and evaluation
reads their text directly (evals/eval_pretrain_context_length.py) with no token stream.

    ./run_scripts/run_tokenize_data.sh --only llm-hellaswag

builds one anyway, for the case where one is actually wanted. --only resolves a scheme name, a
corpus name, a substring of either, and -- only where none of those matched -- a downloaded
dataset that no scheme trains on.

ONE STREAM PER CORPUS, AND ADDING ONE COSTS ONLY THAT ONE. A scheme's corpora are never
merged: each is tokenised alone, under its own signature, into its own cache directory. So
the way to add a corpus to a model that is already training is simply to add it to the list
and re-run this stage -- every existing stream is found by its unchanged signature and left
exactly as it is, the new one is built beside it, and the run reading them neither restarts
nor re-tokenises. The corpora are combined only in the batch sampler, in proportion to their
token counts.

WHAT IT WRITES. ~100 MB .tokens shards per (corpus, tokenizer, packing), under
cache/tokens/bpe_<encoding>_<fingerprint>/. See helpers/token_store.py. Re-running is cheap: a
stream whose signature still matches its corpus is left alone, an interrupted one resumes from
its last shard, and --force starts again.

Training stages do NOT depend on this having been run. They build a missing stream themselves,
so skipping it costs time rather than correctness -- the same rule stage 1 follows for
downloads, and for the same reason: a pipeline that breaks when a step is skipped teaches
people to run every step blindly.
"""
import argparse
import os
import sys

import default_config as config
import helpers
from tools import config_file as cfgfile


def corpora(args, configured_pairs=()):
    """(name, directory) for every corpus this stage builds a stream for.

    TWO SOURCES, AND THE SECOND IS THE DECLARED ONE. What this run trains on -- PRETRAIN_DIR if
    the yaml sets it, otherwise the corpus belonging to MIRA_CONFIG -- and then every dataset
    whose download-config category is `pretrain` or `instruct-sft`, which is what `by_category`
    below adds. A corpus reached by both is tokenised once.

    --only <name> picks a different scheme by name, and --all does the lot, for the case
    where several sizes are to be trained from one prepared cache. --only also reaches a
    DOWNLOADED DATASET THAT NO SCHEME TRAINS ON -- the evaluation sets -- and only after every
    scheme and corpus has failed to match, so nothing that resolved before resolves elsewhere
    now.

    A scheme owns a LIST of corpora, and each one is tokenised into its OWN stream under its
    own signature. Nothing is merged and nothing is re-read: adding a corpus to a scheme that
    has already been prepared costs exactly the new corpus, because every other stream is
    found in the cache by its unchanged signature and reused. That is what makes it safe to
    extend a run that is already training -- the shards it is reading are never rewritten."""
    if configured_pairs and not args.pretrain_dir and not args.model_scheme:
        configured = [(row["dataset"], config.dataset_dir(row["dataset"]))
                      for row in configured_pairs]
        if not args.only:
            return configured
        selected = [(name, path) for name, path in configured
                    if args.only in name or args.only in os.path.basename(os.path.abspath(path))]
        if selected:
            return selected

    def expand(scheme, roots, out, seen):
        for path in (roots if isinstance(roots, (list, tuple)) else [roots]):
            if not path:
                continue                   # a scheme with no corpus configured at all
            key = os.path.abspath(path)
            if key in seen:                # schemes share corpora; tokenise each one once
                continue
            seen.add(key)
            # The label carries the corpus name as well as the scheme, because a scheme now
            # contributes several rows and "mira-s TOKENS" three times over says nothing.
            name = scheme if roots is path else f"{scheme}/{os.path.basename(key)}"
            out.append((name, path))

    def by_category(out, seen):
        """Every dataset the download config gives a tokenisable category.

        THE CATEGORY IS WHAT DECIDES, NOT A LIST IN A YAML.

        A dataset is added to this stage by giving it `category: pretrain` or
        `category: instruct-sft` in config/download/config_download.yaml and nowhere else. Before this,
        the stage tokenised PRETRAIN_DIR and the scheme's corpora only, so the-stack-smol-xs --
        carrying `category: pretrain` since it was added -- was skipped by every default run
        because no scheme names it, and the category it declared did nothing.

        THE CATEGORY GATE IN main() IS UNCHANGED and still has the last word: instruct-rlhf and
        an entry with no category are still left alone, and they never reach this list.

        NOTHING ALREADY BUILT IS REBUILT. A stream is found by its signature, and a corpus added
        here brings its own; the streams every other corpus already has are untouched. A dataset
        that is not on disk is reported and skipped, so one config still serves every machine."""
        for name in config.DATASETS:
            if config.dataset_category(name) not in ("pretrain", "instruct-sft"):
                continue
            path = config.dataset_dir(name)
            key = os.path.abspath(path)
            if key in seen:
                continue
            seen.add(key)
            out.append((name, path))

    if args.all or args.only:
        out, seen = [], set()
        for scheme, roots in config.PRETRAIN_CORPUS.items():
            expand(scheme, roots, out, seen)
        by_category(out, seen)
        if args.only:
            hits = [(n, p) for n, p in out
                    if args.only in n or args.only in os.path.basename(os.path.abspath(p))]
            # AND A DATASET NO SCHEME TRAINS ON, but ONLY when nothing above matched. The
            # evaluation sets -- llm-hellaswag, llm-lambada -- are downloaded like everything
            # else and are in NO scheme's PRETRAIN_CORPUS, which is exactly what keeps this
            # stage from tokenising them by default and what makes --all leave them alone. They
            # were therefore unreachable: `--only llm-hellaswag` matched nothing and the stage
            # said so, when a token stream of one is a perfectly reasonable thing to ask for.
            #
            # THE FALLBACK IS SECOND AND ONLY SECOND, so no name that resolved before resolves
            # differently now: a scheme name, a corpus name and a substring of either are all
            # answered by the list above, and this is reached only where that list is empty.
            # The names come from config/download/config_download.yaml through config.DATASETS, so there
            # is no second list of datasets here either.
            if not hits:
                hits = [(name, config.dataset_dir(name)) for name in config.DATASETS
                        if args.only in name]
            if not hits:
                have = "\n  ".join(sorted(set(f"{n}" for n, _ in out)))
                spare = "\n  ".join(sorted(n for n in config.DATASETS
                                           if not any(n in m for m, _ in out)))
                raise SystemExit(
                    f"[tokenize] --only {args.only!r} matched no scheme, no corpus and no "
                    f"downloaded dataset.\n"
                    f"  a scheme's corpora:\n  {have}\n"
                    f"  downloaded, trained on by no scheme (evaluation sets, instruction "
                    f"data):\n  {spare}\n"
                    f"  The match is a SUBSTRING, so --only wikitext or --only tiny both work.")
            return hits
        return out
    scheme = config.canonical_scheme(args.model_scheme or config.PRETRAIN["model_scheme"])
    roots = ([d.strip() for d in str(args.pretrain_dir).split(",") if d.strip()]
             if args.pretrain_dir else config.PRETRAIN_CORPUS.get(scheme, []))
    out, seen = [], set()
    expand(scheme, roots, out, seen)
    by_category(out, seen)
    return out



def text_column_for(path, fallback):
    """The column THIS corpus keeps its text in, not the run's one column.

    THE PAIRING IS default_config.BENCHMARK_CORPORA AND IS NOT A SECOND LIST. It already maps
    each corpus name to its directory and its column, for exactly the reason the benchmark states
    there: selecting a corpus must select its column, and the two must not be settable
    inconsistently. A corpus the table does not name keeps the run's configured column, so every
    corpus that read correctly before reads identically now.

    NOTHING ALREADY ON DISK MOVES. The column enters `corpus_signature`, so a stream is found by
    the column it was built with. The three prose corpora are `text` in the table and `text` in
    the yaml, so their signatures are unchanged and every stream they have is still found."""
    entry = config.BENCHMARK_CORPORA.get(os.path.basename(os.path.abspath(path)))
    return entry["text_column"] if entry else fallback


def _check(scheme, path, tok, a, label="check"):
    """Check every shard of one corpus's stream against its manifest. Returns the damage count.

    Three questions per shard: does the file exist, is it at least as long as the manifest
    counts, and -- for shards sealed since digests were recorded -- do its per-chunk md5s still
    match. The third is the one that matters. Length catches the tail a killed writer loses and
    nothing else; a shard whose middle was rewritten, or truncated and refilled to the same
    size, passes a length check and then trains on wrong tokens.

    CHUNKED DIGESTS NAME THE PLACE, not just the fact: the first mismatching 64 MB block is
    reported by index and byte offset, and every chunk before it is known good, which is what
    lets the repair keep them."""
    from helpers import token_store
    files = helpers.corpus_files(path, config.PRETRAIN["exclude_dirs"])
    sig = helpers.corpus_signature(tok, path, files, a.max_words,
                                   text_column_for(path, a.text_column))
    stem = helpers.corpus_stream_path(tok, path, sig)
    idx = token_store.read_index(stem)
    if not idx:
        print(f"[{label}] {scheme}: no stream yet at {stem}_index.json -- nothing to check")
        return 0
    shards, isz = idx.get("shards", []), token_store.itemsize(idx["dtype"])
    print(f"[{label}] {scheme}: {len(shards)} shard(s), {idx.get('n_tokens', 0):,} tokens, "
          f"complete={bool(idx.get('complete'))}\n       {stem}")
    bad, no_digest = 0, 0
    for i, sh in enumerate(shards):
        q = token_store.shard_path(stem, i)
        want = int(sh["n_tokens"]) * isz
        have = os.path.getsize(q) if os.path.exists(q) else -1
        if have < 0:
            print(f"       shard {i:05d}  MISSING  {os.path.basename(q)}"); bad += 1; continue
        if have < want:
            print(f"       shard {i:05d}  SHORT    {have:,} bytes on disk, {want:,} counted "
                  f"({(want - have) / 1048576:.1f} MB missing)"); bad += 1; continue
        recorded = token_store._as_list(sh.get("md5"))
        if not recorded:
            no_digest += 1; continue
        got = token_store.chunk_md5s(q)
        hit = next((c for c in range(min(len(got), len(recorded)))
                    if got[c] != recorded[c]), None)
        if hit is not None:
            lo = hit * token_store.MD5_CHUNK
            print(f"       shard {i:05d}  CORRUPT  chunk {hit} (from byte {lo:,}): "
                  f"md5 {got[hit]} != recorded {recorded[hit]}"); bad += 1
        elif len(got) != len(recorded):
            print(f"       shard {i:05d}  CORRUPT  {len(got)} chunk(s) on disk against "
                  f"{len(recorded)} recorded"); bad += 1
    # THE RESPONSE MASK IS CHECKED WITH THE SHARDS IT DESCRIBES. A conversation corpus carries a
    # `.masks` sidecar of one line per document, and stage 7 refuses a stream whose sidecar and
    # shard disagree -- so a check that looked only at the tokens would report a stream "OK"
    # that the stage it is being prepared for will not accept. Absent everywhere is a
    # pretraining corpus and is not a fault; absent on SOME shards is one.
    seen = [i for i in range(len(shards)) if os.path.exists(token_store.masks_path(stem, i))]
    if seen:
        lines = spans = 0
        for i, sh in enumerate(shards):
            got = token_store.read_masks(stem, i)
            if got is None:
                print(f"       shard {i:05d}  MASK MISSING  "
                      f"{os.path.basename(token_store.masks_path(stem, i))}"); bad += 1; continue
            if len(got) != int(sh["n_docs"]):
                print(f"       shard {i:05d}  MASK SHORT    {len(got):,} line(s) against "
                      f"{int(sh['n_docs']):,} document(s)"); bad += 1; continue
            lines += len(got)
            spans += sum(len(s) for s in got)
        print(f"       response mask: {lines:,} document(s), {spans:,} (offset, length) "
              f"span(s) in {len(seen)} sidecar(s)")
    if no_digest:
        print(f"       {no_digest} shard(s) predate content digests: length checked only")
    if not bad:
        print(f"       OK: {len(shards) - no_digest} verified by digest, "
              f"{no_digest} by length")
    return bad


def install_downloaded_vocabularies():
    """Put a downloaded vocabulary at checkpoints/<name>/bpe.json when stage 3 has not built one.

    Stage 3 writes checkpoints/bpe/bpe.json and stage 12 writes checkpoints/bpe-sft-coder/bpe.json.
    Without them tokenizer.run.resolve_named still finds the copy stage 1 fetched into
    saved_models/ and reads it in place, so this stage works either way -- but every later stage
    repeats the same fallback, and nothing under checkpoints/ records which vocabulary the streams
    were built from. Copying it in once makes the primary path exist.

    A file already at the destination is never replaced: stage 3's own vocabulary always wins, and
    a partially written copy cannot be left behind because the copy lands on a staging name and is
    renamed onto the destination. The source is parsed before it is copied, so an unreadable
    download is skipped rather than installed."""
    import shutil
    from tokenizer import run as train_bpe

    installed = []
    base = config.download_dir(config.BPE_DOWNLOAD_NAME)
    for name, wanted, sources in (
            ("bpe", config.BPE_PATH,
             [os.path.join(base, "bpe.json"), config.BPE_DOWNLOAD_PATH]),
            ("bpe-sft-coder",
             os.path.join(config.CHECKPOINT_DIR, "bpe-sft-coder", "bpe.json"),
             train_bpe.download_candidates("bpe-sft-coder"))):
        if os.path.exists(wanted):
            continue
        for source in sources:
            if train_bpe.peek(source) is None:
                continue
            os.makedirs(os.path.dirname(wanted), exist_ok=True)
            staged = wanted + ".incoming"
            shutil.copyfile(source, staged)
            os.replace(staged, wanted)
            installed.append(name)
            print(f"[tokenize] {name} is not built; installed the downloaded copy "
                  f"{helpers.bpe_subpath(source)} -> {helpers.bpe_subpath(wanted)}", flush=True)
            break
    return installed


def main(argv=None):
    # THE SAME YAML THE TRAINERS READ, AND IT HAS TO BE THE SAME ONE. A token stream is
    # identified by the corpus it came from AND the packing applied to it -- the word budget,
    # the parquet text column, the held-out files excluded. If this stage packed with one set of
    # values and stage 6 with another, the two would name different streams and stage 6 would
    # tokenise the whole corpus again from zero, which for a 10 GB corpus is hours that look
    # exactly like normal progress. That is why --config reaches here and not only the trainers.
    argv = list(sys.argv[1:] if argv is None else argv)
    from_yaml, _cfg = cfgfile.resolve(argv, "tokenize_data")
    argv = from_yaml + argv
    ap = argparse.ArgumentParser(description="tokenise the corpora into token streams")
    ap.add_argument("--config", default=_cfg, metavar="FILE",
                    help="the yaml this run's corpus settings come from (default: "
                         f"{os.path.relpath(_cfg, config.ROOT)})")
    ap.add_argument("--mira_config", dest="model_scheme", default="",
                    help="which scheme's corpus to tokenise; empty = the configured one")
    ap.add_argument("--pretrain_dir", default="",
                    help="the corpus directories themselves, comma-separated, overriding the "
                         "scheme's own list")
    ap.add_argument("--only", default="",
                    help="substring of a scheme OR a corpus name, e.g. tiny, wikitext103, "
                         "llm-pretrain-fineweb-edu-2BT -- or of a downloaded dataset no scheme "
                         "trains on, e.g. llm-hellaswag, llm-lambada")
    # DECLARED HERE, AND THAT IS THE POINT. This parser ends in parse_known_args, so an
    # undeclared --tokenizer was SWALLOWED IN SILENCE: the stage accepted the flag, tokenised
    # with checkpoints/bpe anyway, and wrote the shards under the base vocabulary's cache key.
    # Nothing failed and nothing warned; the only evidence was the directory name.
    ap.add_argument("--tokenizer", default="", metavar="NAME",
                    help="directory under checkpoints/ holding the vocabulary to tokenise "
                         "with, e.g. bpe-sft-coder (default: bpe, the stage-3 vocabulary). "
                         "The stream's cache directory is named after whichever is used")
    ap.add_argument("--all", action="store_true",
                    help="every configured corpus, not just this run's")
    ap.add_argument("--force", action="store_true",
                    help="discard the existing shards and tokenise from the start")
    # --no-resume IS NOT --force HERE, though it was, and that was a trap. This stage receives
    # the yaml's shared settings (see tools/config_file.STAGE_GROUPS), and those carry the
    # trainers' --no-resume, which means "ignore the checkpoints and train from step 0". Treating it as
    # --force meant that starting a model afresh silently deleted and re-tokenised every
    # corpus -- hours of work, and the very shards a concurrent run was reading. Resuming is
    # this stage's only behaviour; rebuilding is --force, typed deliberately, and nothing
    # forwards it.
    ap.add_argument("--no-resume", dest="no_resume", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--list", action="store_true", help="show what would be built, and where")
    ap.add_argument("--check", action="store_true",
                    help="check every shard of this run's streams against the manifest and "
                         "report. READS ONLY: nothing is built, rewritten or repaired. Exits 1 "
                         "if anything is damaged, so it can gate a long job. --repair is the "
                         "same check followed by the fix.")
    ap.add_argument("--repair", action="store_true",
                    help="check every shard of this run's streams against the manifest -- "
                         "length, and the per-chunk md5s recorded when each shard was sealed -- "
                         "then repair what is damaged and carry on tokenising from there.")
    # THE PACKING DEFAULTS ARE RESOLVED AFTER --set, NOT BEFORE IT. An argparse default is
    # evaluated when the parser is built, which is before apply_overrides has run, so
    # `default=config.PRETRAIN["max_words"]` would freeze the value the override was about to
    # change and this stage would pack the corpus differently from every stage that reads
    # config.PRETRAIN directly. A different packing is a different stream, and a different
    # stream is the whole corpus tokenised again. The sentinels defer the question.
    ap.add_argument("--max_words", type=int, default=0,
                    help="words per packed document; 0 = PRETRAIN['max_words'] after --set")
    ap.add_argument("--text_column", default="",
                    help="parquet or JSONL column holding the text; empty = "
                         "PRETRAIN['text_column']. A corpus named in "
                         "default_config.BENCHMARK_CORPORA uses ITS column instead, so "
                         "the-stack-smol-xs is read as `content` whatever this says")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="SEC.key=v", help="override any default_config value, e.g. "
                         "--set PRETRAIN.max_words=512")
    a, _ = ap.parse_known_args(argv)
    config.apply_overrides(a.overrides, print)
    a.max_words = a.max_words or config.PRETRAIN["max_words"]
    a.text_column = a.text_column or config.PRETRAIN["text_column"]

    configured_pairs = []
    if os.path.isfile(_cfg) and "tokenize_pairs" in cfgfile.read(_cfg):
        configured_pairs = cfgfile.read_mapping_sequence(_cfg, "tokenize_pairs")

    # The tokenizer is READ, never built here. A stream is tied to the vocabulary that
    # produced it -- that is what its cache directory is named after -- so a stage that
    # quietly trained its own BPE would write streams no other stage could use.
    from tokenizer import run as train_bpe
    install_downloaded_vocabularies()
    todo = corpora(a, configured_pairs)
    if not todo:
        print(f"[tokenize] no corpus configured for "
              f"{a.model_scheme or config.PRETRAIN['model_scheme']}; nothing to do.\n"
              f"           Set PRETRAIN_DIR in the yaml, or add the scheme to "
              f"PRETRAIN_CORPUS in default_config.py.")
        return 0
    print(f"[tokenize] {'every configured corpus' if (a.all or a.only) else 'this run'}: "
          f"{', '.join(n for n, _ in todo)}")

    pair_tokenizers = {row["dataset"]: row["tokenizer"] for row in configured_pairs}
    tokenizer_for_path = {
        os.path.abspath(path): a.tokenizer or pair_tokenizers.get(
            os.path.basename(os.path.abspath(path)), "bpe")
        for _scheme, path in todo
    }
    tokenizers = {}
    active_tokenizers = dict.fromkeys(
        tokenizer_for_path[os.path.abspath(path)]
        for _scheme, path in todo if os.path.isdir(path))
    for tokenizer_name in active_tokenizers:
        tok, bpe_path = train_bpe.resolve_named(tokenizer_name)
        if tok is None:
            raise SystemExit(
                f"[tokenize] no usable tokenizer for {tokenizer_name}\n"
                f"           Tried checkpoints/{tokenizer_name}/ and the downloaded "
                f"tokenizer. Build or fetch that vocabulary before tokenization.")
        tokenizers[tokenizer_name] = tok, bpe_path
        helpers.set_bpe_source(bpe_path, tokenizer=tok)
        print(f"[tokenize] vocabulary {tokenizer_name}: {len(tok):,} -> "
              f"{helpers.token_cache_root(tok)}", flush=True)

    rows, total, bad = [], 0, 0
    for scheme, path in todo:
        # THE CATEGORY DECIDES WHETHER THIS STAGE TOUCHES THE CORPUS AT ALL.
        #
        #     pretrain        a stream, no mask sidecar
        #     instruct-sft    a stream AND the mask sidecar
        #     instruct-rlhf   nothing; the preference trainers encode their pairs on the fly
        #     absent          nothing; the entry is not a training corpus
        #
        # --only NAMES A CORPUS EXPLICITLY AND IS STILL OBEYED, so a category can be inspected
        # or overridden from the command line without editing the config.
        category = config.dataset_category(path)
        if not a.only and category in ("instruct-rlhf", ""):
            why = ("its category is instruct-rlhf, so its trainers tokenise on the fly"
                   if category else
                   "it carries no category in config/download/config_download.yaml")
            print(f"[tokenize] {scheme}: not tokenised -- {why}")
            rows += [(scheme, "not tokenised: " + (category or "no category"))]
            continue
        if not os.path.isdir(path):
            print(f"[tokenize] {scheme}: corpus directory missing, skipped: {path}")
            rows += [(scheme, "MISSING: " + path)]
            continue
        tokenizer_name = tokenizer_for_path[os.path.abspath(path)]
        tok, bpe_path = tokenizers[tokenizer_name]
        helpers.set_bpe_source(bpe_path, tokenizer=tok)
        if a.check:
            # CHECK AND STOP. Separate from --repair because the two are asked in different
            # moods: "is this sound?" before committing a machine to a run, against "make it
            # sound" once it is not. A check that quietly rewrote a 4 GB corpus would be the
            # wrong thing to reach for when something already looks wrong.
            bad += _check(scheme, path, tok, a, "check")
            continue
        if a.repair:
            # CHECK, REPORT, THEN FALL THROUGH INTO THE BUILD, which is what repairs. The
            # report exists so the damage is named before anything is rewritten; the rewrite is
            # the ordinary resume path, which cuts each damaged shard back to its last complete
            # document and continues from there.
            bad += _check(scheme, path, tok, a, "repair")
        # THE COLUMN IS THE CORPUS'S OWN, AND IT IS SAID OUT LOUD WHEN IT IS NOT THE RUN'S.
        # A corpus read with the wrong column does not fail: it yields the JSON container as
        # text, or no documents at all. Naming the substitution here is what makes it visible
        # in the log rather than only in the cache directory's name.
        column = text_column_for(path, a.text_column)
        if column != a.text_column:
            print(f"[tokenize] {scheme}: text column {column!r}, not the configured "
                  f"{a.text_column!r} -- this corpus keeps its text there")
        if a.list:
            files = helpers.corpus_files(path, config.PRETRAIN["exclude_dirs"])
            sig = helpers.corpus_signature(tok, path, files, a.max_words, column)
            print(f"[tokenize] {scheme}: {len(files):,} files\n"
                  f"           {path}\n"
                  f"        -> {helpers.corpus_stream_path(tok, path, sig)}")
            continue
        stream, n_files = helpers.build_token_stream(
            path, tok, a.max_words, config.PRETRAIN["exclude_dirs"], log=print,
            text_column=column, stage=f"tokenize/{scheme}",
            force=a.force, write=True)
        if stream is None:
            print(f"[tokenize] {scheme}: no corpus files under {path}")
            rows += [(scheme, "no corpus files")]
            continue
        total += stream.n_tokens
        rows += [(f"{scheme} corpus", path),
                 (f"{scheme} files", helpers.count(n_files)),
                 (f"{scheme} documents", helpers.count(stream.n_docs)),
                 (f"{scheme} TOKENS", helpers.count(stream.n_tokens)),
                 (f"{scheme} stream", f"{stream.nbytes / 1048576:.1f} MB {stream.dtype}"),
                 (f"{scheme} path", stream.path)]
    if a.check:
        if bad:
            print(f"\n[check] DAMAGE in {bad} shard(s). Nothing was changed. Run "
                  f"./run_scripts/run_tokenize_data.sh --repair to cut each damaged shard back to its "
                  f"last complete document and re-tokenise from there.")
        else:
            print("\n[check] every shard matches its manifest.")
        return 1 if bad else 0
    if a.repair and bad:
        print(f"\n[repair] {bad} damaged shard(s) were found and cut back to their last "
              f"complete document; the tokenising above continued from there.")
    elif a.repair:
        print("\n[repair] every shard matched its manifest; nothing needed repairing.")
    if a.list:
        return 0
    helpers.table("tokenized corpora", rows + [("TOTAL TOKENS", helpers.count(total))])
    return 0


if __name__ == "__main__":
    sys.exit(main())
