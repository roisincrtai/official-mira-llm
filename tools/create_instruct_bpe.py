"""
tools/create_instruct_bpe.py -- make saved_models/tokenizer-bpe-instruct/.

    python -m tools.create_instruct_bpe
    python -m tools.create_instruct_bpe --force
    python -m tools.create_instruct_bpe --from checkpoints/bpe-sft-coder

It READS checkpoints/bpe-sft-coder/ and falls back to saved_models/tokenizer-bpe-sft-coder/;
--from names a directory instead and is not searched past. It WRITES saved_models/ because that
is where a PUBLISHED vocabulary lives -- the layout stage 1 downloads into, ready to upload as it
stands. The instruct stages copy it to checkpoints/bpe-instruct/ on first use
(tokenizer.run.populate_from_saved).

THE INSTRUCT VOCABULARY IS THE CODER VOCABULARY PLUS FOUR MARKERS.

    <|startofmessage|>  opens any message, whoever is speaking
    <|endofmessage|>    closes it
    <startofthink>      reserved for chain-of-thought; nothing writes it today
    <endofthink>        reserved for chain-of-thought; nothing writes it today

THERE IS NO PER-ROLE TOKEN.

Who is speaking is the ORDINARY TEXT `User:\\n` or `Assistant:\\n` written immediately after the
opening delimiter and encoded like any other text, so a message reads

    <|startofmessage|> User:\\n ...the turn... <|endofmessage|>

and helpers.chat_index assembles that at batch time from stage 4's spans.

THE THINK PAIR IS REGISTERED NOW THOUGH IT IS UNUSED.

An id is fixed by the ORDER a token is registered in. Adding the pair later would append it
behind a vocabulary that stage 7 checkpoints already embed, so the chain-of-thought stage would
begin with a vocabulary change rather than with a change to what is emitted. Two embedding rows
now is the cheaper end of that trade.

It is the vocabulary the INSTRUCT SERIES shares -- instruct-sft, the reward model, RLHF and DPO
-- so the four stages agree on what an id means and a checkpoint handed from one to the next
decodes as the stage that wrote it meant.

THE BASE FILE IS COPIED, NOT EDITED. bpe.json is bpe-sft-coder's, byte for byte; the four
markers go in special_tokens.json beside it, and BPETokenizer.load appends them when it reads the
directory. A consumer sees one tokenizer of 58,263 ids and never learns there were two files.

NOTHING ALREADY IN THE VOCABULARY MOVES. The specials sit last, so an appended token takes the
next id: the 256 raw bytes and all 58,000 merges keep the ids they had, and so do the three
predefined specials.

WHAT IT DOES NOT DO. It does not touch checkpoints/bpe or checkpoints/bpe-sft-coder, trains no
merges, and reads no corpus. It is idempotent.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import default_config as config                                        # noqa: E402
from tokenizer import BPETokenizer                                     # noqa: E402

# THE FOUR, IN THE ORDER THEY ARE REGISTERED, WHICH IS WHAT FIXES THEIR IDS.
#
# SPELLED HERE AND NOT IMPORTED FROM helpers.utils, WHICH IS DELIBERATE AND HAS A COST. This is a
# login-node tool -- it copies a file and writes a sidecar -- and helpers.utils imports torch at
# module scope, so importing CHAT_SPECIALS would make creating a vocabulary need the GPU
# virtualenv. The cost is that this tuple and helpers.utils.CHAT_SPECIALS must stay identical in
# CONTENT AND ORDER, and nothing enforces it: a mismatch appears as a stage-7 tokenizer that
# appends a fifth token and reads every marker as something else. Change the two together.
SPECIALS = ("<|startofmessage|>", "<|endofmessage|>", "<startofthink>", "<endofthink>")

NAME = "bpe-instruct"
BASE = "bpe-sft-coder"
TARGET = os.path.join("saved_models", f"tokenizer-{NAME}")

SOURCES = (os.path.join("checkpoints", BASE),
           os.path.join("saved_models", f"tokenizer-{BASE}"),
           os.path.join("saved_models", f"tokenizer-{BASE}", "bpe"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="source", default="", metavar="DIR",
                   help=f"the vocabulary to extend (default: the first of "
                        f"{', '.join(SOURCES)} that holds a bpe.json)")
    p.add_argument("--out", default=TARGET, metavar="DIR",
                   help=f"where bpe.json is written (default: {TARGET})")
    p.add_argument("--force", action="store_true",
                   help="rewrite the output even when it already carries the four")
    return p.parse_args(argv)


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(config.ROOT, path)


def resolve_source(source=""):
    """The bpe.json to extend: `source`/bpe.json when one was named, else the first of SOURCES.

    A NAMED --from IS NOT SEARCHED PAST. Falling through from a directory the user typed to one
    they did not would build the instruct vocabulary on a base they never asked for, and the
    only symptom is a sidecar pinning ids that belong to another vocabulary."""
    if source:
        here = [_abs(source), os.path.join(_abs(source), "bpe")]
    else:
        here = [_abs(d) for d in SOURCES]
    for d in here:
        candidate = os.path.join(d, "bpe.json")
        if os.path.isfile(candidate):
            return candidate
    raise SystemExit("[bpe-instruct] no vocabulary found. Looked for bpe.json in:\n"
                     + "\n".join(f"          {os.path.relpath(d, config.ROOT)}" for d in here)
                     + "\n      Fetch it:  sh ./run_scripts/run_download_data.sh --only "
                       f"tokenizer-{BASE}")


def create(source="", out=TARGET, force=False, log=print):
    """Copy `source`/bpe.json unchanged and write special_tokens.json beside it.

    THE BASE FILE IS NEVER REWRITTEN.

    bpe-instruct is bpe-sft-coder's 23.9 MB vocabulary BYTE FOR BYTE, beside a sidecar naming
    four tokens. BPETokenizer.load appends them, so a consumer sees one tokenizer of 58,263 ids
    and never learns there were two files -- and the day a merge is added to the base, the
    sidecar is still correct because it describes what is APPENDED, not what the ids are.

    The sidecar pins the ids it expects. load() checks them, so a sidecar that has drifted onto
    a base with a different number of merges is refused rather than silently shifting every id."""
    src = resolve_source(source)
    dst_dir = _abs(out)
    dst = os.path.join(dst_dir, "bpe.json")
    sidecar = os.path.join(dst_dir, BPETokenizer.SPECIAL_TOKENS_FILE)

    base = BPETokenizer.load(src)
    log(f"[bpe-instruct] base {os.path.relpath(src, config.ROOT)}: "
        f"{len(base):,} tokens, {len(base.merges):,} merges, specials {base.specials}")

    if os.path.isfile(sidecar) and not force:
        have = BPETokenizer.load(dst) if os.path.isfile(dst) else None
        if have is not None and all(t in have.special2id for t in SPECIALS):
            log(f"[bpe-instruct] {os.path.relpath(dst_dir, config.ROOT)} is already made: "
                + ", ".join(f"{t}={have.special2id[t]}" for t in SPECIALS)
                + f"; {len(have):,} tokens. Nothing to do (--force rewrites it).")
            return dst_dir

    # THE IDS THE SIDECAR PINS, worked out by registering on a copy of the base rather than by
    # arithmetic: the base's own special count is the base's business.
    probe = BPETokenizer.load(src)
    ids = {t: probe.register_special(t) for t in SPECIALS}

    os.makedirs(dst_dir, exist_ok=True)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)                     # verbatim: not re-serialised, not edited
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(ids, fh, ensure_ascii=False, indent=1)
        fh.write("\n")

    # READ THE DIRECTORY BACK, which is what every consumer will do.
    back = BPETokenizer.load(dst)
    if len(back) != len(base) + len(SPECIALS):
        raise SystemExit(f"[bpe-instruct] {dst_dir} reads back at {len(back):,}, not "
                         f"{len(base) + len(SPECIALS):,}")
    if any(back.special2id.get(t) != i for t, i in ids.items()):
        raise SystemExit(f"[bpe-instruct] the sidecar's ids are not the ids that come back")
    # ONE MESSAGE OF EACH ROLE, IN THE SHAPE chat_index BUILDS. The role prefixes are ordinary
    # text, so this also checks the thing that would silently break the format: `User:\n` must
    # NOT come back as a marker id, and the four markers must each be exactly one id.
    text = ("<|startofmessage|>User:\nhello<|endofmessage|>"
            "<|startofmessage|>Assistant:\nhi<|endofmessage|>"
            "<startofthink>t<endofthink>")
    if sum(1 for i in back.encode(text) if i in ids.values()) != 6:
        raise SystemExit(f"[bpe-instruct] the markers do not encode as single ids")
    if any(i in ids.values() for i in back.encode_ordinary(text)):
        raise SystemExit(f"[bpe-instruct] encode_ordinary produced a marker: it must not")
    # BOTH DIRECTIONS. skip_special_tokens=False because the DEFAULT decode drops specials by
    # design -- that is what keeps a marker out of a reply shown to a person -- so the round trip
    # has to ask for them back to check that each id still stands for the token it was made for.
    if back.decode(back.encode(text), skip_special_tokens=False) != text:
        raise SystemExit(f"[bpe-instruct] the markers do not decode back to what was encoded")
    if os.path.getsize(dst) != os.path.getsize(src):
        raise SystemExit(f"[bpe-instruct] {dst} is not a verbatim copy of {src}")

    log(f"[bpe-instruct] wrote {os.path.relpath(dst_dir, config.ROOT)}/")
    log(f"[bpe-instruct]   bpe.json              copied unchanged "
        f"({os.path.getsize(dst) / 2**20:.1f} MiB)")
    log(f"[bpe-instruct]   {BPETokenizer.SPECIAL_TOKENS_FILE:<21} "
        + ", ".join(f"{t}={i}" for t, i in ids.items()))
    log(f"[bpe-instruct] the directory reads as {len(back):,} tokens; every id below "
        f"{len(base):,} is the base's, unchanged")
    log(f"[bpe-instruct] point the instruct stages at it: --tokenizer {NAME}")
    log(f"[bpe-instruct] they copy it to checkpoints/{NAME}/ themselves on first use")
    return dst_dir


def main(argv=None):
    a = parse_args(argv)
    create(a.source, a.out, a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
