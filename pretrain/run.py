"""
pretrain/run.py -- stage 6 entry point: language-model pretraining.

    python -m pretrain.run [--pretrain_steps N --peak_lr LR]
    ./old_scripts/stage6_pretrain.sh
"""
import os

import default_config as config
import helpers
from helpers import common
from model import parallel

from . import pretrain


def main():
    args = common.parse_args(stage="pretrain")
    ctx = common.setup(args, need_pairs=False)      # the pretraining corpus and nothing else
    path = helpers.ckpt_path(ctx["ckdir"], "pretrain")
    short = os.path.relpath(path, config.ROOT) if os.path.isabs(path) else path
    ctx["log"](f"[pretrain] checkpoint: {short} "
               f"({'resuming from it' if os.path.isfile(path) else 'new, not on disk yet'})")
    corpus, n_files, n_tok = pretrain.load_corpus(ctx["tok"], args.pretrain_dir, ctx["log"],
                                                     args.batch, args.max_len)
    if not corpus:
        ctx["log"](f"no corpus files under {', '.join(args.pretrain_dir)}: "
                   f"nothing to pretrain")
        return
    pretrain.run(ctx, corpus)
    # LEAVE THE PROCESS GROUP EXPLICITLY. A rank that exits without it can leave the others
    # blocked in a collective until the launcher times them out, which reads as a hang at the
    # very end of a run that in fact finished. No-op when there is no group.
    parallel.dist_close()


if __name__ == "__main__":
    try:
        main()
    finally:
        parallel.dist_close()
