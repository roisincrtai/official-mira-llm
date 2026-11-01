"""
eval.py -- held-out evaluation of a trained stage against the SFT reference.

Reads checkpoints/<stage>/<model>_<pe>_<stage-label>/checkpoint.pt and reports, per stage:
    1. teacher-forced preference accuracy, margin and length-shortcut correlation
    2. per-token likelihood displacement of the chosen and rejected responses

Results go to results/eval_<dataset>_<stage>.json.

    python evals/eval.py --stages dpo
"""
import os
import sys

# RUNNABLE AS A SCRIPT, not only as a module. `python evals/<name>.py` is what a person
# types, and without this it fails on `import default_config` -- the project root is on
# the path when python is given a module (-m) and is NOT when it is given a file.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import json
import argparse

import torch
import torch.nn.functional as F

import default_config as config
from helpers import dataset_helpers as dsets
from tokenizer import BPETokenizer
from model import MiraLLM
from helpers import (ckpt_path, resolve_device, run_tag, reward_hist,
                     set_run_tag)
from helpers.common import model_cfg_from_args, translate_cfg
from instruct_dpo.dpo import seq_logp, margin as dpo_margin


def parse_args():
    ap = argparse.ArgumentParser(description="evaluate trained stages; plots -> outputs/plots/eval/")
    ap.add_argument("--dataset", default="hh",
                    help="\"hh\", or a path to your own folder of preference records")
    ap.add_argument("--stages", default="dpo",
                    help="comma-separated stages to evaluate; each is read from "
                         "checkpoints/<stage>/<model>_<pe>_<stage-label>/checkpoint.pt")
    ap.add_argument("--model_dir", default="saved_models")
    ap.add_argument("--data_dir", default=config.DOWNLOAD_DIR)
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--plot_dir", default="plots")
    ap.add_argument("--result_dir", default="results",
                    help="dir for per-method eval JSON: results/eval_<llm>_<dataset>_<method>.json")
    ap.add_argument("--gpu", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="cap #examples loaded (0 = full data)")
    ap.add_argument("--n_hist", type=int, default=256, help="val pairs for reward/margin histograms")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


@torch.no_grad()
def tf_diagnostics(policy, ref, enc, tok, pairs, beta, batch=16):
    margins, len_diffs = [], []
    # ntok_w/ntok_l count RESPONSE TOKENS, not sequences. `ntok += W[0].shape[0]` counted
    # sequences, so dlp_w/dlp_l were per-SEQUENCE despite the name and despite being the axes
    # of the likelihood-displacement panel -- inflated by the mean response length, and by a
    # different factor for the chosen and rejected branches whenever their lengths differ,
    # which is exactly the asymmetry that panel exists to show.
    dlp_w = dlp_l = 0.0
    ntok_w = ntok_l = 0.0
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        W = enc.encode(chunk, "chosen"); L = enc.encode(chunk, "rejected")
        margins.append(dpo_margin(policy, ref, W, L, beta))
        for p in chunk:
            lc = len(tok(" " + p["chosen"], add_special_tokens=False)["input_ids"])
            lr = len(tok(" " + p["rejected"], add_special_tokens=False)["input_ids"])
            len_diffs.append(lc - lr)
        dlp_w += (seq_logp(policy, *W) - seq_logp(ref, *W)).sum().item()
        dlp_l += (seq_logp(policy, *L) - seq_logp(ref, *L)).sum().item()
        # W = (ids, attn, rmask); seq_logp sums over rmask[:, 1:], so that is the token count
        # matching the numerator exactly.
        ntok_w += W[2][:, 1:].sum().item()
        ntok_l += L[2][:, 1:].sum().item()
    M = torch.cat(margins).float().cpu(); ld = torch.tensor(len_diffs, dtype=torch.float)  # CPU (MPS/CUDA-safe)
    corr = torch.corrcoef(torch.stack([M, ld]))[0, 1].item() if M.numel() > 1 else float("nan")
    return {"acc": (M > 0).float().mean().item(), "mu_M": M.mean().item(),
            "var_M": M.var(unbiased=False).item(), "len_corr": corr,
            "dlp_w": dlp_w / max(ntok_w, 1.0), "dlp_l": dlp_l / max(ntok_l, 1.0)}


def main():
    args = parse_args()
    if args.max_len == 0:
        args.max_len = config.MODEL.get("block_size") or config.context_window(config.DEFAULT_SCHEME)
    device = resolve_device(args.gpu)
    methods = [m for m in args.stages.split(",")]
    tag = run_tag("mira_llm", args.dataset)
    # THE RUN TAG HAS TO BE SET OR NOTHING IS FOUND. ckpt_path() composes
    # checkpoints/<stage>/<scheme>_<arch>_<label>/checkpoint.pt from the global helpers sets in
    # set_run_tag; unset, artefact_tag() is empty and every lookup falls back to the stage
    # directory itself, which no stage has ever written into. `tag` above is a different, older convention (<llm>_<dataset>)
    # and only labels this script's own output files.
    #
    # This script exposes no --mira_config/--arch, so the tag is built from default_config.MODEL
    # -- correct for a run trained at the defaults, wrong for an ablation. That case fails as a
    # FileNotFoundError naming the path it looked for, which is recoverable; it was previously an
    # identical failure for every run, including the default one.
    set_run_tag(model_cfg_from_args(args))
    ckdir = config.CHECKPOINT_DIR                      # checkpoints/<stage>/<tag>/checkpoint.pt

    _, ev = dsets.load_pairs(args.dataset, args.data_dir, args.val_frac, args.seed, args.limit)
    # saved_models/ FIRST, THEN checkpoints/ -- tokenizer.run.base_path. config.BPE_PATH
    # names only the checkpoints/ copy, and the base vocabulary is published as
    # tokenizer-bpe-fineweb-edu-2BT, which is where stage 1 puts it.
    from tokenizer.run import base_path
    bpe_file = base_path()
    if os.path.isfile(bpe_file):
        tok = BPETokenizer.load(bpe_file)
    else:
        texts = [p[k] for p in ev for k in ("prompt", "chosen", "rejected")]
        tok = BPETokenizer.build(texts, num_merges=config.BPE["num_merges"])
    enc = dsets.Encoder(tok, device, args.max_len)

    def load_policy(stage):
        p = ckpt_path(ckdir, stage)          # checkpoints/<stage>/<tag>_<label>/checkpoint.pt
        if not os.path.isfile(p):
            raise FileNotFoundError(f"missing {p}; run the {stage} stage first")
        ck = torch.load(p, map_location="cpu")
        # the checkpoint's own architecture, not whatever default_config.py says today --
        # through translate_cfg, the one place that knows the retired spellings. A checkpoint
        # written before `pe` became `arch` carries use_ssm/use_rope, which MiraLLM.__init__
        # does not accept; stripping vocab_size alone left them in and raised TypeError.
        saved_cfg = ck.get("model_cfg")
        if not isinstance(saved_cfg, dict):
            raise SystemExit(f"checkpoint architecture metadata is missing: {p}")
        cfg = translate_cfg(saved_cfg)
        m = MiraLLM(vocab_size=len(tok), **cfg).to(device)
        m.load_state_dict(ck["model"])
        return m.eval()

    print(f"[eval] tag={tag} device={device} loading checkpoints from {ckdir}/ ...")
    ref = load_policy("sft")
    for p in ref.parameters():
        p.requires_grad_(False)

    os.makedirs(args.result_dir, exist_ok=True)
    RES = {"tag": tag, "beta": args.beta, "results": {}}
    for mode in methods:
        pol = load_policy(mode)
        print(f"[eval] {mode}: teacher-forced diagnostics + histograms ...")
        diag = tf_diagnostics(pol, ref, enc, tok, ev, args.beta)
        hist = reward_hist(pol, ref, enc, ev[:args.n_hist], args.beta)
        res = {"llm": "mira_llm", "dataset": args.dataset, "method": mode, "tag": tag,
               "beta": args.beta, "n_val": len(ev), **diag, "hist": hist}
        RES["results"][mode] = res
        out_json = os.path.join(args.result_dir, f"eval_{args.dataset}_{mode}.json")
        json.dump(res, open(out_json, "w"))          # full JSON, re-usable without re-training
        print(f"[eval] wrote {out_json}")


if __name__ == "__main__":
    main()
