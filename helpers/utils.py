"""
helpers.py -- utilities every stage shares: device resolution, progress bars, the cosine
learning-rate schedule, table printing, held-out evaluation and diagnostic probes, the
mixed-precision optimizer, the checkpoint/history IO that resume and the figures are built
on, and the on-disk token cache.
"""
import array
import contextlib
import glob
import json
import math
import os
import re
import shutil

try:                      # progress bars are optional; absence of tqdm must not be fatal
    from tqdm import tqdm
except Exception:                                     # noqa: BLE001
    def tqdm(it, *a, **k): return it

import torch
import torch.nn.functional as F

import default_config as config
# THE CORPUS IDENTITY LIVES IN A TORCH-FREE MODULE, and is imported rather than
# duplicated: tools/rename_dataset.py needs a signature on a login node with no GPU
# virtualenv, and a signature computed two ways is two answers to the one question
# a signature exists to answer. See helpers/corpus_id.py.
from .corpus_id import (_split_token, corpus_files, corpus_name,  # noqa: F401
                        corpus_signature, _sig_tag, _tok_signature)
# THE DPO OBJECTIVE IS IMPORTED WHERE IT IS USED, NOT AT MODULE SCOPE. instruct_dpo.dpo does
# `import helpers` at its own module scope, so a module-scope import here closes the cycle: stage
# 11 starts at instruct_dpo, which imports helpers, which imports instruct_dpo.dpo while it is
# still half-built, and the name is not there yet. helpers/__init__.py already keeps
# helpers.common out of the package for the same reason. The three call sites below take it
# inside the function, by which time both modules are complete whichever was entered first.


def _dpo_objective():
    from instruct_dpo.dpo import margin, rewards
    return margin, rewards


def resolve_device(choice):
    from kernel.hardware import resolve
    return resolve(choice)


# --------------------------------------------------------------------------- #
# THE DATA ROOT IS INPUT. Nothing this program writes may land in it.
# --------------------------------------------------------------------------- #
def input_only(path, what="write"):
    """Refuse `path` if it is inside data/. Returns it otherwise, so it can wrap a target.

    data/ holds the corpora as they were fetched from Hugging Face. They are the INPUT to
    every stage and the one thing in this project that cannot be regenerated from anything
    else here -- re-fetching means the network, the hours and the exact revision being
    available again. A stage READS them; nothing derived belongs among them, no file is added
    beside them, no directory is renamed or resharded under them, and a corpus is never
    rewritten in place. Everything produced goes to cache/, checkpoints/ or outputs/.

    This is a guard rather than a convention because a convention is a thing that holds until
    someone adds one line in a hurry. It sits on the WRITE primitives -- the checkpoint save,
    the history save, the token-store shard writer and its manifest, the pair cache -- so the
    rule is enforced at the only places it can be broken, whatever calls them.

    The one exception is the download tool, which is what puts the corpora there in the first
    place and does not go through these primitives."""
    p = os.path.realpath(path)
    root = os.path.realpath(config.DATA_DIR)
    if p == root or p.startswith(root + os.sep):
        raise SystemExit(
            f"[guard] refusing to {what} inside the data root: {os.path.relpath(p, config.ROOT)}\n"
            f"        data/ holds the corpora as downloaded. They are read-only input; only\n"
            f"        tools/download_data.py may put anything there. Everything this program\n"
            f"        produces belongs under cache/, checkpoints/ or outputs/.")
    tokens = os.path.realpath(os.path.join(config.CACHE_DIR, "tokens"))
    if (p == tokens or p.startswith(tokens + os.sep)) and \
            os.environ.get("MIRALLM_STAGE") != "tokenize_data":
        raise PermissionError(f"Cannot {what} under cache/tokens: only stage 4 writes tokens")
    return path


def progress(it, desc="", initial=0, total=None, bar_format=None):
    """tqdm over `it`, counting in ABSOLUTE steps.

    On resume the caller iterates range(start, total), whose length is the REMAINING count, so
    a stage resumed at 30000 of 40000 rendered as `499/10000` -- a bar that restarts at zero and
    a denominator that is neither the budget nor anything else the run was asked for. Passing
    `initial=start` and `total=<budget>` makes the same iteration render `30499/40000`: the
    numerator is the true step number, the denominator the budget, and the percentage and ETA
    both refer to the whole run rather than to the tail of it.

    ONE BAR UNDER DDP, DRAWN BY RANK 0. Four processes redrawing four bars on one terminal
    interleave into something that is not a bar at all, and the other three carry no
    information the first does not -- every rank runs the same steps. The bar's format is
    untouched; only who prints it is decided here. Off DDP is_main() is true and this is the
    same tqdm it has always been."""
    from model import parallel                  # local: helpers is imported before model is
    if not parallel.is_main():
        return it
    try:
        from tqdm import tqdm
        return tqdm(it, desc=desc, leave=False, initial=initial, total=total, bar_format=bar_format)
    except Exception:
        return it


def bar(desc, unit="it", total=None, unit_scale=False, initial=0, position=0, leave=True):
    """A tqdm bar driven by .update() rather than by wrapping an iterable.

    `progress` wraps an iterable and is right when the count is known. It is wrong when the
    iterable is a handful of very large files: the bar then ticks once per file and sits still
    for minutes, which is indistinguishable from a hang. This counts the units the work is
    actually made of, and the caller puts the coarse position in the postfix.

    ALWAYS PASS A TOTAL. Without one tqdm can print a count and a rate but no percentage and
    no estimate, and "156163doc [02:57, 900doc/s]" does not answer the only question being
    asked, which is how much is left. Where the exact count is unknowable in advance, measure
    the work in a unit that IS knowable -- bytes, usually, via corpus_bytes below.

    `initial` IS THE WORK ALREADY DONE, for a resumed job. A build that restarts its bar at
    zero having already written a quarter of the corpus reports 0% and an ETA for the whole
    corpus -- the two numbers a person uses to decide whether the resume worked. Passing the
    bytes already accounted for makes the percentage and the estimate refer to the corpus
    rather than to the remainder of it.

    Absence of tqdm must never be fatal, so a no-op stand-in with the same interface is
    returned instead.

    AND ONE BAR UNDER DDP, DRAWN BY RANK 0, for the reason `progress` gives above: four
    processes redrawing four bars on one terminal interleave into something that is not a bar.
    The other ranks get the same stand-in that a machine without tqdm gets, so every call site
    keeps working with no branch of its own. Off DDP is_main() is true and nothing changes."""
    from model import parallel                  # local: helpers is imported before model is

    class _Null:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def update(self, n=1): pass
        def set_postfix_str(self, s): pass
        def close(self): pass
    if not parallel.is_main():
        return _Null()
    try:
        from tqdm import tqdm
        return tqdm(desc=desc, unit=unit, total=total, unit_scale=unit_scale,
                    initial=initial, position=position, leave=leave)
    except Exception:                                          # noqa: BLE001
        return _Null()


class CosineLR:
    """Linear warmup then cosine decay, shared by every trainer.

        lr(t) = lr (t + 1) / W                                             t <  W
        lr(t) = lr_min + 1/2 (lr - lr_min) (1 + cos(pi (t-W) / (T-W))),    t >= W

    with lr_min = last_lr. The rate ramps from near zero to the stage's `lr` over W steps,
    then decays smoothly to `last_lr` (2e-5 by default, an absolute rate) across the steps
    that remain.

    WHY THE RAMP.

    The opening steps are not like the rest. Adam's second moment is an exponential average
    with beta2 = 0.999, so for the first several hundred steps `sqrt(v)` is built from a
    handful of samples and the ratio m/sqrt(v) can be far larger than 1 for any parameter whose
    gradient happened to be small in those batches -- the bias correction fixes the expectation,
    not the variance. A small rate while that denominator is unreliable is what warmup buys.

    THE DECAY SPANS WHAT IS LEFT, T - W, NOT T. Otherwise the ramp eats the head of the cosine
    and the run reaches the floor W steps early, which is a different schedule from the one the
    budget describes.

    The schedule is a function of the ABSOLUTE step, not of a counter inside the object, so a
    run resumed at step 30000 of 40000 continues on the same curve instead of restarting it --
    which is why `step()` takes the step rather than incrementing. A resume past W is past the
    warmup and does not repeat it, which is the point: a re-warm on every restart would hand an
    annealed model a rate it had already left behind.

    `last_lr >= lr` or schedule "constant" degenerates to a flat `lr` AFTER THE RAMP, so a caller
    never has to branch on whether scheduling is enabled; `warmup = 0` restores the curve this
    class had before the ramp existed, exactly.
    """
    def __init__(self, opt, lr, total, last_lr=2e-5, schedule="cosine", warmup=0):
        self.opt, self.lr, self.total = opt, lr, max(int(total), 1)
        self.warmup = max(int(warmup or 0), 0)
        self.cosine = (schedule or "cosine").lower().startswith("cos") and last_lr < lr
        self.lr_min = float(last_lr) if self.cosine else lr

    def value(self, step):
        s = max(int(step), 0)
        # `s + 1`, so the first step trains at lr/W rather than at nothing: a step at rate zero
        # spends a whole batch and moves no parameter.
        if s < self.warmup:
            return self.lr * (s + 1) / self.warmup
        if not self.cosine:
            return self.lr
        span = max(self.total - self.warmup, 1)
        t = min(s - self.warmup, span) / span
        return self.lr_min + 0.5 * (self.lr - self.lr_min) * (1.0 + math.cos(math.pi * t))

    def step(self, step):
        """Set the optimizer's learning rate for absolute step `step`; returns the rate."""
        v = self.value(step)
        for g in self.opt.param_groups:
            g["lr"] = v
        return v

    def describe(self):
        ramp = f"warmup {self.warmup:,} -> " if self.warmup else ""
        return (f"{ramp}cosine {self.lr:g} -> {self.lr_min:g} over "
                f"{self.total - self.warmup:,} of {self.total:,} steps"
                if self.cosine else f"{ramp}constant {self.lr:g}")


def schedule_total(ck, configured, log=None, stage=""):
    """THE BUDGET THE CURVE WAS STARTED ON, which a resume may not change. Returns an int.

    THE BUG THIS EXISTS FOR, and it was seen in a plot rather than in a log. The rate fell to
    2.5e-6 by step ~15,300 and then JUMPED to ~1.07e-5:

        lr(t) = lr_min + 1/2 (lr - lr_min) (1 + cos(pi t / T))

    is correct, and T was SCHEME_STEPS -- edited from 17,100 to 30,000 to 60,000 while a run was
    resumable. At step 15,300, T = 17,100 gives 2.5e-6 and T = 30,000 gives 1.07e-5. Nothing
    pinned T across the resume, so the run changed curve mid-flight and the optimiser was handed
    a rate four times what it had been converging under.

    THE FIX IS THE ONE THIS REPOSITORY ALREADY USES FOR THE SAME CLASS OF PROBLEM. The CONTEXT
    WINDOW is recorded in the checkpoint and ratcheted so it can never go backwards -- see
    model.lm.context_floor. This does the same for T, and reads the same two sources, larger
    specificity first:

      * `sched_total`, written by save_ckpt at every save since 2026-08-16. Authoritative: it is
        the T the curve was actually being evaluated at.
      * `total`, the step BUDGET the checkpoint was written under, which every checkpoint has
        always carried. This is the fallback for a checkpoint written before `sched_total`
        existed -- and it is a WEAK one, because it is the budget in force at the LAST SAVE, not
        at the first step. A run whose budget was already edited twice has the LATEST of those
        in `total`, so the original curve is not recoverable from it. What it can still do is
        stop the NEXT jump, which is worth having and is all it claims.

    NO LEARNING RATE IS CHOSEN HERE. `lr`, `last_lr`, `lr_schedule` and `warmup` are not
    read, not written and not touched. This returns a STEP COUNT, and it returns the one the run
    already had; it preserves a curve, it does not pick one.

    A FRESH RUN IS UNCHANGED: no checkpoint means the configured budget, exactly as before.

    IT IS UNCONDITIONAL ONCE THERE IS A CHECKPOINT, AND THAT HAS A CONSEQUENCE WORTH SAYING.
    Deliberately RAISING the budget of a finished run to train it further now extends it at the
    floor of the original curve rather than lifting the rate back up -- the curve is monotone,
    which is the property that was missing. To put a run on a genuinely new curve, start one:
    move the checkpoint aside, or pass --no-resume.
    """
    configured = int(configured)
    if not ck:
        return configured
    rec = int(ck.get("sched_total", 0) or 0)
    src = "recorded with the schedule"
    if rec <= 0:
        rec = int(ck.get("total", 0) or 0)
        src = "checkpoint total"
    if rec <= 0 or rec == configured:
        return configured
    if log is not None:
        pre = f"{stage}: " if stage else ""
        log(f"{pre}lr schedule over {rec:,} steps ({src}); configured budget {configured:,}")
    return rec


def table(title, rows, out=None):
    """Print `rows` (a list of (key, value) pairs, or a dict) as a bordered two-column
    table, ONE FIELD PER LINE. Used for the DATASET STATISTICS and the CORPUS SCAN RESULTS
    only -- the summaries whose many numbers are unreadable on one line. Everything else
    (per-stage settings, done lines) stays as plain single-line logging.

        +-- title ---------------------+
        | key                  value   |
        +------------------------------+
    """
    out = out or (lambda m: print(m, flush=True))
    items = list(rows.items()) if isinstance(rows, dict) else list(rows)
    items = [(str(k), "" if v is None else str(v)) for k, v in items]
    kw = max([len(k) for k, _ in items] + [1])
    vw = max([len(v) for _, v in items] + [1])
    inner = kw + vw + 4                              # "| " + k + "  " + v + " |"
    inner = max(inner, len(title) + 4)               # a long title must not overflow the box
    out("+" + f"-- {title} ".ljust(inner, "-") + "+")
    for k, v in items:
        out("| " + f"{k.ljust(kw)}  {v.ljust(vw)}".ljust(inner - 2) + " |")
    out("+" + "-" * inner + "+")


def human(n):
    """A count as a short magnitude string: 2_034_110_928 -> "2.03B".

    Reported ALONGSIDE the exact number, never instead of it. A corpus is the one thing about a
    run that cannot be recovered from the checkpoint, so the log has to say plainly how much
    text went in -- and "2,034,110,928" is a figure the eye slides over, while "2.03B" is one
    it checks against what was intended. Thresholds are decimal (1e3/1e6/1e9/1e12), which is
    how token counts are quoted everywhere in this literature."""
    n = float(n)
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{suffix}"
    return f"{n:.0f}"


def count(n):
    """`123,456,789 (123.46M)` -- the exact count and its magnitude, which is the form every
    corpus statistic in this project is reported in. Below a thousand the magnitude would just
    repeat the number, so it is omitted: "78 (78)" is noise, not information."""
    n = int(n)
    return f"{n:,}" if abs(n) < 1000 else f"{n:,} ({human(n)})"


def run_tag(llm, dataset):
    """<llm>_<dataset> tag, filesystem-safe (e.g. gpt2_hh-rlhf, mira_llm_hh)."""
    short = dataset.split("/")[-1]
    safe = lambda s: "".join(c if (c.isalnum() or c in "-._") else "-" for c in s)
    return f"{safe(llm)}_{safe(short)}"


@torch.no_grad()
def evaluate(policy, ref, enc, pairs, beta, batch=16):
    """Held-out teacher-forced preference accuracy / mean margin / reversal rate."""
    dpo_margin, _ = _dpo_objective()
    Ms = []
    for i in progress(range(0, len(pairs), batch), desc=tag("eval") + " held-out margins",
                      total=(len(pairs) + batch - 1) // batch):
        W = enc.encode(pairs[i:i + batch], "chosen")
        L = enc.encode(pairs[i:i + batch], "rejected")
        Ms.append(dpo_margin(policy, ref, W, L, beta))
    M = torch.cat(Ms)
    return {"acc": (M > 0).float().mean().item(), "mu_M": M.mean().item(),
            "P_rev": (M <= 0).float().mean().item()}


@torch.no_grad()
def reward_hist(policy, ref, enc, pairs, beta, batch=16):
    """Per-pair teacher-forced distributions for histograms:
    r_w (chosen reward; win = r_w>0), r_l (rejected reward; lose = r_l<0), margin = r_w-r_l."""
    _, dpo_rewards = _dpo_objective()
    rw, rl = [], []
    for i in progress(range(0, len(pairs), batch), desc=tag("eval") + " reward histogram",
                      total=(len(pairs) + batch - 1) // batch):
        W = enc.encode(pairs[i:i + batch], "chosen")
        L = enc.encode(pairs[i:i + batch], "rejected")
        cr, rr = dpo_rewards(policy, ref, W, L, beta)
        rw += [round(x, 5) for x in cr.tolist()]
        rl += [round(x, 5) for x in rr.tolist()]
    return {"r_w": rw, "r_l": rl, "margin": [round(a - b, 5) for a, b in zip(rw, rl)]}


def mixed_precision_ok(device):
    """Can this device carry bf16 activations? (available, supported, and CUDA).

    CUDA ONLY, and deliberately. bfloat16 exists on CPU and on MPS as a dtype, but without the
    hardware paths it is emulated -- slower than fp32 and with none of the bandwidth saving the
    whole exercise is for. A card older than Ampere reports no bf16 support, and there the
    answer is fp32 rather than fp16: fp16 needs loss scaling, and a scaler is a second thing to
    get wrong for a saving this pipeline does not need."""
    dev = device if isinstance(device, str) else getattr(device, "type", str(device))
    if str(dev).startswith("npu"):
        from kernel.hardware import npu_available
        if not npu_available():
            return False
        with torch.npu.device(device):
            return bool(getattr(torch.npu, "is_bf16_supported", lambda: False)())
    if not str(dev).startswith("cuda") or not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:                                             # noqa: BLE001
        return False


def amp(args):
    """The autocast context every training forward runs inside; a no-op when it is off.

    MIXED PRECISION, AS THE STANDARD RECIPE HAS IT: bfloat16 for the activations, the matmuls
    and the backward pass -- which is where the memory and the bandwidth go -- and fp32 for
    everything that ACCUMULATES. autocast draws that line itself and draws it correctly: matmul
    and the convolutions run in bf16, while softmax, log_softmax, layer normalisation and the
    reductions are on its fp32 list and stay there. The weights are never cast, so the master
    copy the optimiser updates is fp32 by construction, and AdamW's moments with it.

    WHY bf16 AND NOT fp16. bf16 keeps fp32's eight exponent bits and spends mantissa instead, so
    gradients neither overflow nor underflow and NO LOSS SCALING IS NEEDED. fp16 needs a
    GradScaler, which is a second mechanism to misconfigure; this pipeline has none and needs
    none. Pure bf16 -- bf16 optimiser state as well -- is the thing to avoid: once the weights
    are large next to the updates, an update below bf16's ~3 significant digits rounds to
    nothing and training stalls with a loss curve that looks merely flat. fp32 masters are
    exactly what stops that.

    WRAP THE FORWARD, NOT THE BACKWARD. autocast records the dtype each operation ran in, and
    the backward pass follows that record; wrapping backward as well is at best redundant. The
    optimiser step belongs OUTSIDE, where it operates on fp32 parameters and fp32 gradients."""
    if not getattr(args, "mixed_precision", False):
        return contextlib.nullcontext()
    kind = getattr(args, "_device_type", None) or resolve_device(getattr(args, "gpu", "auto")).type
    return torch.autocast(device_type=kind, dtype=torch.bfloat16)


class MasterAdamW:
    """AdamW with fp32 master weights.

    Live parameters may be bf16 (or any dtype); each step casts their gradients to fp32,
    updates fp32 masters with a stock AdamW (fp32 moments), and copies the masters back
    into the live parameters. For fp32 live parameters this reduces to plain AdamW at the
    cost of one extra weight copy. Accepts either an iterable of parameters or AdamW-style
    param groups; `param_groups` is the inner optimizer's (so lr edits pass through).

    THE BETAS, THE DECAY AND THE EPSILON COME FROM config.TRAIN unless the caller names them,
    so every trainer runs one recipe. A caller that passes `eps` explicitly overrides
    TRAIN["adam_eps"]."""

    def __init__(self, params, lr=1e-3, weight_decay=None, betas=None, eps=None,
                 mixed_precision=False):
        if weight_decay is None:
            weight_decay = float(config.TRAIN["weight_decay"])
        if betas is None:
            betas = (float(config.TRAIN["adam_beta1"]), float(config.TRAIN["adam_beta2"]))
        if eps is None:
            eps = float(config.TRAIN["adam_eps"])
        params = list(params)
        groups = params if (params and isinstance(params[0], dict)) else [{"params": params, "lr": lr}]
        self._pairs = []                    # [(live, master)] in group order
        inner_groups = []
        for g in groups:
            live = list(g["params"])
            masters = [p.detach().clone().float() for p in live]
            self._pairs += list(zip(live, masters))
            ig = {k: v for k, v in g.items() if k != "params"}
            ig.setdefault("lr", lr)
            ig["params"] = masters
            inner_groups.append(ig)
        self.inner = torch.optim.AdamW(inner_groups, lr=lr, betas=betas, eps=eps,
                                       weight_decay=weight_decay)
        self.settings = {"betas": betas, "eps": eps, "weight_decay": weight_decay}

    @property
    def param_groups(self):
        return self.inner.param_groups

    def zero_grad(self, set_to_none=True):
        for p, _ in self._pairs:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for p, m in self._pairs:
            m.grad = None if p.grad is None else p.grad.detach().float()
        self.inner.step()
        for p, m in self._pairs:
            p.copy_(m.to(p.dtype))

    def state_dict(self):
        return {"inner": self.inner.state_dict(), "masters": [m for _, m in self._pairs]}

    @staticmethod
    def _fit(dst, src):
        """`src` into `dst`, PADDING when `dst` has grown under it.

        Registering a special appends embedding rows, so a checkpoint written before it holds a
        master copy and moments that are correct in every row they have and short by the rows
        that were added. The saved block is copied into the leading corner and the rest of `dst`
        is left as it already was -- which for a master is the live parameter, whose new rows
        resize_token_embeddings initialised at the embedding mean, and for a moment is zero,
        which is the momentum a row that has never been stepped actually has.

        ANY OTHER SHAPE IS STILL AN ERROR. Growing is the one direction this accepts: a saved
        tensor LARGER in any dimension, or of a different rank, belongs to a different model."""
        if tuple(dst.shape) == tuple(src.shape):
            dst.copy_(src.to(dst.device))
            return
        if len(dst.shape) == len(src.shape) and all(a <= b for a, b in
                                                    zip(src.shape, dst.shape)):
            dst[tuple(slice(0, n) for n in src.shape)].copy_(src.to(dst.device))
            return
        raise RuntimeError(
            f"optimiser state {tuple(src.shape)} does not fit a parameter of "
            f"{tuple(dst.shape)}: this state belongs to a different model")

    def _grown_inner(self, inner):
        """`inner` with every state tensor padded to the parameter it belongs to.

        torch.optim.Optimizer.load_state_dict does not check shapes -- it assigns -- so an
        exp_avg three rows short would attach cleanly and fail on the first step() instead of
        here. State keys are positions in param_groups, which is the order of self._pairs."""
        masters = [m for _, m in self._pairs]
        out = dict(inner)
        state = {}
        for key, entry in (inner.get("state") or {}).items():
            if not isinstance(entry, dict) or not isinstance(key, int) or key >= len(masters):
                state[key] = entry
                continue
            want = tuple(masters[key].shape)
            fixed = {}
            for name, value in entry.items():
                if (torch.is_tensor(value) and value.dim() > 0
                        and tuple(value.shape) != want
                        and len(value.shape) == len(want)
                        and all(a <= b for a, b in zip(value.shape, want))):
                    grown = value.new_zeros(want)
                    grown[tuple(slice(0, n) for n in value.shape)] = value
                    fixed[name] = grown
                else:
                    fixed[name] = value
            state[key] = fixed
        out["state"] = state
        return out

    @torch.no_grad()
    def load_state_dict(self, sd):
        if isinstance(sd, dict) and "masters" in sd:
            removed = {}
            offset = 0
            for p, _ in self._pairs:
                for shape in getattr(p, "_mira_removed_input_bias_shapes", ()):
                    removed[offset] = shape
                    offset += 1
                offset += 1
            if removed and len(sd["masters"]) == len(self._pairs) + len(removed):
                for index, shape in removed.items():
                    if tuple(sd["masters"][index].shape) != shape:
                        raise ValueError("MIRA input-bias optimizer removal has incompatible shapes")
                inner = sd["inner"]
                groups, state, offset = [], {}, 0
                if len(inner["param_groups"]) != len(self.inner.param_groups):
                    raise ValueError("MIRA input-bias optimizer removal has incompatible groups")
                for old_group, live_group in zip(inner["param_groups"], self.inner.param_groups):
                    old_ids = [old_id for i, old_id in enumerate(old_group["params"], offset)
                               if i not in removed]
                    offset += len(old_group["params"])
                    if len(old_ids) != len(live_group["params"]):
                        raise ValueError("MIRA input-bias optimizer removal has incompatible groups")
                    group = dict(old_group, params=list(live_group["params"]))
                    for old_id, new_id in zip(old_ids, group["params"]):
                        if old_id in inner["state"]:
                            state[new_id] = inner["state"][old_id]
                    groups.append(group)
                sd = dict(sd, masters=[m for i, m in enumerate(sd["masters"]) if i not in removed],
                          inner=dict(inner, state=state, param_groups=groups))
            if len(sd["masters"]) != len(self._pairs):
                additions = [i for i, (p, _) in enumerate(self._pairs)
                             if getattr(p, "_mira_input_bias", False)]
                if len(sd["masters"]) + len(additions) == len(self._pairs):
                    source = iter(sd["masters"])
                    masters = [m.clone() if i in additions else next(source)
                               for i, (_, m) in enumerate(self._pairs)]
                    inner = sd["inner"]
                    old_groups = inner["param_groups"]
                    groups, state, offset = [], {}, 0
                    for old_group, live_group in zip(old_groups, self.inner.param_groups):
                        old_ids = iter(old_group["params"])
                        group = dict(old_group, params=list(live_group["params"]))
                        for new_id in group["params"]:
                            if offset not in additions:
                                old_id = next(old_ids)
                                if old_id in inner["state"]:
                                    state[new_id] = inner["state"][old_id]
                            offset += 1
                        if next(old_ids, None) is not None:
                            raise ValueError("MIRA input-bias optimizer migration has incompatible groups")
                        groups.append(group)
                    sd = dict(sd, masters=masters,
                              inner=dict(inner, state=state, param_groups=groups))
            from model.distributed import distributed_fsdp_channel
            sd = distributed_fsdp_channel.channel_local_optimizer_state(self._pairs, sd)
            self.inner.load_state_dict(self._grown_inner(sd["inner"]))
            for (_, m), s in zip(self._pairs, sd["masters"]):
                self._fit(m, s)
            for p, m in self._pairs:
                p.copy_(m.to(p.dtype))          # live follows the restored masters
        else:
            # legacy plain-AdamW checkpoint: adopt the (already-loaded) live weights as
            # masters, then restore the fp32 moments if their shapes are compatible
            for p, m in self._pairs:
                m.copy_(p.float())
            try:
                self.inner.load_state_dict(sd)
            except Exception:
                pass                            # moments restart; weights are correct
        for group in self.inner.param_groups:
            group["eps"] = self.settings["eps"]


# --------------------------------------------------------------------------- #
# checkpoint / history IO
# --------------------------------------------------------------------------- #
# Checkpoint layout, one directory per stage and one DIRECTORY per configuration under it:
#
#     checkpoints/<stage>/<model>_<arch>_<stage-label>/checkpoint.pt
#     checkpoints/<stage>/<model>_<arch>_<stage-label>/history.json
#     checkpoints/<stage>/<model>_<arch>_<stage-label>/config.json
#     checkpoints/<stage>/<model>_<arch>_<stage-label>/tokenizer.json
#     checkpoints/<stage>/<model>_<arch>_<stage-label>/dynamics.jsonl
#     outputs/plots/<stage>/dynamics_<model>_<arch>_<stage-label>.pdf
#
# where <model> is the size (mira-s / -m / -l), <arch> is the architecture and with it how
# position enters (ssm, mira or rope),
# and <stage-label> names the training stage the artefact came out of. The configuration is the
# DIRECTORY, so an ablation run sits beside the run it is being compared with instead of
# overwriting it -- which is the whole point of being able to set arch="rope". See artefact_dir
# for the two cases in which the level is not added.
#
# The stage is ALSO the directory, so the label in the name is redundant while a file sits
# where it was written. It stops being redundant the moment a file leaves: checkpoints get
# copied into saved_models/, attached to an issue, dropped in a shared folder, and a bare
# checkpoint_mira_llm-s_ssm.pt is then indistinguishable from the six other stages' output.
# The name is self-describing on its own.
# --------------------------------------------------------------------------- #

# The label each stage puts in its filenames. Deliberately NOT the internal stage key: "rlhf"
# and "dpo" both act on the instruction-following model and read better carrying that, and the
# chain-of-thought stage is named for its algorithm because that is what distinguishes it. A
# stage missing from this table falls back to its own key, so adding a stage cannot break the
# path helpers -- it only gets a plainer name until it is listed here.
STAGE_LABELS = {
    "pretrain": "pretrain",
    "sft": "sft",
    "reward": "reward",
    "rlhf": "instruct-rlhf",
    "dpo": "instruct-dpo",
    "cot_sft": "cot-sft",
    "cot": "cot-grpo",
    "distill": "distill",
    # Stage 11's label carries the LANGUAGE, so two adaptations of one model are told apart by
    # their filenames and not only by their directories. lang_sft/run.py rewrites this entry
    # from --lang_sft_short before any artefact is named.
    "lang_sft": "lang-sft",
}

# WHERE A STAGE'S ARTEFACTS LIVE, when that is not its own key. Chain of thought is TWO
# training runs -- supervised fine-tuning on demonstrations, then GRPO on the result -- and
# they are one stage of the pipeline, so they share a directory and are told apart by their
# labels: checkpoints/cot/<run>_cot-sft/ beside checkpoints/cot/<run>_cot-grpo/.
# Splitting them into checkpoints/cot/ and checkpoints/cot_sft/ would hide the fact that the
# second reads the first, and would leave a stage of the pipeline occupying two directories.
STAGE_DIRS = {
    "cot_sft": "cot",
    # Stage 7 is handed a directory that ALREADY names it --
    # checkpoints/instruct_sft/<scheme>_<dataset>/ -- because one model fine-tuned on one corpus
    # is what the artefact is, and checkpoints/sft/ held whichever pair ran last under one name.
    # An empty mapping means "write into the directory you were handed", exactly as lang_sft
    # below, so the path does not become .../<scheme>_<dataset>/sft/.
    "sft": "",
    # Stage 8 is handed a directory that ALREADY names it --
    # checkpoints/instruct_reward/<scheme>_<dataset>/ -- exactly as stage 7 is, so nothing is
    # appended to it.
    "reward": "",
    # Stage 11 is handed a directory that ALREADY names it --
    # checkpoints/instruct_dpo/<scheme>_<dataset>/ -- exactly as stages 7 and 8 are, so nothing
    # is appended to it. See dpo_dir.
    "dpo": "",
    # Stage 11 is given a directory that ALREADY names it -- checkpoints/lang-sft/<dataset>/,
    # because a language adaptation's artefacts belong with the corpus they adapted to. An
    # empty mapping means "write into the directory you were handed", so the path stays flat
    # instead of becoming checkpoints/lang-sft/<dataset>/lang_sft/.
    "lang_sft": "",
}


def stage_label(stage):
    """The filename label of a stage key ("rlhf" -> "instruct-rlhf")."""
    return STAGE_LABELS.get(stage, stage)


def sft_scheme_name(scheme=None):
    """The `<scheme>` of stage 7's path: the display name, lowercase -- `mira-s`.

    display_name is what a FIGURE calls the model, and it is the spelling the rule gives; the
    identifier `mira-s` is what model_name produces and is not what was asked for."""
    import default_config as config
    return display_name(scheme or config.DEFAULT_SCHEME).lower()


def sft_dataset_name(corpus_dir=None):
    """The `<dataset>` of stage 7's path: the name of the fine-tuning corpus directory."""
    import os as _os
    import default_config as config
    path = str(corpus_dir or config.SFT_DIR).rstrip("/")
    return _os.path.basename(path) or "corpus"


def sft_dir(scheme=None, corpus_dir=None):
    """checkpoints/instruct_sft/<scheme>_<dataset>/ -- where stage 7 writes and stages 8, 9 and
    11 read.

    ONE FUNCTION, BECAUSE FIVE PLACES RESOLVE THIS PATH. Stage 7 writes it; instruct_dpo,
    instruct_reward, instruct_rlhf and evals/eval.py each load the checkpoint under it as their
    starting policy or their frozen reference. A second copy of the arithmetic is a second answer
    to where the file is, and the stage that disagrees reports it missing rather than wrong.

    The root is SFT["ckpt_dir"] from the yaml and is configurable; the two leaf levels are the
    rule and are not."""
    import os as _os
    import default_config as config
    root = str(config.SFT.get("ckpt_dir") or _os.path.join("checkpoints", "instruct_sft"))
    if not _os.path.isabs(root):
        root = _os.path.join(config.ROOT, root)
    return _os.path.join(root, f"{sft_scheme_name(scheme)}_{sft_dataset_name(corpus_dir)}")


def find_sft_checkpoint():
    """The newest checkpoint anywhere under checkpoints/instruct_sft/, or "".

    RECURSIVE, because stage 7 writes one directory per fine-tune underneath the root --
    checkpoints/instruct_sft/<scheme>_<dataset>/ -- so the file is a level down and a
    single-level glob finds nothing. Nor can the path be COMPUTED: sft_dir names
    <scheme>_<dataset> from THIS run's configuration, and stage 8's is not stage 7's, so a stage
    that asked for the exact path would report "no sft checkpoint" beside a full tree.

    Newest by mtime: two fine-tunes of the same model on different corpora are both valid
    starting points, and the one written most recently is the one being worked on.

    ONE DEFINITION, because stages 8, 9 and 11 all start from stage 7's output and three copies
    of this glob are three answers to where that output is."""
    import glob as _glob
    import os as _os
    import default_config as config
    root = str(config.SFT.get("ckpt_dir") or _os.path.join("checkpoints", "instruct_sft"))
    if not _os.path.isabs(root):
        root = _os.path.join(config.ROOT, root)
    found = set(_glob.glob(_os.path.join(root, "**", "checkpoint.pt"), recursive=True))
    return max(found, key=_os.path.getmtime) if found else ""


def reward_dataset_name(dataset=None, data_dir=None):
    """The `<dataset>` of stage 8's path: the preference corpus it trains on.

    "hh" is the downloaded tree, and the name that means something to a reader is the DATASET
    directory -- llm-rlhf-instruction-following -- not the rlhf_hh subdirectory inside it that
    config.HH_DIR points at. A --dataset given as a path is named by its own last component."""
    import os as _os
    import default_config as config
    name = str(dataset or "hh").strip()
    if not name or name == "hh":
        return _os.path.basename(str(config.INSTRUCT_DIR).rstrip("/")) or "hh"
    return _os.path.basename(_os.path.normpath(name)) or "corpus"


def reward_dir(scheme=None, dataset=None):
    """checkpoints/instruct_reward/<scheme>_<dataset>/ -- where stage 8 writes and stage 9 reads.

    ONE FUNCTION, as sft_dir is one, because stage 9 loads this checkpoint as its scorer and a
    second copy of the arithmetic is a second answer to where the file is.

    The root is REWARD["ckpt_dir"] from the yaml and is configurable; the two leaf levels are
    the rule and are not."""
    import os as _os
    import default_config as config
    root = str(config.REWARD.get("ckpt_dir")
               or _os.path.join("checkpoints", "instruct_reward"))
    if not _os.path.isabs(root):
        root = _os.path.join(config.ROOT, root)
    return _os.path.join(root, f"{sft_scheme_name(scheme)}_{reward_dataset_name(dataset)}")


def dpo_dir(scheme=None, dataset=None):
    """checkpoints/instruct_dpo/<scheme>_<dataset>/ -- where stage 11 writes.

    THE SAME TWO LEAF LEVELS STAGES 7 AND 8 GET, and for the same reason: one model moved by one
    preference corpus is what the artefact is, and a directory named by the run tag alone held
    whichever pair ran last under one name.

    The root is DPO["ckpt_dir"] from the yaml and is configurable; the two leaf levels are the
    rule and are not."""
    import os as _os
    import default_config as config
    root = str(config.DPO.get("ckpt_dir") or _os.path.join("checkpoints", "instruct_dpo"))
    if not _os.path.isabs(root):
        root = _os.path.join(config.ROOT, root)
    return _os.path.join(root, f"{sft_scheme_name(scheme)}_{reward_dataset_name(dataset)}")


def dpo_plot_dir(scheme=None, dataset=None):
    """outputs/plots/instruct_dpo/<scheme>_<dataset>/ -- stage 11's figures."""
    import os as _os
    import default_config as config
    return _os.path.join(config.PLOT_DIR, "instruct_dpo",
                         f"{sft_scheme_name(scheme)}_{reward_dataset_name(dataset)}")


def reward_plot_dir(scheme=None, dataset=None):
    """outputs/plots/instruct_reward/<scheme>_<dataset>/ -- stage 8's figures."""
    import os as _os
    import default_config as config
    return _os.path.join(config.PLOT_DIR, "instruct_reward",
                         f"{sft_scheme_name(scheme)}_{reward_dataset_name(dataset)}")


def sft_plot_dir(scheme=None, corpus_dir=None):
    """outputs/plots/instruct_sft/<scheme>_<dataset>/ -- stage 7's figures, beside nothing else's."""
    import os as _os
    import default_config as config
    return _os.path.join(config.PLOT_DIR, "instruct_sft",
                         f"{sft_scheme_name(scheme)}_{sft_dataset_name(corpus_dir)}")


def stage_dirname(stage):
    """The directory a stage's artefacts live in -- its own key unless STAGE_DIRS says
    otherwise. Used for both checkpoints and figures, so the two cannot disagree.

    An empty string means the caller already supplied the directory and nothing should be
    appended to it."""
    return STAGE_DIRS.get(stage, stage)

# The named sizes, keyed by (n_layer, n_embd). A configuration that is not one of these gets a
# descriptive name rather than being forced into the nearest label -- so a model trained before
# these values changed is reported as, say, mira_llm-6x384 rather than silently adopting the
# label of a size it is not.
#
# DEPTH AND WIDTH ALONE, because n_head is not part of a MIRA description: the lattice has no
# heads, so a MIRA cfg does not record one, and keying on it resolved every checkpoint to the
# descriptive fallback. Nothing is lost by dropping it from the key -- every scheme has a head
# dimension of 64 (512/8 = 768/12 = 1024/16; see model.mira_llm.HEAD_DIM), so n_head is n_embd/64
# and carries no information the width does not. The pair is still unique across the ladder: M and
# L share a width and differ in depth, XS and S differ in both.
def _model_sizes():
    """(n_layer, n_embd) -> scheme name, DERIVED FROM SCHEMES rather than repeated.

    This was a hand-written table beside default_config.SCHEMES, and the two fell out of step
    the first time the schemes changed: every run then reported itself as `mira_llm-24x512`,
    the descriptive fallback, and wrote its checkpoints and figures under that name. A scheme
    is defined in exactly one place, so its name must be looked up from that place."""
    return {(s["n_layer"], s["n_embd"]): name for name, s in config.SCHEMES.items()}


def model_name(cfg):
    """The size label of a model configuration dict (or a MiraLLM's .cfg).

    Falls back to `mira_llm-<layers>x<width>` for an architecture no scheme describes, which is
    what a hand-built or checkpoint-restored configuration may be. Seeing that name where a
    scheme was expected means the two have diverged."""
    key = (cfg.get("n_layer"), cfg.get("n_embd"))
    return _model_sizes().get(key, f"mira_llm-{cfg.get('n_layer')}x{cfg.get('n_embd')}")


def display_name(cfg_or_name):
    """The name a FIGURE calls this model: `Mira-S`, `Mira-M`, `Mira-L`.

    SEPARATE FROM model_name() BECAUSE THE TWO ARE READ BY DIFFERENT AUDIENCES. model_name is an
    IDENTIFIER -- it goes into checkpoint filenames, history paths and run tags, where it must be
    lowercase, stable and filesystem-safe, and where `mira-s` is exactly right. This one goes
    into legends, axis labels and captions, where the internal product name is noise: a reader of
    the paper wants the size, not the codebase's name for itself.

    TAKES EITHER a config dict or a scheme name already resolved, so a caller that has only the
    string does not have to fabricate a config to ask.

    ANYTHING IT DOES NOT RECOGNISE COMES BACK UNCHANGED, which is deliberate. The fallback
    `mira_llm-<layers>x<width>` means a configuration no scheme describes, and inventing a pretty
    name for it would hide exactly the divergence model_name's fallback exists to show."""
    name = cfg_or_name if isinstance(cfg_or_name, str) else model_name(cfg_or_name)
    name = config.canonical_scheme(name)
    prefix = "mira-"
    if not name.startswith(prefix):
        return name
    size = name[len(prefix):]
    return f"Mira-{size.upper()}" if size in ("s", "m", "l") else name


def arch_of(cfg):
    """The architecture a config asks for.

        pe    the KEY before `arch` existed         -> "ssm" / "rope"

    Every site that branches on the architecture without building a model first -- the cache
    sizer, the run banner, mirachat.py's header, the eval harness -- goes through here, so an old
    checkpoint takes the same branch it always did."""
    named = cfg.get("arch") or cfg.get("pe")
    # A cfg THAT NAMES NOTHING IS MIRA'S. `arch` is not recorded any more -- one architecture
    # names itself -- so its absence is the answer rather than a gap to fill with the old "ssm"
    # default. Anything that is not MIRA (the rope ablation, every benchmark baseline) still
    # records what it is, and comes back unchanged.
    if config.is_lattice_arch(named):
        return "mira-lattice"
    return config.canonical_arch(named) if named else "mira-lattice"


def run_name(cfg):
    """`<model>_<arch>`, the tag that identifies a run in every filename it writes. The stage
    label is appended per artefact by ckpt_path/hist_path, not here, because one run passes
    through several stages and the tag is set once.

    `arch` FIRST, `pe` AS THE FALLBACK, and for the two architectures that existed under the
    old name the string is the same either way -- so a run in flight keeps writing to
    checkpoint_mira_llm-s_ssm_pretrain.pt and nothing on disk moves. arch="mira" is the only
    value that produces a name no earlier run could have written."""
    return f"{model_name(cfg)}_{arch_of(cfg)}"


# Set once per run, by helpers.common.setup(), from the model configuration in force. Left
# empty, no directory is named after it and the artefacts sit in the stage directory itself --
# see artefact_dir.
_RUN_TAG = ""
# The scheme part of the displayed tag: for a string tag it is the tag itself, for a cfg tag it
# is run_name() with the trailing arch removed. A display-only suffix can follow the context
# without changing checkpoint or history filenames.
_SCHEME_TAG = ""
_DISPLAY_SUFFIX = ""


def set_run_tag(cfg_or_str, display_suffix="", scheme=""):
    """Called once at startup with the model configuration; every artefact path follows.

    `scheme` NAMES WHAT THE BAR SAYS when that is not the run tag. Stage 7's run tag carries
    the stage -- `mira-s-instruct`, because checkpoint_mira-s-instruct_sft.pt is the filename
    the rule gives it -- and a bar that repeats it reads `[instruct-sft @mira-s-instruct]`."""
    global _RUN_TAG, _SCHEME_TAG, _DISPLAY_SUFFIX
    if isinstance(cfg_or_str, str):
        # A STRING TAG IS ALREADY THE NAME AND CARRIES NO ARCH SUFFIX TO STRIP. Family names
        # contain underscores -- mamba_v1, mamba_v2, rwkv_v6, rwkv_v7, rwkv_v8, hgrn_v2,
        # gated_deltanet -- and removing the last one collapses each version pair to a single
        # indistinguishable label.
        _RUN_TAG = cfg_or_str
        _SCHEME_TAG = cfg_or_str
    else:
        _RUN_TAG = run_name(cfg_or_str)
        _SCHEME_TAG = _RUN_TAG.rsplit("_", 1)[0]      # run_name is <model>_<arch>
    if scheme:
        _SCHEME_TAG = str(scheme)
    _DISPLAY_SUFFIX = str(display_suffix).strip()
    return _RUN_TAG


def get_run_tag():
    return _RUN_TAG


# --------------------------------------------------------------------------- #
# THE TOKENIZER TRAVELS WITH THE WEIGHTS
# --------------------------------------------------------------------------- #
# A checkpoint's ids are meaningless without the vocabulary that assigned them. Storing the
# architecture alongside the weights (model_cfg, below) already saved this project from loading
# a differently-shaped model in silence; the tokenizer is the same class of dependency and a
# worse failure when it is wrong -- a model loaded against a different vocabulary does not
# crash, it TALKS NONSENSE, fluently, and nothing in the log says why.
#
# So every checkpoint carries the whole of bpe.json, verbatim. It is ~20 MB against a 2.1 GB
# checkpoint -- under one percent -- and it makes a checkpoint self-describing: copy the single
# .pt file to a machine with no checkpoints/bpe/ at all and it still decodes.
#
# WHICH tokenizer is set once per run, beside the run tag, because a stage may not be using the
# default one: lang_sft trains on a vocabulary EXTENDED for another language, and a checkpoint
# that carried the English bpe.json while its embedding had the Irish rows would be exactly the
# silent mismatch this exists to prevent.
_BPE_SOURCE = ""
# THE LIVE TOKENIZER, when a stage has one. See _bpe_blob.
_BPE_TOKENIZER = None


def bpe_subpath(path=None):
    """WHICH tokenizer, as the short name a person recognises: `bpe`, `bpe-sft-coder`.

    The directory holding bpe.json, relative to checkpoints/. That directory IS the tokenizer's
    identity -- stage 12 writes checkpoints/<name>/bpe.json and every later stage is pointed at
    one by name -- so the name is what a log line has to carry. A tokenizer from outside the
    checkpoint tree (the downloaded fallback under data/download/) is reported relative to the
    project root instead, because there is no short name to give it and a half-path that looked
    like one would be worse than the truth."""
    p = _BPE_SOURCE if path is None else (path or "")
    if not p:
        return "(none)"
    directory = os.path.dirname(os.path.abspath(p))
    for root, label in ((config.CHECKPOINT_DIR, None), (config.ROOT, None)):
        try:
            rel = os.path.relpath(directory, root)
        except ValueError:                                     # different drive on Windows
            continue
        if not rel.startswith(os.pardir):
            return rel if rel != os.curdir else os.path.basename(directory)
    return directory


def set_bpe_source(path, log=None, tokenizer=None):
    """The tokenizer file every checkpoint written from here on will carry. Called by
    helpers.common.setup(), and again by any stage that swaps the tokenizer mid-run.

    ANNOUNCES WHICH ONE, because every trainer reaches this function and a run that does not say
    which vocabulary it loaded cannot be read back. There is more than one now -- checkpoints/bpe
    from stage 3, and a checkpoints/bpe-sft-<name> from stage 12 for every corpus adapted -- and
    they differ in their id layout, so a checkpoint trained against the wrong one is wrong in a
    way that decodes without complaint. The structured lines are printed HERE rather than in
    each trainer so that a stage cannot forget them, including a stage that swaps the tokenizer
    part-way through."""
    global _BPE_SOURCE, _BPE_TOKENIZER
    _BPE_SOURCE = path or ""
    if tokenizer is not None:
        _BPE_TOKENIZER = tokenizer
    emit = log or print
    try:
        emit("Tokenizer: BPE")
        emit(f"Tokenizer checkpoint: {_BPE_SOURCE or '(none)'}")
        if tokenizer is not None:
            emit(f"Tokenizer vocabulary size: {len(tokenizer):,}")
    except Exception:                                          # noqa: BLE001
        pass                                                   # a log line is never fatal
    return _BPE_SOURCE


def get_bpe_source():
    return _BPE_SOURCE


def chat_specials_present():
    """The chat markers the live tokenizer actually has, in registration order.

    Read from the tokenizer rather than from CHAT_SPECIALS, so the list describes THIS
    checkpoint: a stage that registered none gets [], and a stage that registered them gets the
    three it registered. Never raises -- a checkpoint that could not name its markers is still a
    checkpoint."""
    try:
        have = getattr(_BPE_TOKENIZER, "special2id", None) or {}
        return [t for t in CHAT_SPECIALS if t in have]
    except Exception:                                          # noqa: BLE001
        return []


def _bpe_blob():
    """The tokenizer file's bytes, as text, or None. Never raises: a checkpoint that could not
    embed its tokenizer is still a checkpoint, and the fallback below covers it."""
    # THE LIVE ONE FIRST, AND THE FILE ONLY AS A FALLBACK. A stage that registers a special has
    # a tokenizer in memory that no file describes: stage 7 registers <|user|>, <|system|> and
    # <|end|>, so re-reading checkpoints/bpe/bpe.json here would embed a vocabulary three tokens
    # short of the model saved beside it -- and a checkpoint whose tokenizer disagrees with its
    # embedding talks nonsense fluently, which is the failure this whole mechanism exists for.
    if _BPE_TOKENIZER is not None and hasattr(_BPE_TOKENIZER, "dumps"):
        try:
            return _BPE_TOKENIZER.dumps()
        except Exception:                                      # noqa: BLE001
            pass
    p = _BPE_SOURCE
    if not p or not os.path.isfile(p):
        return None
    try:
        return open(p, "r", encoding="utf-8").read()
    except OSError:
        return None


def bpe_from_ckpt(ck, fallback=None, log=print, checkpoint_path=""):
    """The tokenizer a checkpoint was trained with.

    Prefers the copy INSIDE the checkpoint; falls back to `fallback` (checkpoints/bpe/bpe.json)
    when the checkpoint predates this or could not embed one. Returns None when neither is
    available, so a caller can still decide what to do.

    The fallback is a convenience and is announced as one. A checkpoint written before its
    tokenizer travelled with it is not KNOWN to match checkpoints/bpe/bpe.json -- it is merely
    the only candidate there is -- and a line saying so is the difference between a reader who
    checks and a reader who assumes."""
    from tokenizer import BPETokenizer              # here: helpers is imported by the tokenizer
    blob = (ck or {}).get("bpe")
    if blob:
        # STRAIGHT FROM THE STRING. This wrote the blob to a SYSTEM TEMPORARY FILE and read it
        # back, for a value already in memory -- a path outside the project, on a filesystem
        # that may be small, read-only, or cleared under a long run, and one that no rule about
        # where this project keeps its files covers. BPETokenizer.loads parses the JSON it was
        # handed; the round trip through /tmp was never anything but the absence of that method.
        tok = BPETokenizer.loads(blob, where="the tokenizer embedded in this checkpoint")
        tok.loaded_from = checkpoint_path or "(the checkpoint)"
        tok.loaded_how = f"embedded; names {(ck or {}).get('tokenizer') or '(not recorded)'}"
        return tok
    named = str((ck or {}).get("tokenizer") or "").strip()
    if named:
        for candidate in _saved_vocabularies(named):
            if os.path.isfile(candidate):
                log(f"[bpe] no tokenizer in the checkpoint; it names {named!r}: "
                    f"{os.path.relpath(candidate, config.ROOT)}")
                tok = BPETokenizer.load(candidate)
                tok.loaded_from = candidate
                tok.loaded_how = f"named by the checkpoint as {named!r}; not embedded"
                return tok
    if fallback and os.path.isfile(fallback):
        log(f"[bpe] no tokenizer in the checkpoint"
            + (f", and nothing provides {named!r}" if named else "")
            + f"; fallback {os.path.relpath(fallback, config.ROOT)}")
        tok = BPETokenizer.load(fallback)
        tok.loaded_from = fallback
        tok.loaded_how = "fallback; the checkpoint embeds and names none"
        return tok
    return None


def _saved_vocabularies(name):
    """Where the vocabulary called `name` may sit, saved_models/ first, then checkpoints/.

    The same order tokenizer.run.resolve_named uses, spelled here rather than imported: this
    module is what tokenizer/run.py imports, so importing it back would close the cycle. The
    name may already be a path relative to checkpoints/ -- bpe_subpath returns one -- which the
    last candidate takes as it is."""
    leaf = os.path.basename(os.path.normpath(name))
    directory = os.path.join(config.MODEL_DIR, f"tokenizer-{leaf}")
    return [os.path.join(directory, "bpe.json"),
            os.path.join(directory, "bpe", "bpe.json"),
            os.path.join(config.CHECKPOINT_DIR, name, "bpe.json"),
            os.path.join(config.CHECKPOINT_DIR, leaf, "bpe.json")]



def scheme_tag():
    """The model scheme this run is training -- `mira-s`. Empty before setup() has run.

    Set beside the run tag by set_run_tag, so the two cannot drift apart, and NOT derived by
    splitting the run tag here: family names contain underscores and a trailing split removes
    the version."""
    return _SCHEME_TAG


def tag(stage, ctx=None):
    """`[<stage> @<scheme>]`, or `[<stage> @<scheme> <context>]` -- the prefix every line a
    stage prints should carry.

    THE STAGE ALONE IS NOT ENOUGH TO READ A LOG. Nine stages write progress bars that look
    alike, four model sizes share every one of them, and the same stage is run again and again
    at different sizes; a bar that says only `[pretrain]` leaves the reader unable to tell which
    run they are watching, which is exactly when a misconfigured scheme goes unnoticed for
    hours. `[pretrain @mira-s]` answers both questions in the width of a few characters.

    Falls back to `[<stage>]` when no run tag has been set -- a tool that never builds a model
    has no scheme to name, and should not print an empty one.

    STAGE 7 CARRIES NO CONTEXT AND IS NAMED FOR ITSELF:

        "[sft @mira-s-instruct 512]: should drop 512, because instruct-sft has no context
         limit!!!  also [sft @mira-s-instruct 512] --> [instruct-sft @mira-s]"  -- 2026-09-05

    Its width is the longest conversation in the batch that was drawn, so a number on the bar
    describes a window it does not have. No other stage changes."""
    s = scheme_tag()
    name = "instruct-sft" if stage == "sft" else stage
    head = f"{name} @{s}" if s else name
    # THE CONTEXT TOO, WHERE IT MOVES. Pretraining lengthens its window as it goes, and the
    # window decides the batch, the memory and the speed; a bar that showed only the scheme
    # would leave every one of those changing with nothing on screen to say why.
    if stage == "sft":
        ctx = None
    details = ([f"{int(ctx):,}"] if ctx else []) + ([_DISPLAY_SUFFIX] if _DISPLAY_SUFFIX else [])
    return f"[{head}{' ' + ' '.join(details) if details else ''}]"


def stage_dir(ckdir, stage, flat=False):
    if flat:
        return ckdir
    # STAGE 7 OWNS ITS DIRECTORY AND IT IS RESOLVED HERE, NOT AT THE CALL SITES. Five places
    # reach for that checkpoint -- instruct_sft writes it, instruct_dpo, instruct_reward,
    # instruct_rlhf and evals/eval.py read it as their starting policy or frozen reference --
    # and each of them passes ckdir and the stage key to this one function. Resolving it here
    # means none of them can look in the wrong place, and `ckdir` is ignored for this stage
    # because the rule names the whole path: checkpoints/instruct_sft/<scheme>_<dataset>/.
    if stage == "sft":
        return sft_dir()
    # STAGE 8 OWNS ITS DIRECTORY TOO, for the reason stage 7 does: one model trained on one
    # preference set is what the artefact is, and checkpoints/reward/ held whichever pair ran
    # last under one name.
    if stage == "reward":
        return reward_dir()
    # STAGE 11 OWNS ITS DIRECTORY TOO, for the reason stages 7 and 8 do.
    if stage == "dpo":
        return dpo_dir()
    sub = stage_dirname(stage)
    return os.path.join(ckdir, sub) if sub else ckdir


def artefact_tag(stage):
    """`<model>_<pe>_<stage-label>`, the full identifier in every artefact name. Empty when no
    run tag has been set, which is what selects the pre-convention names below."""
    return f"{_RUN_TAG}_{stage_label(stage)}" if _RUN_TAG else ""



def names_its_own_artefact(stage):
    """True when the directory a stage is HANDED already identifies the artefact, so no level
    is added below it.

    STAGE_DIRS ALREADY SAYS THIS AND HAS SINCE STAGE 7 WAS WRITTEN. An empty mapping there means
    "write into the directory you were handed", and the three stages that carry one --
    instruct SFT, the reward model, the language adaptation -- are handed
    checkpoints/instruct_sft/<scheme>_<dataset>/ and its siblings, paths whose leaf IS the
    identity. Reading the same table for this decision keeps one statement of the fact instead
    of two that can disagree."""
    return stage in STAGE_DIRS and not STAGE_DIRS[stage]


def artefact_dir(ckdir, stage, flat=False):
    """The directory holding ONE artefact group: checkpoint.pt, history.json, config.json,
    legender.json, tokenizer.json and dynamics.jsonl, all under bare names.

    THE DIRECTORY IS THE IDENTITY, NOT THE FILENAME.

    SO THE TAG THAT WAS IN THE FILENAME BECOMES A DIRECTORY, and it is still there: a stage
    directory is shared by every run that ever trained under it --
    checkpoints/pretrain/ holds mira-s beside mira-xs, checkpoints/cot/ holds the
    CoT SFT beside the GRPO that reads it -- so bare filenames directly under it would have
    each run silently overwrite the last. checkpoints/pretrain/<tag>/checkpoint.pt keeps
    every one of those distinctions and moves them one level out of the filename.

    TWO CASES ADD NOTHING. A caller that already holds an identifying directory says so with
    `flat` -- the benchmark hands
    checkpoints/benchmark/<dataset>/mira-s_K2_d768_de1536_dr48_dm16/, where a level below it
    would repeat what the path says twice -- and a stage whose STAGE_DIRS entry is empty is
    handed one by the rule (see names_its_own_artefact). A run with NO tag set at all has
    nothing to name a directory with and writes into the stage directory itself, which is
    where an untagged run's artefacts have always sat."""
    base = stage_dir(ckdir, stage, flat)
    if flat or names_its_own_artefact(stage):
        return base
    tag = artefact_tag(stage)
    if not tag:
        return base
    path = os.path.join(base, tag)
    legacy = os.path.join(base, legacy_scheme_tag(tag))
    return legacy if not os.path.isdir(path) and os.path.isdir(legacy) else path


def legacy_scheme_tag(tag):
    """`tag` as it was spelt before 2026-10-06, when the scheme ids were mira_llm-<size>:
    mira-s_mira-lattice_pretrain -> mira_llm-s_mira-lattice_pretrain. A tag that does not start
    with a scheme id comes back unchanged."""
    for name in config.SCHEMES:
        if tag == name or tag.startswith(name + "_"):
            return "mira_llm-" + tag[len("mira-"):]
    return tag


def ckpt_path(ckdir, stage, flat=False):
    return os.path.join(artefact_dir(ckdir, stage, flat), "checkpoint.pt")


def hist_path(ckdir, stage, flat=False):
    return os.path.join(artefact_dir(ckdir, stage, flat), "history.json")


def config_path(ckdir):
    """The architecture sidecar beside the checkpoint. DERIVED, NEVER AUTHORITATIVE: the
    weights and the model_cfg inside checkpoint.pt are what a resume reads, and this is a copy
    of that metadata in a form a person -- or a plotting script, or a model card -- can read
    without opening two and a half gigabytes of tensors. Rewritten from the same dict on every
    save, so it cannot drift from the file it describes."""
    return os.path.join(ckdir, "config.json")


def legender_path(ckdir):
    """The training legend beside the checkpoint. WHAT THIS FILE IS, NOT WHAT THE MODEL IS.

    config.json answers "which architecture"; this answers "which run, how far". They are two
    questions and they were being answered in one file, which is how a step counter came to sit
    beside a lattice order.

    NO MAX STEPS, AND THAT IS A RULE RATHER THAN AN OMISSION:

    A ceiling a run may raise tomorrow is not a property of a file written today, so `total`,
    `sched_total` and the `done` flag derived from them are not here and are not anywhere in the
    sidecars. The schedule a run is on lives in that stage's yaml, where it can be changed, and
    in the .pt, where a resume reads the value it was actually written with.

    WHAT IS HERE IS TRUE OF THIS FILE FOREVER: the stage that wrote it, the step it reached, the
    window it was trained at by then, the tokenizer it embeds and how many parameters it holds.
    None of it can be changed by editing a yaml."""
    return os.path.join(ckdir, "legender.json")


def tokenizer_path(ckdir):
    """The tokenizer sidecar beside the checkpoint. A REAL COPY of the vocabulary the
    checkpoint embeds, written by save and read back by nothing: tokenizer.run.resolve decides
    which tokenizer a stage uses, and a second place for it to look would be a second answer to
    a question that must have one."""
    return os.path.join(ckdir, "tokenizer.json")


def existing_artifact_path(path):
    """The canonical artifact path. There is one spelling and this is it."""
    return path


def _atomic_torch_save(obj, path):
    """torch.save via a temporary file and an atomic rename.

    A DIRECT torch.save to `path` is not atomic: if the write fails part way -- a full disk, an
    exceeded quota, a killed job -- `path` is left truncated and the previous GOOD checkpoint is
    gone with it. That turns a recoverable interruption into a lost run, and it is how

        RuntimeError: basic_ios::clear: iostream error

    The failure is re-raised with the free space attached, because the underlying iostream error
    names neither the file nor the cause."""
    input_only(path, "save a checkpoint")
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)                 # never leave a partial file behind
        except OSError:
            pass
        free = shutil.disk_usage(os.path.dirname(path) or ".").free
        raise RuntimeError(
            f"failed writing {path}: {type(e).__name__}: {e}. "
            f"Free space on that filesystem: {free / 2**30:.2f} GiB. "
            f"A checkpoint here is model + optimizer state (~4x the parameter count in fp32 "
            f"under MasterAdamW), and the previous checkpoint is NOT overwritten until the new "
            f"one is complete, so the write needs room for both at once. "
            f"The last good checkpoint at {path} is intact."
        ) from e


def model_cfg_of(model):
    """The constructor arguments of a MiraLLM (or of the MiraLLM inside a wrapper such as the
    reward model), or None for a foreign model like the distilled HF student."""
    model = unwrap(model)              # the cfg belongs to the model, not to a wrapper
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        cfg = getattr(getattr(model, "base", None), "cfg", None)
    return _lower_cfg_keys(cfg) if isinstance(cfg, dict) else None


def _lower_cfg_keys(value):
    if isinstance(value, dict):
        return {str(key).lower(): _lower_cfg_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_lower_cfg_keys(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_lower_cfg_keys(item) for item in value)
    return value


def _is_mira_model_cfg(cfg):
    """Is this architecture description MIRA's?

    BY THE GEOMETRY, NOT BY A NAME, because a MIRA cfg no longer records one: there is one
    architecture, so `arch` said nothing and was dropped with the other ten fields config.json
    does not write. `lattice_order` is what a MIRA cfg has and no other family's does -- the
    rope ablation and every benchmark baseline are built on an attention scaffold that never
    enters the lattice branch, so none of them carries it. A checkpoint written while `arch`
    was still recorded is recognised by that name as before."""
    named = cfg.get("experimental_arch") or cfg.get("arch")
    if named is not None:
        return config.is_lattice_arch(named)
    return "lattice_order" in cfg


def _mira_cfg_leaks(cfg):
    """MIRA-only fields present in a non-MIRA architecture description."""
    if _is_mira_model_cfg(cfg):
        return []
    return sorted(config.MIRA_ONLY_MODEL_CFG_KEYS.intersection(cfg))


def grown_vocabulary(saved, current):
    """(have, want) when `current` is `saved` with a LARGER vocabulary and nothing else changed.

    Returns None when this is not that case, so the caller refuses exactly as it did before."""
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return None
    have, want = saved.get("vocab_size"), current.get("vocab_size")
    if not isinstance(have, int) or not isinstance(want, int) or not 0 < have < want:
        return None
    if {k: v for k, v in saved.items() if k != "vocab_size"} != \
       {k: v for k, v in current.items() if k != "vocab_size"}:
        return None
    return have, want


def resolve_ffn_hidden(cfg):
    """A null `ffn_hidden` replaced, in place, by the width it stood for.

    IT USED TO MEAN "UNSET, DERIVE THE WIDTH FROM ffn_factor", so a checkpoint written then and a
    model built now describe the same FFN with two different values -- null against the number --
    and every comparison between them fails on a difference that is not one. Both sides go
    through here first.

    TWO PLACES COMPARE A SAVED cfg AGAINST A BUILT ONE and they must agree on this:
    require_checkpoint_model_cfg below, and model/lm.py's `require_model_cfg`, which ablation
    hands its whole built cfg. A resolution in one of them alone is a resume that passes the
    first check and fails the second."""
    if isinstance(cfg, dict) and cfg.get("ffn_hidden") in (None, 0) and cfg.get("n_embd"):
        # MIRA: 4 x n_embd, never ffn_factor x n_embd. A baseline keeps its own ffn_factor.
        factor = 4 if arch_of(cfg) == "mira-lattice" else int(cfg.get("ffn_factor") or 4)
        cfg["ffn_hidden"] = factor * int(cfg["n_embd"])
    return cfg


def require_checkpoint_model_cfg(model, checkpoint, stage):
    raw = checkpoint.get("model_cfg")
    saved = _lower_cfg_keys(raw) if isinstance(raw, dict) else None
    current = model_cfg_of(model)
    if saved is None or current is None:
        raise RuntimeError(f"cannot resume {stage}: checkpoint architecture metadata is missing")
    if "init_horizon" in saved and "lattice_init_horizon" not in saved:
        saved["lattice_init_horizon"] = saved.pop("init_horizon")
    saved.pop("lattice_norm_module", None)      # lattice_norm_module (retired 2026-10-05; MIRA is RMSNorm only)
    if _is_mira_model_cfg(saved):
        saved_method = config.canonical_lattice_initializer(
            saved.get("lattice_initializer", saved.pop("lattice_init_method", "schur_init")), recorded=True)
        saved["lattice_initializer"] = saved_method
        current["lattice_initializer"] = saved_method
        owner = unwrap(model)
        cfg = getattr(owner, "cfg", None)
        if cfg is None:
            cfg = getattr(getattr(owner, "base", None), "cfg", None)
        if isinstance(cfg, dict):
            cfg["lattice_initializer"] = saved_method
        if "lattice_init_horizon" in saved:
            current["lattice_init_horizon"] = saved["lattice_init_horizon"]
            if isinstance(cfg, dict):
                cfg["lattice_init_horizon"] = saved["lattice_init_horizon"]
        else:
            saved["lattice_init_horizon"] = current.get("lattice_init_horizon")
        # MIRA records its modulus and argument maps separately.
        config.mira_map_arch(saved)
        config.mira_map_arch(current)
        if "arch" in saved or "lattice_arch" in saved:
            raise ValueError("this MIRA checkpoint records arch instead of lattice_modulus_map and "
                             "lattice_argument_map, the format before those keys, and cannot be loaded")
    # BEFORE ffn_factor IS DROPPED BELOW, because resolving a null width needs it.
    resolve_ffn_hidden(saved)
    resolve_ffn_hidden(current)
    # THE MIRA-ONLY RETIREMENTS ARE DECIDED BEFORE `arch` IS POPPED, because popping it is one
    # of them: a checkpoint written while MIRA still recorded an architecture name is MIRA by
    # that name, and one written since is MIRA by its lattice geometry. Asking afterwards would
    # ask of a dict the answer had just been removed from.
    mira = (_is_mira_model_cfg(saved), _is_mira_model_cfg(current))
    for cfg, is_mira in zip((saved, current), mira):
        for key in config.RETIRED_MODEL_CFG_KEYS:
            cfg.pop(key, None)
        if is_mira:
            for key in config.RETIRED_MIRA_MODEL_CFG_KEYS:
                cfg.pop(key, None)
    grown = grown_vocabulary(saved, current)
    if grown is not None:
        return grown
    # Old baselines inherited every MIRA default from the temporary MiraLLM scaffold. Those
    # values configured no baseline weights, so discard them symmetrically for compatibility.
    # New checkpoints cannot acquire them: save_ckpt rejects such schema leakage below.
    for cfg in (saved, current):
        if not _is_mira_model_cfg(cfg):
            for key in config.MIRA_ONLY_MODEL_CFG_KEYS:
                cfg.pop(key, None)
    # THE COMPARISON IS EXACT AND CARRIES NO COMPATIBILITY. A loader that quietly accepts a
    # mismatched architecture is a loader that cannot say which model a set of weights belongs
    # to.
    if saved != current:
        keys = sorted(set(saved) | set(current))
        key = next(key for key in keys if saved.get(key) != current.get(key))
        raise RuntimeError(
            f"cannot resume {stage} with a different architecture "
            f"({key}: checkpoint={saved.get(key)!r}, requested={current.get(key)!r})")


def unwrap(model):
    """The real model inside any parallel wrapper. Returns `model` when there is no wrapper.

    A CHECKPOINT MUST NOT KNOW HOW THE RUN WAS PARALLELISED. nn.DataParallel, our
    model.distributed.distributed_ddp.DataParallelModel and DistributedDataParallel all keep the real module at
    `.module`; following it means one file serves a 1-GPU run and a 4-GPU run alike. Loops
    rather than unwrapping once, so a doubly wrapped model is still reduced to the model."""
    seen = 0
    while hasattr(model, "module") and seen < 8:
        model, seen = model.module, seen + 1
    return model


def mixer_geometry(model):
    """The sequence mixer's own geometry, READ OFF THE BUILT MODEL, or () for an ordinary one.

    IT RETURNS () RATHER THAN RAISING for anything that is not a lattice, so attention, S6 and
    the RWKVs print exactly what they printed before and a foreign model still gets a summary."""
    blocks = getattr(unwrap(model), "blocks", None)
    if not blocks:
        return ()
    try:
        mixer = getattr(blocks[0], "mixer", None)
    except Exception:                                              # noqa: BLE001
        return ()
    if mixer is None or not hasattr(mixer, "lattice_order") or not hasattr(mixer, "inner"):
        return ()
    rows = [("lattice_order", f"{mixer.lattice_order}  (N stages, N+1 ladder taps, N states per channel)"),
            ("d_e", f"{getattr(mixer, 'full_inner', mixer.inner):,}"),
            ("coefficients",
             "alpha [d_e, N] and rho [d_e, N+1], one per channel per stage")]
    if hasattr(mixer, "coefficient_bottleneck"):
        rows.append(("coefficient_bottleneck", f"{mixer.coefficient_bottleneck}  (alpha and rho; the gate is full rank)"))
    if hasattr(mixer, "memory_dimension"):
        rows.append(("memory_dimension", f"{mixer.memory_dimension}"))
    if hasattr(mixer, "state_values"):
        rows.append(("recurrent state",
                     f"{mixer.state_values():,} FP32 values per sequence per layer"))
    return tuple(rows)


def mixer_state_total(model):
    """The recurrent state carried across EVERY layer, in values, or 0 when a mixer does not
    report one.

    It is summed rather than multiplied by the depth because nothing here requires every block
    to carry the same geometry, and a summary that assumes it would be wrong for a model that
    does not."""
    blocks = getattr(unwrap(model), "blocks", None)
    if not blocks:
        return 0
    total = 0
    for block in blocks:
        values = getattr(getattr(block, "mixer", None), "state_values", None)
        if not callable(values):
            return 0
        try:
            total += int(values())
        except Exception:                                          # noqa: BLE001
            return 0
    return total


def model_summary(model, log=print, prefix="[model]"):
    """WHAT IS ABOUT TO BE TRAINED, said in full and in one place, before the first step.

    ONE FUNCTION RATHER THAN TEN PRINTS. Every stage of the pipeline builds its model through
    helpers.common.build_model, which calls this; the two trainers that construct a MiraLLM
    outside it -- scaling_laws/train.run_point, which trains a grid of sizes with its own loop,
    and benchmark/pretrain_benchmark.py, which builds baselines through baselines.build_lm -- call
    it themselves with their own prefix. Nothing is duplicated: the format lives here.

    THE TRAINABLE COUNT IS MEASURED, NEVER CONFIGURED. It is
    `sum(p.numel() for p in model.parameters() if p.requires_grad)` over the model as it stands,
    so a stage that has frozen part of what it holds reports what it will actually update rather
    than what the configuration implies. The total is printed beside it, and the frozen remainder
    named explicitly, so the two disagreeing is a visible fact and not a silent one.

    THE VOCABULARY IS IN THAT COUNT ONCE, NOT TWICE. The output head is WEIGHT-TIED to the token
    embedding (model/mira_llm.py), so one vocab_size x d_model matrix serves as both and is
    counted a single time -- which is why the total is smaller than a per-layer sum plus two
    embedding matrices would suggest.

    `log` IS THE CALLER'S ALREADY-GATED LOGGER. Under DDP that is helpers.common._say, so this
    is rank 0 alone; it is routine output, not an error, and four interleaved copies of it are
    not four times the information. The shape is the one model/lm.py prints its context window
    in -- an `===` opener, one field per line under a 15-column label, an `===` closer -- so a
    log reads as one document rather than as several.

    NOTHING HERE CAN RAISE. Every field is read with .get() and rendered through `_fld`, which
    answers "(unknown)" for a missing one, so a model whose cfg is partial, or a foreign model
    with no cfg at all, still gets a summary rather than killing the run that was about to
    start."""
    m = unwrap(model)
    cfg = model_cfg_of(m) or {}
    params = list(m.parameters()) if hasattr(m, "parameters") else []
    from model.distributed.distributed_fsdp_channel import full_numel
    total = sum(full_numel(p) for p in params)
    trainable = sum(full_numel(p) for p in params if getattr(p, "requires_grad", False))
    frozen = total - trainable

    def _fld(v, fmt="{:,}"):
        return "(unknown)" if v in (None, "") else fmt.format(v)

    n_layer, n_head = cfg.get("n_layer"), cfg.get("n_head")
    d_model, vocab = cfg.get("n_embd"), cfg.get("vocab_size")
    # THE TRAINED WINDOW COMES OFF THE MODEL, not off the architecture. block_size is a training
    # choice -- it moves between runs and between stages -- so a MIRA cfg does not record it; the
    # model it built still knows what it was constructed at, and that is what the reader wants.
    block = cfg.get("block_size") or getattr(m, "block_size", None)
    # A LATTICE HAS NO HEADS, and saying so is more use than "(unknown)". The cfg omits n_head
    # because there is nothing for it to configure, not because the value was lost.
    heads = "(none: the lattice mixer is elementwise per channel)" \
        if n_head is None and _is_mira_model_cfg(cfg) else _fld(n_head)
    if n_head and d_model:
        heads += f"  (head dim {d_model // n_head})"
    geometry = mixer_geometry(m)
    state = mixer_state_total(m)
    if geometry:
        # A HEAD COUNT THAT REACHES NOTHING IS WORSE THAN NO HEAD COUNT. baselines.build_lm
        # builds every family on a disposable RoPE scaffold and then REPLACES its attention, so
        # a headless mixer still carries the split that scaffold was constructed with. Printing
        # it unqualified said "heads 8 (head dim 64)" of a model that has neither.
        if n_head is not None:
            heads += ("  (nominal: this mixer has no heads; the split is the discarded "
                      "scaffold's)")
    name = model_name(cfg) if cfg.get("n_embd") else type(m).__name__
    rows = [
        # THE ARCHITECTURE IS THE ONE THAT WAS BUILT, not the scaffold's. cfg["arch"] stays
        # "rope" for every late-bound family because that is what MiraLLM constructed;
        # cfg["experimental_arch"] is what replaced it, and it is what the reader wants.
        ("architecture", cfg.get("experimental_arch") or (arch_of(cfg) if cfg else "(unknown)")),
        ("layers", _fld(n_layer)),
        ("heads", heads),
        ("d_model", _fld(d_model)),
        *geometry,
        ("context", f"{block:,} tokens" if block else "(unknown)"),
        ("vocabulary", f"{vocab:,}  (tied to the output head: one matrix, counted once)"
                       if vocab else "(unknown)"),
        ("parameters", count(total)),
        ("TRAINABLE", f"{count(trainable)}" +
                      ("  -- the whole model" if frozen <= 0 else
                       f"  -- {count(frozen)} frozen")),
        ("state/parameter",
         f"{state / total:.6f}" if state and total else
         "(this mixer does not report a recurrent state size)"),
    ]
    log(f"{prefix} {'=' * 18} model summary: {name} {'=' * 18}")
    for label, value in rows:
        log(f"{prefix}   {label.ljust(13)}  {value}")
    log(f"{prefix} {'=' * (53 + len(name))}")


def cpu_state(obj):
    """The same structure with every tensor on the CPU. For the OPTIMISER state.

    AND THAT WAS EXACTLY TRUE. fp32_state below moves the weights to the CPU and explains at
    length why a file must not remember the machine that wrote it -- and then save_ckpt wrote
    `"opt": opt.state_dict()` RAW. MasterAdamW's masters and AdamW's exp_avg/exp_avg_sq are
    live CUDA tensors, so every checkpoint said `cpu` for the weights and `cuda:0` for the
    optimiser. Half a hardware-independent file is not one.

    It has not bitten this pipeline because load_ckpt passes map_location="cpu". That is the
    same defence fp32_state's docstring calls insufficient, for the same reason: any other
    reader -- a notebook, a conversion script, a colleague -- that forgets the argument gets a
    checkpoint that only opens on a CUDA box.

    RECURSIVE, because an optimiser state_dict is nested: {"inner": {"state": {i: {"step",
    "exp_avg", "exp_avg_sq"}}, "param_groups": [...]}, "masters": [...]}, and in recent torch
    `step` is itself a tensor. DTYPES ARE NOT TOUCHED -- the moments are already fp32 and
    casting them would change the optimiser's arithmetic, which is not what this is for."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: cpu_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(cpu_state(v) for v in obj)
    return obj


def fp32_state(model):
    """A state dict with every floating tensor in fp32, whatever the live weights are.

    A CHECKPOINT IS FP32 EVEN WHEN THE RUN IS NOT. bf16 is a way to fit a step on a card; it is
    not a property of the model, and writing it into the file would make the saved model
    permanently lossier than the one that produced it -- every later stage loading a vocabulary
    projection that has already lost eight bits of mantissa, for a memory saving that only
    mattered while the step was in flight. MasterAdamW keeps fp32 masters for the same reason,
    and this is the same argument applied to the artefact.

    Integer and boolean buffers are left alone: casting them would change what they mean.

    ON CPU, AND UNWRAPPED, SO THE FILE IS A PILE OF STATES AND NOTHING ELSE.

    Two things would otherwise leak the machine that wrote it into the file:

      * THE WRAPPER. nn.DataParallel's state_dict prefixes every key with "module.", so a
        checkpoint written on four GPUs would not load on one and vice versa. `unwrap` follows
        `.module` to the real model, so the keys are the model's own however it was wrapped.
      * THE DEVICE. torch.save records which device each tensor was on. Loading is done with
        map_location="cpu" so that already works, but a file whose tensors say "cuda:3" is a
        file that remembers a machine it should know nothing about -- and any other reader,
        one that forgets map_location, gets a CUDA-only checkpoint. Saved on CPU, it loads
        anywhere with no argument at all."""
    return {k: (v.detach().float().cpu() if torch.is_tensor(v) and v.is_floating_point()
                else v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in unwrap(model).state_dict().items()}


def _atomic_text_save(text, path):
    """Write a text file through a temporary and an atomic rename, as the checkpoint is."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def architecture_cfg(cfg):
    """`cfg` reduced to the keys config.json holds, under the names it holds them.

    THE KEYS ARE LISTED, NOT FILTERED FOR. config.ARCHITECTURE_CFG_KEYS names what the file
    carries and where each value is read from; anything not named there does not reach it. A
    run's choices, the benchmark's bookkeeping, the kernel build's paths, a second spelling of
    something already present and a ratio derivable from two keys beside it are all absent by
    construction rather than by an exclusion that has to be kept complete.

    THE BENCHMARK PLAN IS A SECOND SOURCE, NOT A SECOND SET OF KEYS. `benchmark_architecture`
    records the geometry under its own spellings, so it is searched after the cfg for the same
    named value and then plays no further part; the block itself never reaches the file.

    THE .pt IS NOT TOUCHED. A resume compares the saved cfg against the built one key for key and
    reads the checkpoint, not this, so the architecture record it checks is as complete as it was.
    """
    if not isinstance(cfg, dict):
        return {}
    # A cfg THAT NAMES NO ARCHITECTURE IS MIRA'S, the same rule arch_of follows: there is one
    # architecture, so a MIRA cfg stopped recording the name, and reading the key alone left
    # every MIRA checkpoint with no row and an EMPTY config.json. A baseline still records what
    # it is and still picks its own row.
    arch = arch_of(cfg) if not cfg.get("experimental_arch") else \
        config.canonical_arch(cfg["experimental_arch"])
    listed = config.ARCHITECTURE_CFG_KEYS.get(arch)
    if listed is None:
        return {}
    block = cfg.get("benchmark_architecture")
    block = block if isinstance(block, dict) else {}
    # EVERY LISTED KEY IS WRITTEN, ALWAYS, SO TWO CHECKPOINTS OF ONE ARCHITECTURE HAVE THE SAME
    # FILE. A key was emitted only when the recorded cfg had a source for it, so a checkpoint
    # written before a field existed produced a SHORTER file -- mira-s_K4 has no
    # lattice_initializer or lattice_init_horizon and its sidecar was nine keys against another
    # MIRA's eleven. Two files with different keys cannot be compared field by field, which is
    # what a record of the architecture is for. `null` is the honest value: it says this
    # checkpoint recorded nothing for that key, which is the constructor's own default.
    reduced = {}
    for name, sources in listed:
        reduced[name] = None
        for source in sources:
            if source in cfg and cfg[source] is not None:
                reduced[name] = cfg[source]
                break
            if source in block and block[source] is not None:
                reduced[name] = block[source]
                break
    # A MIRA cfg THAT NAMES NO arch, OR "mira-lattice", IS CAYLEY.
    if arch == "mira-lattice":
        config.mira_map_arch(reduced)
    return reduced


def _save_sidecars(adir, model, model_cfg, blob, stage, step, ctx_window):
    """config.json, legender.json and tokenizer.json beside the checkpoint.

    THREE FILES, THREE QUESTIONS, WRITTEN IN ONE CALL SO THEY CANNOT DIVERGE. config.json is
    WHAT THIS MODEL IS and holds nothing else -- config.ARCHITECTURE_CFG_KEYS names its keys.
    legender.json is HOW FAR THIS CHECKPOINT GOT: the step it reached and the window it had by
    then, and nothing else. A parameter count is DERIVED from the architecture, so it belongs to
    neither file -- config.json states the geometry and the count follows from it.
    tokenizer.json is the vocabulary; its own `specials` and `vocab_size` describe the
    tokenizer, so neither is repeated next door.

    WRITTEN FROM THE SAME VALUES THE CHECKPOINT WAS WRITTEN FROM, in the same call, every save.
    Neither file is ever read back by a resume: the .pt is authoritative and these are the copy
    a person opens. Deriving them from anything else -- default_config.py, a second read of the
    tokenizer -- is how a sidecar comes to describe a checkpoint that is not there.

    THE SCHEDULE'S CEILING IS IN NEITHER. `total` and `sched_total` and the `done` flag derived
    from them describe a limit a run may raise tomorrow, so they are not a property of a file
    written today; see legender_path. A resume reads them from the .pt, which is where they are
    written and where they belong.

    AFTER THE CHECKPOINT, NOT BEFORE. An interrupted save then leaves the previous complete
    group on disk plus at most one sidecar describing a step the .pt has not reached, which the
    next save overwrites; the reverse order would leave sidecars for weights that were never
    written."""
    # NOT SORTED. The keys are written in the order config.ARCHITECTURE_CFG_KEYS lists them,
    # which is the order the architecture is read in -- vocabulary, depth, width, then the
    # lattice's own geometry and its initialiser.
    _atomic_text_save(
        json.dumps(architecture_cfg(model_cfg), indent=2) + "\n",
        config_path(adir))
    _atomic_text_save(json.dumps({
        "step": int(step),
        "ctx_window": int(ctx_window or 0),
    }, indent=2, sort_keys=True) + "\n", legender_path(adir))
    if blob:
        _atomic_text_save(blob, tokenizer_path(adir))


def save_ckpt(ckdir, stage, model, opt, step, total, gens, evald=None, extra=None,
              ctx_window=0, sched_total=0, flat=False):
    """Write the checkpoint. CALLED ON EVERY RANK, WRITTEN BY RANK 0, BARRIERED BOTH SIDES.

    THE GATE IS HERE AND NOT AT THE CALL SITES so that every stage gets it and none of them has
    to remember -- and so that "every rank calls it" stays a rule with no exceptions, which is
    what makes the barriers safe. A caller that wrapped this in `if is_main()` would leave the
    other ranks past the barrier rank 0 is inside.

    WHY RANK 0 ALONE. Four ranks writing one path race, and _atomic_torch_save's rename makes
    that quiet rather than safe: last writer wins and the others' work vanishes without an
    error. The weights are identical on every rank -- DDP all-reduces the gradients -- so there
    is nothing to choose between the copies, only a file to stop them fighting over.

    WHY BOTH BARRIERS. The trailing one is the half that matters, and it matters on RESUME: a
    fast rank that reaches the next load while rank 0 is mid-write reads the PREVIOUS
    checkpoint, because the rename has not happened yet -- a silent step backwards rather than
    an exception. The leading one keeps the file from describing a step some rank has already
    left. Both are no-ops off DDP, where is_main() is true and the path is unchanged."""
    model_cfg = model_cfg_of(model)
    if model_cfg is None:
        raise RuntimeError(f"cannot save {stage}: model architecture metadata is missing")
    leaked = _mira_cfg_leaks(model_cfg)
    if leaked:
        raise RuntimeError(
            f"cannot save {stage}: non-MIRA checkpoint contains MIRA-only architecture "
            f"metadata: {', '.join(leaked)}")
    from model import parallel                  # local: helpers is imported before model is
    channel = parallel.channel_world() > 1
    if channel:
        from model.distributed import distributed_fsdp_channel
        model_state = distributed_fsdp_channel.channel_fp32_state(unwrap(model))
        opt_state = (distributed_fsdp_channel.channel_optimizer_state(opt)
                     if opt is not None else None)
    if parallel.dist_active():
        gen_states = parallel.broadcast_ckpt(
            [g.get_state() for g in gens] if parallel.is_main() else None)
        for generator, state in zip(gens, gen_states):
            generator.set_state(state)
        rng_state = parallel.broadcast_ckpt(
            torch.get_rng_state() if parallel.is_main() else None)
        torch.set_rng_state(rng_state)
        if torch.cuda.is_available():
            cuda_rng_state = parallel.broadcast_ckpt(
                torch.cuda.get_rng_state() if parallel.is_main() else None)
            torch.cuda.set_rng_state(cuda_rng_state)
    parallel.barrier()
    failure = None
    # ONE READ OF THE TOKENIZER FOR BOTH COPIES. The blob goes into the checkpoint and, in the
    # flat layout, into tokenizer.json beside it; reading it twice would let the two disagree if
    # a stage registered a special between the calls.
    blob = _bpe_blob()
    if parallel.is_main():
        try:
            os.makedirs(artefact_dir(ckdir, stage, flat), exist_ok=True)
            _atomic_torch_save({"model": model_state if channel else fp32_state(model),
                "special_tokens": chat_specials_present(),
                # THE NAME OF THE VOCABULARY, so a reader that cannot use the embedded copy
                # knows which one to look for instead of guessing. bpe_subpath is the short name
                # the reader recognises -- `bpe`, `bpe-sft-coder`, `bpe-instruct`.
                "tokenizer": bpe_subpath(),
                "model_cfg": model_cfg,
                # AND SO DOES THE TOKENIZER, for the same reason and against a worse failure:
                # a wrong architecture refuses to load, a wrong vocabulary loads and talks
                # nonsense. See set_bpe_source / bpe_from_ckpt above.
                "bpe": blob,
                # ON THE CPU, LIKE THE WEIGHTS. The optimiser holds one fp32 master and
                # two moments PER PARAMETER -- a count fixed by the architecture, not by the
                # hardware -- so the only thing device-shaped about it was where the tensors
                # happened to live. cpu_state removes that. See cpu_state for what this was.
                "opt": (opt_state if channel else cpu_state(opt.state_dict()))
                if opt is not None else None,
                "step": step, "total": total, "done": step >= total,
                # THE CONTEXT WINDOW THIS STEP WAS TRAINED AT. The schedule divides the budget
                # evenly among its windows, so its boundaries move when the budget does, and a
                # resumed run could otherwise land in a SHORTER window than the one it had
                # reached -- un-learning a context that cost tens of thousands of steps to
                # acquire. Recorded here, it becomes a floor the schedule may never go below
                # (model.lm.context_floor / ratchet). 0 for the stages that have no
                # schedule, which is what those stages pass and what old checkpoints lack.
                "ctx_window": int(ctx_window or 0),
                "sched_total": int(sched_total or 0),
                "gens": [g.get_state() for g in gens], "eval": evald,
                # The explicit generators cover data order and corruption, but DROPOUT draws
                # from the global RNG, so a resumed run saw a different dropout stream from an
                # uninterrupted one and the two diverged. Saving the global state makes resume
                # exact. CPU state is always present; the CUDA states only when CUDA is in use.
                "rng": torch.get_rng_state(),
                "rng_cuda": (torch.cuda.get_rng_state()
                             if torch.cuda.is_available() else None),
                "extra": cpu_state(extra)},   # method-specific side state, e.g. a mask head
               ckpt_path(ckdir, stage, flat))
            _save_sidecars(artefact_dir(ckdir, stage, flat), model, model_cfg, blob,
                           stage, step, ctx_window)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    failure = parallel.broadcast_ckpt(failure)
    if failure is not None:
        raise RuntimeError(f"failed writing distributed checkpoint: {failure}")
    parallel.barrier()             # the file is complete before any rank may read it


def restore_rng(ck):
    """Restore the global RNG streams saved by save_ckpt. Silently does nothing for
    checkpoints written before `rng` was recorded, so old runs still resume."""
    if not ck:
        return
    if ck.get("rng") is not None:
        torch.set_rng_state(ck["rng"].cpu() if hasattr(ck["rng"], "cpu") else ck["rng"])
    cu = ck.get("rng_cuda")
    if cu is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state(cu.cpu() if hasattr(cu, "cpu") else cu)
        except Exception as e:                                     # noqa: BLE001
            print(f"[rng] cuda stream not restored ({type(e).__name__}: {e}); "
                  f"the CPU stream and the data generators are unaffected", flush=True)


def load_ckpt(ckdir, stage, flat=False):
    """The stage's checkpoint, or None. RANK 0 READS THE DISK AND EVERY RANK RECEIVES IT.

    THE MIRROR OF save_ckpt, WHICH RANK 0 ALSO OWNS. Four processes reading one file is four
    times the traffic on the shared filesystem a cluster checkpoint lives on, at the one moment
    every rank wants the same tens of gigabytes; and it is four independent answers to a
    question that must have one, since a file being written by a job elsewhere -- or a rename
    that lands between two ranks' opens -- gives them different states of the same path. One
    reader and a broadcast makes that impossible rather than unlikely.

    EVERYTHING GOES ACROSS, NOT ONLY THE WEIGHTS. `step`, the optimiser state, the data
    generators, the global RNG streams, `ctx_window`, `sched_total`, `model_cfg`, the embedded
    tokenizer and `extra` all come back from here, and every one of them is consumed on every
    rank: the step decides where the loop starts and which segment of the context schedule it is
    in, `gens` decides which tokens are drawn, `ctx_window` is the floor the ratchet may not go
    below, `sched_total` is the T of the cosine the run is on (see schedule_total), and
    the optimiser state is what makes the first step after a resume the step it would have been.
    A rank short of any of them disagrees with the others about what it is training and nothing
    says so. See model/parallel.broadcast_ckpt.

    GUARDED BY dist_active(), SO A SINGLE-PROCESS CALLER NEVER MEETS A COLLECTIVE. mirachat.py, the
    evals, tools and load_stage_model before any model has been built are all outside DDP's
    scope, and off DDP this is the one line it has always been -- the same read, the same
    map_location, the same None when there is no file."""
    from model import parallel                  # local: helpers is imported before model is
    canonical = ckpt_path(ckdir, stage, flat)
    p = existing_artifact_path(canonical)
    if p != canonical and (not parallel.dist_active() or parallel.is_main()):
        print(f"[compat] loading former-name checkpoint {p}", flush=True)
    return read_ckpt(p)


def read_ckpt(path):
    """One checkpoint FILE, or None when it is not there. Rank 0 reads, every rank receives.

    SPLIT OUT OF load_ckpt SO A PATH CAN BE LOADED WITHOUT BEING DERIVED. load_ckpt answers
    "this stage's checkpoint", which is the right question almost everywhere; stage 11 asks a
    different one -- "the file the person named with --checkpoint", or the newest under
    checkpoints/instruct_sft/ whatever its <scheme>_<dataset> directory is called -- and it must
    not reach for torch.load itself, or a distributed run would have every rank open the same
    tens of gigabytes and, worse, possibly disagree about what it found."""
    from model import parallel
    if not parallel.dist_active():
        return torch.load(path, map_location="cpu") if os.path.isfile(path) else None
    # ONLY RANK 0 LOOKS AT THE PATH AT ALL, existence included: the others learn whether there
    # was a checkpoint from the broadcast, so "no checkpoint" is one decision rather than N.
    ck, failure = None, None
    if parallel.is_main() and os.path.isfile(path):
        try:
            ck = torch.load(path, map_location="cpu")
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    failure = parallel.broadcast_ckpt(failure)
    if failure is not None:
        raise RuntimeError(f"failed loading distributed checkpoint {path}: {failure}")
    return parallel.broadcast_ckpt(ck)


def save_hist(ckdir, stage, records, flat=False):
    """Atomic for the same reason as the checkpoint: history.json is what every figure is drawn
    from, and a truncated one is invalid JSON, so a failed write here loses the whole training
    curve rather than one record.

    AND RANK 0 ALONE, for the same reason again: the records are identical on every rank -- the
    loss is all-reduced before it is appended -- so the other ranks have nothing to add and
    would only be a second writer on one temporary path. No barrier: nothing reads this file
    during a run, only at the start of the next one."""
    from model import parallel                  # local: helpers is imported before model is
    failure = None
    if parallel.is_main():
        p = hist_path(ckdir, stage, flat)
        tmp = p + ".tmp"
        try:
            os.makedirs(artefact_dir(ckdir, stage, flat), exist_ok=True)
            input_only(p, "save a history")
            with open(tmp, "w") as f:
                json.dump(records, f)
            os.replace(tmp, p)
        except Exception as exc:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            failure = f"{type(exc).__name__}: {exc}"
    failure = parallel.broadcast_ckpt(failure)
    if failure is not None:
        raise RuntimeError(f"failed writing distributed history: {failure}")


def load_hist(ckdir, stage, upto=None, flat=False):
    """The stage's history. With `upto` set, records at or after that step are DROPPED.

    Resume replays from the checkpoint's step, so any record at or beyond it is about to be
    produced again. Appending to them would leave the file with duplicated and then
    non-monotone steps, and the figure -- which keeps only the last monotone run -- would
    silently restart at the resume point instead of showing the whole training curve. The
    truncation makes that impossible by construction: after it, appending can only extend a
    strictly increasing sequence.

    The history can legitimately extend past the checkpoint, because a checkpoint is written
    every --checkpoint_every_steps while the history is appended every step, and the two are
    saved in that order."""
    canonical = hist_path(ckdir, stage, flat)
    p = existing_artifact_path(canonical)
    if p != canonical:
        print(f"[compat] loading former-name history {p}", flush=True)
    if not os.path.isfile(p):
        return []
    h = json.load(open(p, "r"))
    if upto is None:
        return h
    keep = [r for r in h if r.get("step", 0) < upto]
    if len(keep) != len(h):
        print(f"[resume] history: dropped {len(h) - len(keep)} record(s) at or after step "
              f"{upto}, which this run is about to recompute", flush=True)
    return keep


@torch.no_grad()
def holdout_probe(policy, ref, enc, pairs, beta, batch=16):
    """Held-out preference accuracy and mean margin, teacher forced, against `ref`.

    Cheap by construction -- `pairs` is expected to be a small held-out slice, and the caller
    decides the cadence. Returns a dict so the keys land in the history and in the figures under
    the same names."""
    dpo_margin, _ = _dpo_objective()
    Ms = []
    for i in progress(range(0, len(pairs), batch), desc=tag("probe") + " held-out",
                      total=(len(pairs) + batch - 1) // batch):
        chunk = pairs[i:i + batch]
        W = enc.encode(chunk, "chosen")
        L = enc.encode(chunk, "rejected")
        Ms.append(dpo_margin(policy, ref, W, L, beta))
    M = torch.cat(Ms)
    return {"eval_acc": (M > 0).float().mean().item(),
            "eval_margin": M.mean().item()}


# --------------------------------------------------------------------------- #
# PRE-TOKENISED CORPORA, cached on disk.
#
# Tokenising the corpus once per training STEP is pure waste: the same documents are encoded
# thousands of times by the same tokenizer. A corpus is tokenised ONCE into shards named for
# THE DATASET, in a directory named for THE TOKENIZER THAT PRODUCED IT:
#
#     data/download/<corpus>/**            (any depth, any of CORPUS_EXTENSIONS)
#         ->  cache/tokens/bpe_<256+merges>_<fp>/<corpus>/<corpus>_<sig>_00000.tokens
#                                                        <corpus>_<sig>_index.json
#
# THE DATASET NAMES THE DIRECTORY, NOT ITS PATH. An earlier version mirrored the corpus's path
# under the cache root -- cache/tokens/<tok>/data/download/<corpus>/ -- so two machines that had
# put the same corpus in different places looked in different directories and the second
# retokenised what the first had finished. The dataset name travels with the data; the path does
# not. Streams written under the old rule used to be found anyway, by a lookup that scanned the
# mirrored directory; that lookup is gone, so a stream under the old layout is not found and the
# corpus is tokenised again.
#
# THE TOKENIZER IS THE DIRECTORY, not a field inside the file, so two vocabularies coexist
# instead of overwriting each other: retrain the BPE, try the new vocabulary, go back, and the
# old tokens are still there. The fingerprint is the first 8 characters of GIT'S BLOB HASH of
# the tokenizer's MERGES, so the directory names the encoding rather than merely its size: a
# vocabulary rebuilt to the same size lands somewhere else, which is the whole point. It names
# the merges rather than the whole bpe.json so that REGISTERING A SPECIAL LEAVES IT ALONE -- a
# chat token added after pretraining must not throw away a corpus that has not changed.
# THE STAGE IS NOT IN THE PATH: tokens are a property of (file,
# tokenizer, packing) and of nothing else, so a file tokenised for pretraining is reused by any
# other stage that reads it rather than tokenised again from scratch.
#
# `<sig>` is a short hash of the packing (max_words, the parquet text column). It is in the
# NAME rather than only in the header so that two settings do not evict each other's entries
# turn by turn; the same signature is also stored inside and checked on read.
#
# Format: `<corpus>_<sig>_index.json` is the manifest -- signature, dtype, eos, the shard list
# and the resume cursor -- and each `_NNNNN.tokens` shard holds the ids of the packed documents
# it covers, concatenated, memory-mapped on first touch. The manifest names its shards by BARE
# FILENAME and resolves them against its own directory, which is what lets a cache directory be
# moved or renamed without breaking a single stream.
# --------------------------------------------------------------------------- #
_FINGERPRINTS = {}


def blob_hash(data):
    """GIT'S object hash of `data`: sha1(b"blob <size>\\0" + data).

    Git's, deliberately, rather than a bare digest of the bytes. It costs one extra header to
    compute and buys a fingerprint anyone can CHECK against the tool already on their machine:

        git hash-object checkpoints/bpe/bpe.json

    prints the full 40-character digest whose first characters name the cache directory. A
    fingerprint you cannot independently reproduce is one you end up trusting; this one can be
    verified in a second, which matters when the question is "did this cache come from the
    tokenizer I am holding"."""
    import hashlib
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def encoding_blob(tok):
    """The canonical bytes that DEFINE this tokenizer: its SORTED base vocabulary, and nothing
    else.

    SORTED, SO SERIALISATION ORDER CANNOT REACH THE SIGNATURE. Two files that hold the same set
    of tokens are the same tokenizer however they were written out, and the signature has to say
    so. The bytes are the sort key -- the exact, lossless byte sequence of each token, not its
    readable form, which is lossy for any token that is not valid UTF-8.

    THE BASE VOCABULARY ONLY. The specials carry no bytes and are excluded, because they do not
    participate in corpus tokenisation at all: the packer calls encode_ordinary, so a document is
    bytes and merges. Registering <|im_start|> after pretraining must leave every cached token
    stream valid, which is the entire point of being able to register one.

      * MERGE ORDER decides what a corpus tokenises to -- the same token set applied in a
        different order gives different ids -- so two tokenizers with one vocabulary and two
        merge orders now share a signature and a cache directory.
      * THE PRE-TOKENIZER decides the chunking before any merge applies, so editing _PAT no
        longer invalidates a cached stream by itself.

    Neither is reachable by any path this project takes: the vocabulary is trained once and the
    merges are derived from it, and PRETOK_VERSION has a bump of its own that is the thing to
    reach for if _PAT changes. Both are stated here rather than discovered later."""
    return json.dumps(
        {"vocab": sorted(list(bytes(tok.id2sym[i])) for i in range(tok.n_base))},
        separators=(",", ":")).encode("utf-8")


def tokenizer_fingerprint(path=None, tok=None, n=8):
    """`n` characters of the git blob hash of the tokenizer's ENCODING.

    Abbreviated the way git abbreviates a commit, and for the same reason: the full digest is
    what makes it unique, the first few characters are what makes it usable in a name. Eight
    hex characters is 4.3 billion values against the handful of tokenizers a project ever has,
    so a collision is not a practical concern; git itself defaults to seven.

    Any change to the merges -- a retrained vocabulary, a different merge order, one merge more
    -- lands the cache somewhere else, because those are the changes that alter what a corpus
    tokenises to. Registering a special does NOT, by construction: see encoding_blob.

    The value is reproducible from the shell, which is why git's construction is used rather
    than a bare digest:

        python -c "import json,hashlib,sys; m=json.load(open('checkpoints/bpe/bpe.json', "r"))['merges']; \
                   b=json.dumps(m,separators=(',',':')).encode(); \
                   print(hashlib.sha1(b'blob %d\\0'%len(b)+b).hexdigest()[:8])"

    Memoised on the tokenizer object, since it is asked for once per cached file and a 50k-merge
    list is not free to serialise tens of thousands of times."""
    if tok is not None and hasattr(tok, "merges"):
        # Cached ON THE OBJECT, not in a dict keyed by id(): CPython reuses an id once an
        # object is collected, so an id-keyed cache can hand a new tokenizer the fingerprint
        # of a dead one -- and the failure would be a corpus silently read from the wrong
        # cache. The merge count is part of the key so an incrementally extended vocabulary
        # recomputes.
        cached = getattr(tok, "_fingerprint_cache", None)
        if not cached or cached[0] != len(tok.merges):
            cached = (len(tok.merges), blob_hash(encoding_blob(tok)))
            try:
                tok._fingerprint_cache = cached
            except AttributeError:                 # a tokenizer that forbids new attributes
                pass
        return cached[1][:n]
    path = path or config.BPE_PATH
    try:
        st = os.stat(path)
        key = (path, int(st.st_mtime), st.st_size)     # a MEMO key only; never part of the hash
        if key not in _FINGERPRINTS:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)["vocab"]
            # THE SAME SET encoding_blob TAKES, SELECTED THE SAME WAY. vocab_records writes
            # "bytes": null for the specials and the exact byte list for every base token, so a
            # null IS the test for "this token has no bytes and does not tokenise a corpus" --
            # the id boundary is not needed and is not consulted, which keeps this branch from
            # having a second opinion about where the specials start.
            _FINGERPRINTS[key] = blob_hash(json.dumps(
                {"vocab": sorted(r["bytes"] for r in records if r.get("bytes") is not None)},
                separators=(",", ":")).encode("utf-8"))
        return _FINGERPRINTS[key][:n]
    except (OSError, ValueError, KeyError):
        return "nofile00"[:n].ljust(n, "0")


def tokenizer_tag(tok, path=None):
    """The cache directory name: bpe_<encoding size>_<fingerprint>.

    The size counted is 256 + #merges -- the ids a corpus can actually contain -- NOT len(tok),
    which includes the specials. That distinction is the difference between a cache that
    survives registering a chat token and one that does not: len(tok) rises by one the moment
    <|im_start|> is registered, and naming the directory after it would throw away a corpus
    that had not changed by a single token.

    `bpe` says what kind of tokenizer it is, the size is the number a human recognises the
    directory by, and the fingerprint is what makes the name correct rather than merely
    descriptive -- two vocabularies of the same size are different tokenizers."""
    n_base = 256 + len(tok.merges) if hasattr(tok, "merges") else len(tok)
    return f"bpe_{n_base}_{tokenizer_fingerprint(path, tok)}"


def token_cache_root(tok, path=None):
    """cache/tokens/<tokenizer>/. The `tokens` level names WHAT is cached, so anything else
    the pipeline caches later gets its own sibling rather than being mixed in with a corpus."""
    return os.path.join(config.CACHE_DIR, "tokens", tokenizer_tag(tok, path))


def _portable_src(src):
    """A source named the way it travels: <dataset>/<file>, and never where a machine put it.

    The last two components, because the file name alone collides (every preference tree has a
    batch_0.json) while the absolute path is different on every machine -- which is the whole
    fault this replaces. Anything further up is that host's directory layout, and a cache keyed
    on it is a cache the next host cannot find."""
    src = os.path.normpath(src)
    parent = os.path.basename(os.path.dirname(src))
    return os.path.join(parent, os.path.basename(src)) if parent else os.path.basename(src)


def _tok_path(tok, src, sig):
    return os.path.join(token_cache_root(tok), f"{_portable_src(src)}.{_sig_tag(sig)}.tokens")


def _parquet_docs(path, column):
    """The rows of `column` as DOCUMENTS -- one document per row, the cell's text VERBATIM.

    Row boundaries are kept rather than flattened into one stream of lines. A parquet corpus
    holds one document per row, exactly as a directory of .txt holds one per file, and the
    packer must treat the two the same way: a corpus converted from text to parquet has to
    produce the SAME training documents, or the conversion silently changed what the model
    sees. Flattening would let a packed document begin in one article and end in another,
    teaching the model transitions that exist nowhere in the corpus.

    A wrong column name is reported WITH THE FILE'S ACTUAL COLUMNS. The alternative -- a bare
    KeyError -- arrives after the scan has already opened some fraction of a 78-file corpus and
    says nothing about what to put right."""
    try:
        import pyarrow.parquet as pq
    except Exception as e:                                     # noqa: BLE001
        raise SystemExit(
            f"[corpus] {os.path.basename(path)} is parquet and pyarrow is not importable "
            f"({e}).\n         pip install pyarrow") from e
    pf = pq.ParquetFile(path)
    names = list(pf.schema_arrow.names)
    if column not in names:
        raise SystemExit(
            f"[corpus] {os.path.basename(path)} has no column {column!r}.\n"
            f"         Columns present: {', '.join(names)}\n"
            f"         Set PRETRAIN[\"text_column\"] in default_config.py to the right one.")
    # ROW GROUP BY ROW GROUP, SO ONE CORRUPT PAGE COSTS A ROW GROUP AND NOT THE JOB.
    #
    #     OSError: Corrupt snappy compressed data
    #
    # killed a 4h21m tokenisation at 61% on 2026-08-15, and would have killed it again on every
    # retry: the bad bytes are on disk, so the failure is deterministic and a resumed run walks
    # straight back into it. The corpus is 14 files of a 10B-token dataset and the damage is a
    # few pages; losing the remaining 39% of the corpus to them is the wrong trade.
    #
    # ITERATING ROW GROUPS IS WHAT MAKES THE FAILURE CONTAINABLE. A single iter_batches() over
    # the whole file gives one generator, and once it raises it is dead -- there is no way to
    # step past the bad page. Per row group, a failure costs only that group, and the batches
    # inside it stay small, which is what the original comment was protecting:
    #
    # A SMALL BATCH ON PURPOSE. pyarrow's default is 65,536 rows, and a row here is a whole
    # article: that default materialises hundreds of megabytes of Python strings at a time,
    # which is precisely the cost this module exists to avoid paying.
    #
    # ROWS ALREADY YIELDED FROM A PARTIALLY READ GROUP ARE KEPT. The exception arrives part way
    # through, so everything decoded before it is good data and is passed on.
    #
    # THE SKIP IS DETERMINISTIC, which is what makes it safe for the resume cursor. The cursor
    # counts DOCUMENTS YIELDED, so skipping shifts the numbering -- but the same bytes fail the
    # same way on every run, so a resumed build lands exactly where the first one would have.
    # A corrupt file that is later REPLACED invalidates the cursor, and the signature in the
    # manifest does not cover file contents, so re-tokenise that corpus from scratch after any
    # repair rather than resuming into it.
    #
    # LOUD, NEVER SILENT. Skipping data changes what a checkpoint was trained on, so it is
    # reported per occurrence and again as a total; a run that lost pages says so in its log.
    bad = 0
    for rg in range(pf.num_row_groups):
        try:
            for batch in pf.iter_batches(batch_size=64, columns=[column], row_groups=[rg]):
                for value in batch.column(0).to_pylist():
                    if value:
                        yield str(value)
        except Exception as e:                                 # noqa: BLE001
            bad += 1
            print(f"\n[corpus] CORRUPT DATA SKIPPED: {os.path.basename(path)} row group "
                  f"{rg}/{pf.num_row_groups} -- {type(e).__name__}: {e}\n"
                  f"[corpus] the rest of this file is still being read. This is a damaged "
                  f"download, not a code fault: re-fetch the file and re-tokenise this corpus "
                  f"from scratch if you need the missing documents.", flush=True)
    if bad:
        print(f"[corpus] {os.path.basename(path)}: {bad} of {pf.num_row_groups} row group(s) "
              f"were unreadable and their documents are ABSENT from the token stream.",
              flush=True)


def _source_documents(path, text_column):
    """One corpus file -> its documents, EACH THE TEXT AS THE FILE HOLDS IT.

    A .parquet holds one document per row; every other format is one document per FILE. That
    is the whole difference between the two, and stating it in one place is what lets the
    packer be written once.

    A DOCUMENT IS A STRING, NOT A LIST OF LINES. Newlines, blank lines, indentation and runs of
    spaces reach the encoder as the corpus holds them."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return _parquet_docs(path, text_column)
    if ext in (".json", ".jsonl"):
        docs = _jsonl_docs(path, text_column)
        if docs is not None:
            return docs
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return [handle.read()]
    except Exception:                                          # noqa: BLE001
        raise


def _jsonl_docs(path, text_column):
    """Line-delimited JSON -> one document per row, from `text_column`. None if it is not that.

    RETURNS None RATHER THAN GUESSING, and that is what keeps this safe to add. A .json file
    that is not line-delimited, or whose rows do not carry `text_column`, is handed back to the
    raw-text branch exactly as before -- so no corpus that reads correctly today changes, and
    only the case that was already being read wrongly is affected. `data/download/
    the-stack-smol-xs/data/<lang>/data.json` is JSONL with the code in `content`.

    A generator would be wrong here: the caller needs to know whether this branch applies
    before it commits, so the first line is parsed eagerly and the rest streamed after it."""
    import json
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            first = handle.readline()
            if not first.strip():
                return None
            try:
                row = json.loads(first)
            except ValueError:
                return None                                    # not line-delimited JSON
            if not isinstance(row, dict) or text_column not in row:
                return None                                    # not the column we were asked for
            docs = []
            handle.seek(0)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue                                   # one bad row is not a bad corpus
                value = row.get(text_column)
                if isinstance(value, (list, tuple)):
                    turns = [str(turn) for turn in value if isinstance(turn, str) and turn]
                    if turns:
                        # one newline between turns; each turn's own text is untouched
                        docs.append("\n".join(turns))
                elif isinstance(value, str) and value:
                    docs.append(value)
            return docs or None
    except OSError:
        return None


def _pack(path, max_words, text_column="text"):
    """Source file -> ~max_words-word documents, NEVER CROSSING A SOURCE DOCUMENT.

    A GENERATOR, not a list. A single parquet shard is ~100 MB compressed and several hundred
    megabytes of Python strings once expanded, and returning them all at once would put the
    peak cost of tokenising a corpus at one shard rather than at one document -- which is the
    same mistake, one order of magnitude down, as holding the corpus in RAM.

    The boundary is the file for text and the row for parquet, so the same corpus packs
    identically in either format. The tail of each document is kept even when it is short: a
    remainder discarded for being under the budget would drop the end of every article in the
    corpus, which is a systematic loss rather than a rounding one.

    IT CUTS, IT DOES NOT REWRITE. The pieces concatenate back to the document byte for byte.
    The cut lands on a word boundary and the whitespace before a word goes with the piece that
    follows it, so no piece begins mid-word."""
    for text in _source_documents(path, text_column):
        if not text:
            continue
        # word starts, so a cut is a slice of the original string
        starts = [m.start() for m in re.finditer(r"\S+", text)]
        if not starts:
            yield text
            continue                                           # whitespace only: no words to count
        budget = max(int(max_words), 1)
        at = 0
        for k in range(budget, len(starts), budget):
            yield text[at:starts[k]]
            at = starts[k]
        yield text[at:]


CHAT_START, CHAT_END = "<|startofmessage|>", "<|endofmessage|>"
CHAT_USER_PREFIX, CHAT_ASSISTANT_PREFIX = "User:\n", "Assistant:\n"
CHAT_THINK_START, CHAT_THINK_END = "<startofthink>", "<endofthink>"
# THE ORDER IS THE ORDER THEY WERE NAMED IN, and it is the order they are registered in, which is
# what fixes their ids. THE SET AND THE ORDER ARE THE SIDECAR'S, and they must be:
# tools/create_instruct_bpe.py pins these four in
# saved_models/tokenizer-bpe-instruct/special_tokens.json, and a stage that registered a
# DIFFERENT four would append a fifth token and read every marker as something else.
CHAT_SPECIALS = (CHAT_START, CHAT_END, CHAT_THINK_START, CHAT_THINK_END)


def stop_token_id(tok):
    """The id that ENDS A REPLY: <|end|> when the tokenizer has it, else <|endoftext|>.

    ONE ANSWER FOR EVERY GENERATOR. Stage 7 supervises <|end|> at the close of each assistant
    turn and never supervises <|endoftext|>, so a generator stopping on eos is stopping on a
    token this model has not been taught to emit -- it runs to its budget every time and the
    preview never shows the model finishing. mirachat, the stage previews and the end-of-stage
    samples all ask here.

    Falls back to eos, so a checkpoint from before the markers stops exactly where it did."""
    end = (getattr(tok, "special2id", None) or {}).get(CHAT_END)
    return end if end is not None else getattr(tok, "eos_token_id", None)


def chat_corpus(files, text_column="text", log=None):
    """Is this corpus CONVERSATIONS -- records carrying a list of turns?

    ONE RECORD DECIDES IT, as dataset_helpers._has_messages decides the same question for the
    fine-tuning loaders: a corpus is homogeneous, and reading ten gigabytes to learn its shape is
    the check that gets deleted later for being slow.

    ONE LINE IS READ, NOT ONE FILE. dataset_helpers._read_records reads a whole file and parses
    every line before its first record can be looked at, which on UltraChat's first shard is a
    gigabyte of json parsed to answer a yes/no question -- silently, before stage 4 has printed
    anything. The first non-empty line is the whole cost.

    THE LINE IS READ FOR BOTH EXTENSIONS, BECAUSE `_jsonl_docs` DOES. Stage 4 reads a .json and
    a .jsonl the same way -- one record per line -- so a line-delimited file named .json is a
    conversation corpus to the packer and must be one here too.

    AND THE ANSWER IS EXACTLY `_jsonl_docs`' ANSWER, WHICH IS WHY THERE IS NO FALLBACK TO
    `_read_records`. That reader also understands a json ARRAY and a wrapper object; the packer
    does not -- `_jsonl_docs` requires the first line to be a dict and hands anything else to the
    raw-text branch. A probe that said "conversations" where the packer then finds none would
    build an EMPTY stream. Disagreeing with the packer is worse than reading a corpus the packer
    already reads wrongly, and stage 7 catches that case loudly: no sidecar, and it stops.

    NO SINGLE FILE MAY DECIDE IT, AND THAT IS WHAT THE FIRST TWO ATTEMPTS GOT WRONG. A downloaded
    dataset directory is not only its shards: `corpus_files` returns everything with a
    CORPUS_EXTENSIONS extension, so `README.md` and `dataset_infos.json` are part of the corpus,
    and both sort BEFORE `train_0.jsonl`. A probe that answered from the first file read one
    metadata record, decided UltraChat is not conversations, and packed 9.29 GB of dialogue as
    raw JSON with no sidecar. So EVERY json and jsonl file is asked for its first record, and the
    corpus is conversations when ANY of them carries a turn list.

    THE FILES THAT SAY NO COST NOTHING AND LOSE NOTHING. `_pack_chat` reads every file through
    the same record reader and yields only rows with a turn list, so a metadata file contributes
    zero documents to a conversation corpus -- which is correct: it is metadata, not dialogue.

    AND IT COSTS NOTHING FOR A PRETRAINING CORPUS. Parquet and text files carry no
    line-delimited records, so they are skipped without being opened; a corpus of those alone
    yields no record from any file and is not conversations, which is the answer it always had.
    One line is read from each json file that does exist.

    THIS IS WHAT SEPARATES THE MASK PATH FROM THE PRETRAINING PATH, and nothing else does. A
    pretraining corpus -- parquet shards of a `text` column, a directory of .txt -- has no turn
    list in it, so it takes the branch it always took, is packed by the same `_pack`, is encoded
    by the same call, and gains no sidecar. Only a corpus whose records hold turns is read as a
    conversation and gets a `.masks` file beside its shards.

    THE DECISION IS PRINTED WITH THE FILE AND THE FIELDS IT WAS MADE FROM. A corpus read as the
    wrong kind is silent by construction -- it tokenises, it does not error, and the only symptom
    is a document count that matches nothing -- so `log` gets one line naming the file that
    decided it and what its first record carries."""
    from . import dataset_helpers as dsets

    def first_record(path):
        """The first parseable record of a json/jsonl file, or None. ONE LINE, never the file."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    values = dsets.json_objects(line)
                    return values[0] if values else None
        except OSError:
            return None
        return None

    def fields(record):
        return (", ".join(sorted(record)[:8]) if isinstance(record, dict) else "(not an object)")

    seen = 0
    for path in files:
        if os.path.splitext(path)[1].lower() not in (".json", ".jsonl"):
            continue                           # parquet, text: no line-delimited records to read
        record = first_record(path)
        if record is None:
            continue
        seen += 1
        if dsets._turn_list(record) is not None:
            if log is not None:
                log(f"[corpus] {os.path.basename(path)}: its first record carries "
                    f"{fields(record)} -- a turn list, so this corpus is CONVERSATIONS")
            return True
    if log is not None and seen:
        log(f"[corpus] none of the {seen} json file(s) carries a turn list in its first record, "
            f"so this corpus is read as text")
    return False


def _pack_chat(path, text_column="text"):
    """Source file -> ONE DOCUMENT PER CONVERSATION, as [(role, line), ...].

    NO max_words HERE, AND THAT IS THE DIFFERENCE FROM `_pack`. A pretraining corpus is cut into
    ~max_words-word pieces because its documents are articles of any length and a window has to
    be drawn from somewhere; a conversation is a unit of meaning, and cutting one in three would
    put a reply in a document that does not contain the question it answers. So the boundary is
    the conversation, the stream's document count IS the conversation count, and the mask file
    has exactly one line per conversation.

    THE ROLE COMES FROM THE POSITION IN THE ORIGINAL TURN LIST, which is what
    dataset_helpers._role_tagged does: UltraChat stores bare alternating strings and the first is
    the user's. It is read BEFORE empty turns are dropped, so dropping one cannot swap the
    speakers of everything after it.

    NOTHING IS STRIPPED AND NOTHING IS DROPPED, as in `_pack`, so the two paths agree about what
    text a document holds and differ only about where documents end.

    IT YIELDS (lines, bytes consumed), AND THE SECOND HALF IS FOR THE BAR. `_pack` yields the
    document text, whose length is what the corpus bar counts; a conversation's text is a
    fraction of the bytes its jsonl line occupies -- field names, quoting, escapes -- so
    measuring the same way here would leave the bar short by the whole JSON overhead and stop it
    climbing somewhere near 80% for the rest of a run that takes hours. A bar that never reaches
    its total is indistinguishable from one that has stopped, which is the failure the silent
    execution rule exists to prevent. The count is the RAW BYTES READ since the previous
    document, so it adds up to the file."""
    from . import dataset_helpers as dsets
    for record, consumed in _chat_records(path):
        lines = []
        for turn in dsets._role_tagged(dsets._turn_list(record)) or []:
            got = dsets._turn(turn)
            if not got:
                continue
            role, body = got
            # one entry per turn, the turn's text as the corpus holds it; nothing is stripped
            body = str(body)
            if body:
                lines.append((role, body))
        if lines:
            yield lines, consumed


def _chat_records(path):
    """(record, raw bytes consumed) for a conversation json/jsonl file, in file order.

    A LINE HOLDING SEVERAL RECORDS IS RECOVERED, not skipped: three lines of UltraChat's
    train_3/4/7 hold two objects each, and dropping them would lose twenty conversations and --
    worse -- put every document after them at a boundary the stream does not have.

    THE BYTE COUNT IS EVERY RAW BYTE READ SINCE THE PREVIOUS RECORD, blank lines and lines that
    hold no conversation included, so summing it over a file gives the file. It is the corpus
    bar's unit, and a bar whose parts do not add up to its total stops short and reads as a hang.
    A line holding two records attributes its bytes to the first: the second consumed none."""
    from . import dataset_helpers as dsets
    pending = 0
    with open(path, "rb") as handle:
        for raw in handle:
            pending += len(raw)
            text = raw.strip()
            if not text:
                continue
            for row in dsets.json_objects(text.decode("utf-8")) or []:
                if dsets._turn_list(row) is not None:
                    yield row, pending
                    pending = 0



def corpus_bytes(files, text_column="text"):
    """Total UNCOMPRESSED bytes of `files` -- the denominator a progress bar needs.

    A parquet file's size on disk is its compressed size, which for prose is two or three
    times smaller than the text it holds; using it would drive the bar past 100%. Parquet
    records the uncompressed size of every row group in its FOOTER, so the true figure costs
    a metadata read and no data. Anything else is its size on disk, which is already the
    number of bytes that will be read.

    Best-effort: a file that cannot be measured contributes its on-disk size, and a total that
    is slightly wrong is a bar that is slightly wrong, never a failure."""
    total = 0
    for fp in files:
        try:
            if os.path.splitext(fp)[1].lower() == ".parquet":
                import pyarrow.parquet as pq
                md = pq.ParquetFile(fp).metadata
                total += sum(md.row_group(i).total_byte_size
                             for i in range(md.num_row_groups))
            else:
                total += os.path.getsize(fp)
        except Exception:                                      # noqa: BLE001
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def corpus_stream_path(tok, root, sig):
    """The STEM every shard of a corpus shares:

        cache/tokens/<tokenizer>/<dataset>/<dataset>_<sig>_00000.tokens
                                           <dataset>_<sig>_index.json

    Shards rather than one file because a corpus can be 10 TB, and one file that size cannot
    be resumed after an interruption, cannot be copied incrementally, and loses everything to
    one bad byte. They sit in the dataset's own directory, so every file there ends in .tokens;
    the signature is in each NAME rather than in a directory above them, which is what lets
    two packings of one corpus coexist without a level of nesting that carries no extension."""
    name = corpus_name(root)
    return os.path.join(token_cache_root(tok), name, f"{name}_{_sig_tag(sig)}")


def corpus_writes_masks(root, files, text_column, log=print):
    """Does this corpus get a `.masks` sidecar beside every shard?

    THE DOWNLOAD CONFIG'S `category` IS THE ANSWER WHERE THERE IS ONE: "instruct-sft" yes,
    "pretrain" no. A corpus with no category -- one outside config/download/config_download.yaml, or an
    entry that names none -- falls back to reading the first record of every json and jsonl file
    and asking whether it carries a turn list, which is what decided this before.

    The reading is printed either way, because a corpus read as the wrong kind is silent by
    construction: it tokenises, it does not error, and the only symptom is a document count that
    matches nothing."""
    category = config.dataset_category(root)
    if category == "instruct-sft":
        log(f"[tokens] {corpus_name(root)}: category instruct-sft, so the response mask is "
            f"written beside every shard")
        return True
    if category:
        log(f"[tokens] {corpus_name(root)}: category {category}, so no response mask is written")
        return False
    return chat_corpus(files, text_column, log=log)


def build_token_stream(root, tok, max_words=200, exclude_dirs=(), log=print,
                       extensions=None, text_column="text", stage="tokenize", force=False,
                       resume=True, write=False):
    """Tokenise a corpus into one memory-mapped stream; return (stream, n_files).

    Reused when a current stream already exists, which is what makes this cheap to re-run and
    what lets the training stages simply OPEN the corpus rather than build it. The tokenising
    itself is a generator, so at no point does the whole corpus exist as Python objects: one
    source file is packed, tokenised and appended, and then released.

    `write=False` IS THE DEFAULT AND EVERY CONSUMER TAKES IT: without a stream to open, a
    consumer stops and names the script that writes one. Stage 4 passes `write=True`."""
    from . import token_store
    if write and os.environ.get("MIRALLM_STAGE") != "tokenize_data":
        raise PermissionError("Only run_scripts/run_tokenize_data.sh may write token streams")
    files = corpus_files(root, exclude_dirs, extensions)
    if not files:
        return None, 0
    if force and not write:
        raise SystemExit(f"[{stage}] --force rebuilds a token stream; only "
                         f"./run_scripts/run_tokenize_data.sh writes tokens")
    packing = f"{_tok_signature(tok, max_words, text_column)}|files={len(files)}"
    sig = corpus_signature(tok, root, files, max_words, text_column)
    path = corpus_stream_path(tok, root, sig)
    if not force:
        st = token_store.open_if_current(path, sig)
        if st is not None:
            log(f"[{stage}] token stream: {st.describe()}")
            log(f"[{stage}]               {path}")
            return st, len(files)

    if not write:
        raise SystemExit(
            f"[{stage}] no token stream for {corpus_name(root)} under\n"
            f"          {os.path.dirname(path)}\n"
            f"      This stage reads tokens; it does not write them:\n"
            f"          sh ./run_scripts/run_tokenize_data.sh")

    # A tokenizer without encode_ordinary (the gpt2 student's, say) falls back to __call__;
    # the distinction only exists for tokenizers that HAVE registered specials to protect.
    ordinary = (tok.encode_ordinary if hasattr(tok, "encode_ordinary")
                else lambda t: tok(t, add_special_tokens=False)["input_ids"])
    # THE CATEGORY IN THE DOWNLOAD CONFIG DECIDES IT, and the records are only consulted for a
    # corpus that carries none. "instruct-sft" writes the mask sidecar, "pretrain" does not, and
    # the reading is printed rather than assumed.
    is_chat = corpus_writes_masks(root, files, text_column, log)
    if is_chat:
        log(f"[{stage}] conversation corpus: ONE DOCUMENT PER CONVERSATION (max_words does not "
            f"cut one), encoded line by line, with a response mask written beside every shard "
            f"as {os.path.basename(path)}_NNNNN.masks")

    def documents(cursor=None, skip_docs=0):
        """Every packed document of the corpus, tokenised, one at a time, from `cursor` on.

        Counted in DOCUMENTS rather than files: a corpus is three parquet shards now, not
        thirty thousand text files, so a per-file bar sits still while half a billion tokens
        are encoded -- indistinguishable from a hang. The file position goes in the postfix.

        RESUMING SKIPS WORK, IT DOES NOT REDO IT. The cursor names a file and a document
        within it, so files already consumed are never opened and documents already stored
        are never handed to the tokenizer. Encoding is the expensive half by a wide margin --
        a corpus reads at hundreds of MB/s and encodes at a fraction of that -- so an
        interrupted build that re-encoded what it had already written would spend longer
        catching up than it spent getting there. Without a cursor (a manifest written before
        this existed) the corpus is re-read, which is I/O, but `skip_docs` still keeps those
        documents away from the encoder.

        A PAIR IS YIELDED FOR EVERY PACKED DOCUMENT, even one that encodes to nothing, so the
        position advances in step with what was read rather than with what happened to be
        written. The store drops the empty ones itself.

        encode_ordinary, NOT encode: a pretraining corpus is scraped text, and a document that
        merely MENTIONS <|endoftext|> -- any page discussing GPT-2 does -- would otherwise
        contribute a real end-of-text in its middle. In a packed stream that is worse than a
        stray token: it forges a document boundary, and contiguous sampling then trains across
        a join the corpus does not contain. The stream inserts its own separators; the text
        does not get a vote. Curated data (the SFT and preference sets, where a registered
        token is deliberately written) still goes through encode.

        The leading space on a PRETRAINING document matches Encoder's convention for a
        response, so a document read from the stream and the same document encoded on the fly
        are the same ids. It is added between documents, never inside one.
        """
        first = int((cursor or {}).get("file", 0))
        in_file = int((cursor or {}).get("doc_in_file", 0))
        n_docs = int((cursor or {}).get("n_read", 0))
        left = 0 if cursor else max(int(skip_docs), 0)     # count fallback, no cursor
        done = corpus_bytes(files[:first], text_column) if first else 0
        total_bytes = corpus_bytes(files, text_column)
        # TWO BARS. The outer one is the corpus and stays put: total bytes, already-done bytes
        # on a resume, percentage and ETA for the whole job. The inner one is the file being
        # read right now and is replaced by the next -- because "62% of the corpus" and "which
        # file is this and how far into it" are different questions and a run needs both. The
        # per-file bar alone lost the first; the corpus bar alone was what made a single stuck
        # file indistinguishable from slow progress.
        with bar(f"[{stage}] corpus", unit="B", unit_scale=True, total=total_bytes,
                 initial=done, position=0) as corpus_b:
            for i in range(first, len(files)):
                # documents of THIS file already stored: `in_file` of them on the first file after
                # a seek, none thereafter
                ahead, in_file = in_file, 0
                name = os.path.basename(files[i])
                text_b = corpus_bytes([files[i]], text_column)
                try:
                    disk_b = os.path.getsize(files[i])
                except OSError:
                    disk_b = 0
                log(f"[{stage}] file {i + 1}/{len(files)}: {name}  "
                    f"{disk_b / 1048576:.1f} MB on disk, {text_b / 1048576:.1f} MB of text")
                with bar(f"[{stage}] {name}", unit="B", unit_scale=True, total=text_b,
                         position=1, leave=False) as b:
                    packed = (_pack_chat(files[i], text_column) if is_chat
                              else _pack(files[i], max_words, text_column))
                    read_b = 0
                    for j, d in enumerate(packed):
                        # THE BAR IS UPDATED BEFORE THE SKIPS AND THE ENCODER IS REACHED AFTER
                        # THEM, which is the property the resume depends on: a document already
                        # in a shard is read past but never handed to the tokenizer.
                        #
                        # THE CONVERSATION PATH COUNTS BYTES READ, NOT BYTES KEPT. `_pack` yields
                        # the document text and its length is the progress; a conversation's text
                        # is a fraction of the jsonl line it came from -- field names, quoting,
                        # escapes -- so counting it would leave the bar short by the whole JSON
                        # overhead and stall it near 80% for the rest of a run that takes hours.
                        # `_pack_chat` reports the raw bytes it consumed instead, which add up.
                        if is_chat:
                            d, nb = d
                        else:
                            nb = len(d.encode("utf-8", "ignore"))
                        b.update(nb); corpus_b.update(nb); done += nb; read_b += nb
                        if j < ahead:
                            continue                           # stored; not read past, not encoded
                        if left:
                            left -= 1; n_docs += 1
                            continue                           # stored; re-read but not encoded
                        n_docs += 1
                        if n_docs % 200 == 0:
                            b.set_postfix_str(f"{n_docs:,} docs, corpus "
                                              f"{100 * done / max(total_bytes, 1):.1f}%")
                        # THE CONVERSATION PATH ENCODES TURN BY TURN, AND THAT IS WHAT MAKES
                        # THE MASK EXACT. The pretraining path encodes a document in one call;
                        # where a response has to be located inside the result, the only offsets
                        # that can be trusted are the ones the encoder itself produced. So each
                        # turn is encoded on its own and the pieces are concatenated: the ids ARE
                        # the sum of the pieces, and a span is the sum of the lengths before it.
                        #
                        # ADJACENT ASSISTANT TURNS BECOME ONE SPAN. They are contiguous in the
                        # stream, and two rows would say the same thing in twice the memory.
                        if not is_chat:
                            yield ordinary(d), {"file": i, "doc_in_file": j + 1,
                                                      "n_read": n_docs}, None
                            continue
                        # One piece per turn, encoded as the corpus has it; the boundary is
                        # the marker helpers.chat_index inserts at batch time.
                        ids, spans, at = [], [], 0
                        for role, line in d:
                            piece = ordinary(line)
                            if piece and role in ("assistant", "gpt"):
                                if spans and spans[-1][0] + spans[-1][1] == at:
                                    spans[-1][1] += len(piece)
                                else:
                                    spans.append([at, len(piece)])
                            ids += piece
                            at += len(piece)
                        yield ids, {"file": i, "doc_in_file": j + 1, "n_read": n_docs}, spans
                    # THE FILE IS CLOSED OUT TO ITS FULL LENGTH. A conversation corpus contains
                    # files that yield no conversation at all -- the dataset's own README.md and
                    # dataset_infos.json are part of it -- and a file that yields nothing reports
                    # no bytes. Without this the corpus bar is short by exactly those files and
                    # never reaches its total, which reads as a job that stopped.
                    if is_chat and read_b < text_b:
                        b.update(text_b - read_b)
                        corpus_b.update(text_b - read_b)
                        done += text_b - read_b
                    b.set_postfix_str(f"{n_docs:,} docs, corpus "
                                      f"{100 * done / max(total_bytes, 1):.1f}%")

    log(f"[{stage}] tokenizing {len(files):,} files -> {path}")
    from model import parallel
    failure = None
    if parallel.is_main():
        try:
            token_store.build(
                path, sig, documents, tok.eos_token_id, len(tok), log=log,
                shard_bytes=int(getattr(config, "TOKENS", {}).get("shard_mb", 100))
                * 1024 * 1024, resume=resume and not force, packing=packing,
                masks=is_chat)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
    failure = parallel.broadcast_ckpt(failure)
    if failure is not None:
        raise RuntimeError(f"distributed token-cache build failed: {failure}")
    parallel.barrier()
    stream = token_store.TokenStream(path)
    # A CONVERSATION CORPUS DOES NOT FINISH WITHOUT ITS SIDECARS, AND THIS IS WHY THE CHECK IS
    # HERE RATHER THAN IN THE STAGE THAT READS THEM. Reading a corpus as the wrong kind is silent
    # by construction: it tokenises, it does not error, and the only symptom is a document count
    # that matches nothing -- so twice now a run of two and a half hours reported success and
    # wrote no mask, and the first thing to say so was stage 7, afterwards. A sidecar that is
    # missing or out of step with its shard is a failure of THIS stage and is raised by it.
    if is_chat:
        for i, shard in enumerate(stream.shards):
            got = token_store.read_masks(path, i)
            if got is None:
                raise RuntimeError(
                    f"{os.path.basename(path)} was read as conversations but shard {i:05d} has "
                    f"no response mask at {token_store.masks_path(path, i)}")
            if len(got) != int(shard.n_docs):
                raise RuntimeError(
                    f"{token_store.masks_path(path, i)} has {len(got):,} line(s) for a shard of "
                    f"{int(shard.n_docs):,} document(s); the two must agree line for line")
        spans = sum(len(s) for i in range(len(stream.shards))
                    for s in (token_store.read_masks(path, i) or []))
        log(f"[{stage}] response mask: {int(stream.n_docs):,} conversation(s), {spans:,} "
            f"(offset, length) span(s) in {len(stream.shards)} sidecar(s)")
    return stream, len(files)


def load_token_corpus(stage, root, tok, max_words=200, exclude_dirs=(), log=print,
                      extensions=None, text_column="text", write=False):
    """The corpus as a memory-mapped TokenStream: (stream, n_files, n_tokens).

    READ-ONLY UNLESS THE CALLER IS STAGE 4; see build_token_stream."""
    st, n_files = build_token_stream(root, tok, max_words, exclude_dirs, log, extensions,
                                     text_column, stage=stage, write=write)
    if st is None:
        return None, 0, 0
    return st, n_files, st.n_tokens


def load_token_corpora(stage, roots, tok, max_words=200, exclude_dirs=(), log=print,
                       extensions=None, text_column="text", write=False):
    """SEVERAL corpora as one: (stream, n_files, n_tokens).

    EACH IS BUILT AND CACHED EXACTLY AS IF IT WERE ALONE, under its own signature, so adding a
    corpus to a run tokenises only the new one and leaves every existing stream untouched --
    same signature, same shards, same cache directory. Nothing is merged on disk, which is what
    makes this safe to do to a run that is already half-way through its budget.

    One corpus returns the plain TokenStream, so a single-corpus run behaves exactly as it did
    and cannot be told apart from before. Several return a MultiStream, which reads across them
    in proportion to their tokens.

    A corpus with no files is REPORTED AND SKIPPED rather than fatal: a list is a list, and
    losing the whole run because the fourth entry has not been downloaded yet would be the
    wrong trade."""
    # IMPORTED HERE, as everything in this module imports token_store: helpers/__init__ does
    # `from .utils import *`, so a module-level import of a sibling would run while the package
    # is still initialising. A function-local import binds the name in THIS function -- the one
    # in build_token_stream does not reach here, which is exactly how MultiStream came to be
    # referenced with nothing bound to it.
    from . import token_store
    roots = [r for r in (roots if isinstance(roots, (list, tuple)) else [roots]) if r]
    streams, names, n_files, missing = [], [], 0, []
    for root in roots:
        if not os.path.isdir(root):
            missing.append(root)
            continue
        st, n = build_token_stream(root, tok, max_words, exclude_dirs, log, extensions,
                                   text_column, stage=stage, write=write)
        if st is None:
            missing.append(root)
            continue
        streams.append(st)
        names.append(corpus_name(root))
        n_files += n
    for m in missing:
        log(f"[corpus] skipped (nothing to read): {m}")
    if not streams:
        return None, 0, 0
    if len(streams) == 1:
        return streams[0], n_files, streams[0].n_tokens
    multi = token_store.MultiStream(streams, names)
    log(f"[corpus] {len(streams)} corpora as one: {multi.describe()}")
    return multi, n_files, multi.n_tokens


# --------------------------------------------------------------------------- #
# preference pairs: the same idea for the preference batches
# --------------------------------------------------------------------------- #
def _pair_stem(tok, src, split, sig):
    """<mirror of the source>.<split>.<sig> -- the stem the pair cache's two files share."""
    return _tok_path(tok, src + (f".{split}" if split else ""), sig)[:-len(".tokens")]


def _pair_read(stem, want_pairs):
    """(records, valid_bytes) already on disk: id lists in the order they were written.

    Each record is LENGTH-PREFIXED -- one uint32 count, then that many uint32 ids -- so the
    file is read by walking forward, and a partial write at the end is simply a record that
    does not complete. No separator is used and none would do: a preference record CAN contain
    the characters of a special token, and a scheme that split on EOS would mis-cut exactly
    those pairs with nothing to say so."""
    import array
    try:
        with open(f"{stem}_index.json", "r", encoding="utf-8") as f:
            head = json.load(f)
        n_rec, valid = int(head["n_records"]), int(head["n_bytes"])
    except (OSError, ValueError, KeyError):
        return [], 0
    try:
        with open(f"{stem}.tokens", "rb") as f:
            raw = array.array("I")
            raw.frombytes(f.read(valid))
    except OSError:
        return [], 0
    out, i = [], 0
    while len(out) < n_rec and i < len(raw):
        n = raw[i]; i += 1
        if i + n > len(raw):
            break
        out.append(raw[i:i + n].tolist()); i += n
    return out[:3 * want_pairs], valid


def _pair_manifest(stem, sig, st, n_records, n_bytes, complete):
    tmp = f"{stem}_index.json.part"
    input_only(stem, "write a pair manifest")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"magic": "mira_llm-pairs", "version": 1, "sig": sig,
                   "src_size": st.st_size,
                   "n_records": n_records, "n_bytes": n_bytes,
                   "complete": bool(complete)}, f)
    os.replace(tmp, f"{stem}_index.json")


def attach_pair_ids(pairs, src, tok, stage="instruct", split="", log=print, resume=True,
                    flush_every=2000):
    """Attach `prompt_ids` / `chosen_ids` / `rejected_ids` to every preference pair, IN PLACE,
    caching them so the reward and DPO trainers never tokenise inside their step loops.

    WRITTEN AS IT GOES, never held. Each pair's three sequences are appended the moment they
    are tokenised, and the manifest catches up every few thousand pairs, so a run killed part
    way leaves every pair it tokenised on disk and RESUMES from there. Building the whole cache
    in memory and writing once at the end -- which is what this did -- means an interrupted run
    leaves nothing at all, and the peak cost is the whole cache rather than one pair.

    The split and the pair count are part of the signature, so the train half can never be
    served the validation half's ids, and a set that grew invalidates rather than silently
    reusing the prefix it happens to agree with."""
    import array
    from contextlib import nullcontext
    write = os.environ.get("MIRALLM_STAGE") == "tokenize_data"
    if not pairs:
        return 0
    # THE SOURCE IS IDENTIFIED BY ITS SIZE, NOT ITS MODIFICATION TIME. An mtime says when the
    # file landed on this filesystem; it moves under rsync, a re-download or a restore, and a
    # cache keyed on it is thrown away every time the data is copied to another machine.
    sig = (f"{_tok_signature(tok, 0)}|pairs={len(pairs)}|split={split}"
           f"|src={_portable_src(src)}|bytes={os.path.getsize(src)}")
    st = os.stat(src)
    stem = _pair_stem(tok, src, split, sig)
    if write:
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    path = f"{stem}.tokens"

    have, valid = [], 0
    if resume and write:
        try:
            with open(f"{stem}_index.json", "r", encoding="utf-8") as f:
                head = json.load(f)
            if head.get("sig") == sig and head.get("src_size") == st.st_size:
                have, valid = _pair_read(stem, len(pairs))
        except (OSError, ValueError):
            have, valid = [], 0
    if not have and write:
        for f in (path, f"{stem}_index.json"):
            if os.path.exists(f):
                os.remove(f)
        valid = 0

    done = len(have) // 3
    for i in range(done):
        pairs[i]["prompt_ids"] = have[3 * i]
        pairs[i]["chosen_ids"] = have[3 * i + 1]
        pairs[i]["rejected_ids"] = have[3 * i + 2]
    n_tokens = sum(len(r) for r in have)
    if done >= len(pairs):
        log(f"[{stage}] pair token cache: reused {done:,} {split or 'pairs'} -> {path}")
        return n_tokens
    if done:
        log(f"[{stage}] pair token cache: {done:,} {split or 'pairs'} reused, "
            f"{len(pairs) - done:,} still to tokenise")

    # bytes written after the manifest's mark belong to a pair it does not count
    if write and os.path.exists(path) and os.path.getsize(path) != valid:
        input_only(path, "truncate a pair cache")
        with open(path, "r+b") as f:
            f.truncate(valid)

    eos = tok.eos_token_id
    n_bytes, n_rec = valid, 3 * done
    if write:
        input_only(path, "append to a pair cache")
    with (open(path, "ab") if write else nullcontext()) as fh:
        for i in progress(range(done, len(pairs)),
                          desc=tag(stage) + f" tokenizing {split or 'pairs'}",
                          total=len(pairs), initial=done):
            p = pairs[i]
            p["prompt_ids"] = tok(p["prompt"], add_special_tokens=False)["input_ids"]
            p["chosen_ids"] = tok(" " + p["chosen"], add_special_tokens=False)["input_ids"] + [eos]
            p["rejected_ids"] = tok(" " + p["rejected"], add_special_tokens=False)["input_ids"] + [eos]
            for seq in (p["prompt_ids"], p["chosen_ids"], p["rejected_ids"]):
                if write:
                    array.array("I", [len(seq)] + list(seq)).tofile(fh)
                n_bytes += 4 * (len(seq) + 1)
                n_tokens += len(seq)
                n_rec += 1
            if write and (i + 1 - done) % flush_every == 0:
                fh.flush()
                _pair_manifest(stem, sig, st, n_rec, n_bytes, complete=False)
        if write:
            fh.flush()
            os.fsync(fh.fileno())
    if write:
        _pair_manifest(stem, sig, st, n_rec, n_bytes, complete=True)
    log(f"[{stage}] pair tokens: {len(pairs):,} {split or 'pairs'} -> "
        f"{path if write else 'memory only; dataset and token caches unchanged'}")
    return n_tokens
