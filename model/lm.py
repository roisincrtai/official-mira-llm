"""
lm.py -- the length-normalized language-model training loop.

ONE objective, used by the two stages that maximise likelihood:

    L(theta) = - E_{y ~ D} [ (1/|y|) sum_t log pi_theta(y_t | y_<t) ]

LENGTH-NORMALIZED per example (the sum over response positions divided by their count), so a
long document does not dominate a short one inside a batch and `ppl = exp(loss)` reads as a
per-token perplexity. That normalization belongs to THIS objective only: the preference
stages score with the unnormalized sum, because the implicit reward
beta*(log pi - log pi_ref) is a difference of sequence log-likelihoods and dividing it by the
length would change the objective rather than the reporting.

    train(model, enc, docs, ckdir, args, log, monitor, stage=..., steps=..., lr=...)

owns its resume, budget extension, checkpointing, live plotting and generation previews.

`--plot_every_steps` IS ONE CADENCE FOR TWO THINGS: it redraws the figures and, on the same
step, profiles the model -- memory horizons, decays, gates and per-bank state of both
recurrences, and under arch=mira the attention scores, distances and metric spectrum (see
model/ssm.py and model/mira.py for what is measured and why). The figure gets the curve; the
full per-bank and per-head detail goes to the append-only record in helpers/dynamics.py, which
history.json deliberately does not carry -- a training run cannot be repeated to answer a
question thought of afterwards, so what is measured is wider than what is currently drawn.

The first step of a run is profiled as well, because initialisation is the one state that
cannot be measured afterwards.

The loop knows nothing about WHICH corpus it is given -- the pretrain and sft packages each
hand it their own -- so the two stages cannot silently drift apart in their treatment of
masking, normalisation or padding.
"""
import importlib
import os

import torch
import torch.nn.functional as F


def _load_weights(model, state, log, stage):
    """`state` into `model`, GROWING the embedding when the vocabulary has since expanded.

    Registering a special appends an id, so a checkpoint written before it has every row it was
    trained with and is short only the new ones. The state dict is loaded at ITS OWN size and the
    model grown afterwards, so the new rows are initialised by resize_token_embeddings rather
    than by whatever this function happened to choose -- the same order helpers.common.
    load_growing uses, because a resume and a fresh load must not differ in how a row is born.

    An EQUAL vocabulary is the ordinary path and is untouched, which is what the second run does:
    the checkpoint it reloads already carries the expanded one."""
    have = int(state["tok.weight"].shape[0]) if "tok.weight" in state else None
    want = int(model.tok.weight.shape[0])
    if have is None or have == want:
        model.load_state_dict(state)
        return 0
    if have > want:
        raise RuntimeError(
            f"cannot resume {stage}: the checkpoint's vocabulary is {have:,} and the tokenizer's "
            f"is {want:,}; ids can be registered but never removed, so these are different "
            f"vocabularies")
    model.resize_token_embeddings(have, allow_shrink=True)
    model.load_state_dict(state)
    model.resize_token_embeddings(want)
    log(f"{stage}: vocabulary {have:,} -> {want:,}; the checkpoint's {have:,} rows are "
        f"unchanged and {want - have} row(s) are new")
    return want - have


def _resume_optimizer(opt, ck):
    """The saved optimiser state.

    THE PADDING IS THE OPTIMISER'S, NOT THIS FUNCTION'S. MasterAdamW keeps fp32 master copies
    beside the live parameters and its own {"inner", "masters"} state dict, so a resume across a
    grown vocabulary has to widen both -- and only that class knows which is which.
    MasterAdamW.load_state_dict does it: the saved block goes into the leading corner and the
    rows that were added keep what they were initialised with. Reaching in from here needed the
    format to be duplicated in two files, and got it wrong, because it padded the plain-AdamW
    shape this optimiser does not use.

    The weights were loaded before the optimiser was built -- see train() -- so `opt` already
    holds the final parameters."""
    opt.load_state_dict(ck["opt"])


def _mira_profiling():
    """model.mira_profiling, or None where MIRA-Matrix cannot be imported."""
    try:
        import importlib
        importlib.import_module("model.mira_mixer")
        return importlib.import_module("model.mira_profiling")
    except Exception:                                          # noqa: BLE001
        return None


def collect_stats(model, on=True):
    """How many dynamical systems had diagnostics switched.

    THE THREE SIGNATURES AND RETURN SHAPES ARE UNCHANGED, because helpers/dynamics.py and
    helpers/visualization.py read them and lm.py's loop calls all three unconditionally."""
    profiling = _mira_profiling()
    return 0 if profiling is None else profiling.collect_stats(model, on)


def layer_stats(model):
    """The scalar diagnostics of the most recent forward pass, one entry per system."""
    profiling = _mira_profiling()
    return [] if profiling is None else profiling.layer_stats(model)


def full_stats(model):
    """Everything measured this step, for the append-only record. See helpers/dynamics.py.

    A reader is expected to look at what a line contains rather than at what its stage was
    called, and an empty "ssm"/"attn" means "not measured"."""
    profiling = _mira_profiling()
    return {"ssm": [], "attn": []} if profiling is None else profiling.full_stats(model)

from helpers import dynamics
from . import parallel as _P               # dist_env / is_main / all_reduce_mean
from helpers.utils import (progress, save_ckpt, load_ckpt, save_hist, load_hist, MasterAdamW,
                     CosineLR, schedule_total, ckpt_path, tag, restore_rng, amp)

NAME = "LM"
STAGE = "sft"          # default stage name; the pretrain package passes its own

_USE_LCE_KERNEL = False
_cuda_lce = None
#     if _USE_LCE_KERNEL:                                        # explicit legacy experiment only
#         try:
#             from kernel.cuda.mira_sdpa.legacies.mira_fused_linear_cross_entropy import (
#                 linear_cross_entropy as _cuda_lce)
#         except Exception:                                      # noqa: BLE001
#             _cuda_lce = None


def _chunk_logprob(head, h, tgt):
    """Log-probability of `tgt` under `head(h)`, for ONE slice of the sequence.

    Called through torch.utils.checkpoint, which is the whole point: the (b, chunk, vocab)
    logits and their log-softmax exist during this call, are thrown away when it returns, and
    are recomputed one slice at a time during the backward pass. What survives in memory is the
    (b, chunk) result. Returns the log-probabilities and, detached, how many positions the model
    would have got right -- computed here because the logits needed to answer that question are
    here and nowhere else."""
    logits = head(h).float()
    lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return lp, (logits.argmax(-1) == tgt).detach()


def normalised_nll(log_probs, rmask):
    """THE OBJECTIVE, IN ONE PLACE: mean over a sequence's own response tokens, then over the batch.

        loss = mean_b [ -sum_t rmask_bt * logp_bt / max(sum_t rmask_bt, 1) ]

    `log_probs` and `rmask` are already shifted -- one entry per PREDICTED position.

    WHY PER SEQUENCE AND NOT PER TOKEN. Summing over the batch's tokens and dividing by their
    count lets one long sequence outweigh several short ones: a 2,000-token conversation would
    count ten times a 200-token one, and the model would be tuned mostly by whichever examples
    happen to be long. Normalising inside each sequence first makes every EXAMPLE contribute
    equally, which is what the corpus is a sample of. It matters most where sequence lengths
    differ most -- instruction data, where a batch is whole conversations of very different
    lengths -- which is the case stage 7 trains on.

    IT IS A FUNCTION SO THAT THERE IS EXACTLY ONE DEFINITION OF IT. model/hf_model.py computes
    the loss HuggingFace asks for from `labels`, and it calls this: a wrapper that computed a
    different objective from the model it wraps would report numbers that cannot be compared
    with anything this pipeline produces, and nothing would say so."""
    log_probs, rmask = log_probs.float(), rmask.float()
    denom = rmask.sum(-1).clamp(min=1)
    return (-(log_probs * rmask).sum(-1) / denom).mean()


def lm_loss(model, ids, attn, rmask, chunk=0):
    """Mean per-sequence negative log-likelihood over the response positions, and accuracy.

    THE OBJECTIVE IS LENGTH-NORMALISED: each sequence contributes the mean log-probability of
    its own response tokens, and the batch averages those. A sum would let one long sequence
    outweigh several short ones.

    `chunk` > 0 evaluates the vocabulary projection in slices of that many positions instead of
    all at once. The result is identical -- the same positions, the same targets, the same
    normalisation -- but the peak memory of the loss becomes 3 x chunk x vocab rather than
    3 x T x vocab, which at T = 32,768 is the difference between 0.6 GB and 18.4 GB."""
    # THE TARGETS FOLLOW THE LOGITS. When the model's layers are split across GPUs the output
    # comes back on the LAST device while the batch was built on the first, and gather() cannot
    # index a tensor on one device with a tensor on another. Unsharded the two already agree
    # and both moves are no-ops. See model/distributed/distributed_fsdp_layer.py.
    home = model.head.weight.device
    tgt, rm = ids[:, 1:].to(home), rmask[:, 1:].to(home).float()
    denom = rm.sum(-1).clamp(min=1).float()
    if not chunk:
        logits = model(input_ids=ids, attention_mask=attn).logits.float()
        lp = F.log_softmax(logits, -1)[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        loss = normalised_nll(lp, rm)
        with torch.no_grad():
            correct = ((logits[:, :-1].argmax(-1) == tgt) & rm.bool()).sum().float()
        return loss, correct, rm.sum().clamp(min=1).float()

    # CHUNKED OVER TOKENS, NOT OVER POSITIONS. The projection produces one vocabulary row per
    # TOKEN, and a batch of 24 sequences at 1,024 positions is 24,576 of them -- slicing the
    # position axis alone still hands the projection every sequence at once and saves nothing
    # the batch does not immediately give back. Flattening first makes `chunk` mean what it
    # says: this many rows of (vocab) exist at a time, whatever shape the batch has.
    from torch.utils.checkpoint import checkpoint
    h = model.hidden_states(input_ids=ids, attention_mask=attn)[:, :-1]
    B, L, D = h.shape
    hf, tf, rf = h.reshape(B * L, D), tgt.reshape(B * L), rm.reshape(B * L)
    # OUR OWN KERNEL, WHEN IT HAS BEEN SWITCHED ON AND BUILT FOR THIS GPU. It answers the same
    # two questions the loop below does -- the per-token log-probability and whether the argmax
    # was right -- from ONE streaming pass over the vocabulary, so there are no slices to
    # checkpoint and nothing to recompute in the backward. It returns None whenever it is not
    # available, and then the loop runs exactly as before. See the note beside _USE_LCE_KERNEL.
    got = (_cuda_lce(hf, model.head.weight, tf)
           if _USE_LCE_KERNEL and _cuda_lce is not None else None)
    if got is not None:
        nll, am = got
        # THE KERNEL RETURNS THE LOSS PER TOKEN; the objective is what it always was. The
        # normalisation is the line at the bottom of this function, which both paths share --
        # a per-sequence mean over response positions, not a mean over the batch's tokens.
        lp = (-nll).view(B, L)
        with torch.no_grad():
            correct = ((am == tf) & rf.bool()).sum().float()
    else:
        parts = []
        correct = torch.zeros((), device=h.device, dtype=torch.float32)
        for a in range(0, B * L, chunk):
            b = min(a + chunk, B * L)
            lp, hit = checkpoint(_chunk_logprob, model.head, hf[a:b], tf[a:b],
                                 use_reentrant=False)
            parts.append(lp)
            correct = correct + (hit & rf[a:b].bool()).sum().float()
        # (B*L,) floats: four bytes a token, against four bytes a token TIMES THE VOCABULARY for
        # the logits that produced them. Keeping these is what lets the per-sequence normalisation
        # below be exact rather than an average of averages.
        lp = torch.cat(parts).view(B, L)
    return (-(lp.float() * rm).sum(-1) / denom).mean(), correct, rm.sum().clamp(min=1).float()


def context_plan_for(args, steps, docs):
    """The (start, stop, context, batch, micro_batch) segments for a budget of `steps`.

    A PURE FUNCTION OF THE BUDGET, and silent, so it can be asked what the schedule WOULD have
    been for some other budget -- which is how a resumed run recovers the window it was last
    training at when the budget has since changed. `context_schedule` is this plus the log line.

    ONE WINDOW UNLESS ALL THREE CONDITIONS HOLD: the corpus is a packed stream (a list of
    records cannot be re-cut to an arbitrary length), the scheme declares a schedule, and the
    run has not pinned a context of its own with --context_window. A stage that
    was told which window to use is not second-guessed."""
    import default_config as config
    one = [(0, steps, max(int(args.max_len), 2), int(args.batch),
            int(getattr(args, "micro_batch", 0) or 0))]
    if not (hasattr(docs, "batch") and hasattr(docs, "n_tokens")):
        return one
    # --context_window WINS, and it may itself be a schedule: "1024,4096" pins those two
    # windows regardless of what the scheme says. Only when nothing was asked for does the
    # scheme's own schedule apply.
    asked = getattr(args, "context_window", "")
    scheme = getattr(args, "model_scheme", "") or config.PRETRAIN["model_scheme"]
    if asked:
        wins = config.windows(asked)
    elif scheme in config.SCHEMES:
        wins = config.context_windows(scheme)
    else:
        wins = []
    if len(wins) < 2:
        return one
    # The batch in force belongs to the LONGEST window, which is what it was sized against.
    return config.context_plan(wins, steps, int(args.batch),
                               config.SCHEME_MICRO_TOKENS.get(scheme, 0)
                               if not getattr(args, "micro_batch", 0) else 0)


def context_schedule(args, steps, docs, log, stage):
    """`context_plan_for`, announced."""
    plan = context_plan_for(args, steps, docs)
    if len(plan) > 1:
        log(f"{stage}: context schedule "
            f"{' -> '.join(f'{w:,}' for _, _, w, _, _ in plan)} "
            f"over {steps:,} steps, {steps // len(plan):,} steps each, "
            f"batch/GPU {plan[0][3]} -> {plan[-1][3]}")
    return plan


def context_floor(args, docs, ck, start):
    """THE WINDOW THE RUN HAS ALREADY REACHED, which the schedule may never fall below.

    The schedule divides the budget evenly among its windows, so the boundaries are a function
    of the budget -- and RAISING the budget pushes them later. A run 82,875 steps into a
    168,317-step budget is at 8,192; extend that budget to 250,000 and step 82,875 falls back
    into the 2,048 segment. The model would then be asked to un-learn a context it has spent
    tens of thousands of steps acquiring, its state-space memory horizon and its attention
    both retuned downwards, and the extension -- meant to buy MORE long-context training --
    would cost some of what was already bought. Hence the ratchet: the window only ever goes
    up.

    Two sources, and the larger wins:"""
    if not ck:
        return 0
    floor = int(ck.get("ctx_window", 0) or 0)
    old_total = int(ck.get("total", 0) or 0)
    if old_total > 0:
        old = context_plan_for(args, old_total, docs)
        for a, b, w, _, _ in old:
            if a <= start < b:
                floor = max(floor, int(w))
                break
    return floor


def _flush_dynamics(ckdir, stage, pending, model, log, flat=False):
    """Write the buffered profiles, beside the checkpoint that makes them true. Clears `pending`
    IN PLACE, so the caller's list and this one cannot diverge.

    Wrapped for the same reason the measurement is: a full disk or a read-only mount is a
    reason to lose the record, not the run. Cleared even when the write fails -- a buffer that
    grows without bound after the first failure would turn a lost diagnostic into a lost job.

    RANK 0 WRITES. The record is append-only and every rank measures the same weights, so N
    ranks appending would put N identical rows in the file for every profiled step -- not a
    corruption, but a file whose step column silently stops being unique. The buffer is still
    cleared on every rank, for the reason above."""
    if not pending:
        return 0
    if not _P.is_main():
        pending.clear()
        return 0
    try:
        n = dynamics.append_many(ckdir, stage, pending, cfg=getattr(model, "cfg", None),
                                 flat=flat)
    except Exception as e:                                          # noqa: BLE001
        log(f"{stage}: {len(pending):,} dynamics record(s) not written: {e}")
        n = 0
    pending.clear()
    return n


def ratchet(plan, seg, start, floor, args, log, stage):
    """`plan` with every segment from `seg` onwards raised to at least `floor`.

    The batch and micro-batch are recomputed by exactly the rules context_plan uses -- batch x
    context constant, so a step stays the same unit of work, and micro_tokens // context
    sequences per forward pass -- rather than being scaled out of the numbers already in the
    plan, which would compound the rounding those numbers already carry.

    Segments already at or above the floor are left untouched, so when nothing has changed
    this returns the plan unaltered and a resumed run is the run it was."""
    import default_config as config
    if floor <= 0 or seg >= len(plan):
        return plan
    scheme = getattr(args, "model_scheme", "") or config.PRETRAIN["model_scheme"]
    longest = max(w for _, _, w, _, _ in plan)
    micro_tokens = (config.SCHEME_MICRO_TOKENS.get(scheme, 0)
                    if not getattr(args, "micro_batch", 0) else 0)
    out, lifted = list(plan), []
    for i in range(seg, len(plan)):
        a, b, w, _, _ = plan[i]
        if w >= floor:
            continue
        batch = max(1, round(int(args.batch) * longest / int(floor)))
        micro = min(batch, max(1, int(micro_tokens) // int(floor))) if micro_tokens else 0
        out[i] = (a, b, int(floor), batch, micro)
        # reported from the RESUME POINT, not from the segment's own start: the run begins
        # inside the first lifted segment, and steps before it are finished and unaffected
        lifted.append((w, max(a, int(start)), b))
    if lifted:
        # NEVER SILENTLY. This changes what the run trains on, so it is stated in full: what
        # was in the plan, what it was raised to, and over which steps.
        log(f"{stage}: context ratchet floor {floor:,}; {len(lifted)} segment(s) raised:")
        for w, a, b in lifted:
            log(f"{stage}:   steps {a:,}-{b - 1:,}: {w:,} -> {floor:,}")
    else:
        # Said even when it does nothing, because "the ratchet is armed and the schedule was
        # already sound" and "the ratchet never ran" look identical in a log that stays quiet.
        log(f"{stage}: context ratchet floor {floor:,}; schedule unchanged")
    return out


def train(model, enc, docs, ckdir, args, log, monitor=None, stage=STAGE, steps=None, lr=None,
          preview=None, stop_at=None, require_model_cfg=None, monitor_on_checkpoint=False,
          optimizer="adamw", profile=True, schedule=True, checkpoint_extra=None, flat=False,
          progress_bar_format=None):
    """Train `model` by maximum likelihood on the response tokens; return it in eval mode.

    `monitor` IS OPTIONAL AND None MEANS "DRAW NOTHING". Five sites below call it -- two on the
    checkpoint cadence, one on --plot_every_steps, one on the pause path and one at the end of
    the stage -- and until this default existed, three of them were reached unconditionally, so
    NO caller could legally pass None even though the parameter had no default saying otherwise.
    ablation/run.py:228 passes None, because a geometry sweep has no per-episode comparison
    figure to render, and paired it with monitor_on_checkpoint=True; that combination is exactly
    the one guaranteed to reach the first site, and it raised

        TypeError: 'NoneType' object is not callable      model/lm.py:758

    Parameterised by `stage`/`steps`/`lr` (defaults reproduce the SFT stage exactly) so the SAME
    length-normalized LM loop also serves the controlled experiment's LM PRETRAIN stage
    (stage="pretrain", steps=args.pretrain_steps, lr=args.peak_lr) over the transcript corpus,
    with its own checkpoint directory and dynamics figure.

    Resume rule, shared with the preference stages: A CHECKPOINT IS FINISHED ONLY FOR THE
    `total` IT RAN. `done` alone ignores the budget, so raising the step budget would load the old
    weights and skip the stage instead of extending it. `!=` rather than `<` so LOWERING the
    budget also re-enters (the step loop is then empty and the stage costs nothing but a re-save)."""
    steps = args.sft_steps if steps is None else steps
    # `steps` REMAINS THE TOTAL TRAINING BUDGET. `stop_at` only bounds this invocation, so a
    # checkpointed round-robin benchmark can release one family and advance the next without
    # changing the schedule, the checkpoint's total, or its definition of completion.
    run_stop = steps if stop_at is None else min(steps, max(0, int(stop_at)))
    lr = args.sft_lr if lr is None else lr
    ck = load_ckpt(ckdir, stage, flat) if args.resume else None
    extend = bool(ck and ck.get("done") and ck.get("total") != steps)
    evaluation_only = bool(ck and ck.get("done") and not extend)
    if ck is not None:
        from helpers import require_checkpoint_model_cfg
        require_checkpoint_model_cfg(model, ck, stage)
    if ck is not None and require_model_cfg:
        def lower_keys(value):
            if isinstance(value, dict):
                return {str(key).lower(): lower_keys(item) for key, item in value.items()}
            return value
        saved = lower_keys(ck.get("model_cfg") or {})
        require_model_cfg = lower_keys(require_model_cfg)
        # THE SAME RESOLUTION require_checkpoint_model_cfg JUST APPLIED. ablation/run.py hands
        # this check its whole built cfg, where ffn_hidden is the width read off the block; a
        # checkpoint written when the field meant "unset" carries null, and the two are the same
        # FFN. Without this the resume passes the check above and fails here.
        from helpers import resolve_ffn_hidden
        resolve_ffn_hidden(saved)
        resolve_ffn_hidden(require_model_cfg)
        import default_config as _config
        if "lattice_order" in require_model_cfg:
            _config.mira_map_arch(saved)
            recorded = saved.pop("lattice_initializer", saved.pop("lattice_init_method", None))
            if recorded is not None:
                saved["lattice_initializer"] = _config.canonical_lattice_initializer(
                    recorded, recorded=True)
        mismatched = {key: (saved.get(key, "<absent>"), expected)
                      for key, expected in require_model_cfg.items()
                      if saved.get(key, "<absent>") != expected}
        if mismatched:
            detail = ", ".join(f"{key}: checkpoint={old!r}, requested={new!r}"
                               for key, (old, new) in mismatched.items())
            raise RuntimeError(f"cannot resume {stage} with a different architecture ({detail})")
    if ck is not None and checkpoint_extra is not None:
        saved_extra = ck.get("extra") or {}
        mismatched = {key: (saved_extra.get(key, "<absent>"), expected)
                      for key, expected in checkpoint_extra.items()
                      if saved_extra.get(key, "<absent>") != expected}
        if mismatched:
            if evaluation_only:
                log(f"{stage}: completed checkpoint predates the current data partition; "
                    "loading its weights for evaluation only")
            else:
                detail = ", ".join(f"{key}: checkpoint={old!r}, requested={new!r}"
                                   for key, (old, new) in mismatched.items())
                raise RuntimeError(
                    f"cannot resume {stage} with a different data partition ({detail})")
    if ck and ck.get("done") and not extend:
        log(f"{stage} done -> loading")
        _load_weights(model, ck["model"], log, stage)
        model.eval()
        return model

    if ck is not None and (not ck["done"] or extend):
        _load_weights(model, ck["model"], log, stage)
    # if str(optimizer).lower() == "muon":
    #     from muon.muon import MasterMuon
    #     opt = MasterMuon(model, lr=lr)
    #     log(f"{stage}: optimiser {opt.describe()}")
    if str(optimizer).lower() == "adamw":
        opt = MasterAdamW(model, lr=lr,
                          mixed_precision=getattr(args, "mixed_precision", False))
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}; expected 'adamw'")
    import default_config as config
    grad_clip = float(config.TRAIN["grad_clip"])
    grad_clip = grad_clip if grad_clip > 0 else float("inf")
    # THE CURVE'S `total` IS THE RUN'S, NOT THE CONFIGURATION'S. The step BUDGET below is still
    # `steps`; only the T in cos(pi t / T) is pinned to what this run was already on, so editing
    # SCHEME_STEPS mid-run extends the training without moving the rate back up the curve. Same
    # ratchet as context_floor, and it changes no learning rate. See helpers.utils.schedule_total.
    sched = CosineLR(opt, lr, schedule_total(ck, steps, log, stage),
                     args.last_lr, args.lr_schedule, args.warmup_steps)
    log(f"{stage}: lr={sched.describe()} steps={steps} batch/GPU={args.batch} "
        f"examples={len(docs):,} -> {ckpt_path(ckdir, stage, flat)}")
    g = torch.Generator().manual_seed(args.seed)
    start = 0
    if ck is not None and (not ck["done"] or extend):
        _resume_optimizer(opt, ck)
        # THE GLOBAL STREAM TOO, not only the data generator. `g` covers the data order; dropout
        # and anything else drawing from torch's default generator do not, so an uninterrupted
        # run and a resumed one saw different dropout masks from the resume point onwards. Every
        # other trainer here already restored it -- reward, rlhf, dpo, cot and distill -- and
        # this one, which serves pretrain, the instruction SFT, the CoT SFT and the language
        # adaptation, did not. save_ckpt has always written it; nothing read it back.
        #
        # Harmless for MIRA, which has no dropout. Not harmless for a model that has one.
        start = ck["step"]; g.set_state(ck["gens"][0]); restore_rng(ck)
        log(f"resume {stage} @ {start}" + (f" (budget {ck.get('total')} -> {steps})"
                                           if extend else ""))
    # Never move a resumable run backwards when a caller asks for a boundary this checkpoint
    # has already reached (the normal state after interruption halfway through a family round).
    run_stop = max(start, run_stop)
    # truncated at the resume point, so the appended records extend a strictly increasing
    # sequence and the figure stays continuous across a restart
    hist = load_hist(ckdir, stage, upto=start, flat=flat) if args.resume else []
    # THE RECORD IS TRUNCATED WITH THE HISTORY, at the same step and for the same reason: a
    # resume is about to re-measure everything from `start` onward. Done here, once, before the
    # loop -- not on the first append, which would have to know whether it was the first.
    if args.resume:
        gone = dynamics.truncate(ckdir, stage, upto=start)
        if gone:
            log(f"{stage}: dropped {gone:,} dynamics record(s) at or after step {start:,}")

    # READ ONCE, HERE, AND NOT INSIDE THE BRANCH BELOW. Both corpus shapes need the rank and
    # the world size -- to shard the data and to form the global batch -- and reading them inside
    # the packed-stream branch left them undefined for a list of records, which is an
    # UnboundLocalError in the step loop rather than a wrong number.
    _rank, _world = _P.data_env(getattr(args, "distributed_parallelism", "ddp"))
    # TWO CORPUS SHAPES, ONE LOOP. A packed TokenStream is memory-mapped and sampled by
    # offset: no padding, no prompt mask, every position a target. A list of records is
    # encoded per batch as before, which is what the prompt/response distinction of the SFT
    # corpus needs. The difference is confined to this closure, so the objective, the
    # resume rule and the diagnostics below cannot drift between the two.
    if hasattr(docs, "batch") and hasattr(docs, "n_tokens"):
        # args.max_len is the model's context window by the time setup() is done with it, and
        # batch() returns seq_len + 1 tokens so the shift produces seq_len targets. Asking for
        # max_len - 1 therefore feeds the model EXACTLY its context window, never one over it.
        T = max(int(args.max_len) - 1, 1)
        if _world > 1:
            n_gpu, my_region = _world, _rank
        else:
            n_gpu = len(getattr(model, "device_ids", []) or []) or 1
            my_region = None                     # DataParallel: all regions, row order kept
        if n_gpu > 1 and _P.is_main():
            log(f"{stage}: {'DDP' if _world > 1 else 'data parallel'} over {n_gpu} "
                f"GPUs -- the {docs.n_tokens:,}-token stream is read as {n_gpu} regions "
                f"of {docs.n_tokens // n_gpu:,}, read-only and never copied")
        # THE BANNER SAYS WHAT IS TRUE OF THIS CORPUS. A fixed window really does give
        # `batch x T` target tokens every step; a corpus that draws whole documents does not,
        # and quoting a per-example length there contradicted the stage's own line four above it.
        if schedule:
            log(f"{stage}: packed stream, {docs.n_tokens:,} tokens, {T + 1} tokens/example, "
                f"batch/GPU {args.batch} -> {args.batch * T:,} target tokens/GPU/step "
                f"({args.batch * n_gpu * T:,} global)")
        else:
            log(f"{stage}: packed stream, {docs.n_tokens:,} tokens in {len(docs):,} documents, "
                f"batch/GPU {args.batch} whole document(s) per step; the width of a step is the "
                f"longest of them")

        def sample(batch, gen, seq_len=None):
            return docs.batch(batch, seq_len or T, generator=gen, device=enc.device,
                              regions=n_gpu, region=my_region)
    else:
        n_gpu = len(getattr(model, "device_ids", []) or []) or 1
        Ntr = len(docs)
        def sample(batch, gen, seq_len=None):
            # ONE STRIDE OF THE DRAW PER RANK. `gen` is seeded from args.seed identically on
            # every rank -- deliberately, so a resumed run reproduces its data order -- so
            # without a slice N ranks would encode the SAME records and compute N copies of one
            # gradient. The whole step's worth is drawn on every rank and each takes its own
            # stride, rather than each drawing only its own share: the generator is then
            # advanced by the same amount everywhere, and since rank 0's is the state that gets
            # checkpointed, that is what makes a resume land every rank where it left off.
            idx = torch.randint(0, Ntr, (batch * _world,), generator=gen).tolist()
            if _world > 1:
                idx = idx[_rank::_world]
            return enc.encode([docs[i] for i in idx], "chosen")

    # THE CONTEXT SCHEDULE. Only a packed stream can serve a window of any length on demand; a
    # list of records has the length its records have, so the fine-tuning stages keep one window
    # and the plan collapses to a single segment. `plan` is a list of
    # (start, stop, context, batch) tiling [0, steps) exactly.
    # `schedule=False` MEANS ONE SEGMENT AT THE CONFIGURED BATCH, and a stage that draws its own
    # window needs it. context_plan_for falls back to the SCHEME's windows when the yaml names
    # none -- mira-s declares 512,1024 -- so removing `context_window` from a yaml does not
    # remove the schedule, it only stops overriding it. For instruct-sft that produced a two
    # segment plan whose first segment ALSO doubled the batch to 16 "so tokens per GPU per step
    # stay constant", against a corpus whose width is the batch's own longest conversation and
    # has nothing to do with either window. Every other caller keeps the behaviour it had.
    plan = (context_schedule(args, steps, docs, log, stage) if schedule else
            [(0, steps, max(int(args.max_len), 2), int(args.batch),
              int(getattr(args, "micro_batch", 0) or 0))])
    # THE SEGMENT THE RUN ACTUALLY STARTS IN, not the first one. A resumed run begins in the
    # middle of the schedule, and seg=0 was only ever corrected INSIDE the loop -- after the
    # progress bar had already been labelled with plan[0]'s window. So a run resumed at 150,000
    # trained at 8,192 while its bar said 1,024, and the line that would have said otherwise
    # only fires at a segment boundary, which a mid-segment resume is not. The training was
    # right and everything a person could see about it was wrong, which is the worst way round.
    seg = 0
    while seg + 1 < len(plan) and start >= plan[seg][1]:
        seg += 1
    floor = context_floor(args, docs, ck, start) if schedule else 0
    plan = ratchet(plan, seg, start, floor, args, log, stage)
    # CHUNKED LOSS. 0 evaluates the vocabulary projection over the whole sequence at once, which
    # is what every stage did before long contexts existed and is still right at 512. Above that
    # the three vocabulary-sized tensors dominate the step, so the projection is sliced.
    chunk = int(getattr(args, "loss_chunk", 0) or 0)
    if chunk:
        V = int(getattr(model, "cfg", {}).get("vocab_size", 0)) or 0
        # the comparison is against the WHOLE STEP -- batch x context tokens -- because that is
        # what the projection would otherwise be handed at once
        most = max(b * w for _, _, w, b, _ in plan)
        note = (f" ({3 * chunk * V * 4 / 1024**3:.2f} GiB peak instead of "
                f"{3 * most * V * 4 / 1024**3:.2f} GiB)") if V else ""
        log(f"{stage}: chunked loss, {chunk:,} TOKENS per slice{note}")
    # initial/total in ABSOLUTE steps: a resumed stage must read 30499/40000, not 499/10000.
    model.train()
    bar = progress(range(start, run_stop),
                   desc=tag(stage, plan[seg][2] if len(plan) > 1 else None),
                   initial=start, total=steps, bar_format=progress_bar_format)
    # THE BAR SAYS WHAT IT IS DOING BEFORE THE FIRST STEP FINISHES. loss, ppl, acc and gnorm
    # exist only once a step has completed, and step 0 is the longest of the run: it is a
    # profile step (want_ssm is true at `step == start`), it builds the first batch, and it
    # loads the CUDA kernel. Until this, all of that happened behind `0/N` with no rate and no
    # postfix, which reads as a hang rather than as work. The four postfix fields replace this
    # the moment the first step lands and are not otherwise touched.
    if hasattr(bar, "set_postfix_str"):
        bar.set_postfix_str("first step: batch, kernel load, profile")
    saved_at = start if ck is not None else -1
    # PROFILES MEASURED SINCE THE LAST CHECKPOINT. Held here rather than written, and flushed
    # by _flush_dynamics beside every save_ckpt -- see helpers/dynamics.append_many for why.
    pending_dyn = []
    for step in bar:
        cur_lr = sched.step(step)              # cosine, keyed off the ABSOLUTE step
        # diagnostics for ONE step at the figure cadence: switched on before the forward that
        # is happening anyway, so the measured pass is a real training pass, not an extra one
        # `profile=False` SWITCHES THE MIRA STATISTICS OFF FOR A WHOLE STAGE, which
        # --profile_every_steps 0 cannot do on its own: the fallback below reads
        # --plot_every_steps when the first is zero, so a stage that wants figures but not
        # diagnostics has no way to say so through the flags. Every stage that does not pass it
        # keeps the behaviour it had.
        prof = 0 if not profile else int(getattr(args, "profile_every_steps", 0)
                                         or getattr(args, "plot_every_steps", 0) or 0)
        want_ssm = bool(prof) and ((step + 1) % prof == 0 or step == start)
        if want_ssm:
            collect_stats(model, True)
        # WHICH WINDOW THIS STEP BELONGS TO. Advanced rather than searched, so a resumed run at
        # step 30,000 lands in the same segment it would have reached by running from zero --
        # the plan is a function of the budget, not of how the budget was consumed.
        while seg + 1 < len(plan) and step >= plan[seg][1]:
            seg += 1
        _, _, cur_ctx, cur_batch, cur_micro = plan[seg]
        # BATCH IS PER GPU. Under torchrun this process owns one GPU and consumes exactly
        # `cur_batch` rows; there is no division by world size. Under the legacy one-process
        # DataParallel path, the process constructs n_gpu times that many rows and the wrapper
        # scatters one configured batch to each card. Both paths therefore implement the same
        # global optimiser batch: cur_batch * n_gpu.
        local_batch, process_batch, global_batch, _ = _P.batch_layout(model, cur_batch)
        if (step == plan[seg][0] or step == start) and hasattr(bar, "set_description"):
            bar.set_description(tag(stage, cur_ctx))
        # ...at a boundary, AND on the first step of a resumed run, which is not one
        if len(plan) > 1 and (step == plan[seg][0] or step == start):
            # EVERY WINDOW IS ITS OWN SCHEDULE SEGMENT. What fits at 1,024 and what fits at
            # 32,768 are different questions -- one sequence at the long end has many more
            # activations than one at the short end. The configured schedule supplies the
            # batch and micro-batch for each segment; nothing rewrites them from a VRAM target.
            #
            # CHECKPOINTED FIRST. A boundary is exactly where a run can be interrupted and
            # resumed under a DIFFERENT yaml, so the weights are on disk before the configured
            # context, batch and micro-batch for the next segment take effect.
            if step == plan[seg][0] and step > start:
                # CALLED ON EVERY RANK AND WRITTEN BY ONE. The rank-0 gate and the barriers
                # that go with it live in helpers/utils.save_ckpt, so every call site in every
                # stage gets them and none of them has to remember; calling it under an
                # `if is_main()` here would leave the other ranks waiting at a barrier rank 0
                # is inside, which is the deadlock the gate exists to avoid.
                save_ckpt(ckdir, stage, model, opt, step, steps, [g],
                          extra=checkpoint_extra, ctx_window=cur_ctx,
                          sched_total=sched.total, flat=flat)
                saved_at = step
                save_hist(ckdir, stage, hist, flat=flat)
                _flush_dynamics(ckdir, stage, pending_dyn, model, log, flat)
            # THE WHOLE CONFIGURATION, AT EVERY SWITCH. A window boundary changes the batch,
            # the pass size, the checkpointing and the memory the run will sit at -- and it is
            # the one moment where a person watching can still intervene before a day of
            # training happens under settings they did not intend. Printed in full, every time,
            # rather than as a diff nobody can reconstruct from a log tail.
            mb_now = (args.micro_batch if getattr(args, "micro_batch", 0) > 0
                      else (cur_micro or cur_batch))
            mb_now = max(1, min(int(mb_now), local_batch))
            local_mb_now = mb_now
            if n_gpu > 1 and _world <= 1:
                mb_now = max(1, min(int(mb_now) * n_gpu, process_batch))
            log(f"{stage}: {'=' * 18} context window {cur_ctx:,} {'=' * 18}")
            log(f"{stage}:   steps          {plan[seg][0]:,}-{plan[seg][1] - 1:,} of {steps:,}")
            log(f"{stage}:   batch/GPU      {local_batch} sequences x {cur_ctx - 1:,} = "
                f"{local_batch * (cur_ctx - 1):,} target tokens/GPU/step")
            log(f"{stage}:   global batch   {global_batch} sequences = "
                f"{global_batch * (cur_ctx - 1):,} target tokens/step")
            log(f"{stage}:   pass/GPU       {local_mb_now} sequences = "
                f"{local_mb_now * cur_ctx:,} tokens, "
                f"{-(-local_batch // local_mb_now)} pass(es)/step")
            log(f"{stage}:   checkpointing  "
                f"{'ON' if getattr(model, 'grad_checkpoint', False) else 'off'}")
            log(f"{stage}:   loss chunk     "
                f"{int(getattr(args, 'loss_chunk', 0) or 0):,} tokens"
                if chunk > 0 else
                f"{stage}:   loss chunk     off")
            log(f"{stage}: {'=' * (38 + len(f'{cur_ctx:,}'))}")
        # ONE STEP, POSSIBLY SEVERAL FORWARD PASSES. A micro-batch splits the step's sequences
        # into groups that are each carried through forward and backward alone, so only one
        # group's activations are ever resident; the gradients add up in the parameters, which
        # is where a batch is combined anyway. The optimiser sees exactly the step it would have
        # seen whole -- each group's loss is weighted by its share of the sequences, so the sum
        # is the mean over all of them, not the mean of the means.
        opt.zero_grad(set_to_none=True)
        mb = (args.micro_batch if getattr(args, "micro_batch", 0) > 0
              else (cur_micro or cur_batch))
        mb = max(1, min(int(mb), local_batch))
        if n_gpu > 1 and _world <= 1:
            mb = max(1, min(int(mb) * n_gpu, process_batch))
        sum_loss = sum_correct = sum_tokens = 0.0
        for off in range(0, process_batch, mb):
            nb = min(mb, process_batch - off)
            ids, attn, rmask = sample(nb, g, max(cur_ctx - 1, 1))
            # THE FORWARD IS INSIDE THE AUTOCAST, THE BACKWARD AND THE STEP ARE NOT. autocast
            # records the dtype each operation ran in and the backward follows that record, so
            # wrapping it too would add nothing; the optimiser must see fp32 parameters and fp32
            # gradients, which is exactly what it sees out here.
            with amp(args):
                loss, correct, ntok = lm_loss(model, ids, attn, rmask, chunk)
            # WEIGHTED BY THIS PROCESS'S SHARE OF ITS OWN BATCH, so the gradient each rank
            # accumulates is the mean over the sequences it saw; the all-reduce below averages
            # those over the ranks, which is the mean over the global batch.
            (loss * (nb / process_batch)).backward()
            sum_loss += float(loss.item()) * nb
            sum_correct += float(correct.item()); sum_tokens += float(ntok.item())
        # ONE ALL-REDUCE PER STEP, HERE, AND NOT FROM AN AUTOGRAD HOOK.
        #
        # THE MODEL IS NOT WRAPPED IN DistributedDataParallel and this is why. DDP arms its
        # reducer from its own forward(), and with a chunked loss the head runs OUTSIDE that
        # call -- lm_loss computes the trunk once through hidden_states and then projects each
        # slice -- so the tied embedding's gradient partly arrives on a graph the reducer is not
        # watching. It raised "Expected to have finished reduction in the prior iteration" on
        # the first step of every four-rank run. Doing it here needs no hooks and makes no
        # assumption about which methods the loss happens to call.
        #
        # AFTER THE LAST MICRO-BATCH AND BEFORE THE CLIP. The gradient the norm is measured on,
        # the update the optimiser applies and the number the log carries are then all the
        # GLOBAL batch's, identically on every rank -- clipping a per-rank gradient instead
        # would give each rank a different scale factor and the replicas would drift apart.
        # A no-op off DDP, where there is nothing to average. See model/parallel.all_reduce_grads
        # for the collective-symmetry argument and for what this costs (the overlap of
        # communication with the backward, which the reducer had and this does not).
        _P.all_reduce_grads(model.parameters())
        gnorm = _P.clip_grad_norm_(model.parameters(), grad_clip); opt.step()
        importlib.import_module("model.mira_block").update_moe_balance(model)
        # THE LOSS IS REDUCED FOR THE RECORD, as the gradients were just above. Each rank's
        # sum_loss is over its OWN shard, so a history written without this is 1/world_size
        # of a batch and every figure and every comparison against a single-GPU run measures
        # a different quantity.
        loss = torch.tensor(_P.all_reduce_mean(sum_loss / process_batch))
        sum_correct = _P.all_reduce_mean(sum_correct)
        sum_tokens = _P.all_reduce_mean(sum_tokens)
        # Next-token accuracy over the RESPONSE positions only (the tokens the loss is on): the
        # fraction of response positions whose argmax prediction is the true next token. Reported
        # alongside loss/ppl so the LM stages (pretrain, sft) have an accuracy curve, not only a
        # loss curve. grad-norm (pre-clip) is tracked too, as an optimisation-health trace.
        acc = sum_correct / max(sum_tokens, 1.0)
        rec = {"step": step, "loss": loss.item(), "ppl": float(loss.exp().item()),
               "acc": acc, "gnorm": float(gnorm), "lr": cur_lr}
        if want_ssm:
            # stored per layer, unaggregated: the history keeps what was measured
            rec["ssm"] = layer_stats(model)
            # AND THE WIDE VERSION GOES TO THE RECORD, which is the only copy of the per-bank
            # and per-head detail. Wrapped, because a diagnostic must not be able to end a run
            # that is otherwise training correctly -- a full disk or a read-only mount is a
            # reason to lose the record, not the week of GPU time.
            # BUFFERED, NOT WRITTEN. A profile describes the weights AT THIS STEP, and those
            # weights are not on disk until the next checkpoint. Writing it now would put rows
            # in the record for a state no checkpoint can account for, and after a kill nothing
            # downstream could tell them from real ones. `_flush_dynamics` writes them beside
            # every save_ckpt, which makes the invariant exact: every row in the file describes
            # a step at or before the checkpoint next to it.
            try:
                pending_dyn.append(dynamics.record(
                    step, full_stats(model),
                    extra={"ctx": cur_ctx, "batch": cur_batch, "lr": cur_lr,
                           "loss": loss.item()}))
            except Exception as e:                                  # noqa: BLE001
                log(f"{stage}: dynamics record skipped at step {step:,}: {e}")
            collect_stats(model, False)
        hist.append(rec)
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=f"{loss.item():.3f}", ppl=f"{loss.exp().item():.1f}",
                            acc=f"{acc:.4f}", gnorm=f"{float(gnorm):.2f}")
        if (step + 1) % args.checkpoint_every_steps == 0:
            # THE WINDOW GOES IN THE CHECKPOINT. It is the one thing about the schedule that
            # cannot be recovered from the weights, and without it a resume has to guess --
            # see context_floor.
            save_ckpt(ckdir, stage, model, opt, step + 1, steps, [g],
                      extra=checkpoint_extra, ctx_window=cur_ctx,
                      sched_total=sched.total, flat=flat)
            saved_at = step + 1
            save_hist(ckdir, stage, hist, flat=flat)
            _flush_dynamics(ckdir, stage, pending_dyn, model, log, flat)
            if monitor and monitor_on_checkpoint:
                # Benchmark episodes are checkpoint intervals. Render while every distributed
                # rank is still inside the successful step, before another family or operation
                # can fail and prevent the durable episode from reaching its comparison PDF.
                monitor(stage, hist, step + 1)
        # --plot_every_steps is a TRAINER flag, so it governs every stage that trains, this one
        # included: same flag, same cadence, same figure writer as the preference stages.
        if (monitor and not monitor_on_checkpoint and args.plot_every_steps > 0
                and (step + 1) % args.plot_every_steps == 0):
            monitor(stage, hist, step + 1)
        # THE GENERATION EXAMPLES HAVE THEIR OWN CADENCE, --print_samples_every_steps. They used
        # to share the figure's, which forced one number to serve two things that are consumed
        # differently: a figure is glanced at, twenty generations are read line by line.
        every = getattr(args, "print_samples_every_steps", args.plot_every_steps)
        if preview and every > 0 and (step + 1) % every == 0:
            preview(model, stage, step + 1)
    model.eval()
    if run_stop < steps:
        # A PARTIAL INVOCATION IS A PAUSE, NOT A FINISHED STAGE. The periodic checkpoint at an
        # exact cadence may already have written this state; writing it again is harmless and
        # closes the same atomic checkpoint/history/dynamics group when stop_at is not itself a
        # configured checkpoint boundary.
        if saved_at != run_stop:
            save_ckpt(ckdir, stage, model, opt, run_stop, steps, [g],
                      extra=checkpoint_extra, ctx_window=plan[seg][2],
                      sched_total=sched.total, flat=flat)
            _flush_dynamics(ckdir, stage, pending_dyn, model, log, flat)
            save_hist(ckdir, stage, hist, flat=flat)
            if monitor and monitor_on_checkpoint:
                monitor(stage, hist, run_stop)
        last = hist[-1] if hist else {}
        log(f"{stage} paused @ {run_stop:,}/{steps:,}: "
            f"loss={last.get('loss', float('nan')):.4f} "
            f"ppl={last.get('ppl', float('nan')):.2f} "
            f"acc={last.get('acc', float('nan')):.4f}")
        return model
    # the LAST segment's window: what the finished model was trained at, and the floor any
    # later extension of this run must respect
    save_ckpt(ckdir, stage, model, opt, steps, steps, [g], extra=checkpoint_extra,
              ctx_window=plan[-1][2], sched_total=sched.total, flat=flat)
    _flush_dynamics(ckdir, stage, pending_dyn, model, log, flat)
    save_hist(ckdir, stage, hist, flat=flat)
    if monitor and monitor_on_checkpoint:
        monitor(stage, hist, steps)
    last = hist[-1] if hist else {}
    log(f"{stage} done: loss={last.get('loss', float('nan')):.4f} "
        f"ppl={last.get('ppl', float('nan')):.2f} acc={last.get('acc', float('nan')):.4f}")
    if monitor and not monitor_on_checkpoint:
        monitor(stage, hist)             # persist the figure the moment the stage ends
    return model
