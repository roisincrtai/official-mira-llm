"""
dynamics.py -- the append-only record of what the two dynamical systems were doing.

    checkpoints/<stage>/<run tag>/dynamics.jsonl

One line per probe step: the full per-channel pole vector and state summaries of every MIIR,
and the score scale, entropy and attention distance of every
attention. Written by model/lm.py at the --plot_every_steps cadence, read by nothing in this
repository.

WHY IT EXISTS AT ALL. A pretraining run costs days of GPU time and cannot be repeated to answer
a question that occurs to someone afterwards. The figures a run draws are decided before it
starts; the questions are not. So the record is deliberately WIDER than any current figure --
what is drawn can change later, what was measured cannot. Anything not written here while the
run was alive is not recoverable from the checkpoint, because these are properties of the
activations and the activations are a function of the batch that produced them.

WHY IT IS NOT history.json. history.json is rewritten WHOLE at every checkpoint, so a record
appended to it costs O(steps^2) bytes written over a run -- at 100k steps and a checkpoint
every 1k that is a hundred rewrites of a file that only grows. This one is opened, appended to
and closed, so a line costs its own length once and nothing else.

WHAT A MIRA LAYER RECORDS. Its `ssm` entry holds model.mira_profiling._summarize's keys (the ones
history.json also keeps) and, from 2026-10-04, _summarize_ext's: the assembled pole pair per sampled
(token, channel) -- pole_horizon_p10/p50/p90, pole_abs_*, pole_angle_p50, pole_complex_frac --, the
zeros of the channel transfer from the write to the rho-weighted level sum (zero_abs_*,
zero_outside_frac, zero_pole_gap_p50), its gains |G(1)| and |G(-1)|, beta's and rho's token spread
and E[beta^2], nu_rms per level, x~/s/gate/control scales, the control's participation ratio, the
mixer's output/input RMS and the RMS of every parameter group. A record written before a key existed
lacks it; readers treat a missing key as not measured.

WHAT IS NOT IN IT: no score matrix, no state tensor and no attention map. The one width-sized
quantity is the full channelwise $a_t$ summary requested for horizon evolution; it is sampled
only at the profiling cadence. Nothing depends on batch size or context length.

FORMAT. JSON Lines, one object per line, UTF-8, newline-terminated. The first line of a fresh
file is a header, {"schema": N, ...}; every later line is a measurement carrying "step". A
reader should skip any line it cannot parse -- see `read`.

    from helpers import dynamics
    dynamics.truncate(ckdir, stage, upto=start)          # once, on resume
    dynamics.append(ckdir, stage, step, full_stats(model), extra={"ctx": 8192})
    for rec in dynamics.read(ckdir, stage): ...          # afterwards, by whatever plots it
"""
import json
import math
import os

from .utils import artefact_dir

SCHEMA = 1

NAME = "dynamics.jsonl"

# Six significant figures. These are statistics of a stochastic quantity measured on two
# sequences of a batch -- the seventh figure is noise, and it is noise that costs a byte per
# number per step for the life of the run.
SIGFIG = 6


def path(ckdir, stage, flat=False):
    return os.path.join(artefact_dir(ckdir, stage, flat), NAME)


def _clean(o):
    """JSON has no NaN. Python's json module writes a bare `NaN` token anyway, which is not
    valid JSON and which json.loads accepts but almost nothing else does -- so a record written
    this way would be readable by this repository and by no other tool, which defeats the point
    of writing it in a portable format.

    NaN and infinity become null. They are not errors: a head with no history long enough to
    measure `mass_beyond_8192` has no value for it, and null is what that means."""
    if isinstance(o, float):
        return None if not math.isfinite(o) else float(f"%.{SIGFIG}g" % o)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (int, str, bool)) or o is None:
        return o
    # torch scalars, numpy scalars, anything else with a float() -- recorded rather than raised
    # on, because a probe must never be the thing that stops a training run.
    try:
        return _clean(float(o))
    except (TypeError, ValueError):
        return str(o)


def header(cfg=None):
    h = {"schema": SCHEMA, "sigfig": SIGFIG}
    if cfg:
        h["cfg"] = _clean(dict(cfg))
    return h


def append(ckdir, stage, step, stats, extra=None, cfg=None, flat=False):
    """One measurement. Creates the file, with its header, if it is not there.

    Opened, written and closed per call. That is a syscall or three every few hundred steps,
    against a step that is doing a forward and a backward over a whole batch -- unmeasurable --
    and it means the file on disk is complete after every probe rather than after the next
    checkpoint. A run killed by the scheduler keeps everything it measured."""
    d = artefact_dir(ckdir, stage, flat)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, NAME)
    new = not os.path.exists(p) or os.path.getsize(p) == 0
    rec = {"step": int(step)}
    if extra:
        rec.update(_clean(dict(extra)))
    rec.update(_clean(stats))
    with open(p, "a", encoding="utf-8") as f:
        if new:
            f.write(json.dumps(header(cfg), separators=(",", ":")) + "\n")
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return p


def append_many(ckdir, stage, records, cfg=None, flat=False):
    """Several measurements at once, in one open. Returns how many were written.

    WRITTEN WITH THE CHECKPOINT, NOT WHEN MEASURED, and that is a correctness rule rather than
    a batching optimisation. A profile describes the WEIGHTS AT A STEP. Appending it the
    instant it is measured puts rows on disk for steps whose weights were never persisted --
    the job is killed between a checkpoint and the next one, and the record then claims a
    history the checkpoint cannot account for. Nothing downstream can tell those rows from
    real ones, because a step number is all they carry.

    Still one open, one write, one close, so an interrupted flush can only truncate the last
    line, which `read` already skips."""
    records = [r for r in records if r]
    if not records:
        return 0
    d = artefact_dir(ckdir, stage, flat)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, NAME)
    new = not os.path.exists(p) or os.path.getsize(p) == 0
    with open(p, "a", encoding="utf-8") as f:
        if new:
            f.write(json.dumps(header(cfg), separators=(",", ":")) + "\n")
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return len(records)


def record(step, stats, extra=None):
    """One measurement as a plain dict, cleaned and ready to be written later.

    Separated from the write so the trainer can hold it until the checkpoint that makes it
    true. Cleaning happens HERE, at measurement time, because it is what turns torch scalars
    into numbers -- holding tensors in a buffer would pin the graph they came from."""
    rec = {"step": int(step)}
    if extra:
        rec.update(_clean(dict(extra)))
    rec.update(_clean(stats))
    return rec


def read(ckdir, stage, upto=None):
    """Every measurement line, in order. The header is skipped; unparseable lines are skipped.

    A LINE CAN BE UNPARSEABLE ONLY AT THE END, and only once: a killed process can leave a
    partial write, and the alternative to skipping it is a reader that raises on a file whose
    first 40,000 lines are perfectly good."""
    p = path(ckdir, stage)
    if not os.path.exists(p):
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict) or "step" not in rec:
                continue
            if upto is not None and rec["step"] >= upto:
                continue
            yield rec


def truncate(ckdir, stage, upto):
    """Drop every measurement at or after `upto`, in place. Returns how many were dropped.

    THE SAME RULE AS load_hist, AND FOR THE SAME REASON. A resume replays from the checkpoint's
    step, so any line at or beyond it describes a step that is about to be measured again.
    Appending past them leaves the file with duplicated and then non-monotone steps, and any
    reader that assumes order -- which every plot of a training curve does -- draws a run that
    appears to jump backwards. After this, appending can only extend a strictly increasing
    sequence.

    Rewritten through a temporary file and renamed, so an interrupted truncation leaves the
    original record rather than half of one."""
    p = path(ckdir, stage)
    if not os.path.exists(p):
        return 0
    kept, dropped, head = [], 0, None
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except ValueError:
                continue
            if isinstance(rec, dict) and "schema" in rec and "step" not in rec:
                head = s
                continue
            if isinstance(rec, dict) and rec.get("step", -1) >= upto:
                dropped += 1
                continue
            kept.append(s)
    if not dropped:
        return 0
    tmp = p + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            if head is not None:
                f.write(head + "\n")
            for s in kept:
                f.write(s + "\n")
        os.replace(tmp, p)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    return dropped


def estimate_bytes(n_layer, d_model, n_head, n_far=4):
    """Bytes per line, near enough, for a model of this shape.

    THE POINT OF HAVING THIS AT ALL is that the record must be checkable BEFORE a run rather
    than discovered to be 40 GB during one. Every term below is a constant times a
    configuration value; none of them is a function of the context window, the batch or the
    step count, which is the property that makes the file safe to append to for a week."""
    num = SIGFIG + 4                       # "0.123457," -- digits, point, comma, exponent room
    miir = 2 * n_layer * (12 * (num + 18) + 4 * d_model * num)
    attn = n_layer * (9 * (num + 18) + (3 * n_head + n_far) * num)
    # model.mira_profiling._summarize_ext (2026-10-04): about 75 named scalars per MIRA layer -- true pole-pair
    # horizons, zeros, channel gains, beta/rho token spread, write/read/gate/control scales and the
    # RMS of every parameter group -- none of them width-, batch- or context-sized.
    extended = n_layer * 75 * (num + 22)
    return int(miir + attn + extended + 64)
