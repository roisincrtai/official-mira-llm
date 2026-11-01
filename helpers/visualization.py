"""
visualization.py -- every figure the pipeline draws, as PDF, with no main title.

Each stage owns one dynamics figure, written to outputs/plots/<stage>/ and named

    dynamics_<model>_<pe>_<stage-label>.pdf

so the size, the positional encoding and the stage are all legible from the filename alone
(see helpers.utils.STAGE_LABELS). By stage:

    bpe                     BPE merge dynamics (pair frequency, corpus size, compression);
                            bpe_dynamics.pdf, as the tokenizer has no model configuration
    pretrain                LM: loss, perplexity, accuracy, gradient norm, learning rate, and
                            the state space diagnostics, ONE LINE PER LAYER, named in a
                            DEDICATED LEGEND SUBPLOT of three columns rather than in-panel
    sft                     the same layout, so the two LM stages are directly comparable
    reward                  BCE, accuracies, class scores, logit margin
    instruct-rlhf           reward score, KL to the reference, PPO/value loss, entropy
    cot-grpo                reward, think length, group advantage, format adherence
    instruct-dpo            loss, implicit rewards, margin, accuracy, held-out probes
    distill                 CE, KL to the student's initialisation, perplexity, accuracy

Entry points:

    method_monitor(mode, records, plotdir, tag)   live redraw during training
    bpe_monitor(history, plotdir, tag)            the tokenizer's own build dynamics

House style throughout: a faint per-step series under a bold moving-average trend, axes
scaled to the trend rather than to outliers, scientific tick labels on the step axis.
"""
import os

from . import utils as _utils

C = {"bpe": "#7b2cbf", "pretrain": "#8338ec", "sft": "#6a994e", "dpo": "#d1495b",
     "reward": "#bc6c25", "rlhf": "#f18701", "cot": "#118ab2", "distill": "#3a86ff"}
NAME = {"bpe": "BPE", "pretrain": "Pretrain", "sft": "SFT", "dpo": "DPO",
        "reward": "Reward model", "rlhf": "RLHF (PPO)", "cot": "CoT (GRPO)",
        "distill": "Distill"}

# Within one stage's figure the stage colour identifies nothing (the filename already does),
# so chosen-versus-rejected contrasts get their own colourblind-safe pair (Okabe-Ito blue /
# vermillion) rather than one hue in two linestyles, which is unreadable at print size.
# Linestyle is kept as a redundant encoding.
C_CHOSEN = "#0072B2"
C_REJECT = "#D55E00"

GREY_NOTE = "#4B5963"      # in-panel annotations: never a data colour


def _sfx(tag):
    """Filename suffix for a run tag: '_<tag>' when one is given, '' when it is empty."""
    return f"_{tag}" if tag else ""


def _figure_path(plotdir, mode, tag):
    """outputs/plots/<stage>/dynamics_<model>_<pe>_<stage-label>.pdf -- the configuration AND
    the stage are in the FILENAME, so an arch="rope" ablation lands beside the run it is compared
    with rather than over it, and a figure dropped into a paper or an issue still says which
    stage produced it. Falls back to <stage>_dynamics.pdf when no run tag has been set."""
    run = tag or _utils.get_run_tag()
    name = (f"dynamics_{run}_{_utils.stage_label(mode)}.pdf" if run
            else f"{mode}_dynamics.pdf")
    return os.path.join(plotdir, name)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9.5, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})
    return plt


# --------------------------------------------------------------------------- #
# training visualization
# --------------------------------------------------------------------------- #
def _live_run(records):
    """Records from the last step restart onward.

    A history file is appended to across runs, so if a run is relaunched from scratch (or
    resumed from a checkpoint older than the last logged step) the "step" column restarts and
    the file holds several segments back to back. Plotting the concatenation draws a line from
    the end of the abandoned segment back to the start of the live one -- a long near-horizontal
    stroke across the early x-range in EVERY panel, which is not data. Keep the live segment."""
    if not records:
        return records
    cut = 0
    for i in range(1, len(records)):
        if records[i].get("step", 0) <= records[i - 1].get("step", 0):
            cut = i
    if cut:
        # the trainers truncate the history at the resume point precisely so this cannot
        # happen; if it does, the figure is about to show a fraction of the run and the
        # reader is entitled to know rather than to be shown a curve that starts late
        print(f"[plot] history is not monotone in step: discarding the first {cut} record(s) "
              f"and drawing from step {records[cut].get('step')}", flush=True)
    return records[cut:]


def _stage_figure(mode, records, plotdir, plt, tag):
    """Dispatch one stage's history to the layout that fits it. One dispatch point, used by
    both the live monitor and the end-of-stage pass, so the two can never disagree about
    which figure a stage owns."""
    if mode == "reward":
        return _reward_figure(mode, records, plotdir, plt, tag)
    if mode == "rlhf":
        return _rlhf_figure(mode, records, plotdir, plt, tag)
    if mode == "cot":
        return _cot_figure(mode, records, plotdir, plt, tag)
    if mode == "distill":
        return _distill_figure(mode, records, plotdir, plt, tag)
    if mode == "sft":
        return _sft_figure(mode, records, plotdir, plt, tag)
    if mode in ("pretrain", "cot_sft"):
        # LM stages: a plain likelihood objective, no chosen/rejected pair, so the preference
        # 1x5 layout does not apply -- they get the LM figure (loss / ppl / accuracy / grad norm).
        # cot_sft belongs here and NOT with cot: it is stage 10's supervised half, and what it
        # optimises is a likelihood, so the GRPO figure has no series to draw for it.
        return _lm_figure(mode, records, plotdir, plt, tag)
    return _method_figure(mode, records, plotdir, plt, tag)


def method_monitor(mode, records, plotdir, tag):
    """Live per-stage dynamics, redrawn every --plot_every_steps during training (PDF).
    Returns the path of the figure, or None if there is nothing to draw."""
    if not records:
        return None
    records = _live_run(records)
    plt = _plt()
    os.makedirs(plotdir, exist_ok=True)
    return _stage_figure(mode, records, plotdir, plt, tag)


def bpe_monitor(history, plotdir, tag=""):
    """bpe_dynamics<_tag>.pdf -- how the byte-level BPE vocabulary is LEARNED over its merges.

    `history` is BPETokenizer.build_history: one record per merge with the merged pair's corpus
    frequency, the number of distinct candidate pairs, the vocabulary size, the corpus token
    count, the new symbol's byte length and the compression ratio (bytes/token). Returns the
    figure path, or None if there is no history (e.g. an HF tokenizer, which has no merges)."""
    if not history:
        return None
    plt = _plt()
    os.makedirs(plotdir, exist_ok=True)
    return _bpe_figure(history, plotdir, plt, tag)


def _bpe_figure(history, plotdir, plt, tag):
    """1 row x 4: merged-pair frequency (log y), corpus size in tokens, compression (bytes/token),
    and distinct candidate pairs -- all against the merge index (the BPE "step"). x is the merge
    number; no main title, house style."""
    if not history:
        return None
    shown = plot_points([r["merge"] for r in history])
    history = [history[i] for i in shown]
    step = [r["merge"] for r in history]
    def col(k): return [r.get(k) for r in history]
    c = C["bpe"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.0))

    ax = axes[0]
    ax.plot(step, col("pair_freq"), color=c, lw=1.6)
    ax.set_yscale("log")
    ax.set_title("Merged-pair frequency", weight="bold")
    ax.set_xlabel("merge"); ax.set_ylabel("corpus count of merged pair"); _sci_x(ax)

    ax = axes[1]
    ax.plot(step, col("tokens"), color=c, lw=1.8)
    ax.set_title("Corpus size (tokens)", weight="bold")
    ax.set_xlabel("merge"); ax.set_ylabel("total tokens"); _sci_x(ax)

    ax = axes[2]
    ax.plot(step, col("bytes_per_token"), color=c, lw=1.8)
    ax.set_title("Compression (bytes / token)", weight="bold")
    ax.set_xlabel("merge"); ax.set_ylabel("mean token length (bytes)"); _sci_x(ax)

    ax = axes[3]
    ax.plot(step, col("distinct_pairs"), color=c, lw=1.8)
    ax.set_title("Distinct candidate pairs", weight="bold")
    ax.set_xlabel("merge"); ax.set_ylabel("#distinct adjacent pairs"); _sci_x(ax)

    fig.tight_layout()
    out = os.path.join(plotdir, f"bpe_dynamics{_sfx(tag)}.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def _lr_panel(ax, step, lrs, c):
    """Learning-rate schedule panel, shared by every stage figure (log y: a cosine decay to
    a tenth of the peak is unreadable on a linear axis next to the peak)."""
    pts = [(s, v) for s, v in zip(step, lrs) if v is not None]
    if pts:
        pts = [pts[i] for i in plot_points([s for s, _ in pts])]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=c, lw=1.8)
        if min(ys) > 0:
            ax.set_yscale("log")
    ax.set_title("Learning rate", weight="bold")
    ax.set_xlabel("step"); _sci_x(ax)


# key, title, y label, y limits, LOGARITHMIC y
#
# FOUR ARE LOGARITHMIC, AND FOR TWO DIFFERENT REASONS.
#
# The last three -- the mean decay and the two channel fractions -- span orders of magnitude
# NEAR THEIR FLOOR: a fraction going from 1e-5 to 3e-3 is a two-order-of-magnitude change that a
# linear axis draws as two points on one line. Memory horizon (p95) spans them near its CEILING:
# it is tau = -1/ln(a), which is 1/(1-a) to first order as a approaches 1, so a = 0.99 is about
# 100 tokens, 0.999 about 1,000 and 0.9999 about 10,000. On a linear axis one long-memory bank
# therefore flattens every other curve onto the x axis. The model's own initialisation spaces the
# banks GEOMETRICALLY over horizons of roughly 2 to 10,000 tokens (model/mira.py), so on a log
# axis the initialised banks are EVENLY SPACED lines and drift away from them is what the eye
# picks up -- which is the whole point of collecting the number.
#
# The cost, shared by all four, is that a value of EXACTLY zero has no place on the axis;
# _ssm_rows blanks it and says so in the panel rather than letting matplotlib drop it silently.
# The horizon is clamped off both ends at source (af.clamp(1e-6, 1 - 1e-7) in model/mira.py and
# model/ssm.py), so it cannot arrive as 0 or infinite; the blanking is the guard, not the fix.
SSM_PANELS = [
    ("horizon_med", "Memory horizon (median)", "tokens", None, False),
    ("horizon_p95", "Memory horizon (p95)", "tokens", None, True),
    ("selectivity", r"Selectivity: std$_t$ of the decay", None, None, False),
    ("write_ratio", "Residual write ratio", r"$\|$out$\|/\|$in$\|$", None, False),
    ("a_mean", "Mean decay $a$", None, None, True),
    ("frac_long", "Long-memory channels ($a > 0.99$)", "fraction", None, True),
    ("frac_local", "Local channels ($a < 0.5$)", "fraction", None, True),
    ("state_norm", "Mean state magnitude", r"$|h|$", None, False),
]

NCOL = 4          # at most four panels per row, in every figure this module draws


def _grid(n, plt, width_per=5.6, height_per=4.0, legend=False):
    """A grid of `n` panels, at most NCOL per row, with the unused slots removed rather than
    left blank. Returns (fig, [axes in reading order], legend axes or None).

    `legend=True` claims ONE MORE CELL, the one straight after the last panel, for the shared
    legend. It is a cell the panels were not using -- the LM figure with the state space
    diagnostics has thirteen panels in a four-by-four grid, so three sat empty -- and only when
    the panels happen to fill their last row exactly does `rows` go up by one to make room. The
    data panels are the same size either way: nothing is reflowed or shrunk to fit the legend in.

    The legend cell's axes are switched off entirely (frame, ticks, tick labels, spines and the
    background patch), so what shows in that cell is the legend and nothing else."""
    cells = n + (1 if legend else 0)
    rows = (cells + NCOL - 1) // NCOL
    fig, grid = plt.subplots(rows, NCOL, figsize=(width_per * NCOL, height_per * rows),
                             squeeze=False)
    axes = [grid[r][c] for r in range(rows) for c in range(NCOL)]
    for spare in axes[cells:]:
        fig.delaxes(spare)
    lax = axes[n] if legend else None
    if lax is not None:
        lax.set_axis_off()
    return fig, axes[:n], lax


def _ssm_rows(records, axes, plt):
    """Draw the state space diagnostics: ONE LINE PER LAYER, plus the across-layer MEAN.

    The history stores what was measured -- a list of per-layer dicts under "ssm", on the
    steps where collection was on -- and nothing else. The averaging happens here, at draw
    time, so the record stays raw and any later analysis can summarise it differently.

    Layers are coloured light to dark with depth; the mean is the bold dashed line on top.
    Both are drawn because they answer different questions: the mean says whether the module
    as a whole is learning to remember, the spread says whether the layers are specialising --
    an early block staying local while a late one grows its horizon is the interesting case,
    and the mean alone would hide it.

    RETURNS THE LEGEND HANDLES: the actual Line2D of each layer -- the first panel that drew
    that layer -- followed by the actual dashed mean line, for the figure's shared legend
    subplot. They are the drawn artists, not proxies built from a copy of the colour list, so a
    swatch in the legend cannot drift from the curve it names."""
    rows = [(r["step"], r["ssm"]) for r in records if r.get("ssm")]
    if not rows:
        return []
    rows = [rows[i] for i in plot_points([s for s, _ in rows])]
    step = [s for s, _ in rows]
    n_layer = max(len(v) for _, v in rows)
    cmap = plt.get_cmap("viridis")
    shades = [cmap(0.12 + 0.76 * (i / max(n_layer - 1, 1))) for i in range(n_layer)]
    def positive(series):
        """On a log axis a value of exactly zero cannot be drawn. Blank those points and count
        them, so the panel can report how many were withheld instead of showing a line that
        mysteriously starts part way along."""
        out = [(q if (q is not None and q > 0) else None) for q in series]
        return out, sum(1 for a, b in zip(series, out) if a is not None and b is None)

    # One handle per layer plus one for the mean, each taken from the FIRST panel that drew it.
    # A layer whose series is all-None in an early panel is skipped there and picked up from a
    # later one, so a layer that appears anywhere in the figure appears in the legend.
    handles = [None] * n_layer
    mean_handle = None
    for (key, title, ylab, ylim, logy), ax in zip(SSM_PANELS, axes):
        drawn, zeros = 0, 0
        for li in range(n_layer):
            y = [(v[li].get(key) if li < len(v) else None) for _, v in rows]
            if logy:
                y, nz = positive(y)
                zeros += nz
            if all(q is None for q in y):
                continue
            line, = ax.plot(step, y, color=shades[li], lw=1.1, alpha=0.85,
                            label=(f"layer {li}" if handles[li] is None else None))
            if handles[li] is None:
                handles[li] = line
            drawn += 1
        mean = []
        for _, v in rows:
            vals = [d.get(key) for d in v if d.get(key) is not None]
            mean.append(sum(vals) / len(vals) if vals else None)
        if logy:
            mean, _ = positive(mean)
        mline, = ax.plot(step, mean, color="k", lw=2.0, ls="--",
                         label=("mean over layers" if mean_handle is None else None))
        if mean_handle is None:
            mean_handle = mline
        if logy:
            ax.set_yscale("log")
            if drawn:
                if zeros:
                    ax.text(0.98, 0.03, f"{zeros:,} point(s) at exactly 0 omitted",
                            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                            color=GREY_NOTE, style="italic")
            else:
                # NOT A LIMIT ON ANY DATA -- there is none. A log axis with no positive point
                # cannot autoscale ("Data has no positive values"), so this is only the frame
                # the message below is written on.
                ax.set_ylim(1e-6, 1.05)
                ax.text(0.5, 0.5, "identically zero\nover the whole run",
                        transform=ax.transAxes, ha="center", va="center", fontsize=8,
                        color=GREY_NOTE, style="italic", linespacing=1.4)
        elif ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title, weight="bold"); ax.set_xlabel("step")
        if ylab:
            ax.set_ylabel(ylab)
        _sci_x(ax)
    return [h for h in handles if h is not None] + ([mean_handle] if mean_handle else [])


def _layer_legend(ax, handles):
    """The shared legend, alone in its own subplot, IN THREE COLUMNS.

        "for visualization legend. we use a dedicated subplot to show legend because there
         are so many layers. we use 3 columns for showing the legend"

    `handles` are the artists the panels actually drew (_ssm_rows returns them), so the swatch
    and the curve cannot disagree about a colour: they are the same Line2D."""
    if ax is None or not handles:
        return
    ax.legend(handles=handles, ncol=3, loc="center", frameon=False, fontsize=8.5,
              handlelength=1.6, handletextpad=0.5, columnspacing=1.4, labelspacing=0.5)


# THE BENCHMARK'S OWN DEFAULT SMOOTHING WINDOW, and it is imported at use rather than repeated.
SFT_SMOOTH_WINDOW = 1001


def _sft_figure(mode, records, plotdir, plt, tag):
    """Stage 7's figures: the benchmark's two, drawn by the benchmark's own code.

        dynamics_<tag>_sft.pdf        benchmark_comparison.pdf's 1 x 3 -- Token accuracy,
                                      Cross-entropy loss, Perplexity
        mira_dynamics_<tag>_sft.pdf   mira_dynamics.pdf's memory horizon, cycles and
                                      oscillation period, one line per layer

    NOTHING IS REDRAWN HERE. make_figure and draw_mira_dynamics ARE the benchmark's, called with
    this stage's history instead of ten families' -- so the panels, the +/-1 s.d. bands, the
    log-space perplexity, the percent axis, the decade ticks, the scientific step axis, the
    grids, the spines, the legend and the typography are not "the same as" the benchmark's, they
    are the benchmark's. A change to that figure arrives here with no edit.

    ONE SERIES, AND IT IS OURS. `series` overrides SERIES with a single entry carrying MIRA's own
    label, colour and solid stroke, so this run is drawn exactly as the MIRA line in the
    comparison is drawn -- same hue, same 1.9pt weight, same zorder.

    THE DYNAMICS FIGURE IS SKIPPED WHEN THERE IS NOTHING IN IT. Stage 7 runs profile=False --
    "for instruct sft, disable mira statistics profiling" -- so its dynamics.jsonl carries no
    `ssm` block and draw_mira_dynamics returns None, exactly as it does for a benchmark episode
    that has not profiled yet. Turn profiling back on and the panel appears with no change here.
    """
    if not records:
        return None
    import numpy as np
    from benchmark.visualize_benchmark import (SERIES, OURS_LABEL, draw_mira_dynamics,
                                               make_figure)
    step = np.asarray([r["step"] for r in records], dtype=float)
    history = {key: np.asarray([r.get(key, np.nan) for r in records], dtype=float)
               for key in ("acc", "loss", "ppl")}
    history["step"] = step
    out = _figure_path(plotdir, mode, tag)
    make_figure({OURS_LABEL: history}, int(step[-1]) + 1 if step.size else 1,
                min(SFT_SMOOTH_WINDOW, max(len(records) // 2, 1)), _Path(out),
                series={OURS_LABEL: SERIES[OURS_LABEL]})
    _sft_dynamics(records, plotdir, mode, tag, draw_mira_dynamics)
    _long_run_insights(records, plotdir, mode)
    _lattice_dynamics(records, plotdir)
    return out


def _sft_dynamics(records, plotdir, mode, tag, draw_mira_dynamics):
    """mira_dynamics_<tag>_sft.pdf, from this stage's own history, or nothing.

    read_mira_dynamics names ONE path -- the benchmark's MIRA run -- so the records are taken
    from the history this stage was handed instead, in the same shape: steps, and a per-step list
    of per-layer statistic dicts."""
    import numpy as np
    rows = [(r["step"], r["ssm"]) for r in records if r.get("ssm")]
    if not rows:
        return None
    steps = np.asarray([s for s, _ in rows], dtype=float)
    name = _figure_path(plotdir, mode, tag)
    return draw_mira_dynamics(steps, [layers for _, layers in rows],
                              _Path(os.path.join(os.path.dirname(name),
                                                 "mira_" + os.path.basename(name))))


def _Path(p):
    from pathlib import Path
    return Path(p)


def _lm_figure(mode, records, plotdir, plt, tag):
    """<mode>_dynamics<_tag>.pdf for the LM stages (pretrain, sft).

    Loss, Perplexity, Next-token accuracy, Gradient norm, Learning rate; and, when the history
    carries state space diagnostics (pretraining does, at --plot_every_steps), eight more
    panels with one line per layer and the across-layer mean over them, plus A DEDICATED
    LEGEND SUBPLOT in three columns naming every layer and the mean. Four panels per row,
    unused slots deleted. x is the training step throughout."""
    if not records:
        return None
    step = [r["step"] for r in records]
    def col(k): return [r.get(k) for r in records]
    c = C.get(mode, "#6a994e")
    has_ssm = any(r.get("ssm") for r in records)
    fig, axes, lax = _grid(5 + (len(SSM_PANELS) if has_ssm else 0), plt, legend=has_ssm)

    lm_panels = [("loss", "Loss (length-normalized LM)", "nats/token", None),
                 ("ppl", "Perplexity", None, None),
                 ("acc", "Next-token accuracy (response)", "accuracy", (0, 1.02)),
                 ("gnorm", "Gradient norm (pre-clip)", None, None)]
    for (key, title, ylab, ylim), ax in zip(lm_panels, axes):
        tr = _plot_trend(ax, step, col(key), c); _fit_to_trend(ax, [tr])
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title, weight="bold"); ax.set_xlabel("step")
        if ylab:
            ax.set_ylabel(ylab)
        _sci_x(ax)
    _lr_panel(axes[4], step, col("lr"), c)
    if has_ssm:
        _layer_legend(lax, _ssm_rows(records, axes[5:], plt))

    fig.tight_layout()
    out = _figure_path(plotdir, mode, tag)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    _long_run_insights(records, plotdir, mode)
    _lattice_dynamics(records, plotdir)
    return out


def _lattice_dynamics(records, plotdir):
    """horizon_dynamics.pdf, radius_dynamics.pdf and theta_dynamics.pdf beside this stage's
    figures (2026-10-08), the benchmark's per-layer heatmaps drawn from the records' `ssm` blocks.
    A failure here is reported and never costs the stage's other figures."""
    try:
        from benchmark.visualize_benchmark import (SERIES, OURS_LABEL, insight_run,
                                                   write_lattice_dynamics)
        run = insight_run(OURS_LABEL, SERIES[OURS_LABEL]["color"], [r["step"] for r in records],
                          [r.get("loss") for r in records], [r.get("acc") for r in records],
                          records=records)
        return write_lattice_dynamics([run], plotdir,
                                      lambda message: print(message, flush=True))
    except Exception as error:                                      # noqa: BLE001
        print(f"[plot] lattice dynamics skipped: {error}", flush=True)
        return None


def _long_run_insights(records, plotdir, mode):
    """<scheme>_long_run_insights.pdf (mira_s_long_run_insights.pdf) beside this stage's figures,
    plotted wherever the comparison/dynamics figures are (2026-10-06): the benchmark's insights
    PDF, drawn by the benchmark's own code from this run's history -- loss/accuracy, a checkpoint
    table, per-layer horizon/gate heatmaps from the records' `ssm` blocks. A failure here is
    reported and never costs the stage's other figures."""
    try:
        import numpy as np
        from benchmark.visualize_benchmark import (SERIES, OURS_LABEL, draw_long_run_insights,
                                                   insight_run, insights_filename)
        def col(key):
            return [np.nan if r.get(key) is None else float(r[key]) for r in records]
        run = insight_run(OURS_LABEL, SERIES[OURS_LABEL]["color"], [r["step"] for r in records],
                          col("loss"), col("acc"), records=records)
        return draw_long_run_insights([run], _Path(os.path.join(plotdir, insights_filename())),
                                      NAME.get(mode, mode))
    except Exception as error:                                      # noqa: BLE001
        print(f"[plot] long-run insights skipped: {error}", flush=True)
        return None


def _reward_figure(mode, records, plotdir, plt, tag):
    """reward_dynamics<_tag>.pdf -- 1 row x 6 for the reward-model stage (sigmoid + BCE,
    chosen -> 1, rejected -> 0). EVERY tracked value gets a panel:

        1 BCE loss
        2 accuracy: pair accuracy Pr(s_w > s_l) and per-class sign accuracies
        3 mean sigmoid score of the chosen and rejected responses (the two class means)
        4 mean logit margin s_w - s_l
        5 gradient norm (pre-clip)
        6 learning rate (cosine)
    """
    if not records:
        return None
    step = [r["step"] for r in records]
    def col(k): return [r.get(k) for r in records]
    c = C[mode]
    fig, axes = plt.subplots(1, 6, figsize=(27, 4.0))

    ax = axes[0]
    t = _plot_trend(ax, step, col("loss"), c); _fit_to_trend(ax, [t])
    ax.set_title("BCE loss", weight="bold"); ax.set_xlabel("step"); _sci_x(ax)

    ax = axes[1]
    _plot_trend(ax, step, col("pair_acc"), c, label=r"pair acc $\Pr(s_w>s_\ell)$")
    _plot_trend(ax, step, col("acc_w"), C_CHOSEN, label="chosen acc", lw=1.4, ls="--")
    _plot_trend(ax, step, col("acc_l"), C_REJECT, label="reject acc", lw=1.4, ls=":")
    ax.set_ylim(0, 1.02); ax.axhline(0.5, ls=":", color="k", alpha=0.4)
    ax.set_title("Classification accuracy", weight="bold")
    ax.set_xlabel("step"); ax.legend(frameon=False, fontsize=8); _sci_x(ax)

    ax = axes[2]
    tw = _plot_trend(ax, step, col("s_w"), C_CHOSEN, label=r"$\sigma(s_w)$ chosen")
    tl = _plot_trend(ax, step, col("s_l"), C_REJECT, label=r"$\sigma(s_\ell)$ rejected", ls="--")
    _fit_to_trend(ax, [tw, tl]); ax.set_ylim(0, 1.02)
    ax.axhline(0.5, ls=":", color="k", alpha=0.4)
    ax.set_title("Mean sigmoid score", weight="bold")
    ax.set_xlabel("step"); ax.legend(frameon=False, fontsize=8); _sci_x(ax)

    ax = axes[3]
    t = _plot_trend(ax, step, col("margin"), c); _fit_to_trend(ax, [t])
    ax.axhline(0, ls=":", color="k", alpha=0.4)
    ax.set_title(r"Logit margin $s_w - s_\ell$", weight="bold")
    ax.set_xlabel("step"); _sci_x(ax)

    ax = axes[4]
    t = _plot_trend(ax, step, col("gnorm"), c); _fit_to_trend(ax, [t])
    ax.set_title("Gradient norm (pre-clip)", weight="bold")
    ax.set_xlabel("step"); _sci_x(ax)

    _lr_panel(axes[5], step, col("lr"), c)

    fig.tight_layout()
    out = _figure_path(plotdir, mode, tag)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def _rlhf_figure(mode, records, plotdir, plt, tag):
    """rlhf_dynamics<_tag>.pdf -- 2 rows x 4 for the PPO stage. EVERY tracked value gets a
    panel:

        1 mean reward-model score of the rollouts (the number PPO maximises)
        2 per-token KL(policy || SFT reference) of the rollouts (what the penalty holds down)
        3 PPO policy (clipped-surrogate) loss
        4 value loss
        5 total loss (policy + vf_coef * value)
        6 policy entropy of the rollouts (collapse indicator)
        7 mean response length (collapse indicator)
    """
    if not records:
        return None
    step = [r["step"] for r in records]
    def col(k): return [r.get(k) for r in records]
    c = C[mode]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.0))

    panels = [("reward", "Reward-model score (rollouts)", None, axes[0, 0]),
              ("kl_ref", r"Per-token KL$(\pi\,\|\,\pi_{\mathrm{sft}})$", "nats/token", axes[0, 1]),
              ("pg_loss", "PPO policy loss", None, axes[0, 2]),
              ("v_loss", "Value loss", None, axes[0, 3]),
              ("loss", "Total loss", None, axes[1, 0]),
              ("entropy", "Policy entropy", "nats/token", axes[1, 1]),
              ("resp_len", "Mean response length", "tokens", axes[1, 2])]
    _lr_panel(axes[1, 3], step, col("lr"), c)
    for key, title, ylab, ax in panels:
        t = _plot_trend(ax, step, col(key), c); _fit_to_trend(ax, [t])
        if key in ("reward", "kl_ref", "pg_loss", "loss"):
            ax.axhline(0, ls=":", color="k", alpha=0.4)
        ax.set_title(title, weight="bold"); ax.set_xlabel("step")
        if ylab:
            ax.set_ylabel(ylab)
        _sci_x(ax)

    fig.tight_layout()
    out = _figure_path(plotdir, mode, tag)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def _cot_figure(mode, records, plotdir, plt, tag):
    """cot_dynamics<_tag>.pdf -- 4 rows x 4 for the chain-of-thought GRPO stage. EVERY quantity
    the stage records has a panel, because a run of this kind fails in several different ways
    and each one is visible in exactly one of them.

    THE FIRST ROW IS WHAT THE STAGE EXISTS TO SHOW.
        1 THINK LENGTH, mean -- words between the think tags. The reported "aha moment" is a
          rising trend here: the policy discovers that deliberating longer earns reward.
        2 THINK LENGTH, longest and 90th percentile. The mean HIDES THE ARRIVAL: longer
          deliberation appears in a few completions first, and a mean over 24 moves by a word
          or two while the best has doubled.
        3 VERIFIED ACCURACY -- extracted answer against gold. Unlike a reward-model score this
          cannot be gamed by sounding fluent.
        4 mean verifier reward, what GRPO maximises.

    THE SECOND ROW IS WHETHER GRPO CAN LEARN AT ALL.
        5 DEAD GROUPS -- the fraction whose completions all scored the same. Their advantages
          are zero, so they contribute nothing; a run sitting near 1.0 here is doing rollouts
          for no gradient, and no other panel will say so. THE FIRST THING TO READ when reward
          will not move.
        6 within-group reward spread, the denominator of the advantage.
        7 mean |advantage|, the size of the signal actually reaching the policy.
        8 format rate: well-formed think/answer structure.

    THE THIRD ROW IS THE REWARD'S PARTS AND THE CONSTRAINT.
        9 grounded rate: thinking long enough AND mentioning the problem's numbers.
        10 reflection rate: a self-correction marker ("wait", "actually") -- a crude proxy.
        11 response length in tokens, beside the think length in words.
        12 KL to the frozen reference (k3), the constraint being paid.

    THE FOURTH ROW IS OPTIMISATION HEALTH.
        13 GRPO policy loss, 14 total loss, 15 policy entropy (collapse indicator),
        16 clipped fraction; the gradient norm and the learning rate share the last panel.
    """
    if not records:
        return None
    step = [r["step"] for r in records]
    def col(k): return [r.get(k) for r in records]
    c = C[mode]
    fig, ax2d = plt.subplots(4, 4, figsize=(18, 15.0))
    axes = [a for row in ax2d for a in row]
    panels = [("think_len", "Think length, mean (the aha-moment curve)", "words"),
              (("think_len_max", "think_len_p90"), "Think length, longest and p90", "words"),
              ("accuracy", "Verified answer accuracy", "fraction correct"),
              ("reward", "Verifier reward", None),
              ("dead_groups", "Dead groups (no spread, no gradient)", "fraction of groups"),
              ("reward_std", "Within-group reward spread", None),
              ("adv_abs", "Mean |advantage|", None),
              (("think_fmt", "answer_fmt"), "Tag rates, think and answer",
               "fraction of rollouts"),
              ("grounded", "Grounded thinking rate", "fraction of rollouts"),
              ("aha", "Reflection rate", "fraction of rollouts"),
              ("resp_len", "Response length", "tokens"),
              ("kl_ref", r"KL$(\pi\,\|\,\pi_{\mathrm{ref}})$", "nats/token"),
              ("pg_loss", "GRPO policy loss", None),
              ("loss", "Total loss", None),
              ("entropy", "Policy entropy", "nats/token"),
              ("clip_frac", "Clipped fraction", "fraction of tokens")]
    UNIT = ("accuracy", "aha", "format", "clip_frac", "dead_groups", "grounded",
            "think_fmt", "answer_fmt")
    for (key, title, ylab), ax in zip(panels, axes):
        if isinstance(key, tuple):
            # two series in one panel: the same hue would make them one line
            labs = {"think_len_max": "longest", "think_len_p90": "p90",
                    "think_fmt": "<think>", "answer_fmt": "<answer>"}
            ts = [_plot_trend(ax, step, col(k), cc, label=labs.get(k, k), ls=st)
                  for k, cc, st in zip(key, (c, "#888888"), ("-", "--"))]
            if all(k in UNIT for k in key):
                ax.set_ylim(0, 1.02)
            _fit_to_trend(ax, ts)
            if any(any(v is not None for v in col(k)) for k in key):
                ax.legend(fontsize=8, frameon=False)
        else:
            t = _plot_trend(ax, step, col(key), c); _fit_to_trend(ax, [t])
            if key in UNIT:
                ax.set_ylim(0, 1.02)
            if key in ("reward", "pg_loss", "loss", "kl_ref", "adv_abs", "reward_std"):
                ax.axhline(0, ls=":", color="k", alpha=0.4)
        ax.set_title(title, weight="bold"); ax.set_xlabel("step")
        if ylab:
            ax.set_ylabel(ylab)
        _sci_x(ax)
    _lr_panel(axes[15], step, col("lr"), c)
    fig.tight_layout()
    out = _figure_path(plotdir, mode, tag)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def _distill_figure(mode, records, plotdir, plt, tag):
    """distill_dynamics<_tag>.pdf -- 2 rows x 4 for the sequence-level distillation stage
    (teacher generates, student maximises the likelihood of the generation under a KL
    constraint to its own frozen initialisation):

        1 total loss  CE + kl_coef * KL
        2 CE on the teacher's generations (nats/token)
        3 KL to the frozen student initialisation (the constraint being paid)
        4 perplexity of the teacher text under the student
        5 next-token accuracy of the student on the teacher text
        6 mean teacher response length (student tokens)
        7 gradient norm (pre-clip)
        8 learning rate (cosine)
    """
    if not records:
        return None
    step = [r["step"] for r in records]
    def col(k): return [r.get(k) for r in records]
    c = C[mode]
    fig, ax2d = plt.subplots(2, 4, figsize=(18, 8.0))
    axes = [a for row in ax2d for a in row]
    panels = [("loss", "Total loss", None),
              ("ce", "CE on teacher generations", "nats/token"),
              ("kl_ref", r"KL$(\pi_{\mathrm{student}}\,\|\,\pi_{\mathrm{init}})$",
               "nats/token"),
              ("ppl", "Perplexity", None),
              ("acc", "Next-token accuracy (student)", "accuracy"),
              ("resp_len", "Teacher response length", "tokens"),
              ("gnorm", "Gradient norm (pre-clip)", None)]
    for (key, title, ylab), ax in zip(panels, axes):
        t = _plot_trend(ax, step, col(key), c); _fit_to_trend(ax, [t])
        if key == "acc":
            ax.set_ylim(0, 1.02)
        if key == "kl_ref":
            ax.axhline(0, ls=":", color="k", alpha=0.4)
        ax.set_title(title, weight="bold"); ax.set_xlabel("step")
        if ylab:
            ax.set_ylabel(ylab)
        _sci_x(ax)
    _lr_panel(axes[7], step, col("lr"), c)
    fig.tight_layout()
    out = _figure_path(plotdir, mode, tag)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def _sci_x(ax):
    """Scientific tick format on the x-axis, typeset as a power of ten: the offset reads
    $\\times 10^{4}$ rather than the plain-text '1e4'."""
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)


def _method_figure(mode, records, plotdir, plt, tag):
    """dynamics_<tag>_<stage-label>.pdf -- 2 rows x 4, EVERY tracked value of the preference
    stage.

    Row 1: Loss, Implicit reward (r_w / r_l), Reward margin, Preference/chosen/reject
    accuracy.
    Row 2 (held-out probes, sparse series filtered to the probed steps, then optimisation):
    accuracy, margin, gradient norm, learning rate.
    x-axis is the training step (scientific format). No main title."""
    if not records:
        return None
    step = [r["step"] for r in records]
    def col(k): return [r.get(k) for r in records]
    c = C[mode]
    fig, ax2d = plt.subplots(2, 4, figsize=(18.0, 8.0))
    axes = [a for row in ax2d for a in row]
    ax = axes[0]
    t = _plot_trend(ax, step, col("loss"), c)
    _fit_to_trend(ax, [t])
    ax.set_title("Loss", weight="bold"); ax.set_xlabel("step"); _sci_x(ax)
    ax = axes[1]
    tw = _plot_trend(ax, step, col("r_w"), C_CHOSEN, label="chosen $r_w$")
    tl = _plot_trend(ax, step, col("r_l"), C_REJECT, label="rejected $r_\\ell$", ls="--")
    _fit_to_trend(ax, [tw, tl])
    ax.axhline(0, ls=":", color="k", alpha=0.4)
    ax.set_title("Implicit reward", weight="bold"); ax.set_xlabel("step")
    ax.legend(frameon=False, fontsize=8); _sci_x(ax)
    ax = axes[2]
    t = _plot_trend(ax, step, col("margin"), c); _fit_to_trend(ax, [t])
    ax.axhline(0, ls=":", color="k", alpha=0.4)
    ax.set_title("Reward margin", weight="bold"); ax.set_xlabel("step"); _sci_x(ax)
    ax = axes[3]
    # acc = Pr(margin > 0) is the PREFERENCE accuracy -- whether the pair is ranked
    # (row 2 starts here)
    # correctly. win_acc = Pr(r_w > 0) and lose_acc = Pr(r_l < 0) are the signs of the two
    # rewards, i.e. the LEVEL, and carry no ranking information: both rewards negative gives
    # (win, lose) = (0, 1) whether the ranking is perfect, chance, or inverted. All three
    # series are drawn, with a chance line, so the level and the ranking can be read apart.
    _plot_trend(ax, step, col("acc"), c, label=r"acc $=\Pr(m>0)$")
    _plot_trend(ax, step, col("win_acc"), C_CHOSEN, label="chosen acc", lw=1.4, ls="--")
    _plot_trend(ax, step, col("lose_acc"), C_REJECT, label="reject acc", lw=1.4, ls=":")
    ax.set_ylim(0, 1.02); ax.axhline(0.5, ls=":", color="k", alpha=0.4)
    ax.set_title("Preference / chosen / reject accuracy", weight="bold")
    ax.set_xlabel("step"); ax.legend(frameon=False, fontsize=8); _sci_x(ax)
    # ---- held-out probes (sparse: every --eval_every / --plot_every_steps) ---- #
    def _sparse(ax, key, title, ylab, chance=None, zero=False):
        pts = [(r["step"], r[key]) for r in records if r.get(key) is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", ms=3, lw=1.6, color=c)
            _fit_to_trend(ax, [list(ys)])
        if chance is not None:
            ax.axhline(chance, ls=":", color="k", alpha=0.4); ax.set_ylim(0, 1.02)
        if zero:
            ax.axhline(0, ls=":", color="k", alpha=0.4)
        ax.set_title(title, weight="bold"); ax.set_xlabel("step")
        if ylab:
            ax.set_ylabel(ylab)
        _sci_x(ax)

    _sparse(axes[4], "eval_acc", "Held-out preference accuracy", "accuracy", chance=0.5)
    _sparse(axes[5], "eval_margin", "Held-out reward margin", None, zero=True)

    ax = axes[6]
    t = _plot_trend(ax, step, col("gnorm"), c); _fit_to_trend(ax, [t])
    ax.set_title("Gradient norm (pre-clip)", weight="bold")
    ax.set_xlabel("step"); _sci_x(ax)

    _lr_panel(axes[7], step, col("lr"), c)

    fig.tight_layout()
    out = _figure_path(plotdir, mode, tag)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def plot_points(steps, window=None):
    import numpy as np
    import default_config as config
    s = np.asarray(steps, dtype=float)
    if s.size == 0:
        return np.zeros(0, dtype=int)
    w = max(int(config.BENCHMARK["smooth_window"] if window is None else window), 1)
    bucket = np.floor((s - s[0]) / w)
    keep = np.flatnonzero(np.r_[True, bucket[1:] != bucket[:-1]])
    if keep[-1] != s.size - 1:
        keep = np.r_[keep, s.size - 1]
    return keep


def _moving_average(y, w=None):
    """Trailing/centered moving average of a per-step series (Nones -> nan, ignored).
    Window auto-scales to the run length so long runs get smooth trends."""
    import numpy as np
    a = np.asarray([np.nan if v is None else v for v in y], dtype=float)
    n = a.size
    if n == 0:
        return a
    if w is None:
        w = max(5, n // 200)             # ~0.5% of the run
    # np.convolve(..., mode="same") returns max(len(a), w) points, not len(a). A run shorter
    # than the window therefore produced a TREND LONGER THAN ITS OWN x-AXIS and every caller
    # died in ax.plot with "x and y must have same first dimension" -- swallowed by the live
    # monitor, fatal in the end-of-run pass. Clamp the window to the series.
    w = min(w, n)
    if w <= 1 or n < 3:
        return a
    mask = ~np.isnan(a)
    a0 = np.where(mask, a, 0.0)
    k = np.ones(w)
    num = np.convolve(a0, k, mode="same")
    den = np.convolve(mask.astype(float), k, mode="same")   # count of valid points per window
    return num / np.maximum(den, 1.0)


def _moving_deviation(y, w=None):
    """The standard deviation of the SAME window `_moving_average` averages over.

        var = mean(x^2) - mean(x)^2,   over the identical NaN-aware window

    IT SHARES THE ESTIMATOR SO THE BAND IS AROUND THE LINE. A deviation computed over a
    different window, or centred where the mean is trailing, sits visibly off the curve wherever
    the curve is steep -- which on a loss curve is exactly where the reader is looking. Reusing
    `_moving_average` for both moments makes that impossible: the same window, the same NaN
    handling, the same length.

    CLIPPED AT ZERO because the subtraction is not exact in floating point: on a flat stretch the
    two means agree to the last bits and the difference can land at -1e-17, whose square root is
    not a number."""
    import numpy as np
    a = np.asarray([np.nan if v is None else v for v in y], dtype=float)
    mean = _moving_average(y, w)
    mean_square = _moving_average(a * a, w)
    return np.sqrt(np.clip(mean_square - mean * mean, 0.0, None))


def exponential_mean(values, span):
    """A causal EMA without an expanding-to-fixed-window boundary kink.

    MOVED HERE FROM benchmark/visualize_benchmark SO EVERY FIGURE SHARES ONE IMPLEMENTATION.

    The benchmark, the ablation, the experimental sweeps and the per-stage trends were drawing
    from two estimators in two files; this is the shared layer, so it is the one place both
    can reach. benchmark.visualize_benchmark re-exports the name, which is how six modules
    already import it."""
    import numpy as np
    if span < 1:
        raise ValueError("smooth-window must be at least 1")
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    alpha = 2.0 / (span + 1.0)
    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for index in range(1, values.size):
        smoothed[index] = alpha * values[index] + (1.0 - alpha) * smoothed[index - 1]
    return smoothed


def exponential_deviation(values, span):
    """The standard deviation of the SAME estimator `exponential_mean` computes.

        var = EMA(x^2) - EMA(x)^2

    THE BAND AND THE LINE MUST SHARE AN ESTIMATOR OR THE BAND IS NOT AROUND THE LINE. A centred
    rolling window would give a tighter deviation, but the line here is a CAUSAL EMA and lags by
    about half a span; a centred band around a lagging line sits visibly off it wherever the
    curve is steep, which on a loss curve is exactly where the reader is looking.

    CLIPPED AT ZERO because the subtraction is not exact in floating point: on a flat stretch the
    two EMAs agree to the last bits and the difference can land at -1e-17, whose square root is
    not a number."""
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    mean = exponential_mean(values, span)
    mean_square = exponential_mean(values * values, span)
    return np.sqrt(np.clip(mean_square - mean * mean, 0.0, None))


def _series(y):
    """A per-step series as float, with None and missing values as NaN. The shape every
    estimator here takes, so a caller never has to know which one it got."""
    import numpy as np
    return np.asarray([np.nan if v is None else v for v in y], dtype=float)


def _span(n, w=None):
    """The window `_moving_average` would choose, so both estimators smooth by the same amount
    and switching between them changes the SHAPE of the curve and not its scale."""
    return max(1, min(int(w) if w else max(5, n // 200), max(n, 1)))


def _ema_mean(y, w=None):
    """`exponential_mean` with `_moving_average`'s signature and its NaN handling.

    NaNs ARE CARRIED, NOT FED IN. One NaN in a recursion poisons every value after it, so a
    missing step holds the previous estimate instead of destroying the rest of the curve --
    which is what the window estimator does too, by ignoring it in its average."""
    import numpy as np
    a = _series(y)
    if a.size == 0:
        return a
    filled, last = np.empty_like(a), np.nan
    for i, v in enumerate(a):
        if not np.isnan(v):
            last = v
        filled[i] = last
    if np.isnan(filled).all():
        return a
    first = np.flatnonzero(~np.isnan(filled))[0]
    filled[:first] = filled[first]
    return exponential_mean(filled, _span(a.size, w))


def _ema_deviation(y, w=None):
    """The deviation of `_ema_mean`, over the same series and the same span."""
    import numpy as np
    a = _series(y)
    mean = _ema_mean(y, w)
    mean_square = _ema_mean(a * a, w)
    return np.sqrt(np.clip(mean_square - mean * mean, 0.0, None))


def _block_points(y, w, reduce):
    import numpy as np
    a = _series(y)
    out = np.full(a.size, np.nan)
    if a.size == 0:
        return out
    width = _span(a.size, w)
    ends = np.arange(width - 1, a.size // width * width, width)
    if ends.size == 0:
        return out
    with np.errstate(all="ignore"):
        values = reduce(a[:ends[-1] + 1].reshape(-1, width), axis=1)
    out[ends[0]:ends[-1] + 1] = np.interp(np.arange(ends[0], ends[-1] + 1), ends, values)
    return out


def _block_mean(y, w=None):
    import numpy as np
    return _block_points(y, w, np.nanmean)


def _block_deviation(y, w=None):
    import numpy as np
    return _block_points(y, w, np.nanstd)


def smoothers(method=None):
    """(mean, deviation) for the configured estimator, for EVERY figure in the repository.

        ema     _ema_mean          /  _ema_deviation          -- the default
        mavg    _moving_average    /  _moving_deviation
        block   _block_mean        /  _block_deviation        -- consecutive blocks of the
                window, each block's mean at its last step, joined by straight lines

    Refused rather than defaulted: a misspelt method that silently fell back would draw a figure
    whose caption said one estimator and whose curve was the other.

    THE SETTING LIVES IN BENCHMARK AND GOVERNS EVERYTHING, which is the convention `smooth_window`
    already set -- ablation/run.py reads config.BENCHMARK["smooth_window"] for its own figures.
    One key, one answer, rather than a second one per stage that can disagree."""
    import default_config as config
    name = str(config.BENCHMARK["plot_smooth_method"] if method is None else method).strip().lower()
    if name == "ema":
        return _ema_mean, _ema_deviation
    if name == "mavg":
        return _moving_average, _moving_deviation
    if name == "block":
        return _block_mean, _block_deviation
    raise SystemExit(
        f"[plot] PLOT_SMOOTH_METHOD {name!r} is not a method; expected 'mavg', 'ema' or 'block'")


def _plot_trend(ax, step, y, color, label=None, lw=1.8, ls="-"):
    """A moving-average trend with a +/- 1 standard-deviation band around it.
    Returns the trend series so the caller can scale the y-axis to the trends.

    THE BAND REPLACES THE RAW PER-STEP LINE, WHICH WAS DRAWN UNDER IT AT alpha=0.1.

    The faint layer was meant to show detail under the trend. With several series on one panel
    it stopped doing that: on the benchmark histories one step moves the loss by about 0.115
    while the whole spread between families at the end is 0.1 to 0.4, so the layer that was
    supposed to add detail covered the result it was drawn under. The band reports the same
    quantity -- how much a step varies -- without occluding anything.

    ONLY THE LINE CHANGES. No axis, label, title, limit, tick formatter, legend or colour is
    touched here or by any caller; `_fit_to_trend` still receives the trend and scales to it
    exactly as before, so the y-axis is the one it always was."""
    import numpy as np
    mean_of, deviation_of = smoothers()
    t = mean_of(y)
    sigma = deviation_of(y)
    shown = plot_points(step)
    x = np.asarray(step, dtype=float)[shown]
    ax.fill_between(x, (t - sigma)[shown], (t + sigma)[shown], color=color, alpha=0.18,
                    linewidth=0)
    ax.plot(x, t[shown], color=color, lw=lw, ls=ls, label=label)
    return t


def _fit_to_trend(ax, trends, pad=0.12, keep_zero=True):
    """Scale the y-axis to the moving-average trends, not to the raw per-step scatter.

    Rationale: methods whose per-step values are heavy-tailed (e.g. RH-DPO under heavy
    masking, |r| up to ~150 while the trend moves in [-46,-20]) otherwise get an axis
    ~50x the trend span, which collapses the chosen/rejected trends onto one another.
    The raw scatter is still drawn and simply clips. `keep_zero` re-includes the y=0
    reference line only when it is close to the trend band (within `pad*8` of a limit)."""
    import numpy as np
    v = np.concatenate([np.asarray(t, dtype=float).ravel() for t in trends if t is not None])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0) * 0.1
    # A log axis cannot represent 0 or a negative limit. Additive padding around a band whose
    # bottom is small relative to its span drives the lower limit negative, matplotlib refuses
    # the whole set_ylim ("non-positive ylim on a log-scaled axis will be ignored"), and the
    # panel silently falls back to autoscale -- or, if the scale is set AFTER this call, keeps
    # a negative limit and renders blank. Pad multiplicatively instead, and never return 0.
    if ax.get_yscale() == "log":
        v = v[v > 0]
        if v.size == 0:
            return
        lo, hi = float(v.min()), float(v.max())
        f = (hi / lo) ** pad if hi > lo else 10.0 ** pad
        ax.set_ylim(lo / f, hi * f)
        return
    if keep_zero and lo - span * pad * 8 <= 0 <= hi + span * pad * 8:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        # RECOMPUTING span here can undo the span <= 0 fallback above. A trend that is
        # identically zero gives lo = hi = 0, the clamp is a no-op, and span goes back to 0, so
        # set_ylim(0, 0) is called with identical limits -- matplotlib warns
        # ("transformation singular"), discards them and autoscales. The panel still rendered,
        # which is why this survived: the warning was the only symptom. Constant NON-zero trends
        # were never affected, because the clamp widens them to include 0.
        span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0) * 0.1
    ax.set_ylim(lo - span * pad, hi + span * pad)




