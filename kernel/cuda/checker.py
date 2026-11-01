"""kernel/cuda/checker.py -- the shared `make check` harness: value AND every gradient, then speed.

    from kernel.cuda.checker import run
    sys.exit(run("rwkv_v6", cases, SHAPES))

WHY THIS FILE EXISTS, and it is not tidiness. On 2026-08-15 three kernel families were added --
mamba_s6, rwkv_v6, rwkv_v7 -- each with a Makefile carrying a `check:` target that ran
`$(SRCDIR)/check.py`. THAT FILE WAS NEVER WRITTEN. So `make check` failed outright for three of
the four families, and old_scripts/stage5_compile_cuda_kernel.sh asserted, in a comment, that

    "The baselines are checked the same way by profiler/baselines.py at run time."

which is FALSE. profiler/baselines.py runs the kernel and reports which path it took; it never
compares against a reference and never touches a gradient. Those three kernels therefore went
into a published comparison having been verified ONLY by numpy transcription on a machine with
no GPU -- never once on hardware, in value or in gradient.

THE REFERENCE IS THE DEFINITION, so the only question that matters is whether the two agree, in
the VALUE and in EVERY GRADIENT -- a backward that is subtly wrong trains a subtly wrong model
and shows up as nothing until the loss curve is unexplainable.

SPEED IS MEASURED ONLY AFTER CORRECTNESS. A faster kernel that disagrees is worthless.

Exit status is 0 when every kernel agrees and 1 when any does not, so `make check` fails loudly.

`make unittest` RUNS THE SAME COMPARISON AND STOPS BEFORE THE SPEED TABLE -- run(..., timing=
False), same cases, same shapes, same tolerance, same exit status. Correctness is what a job
script can gate on; a millisecond is a property of the card and of how busy it was.

The output format matches kernel/cuda/mira-sdpa/check.py line for line, deliberately: four families
printing four different tables would make the one that matters harder to read, not easier.
"""
import json
import math
import os
import time

# fp32's unit of least precision, as a relative step. torch.finfo(torch.float32).eps, written
# out so this module can be imported and reasoned about without torch.
EPS32 = 1.1920928955078125e-07

# HOW MANY ULP A REDUCTION IS ALLOWED, as a function of how many terms it reduces.
#
# THE RULE ABOVE THIS ONE IS NOT WRONG, IT IS INCOMPLETE. `err <= tol * max(1, |ref|max)` judges
# a POINTWISE result, where each output element is a short chain of operations and the two
# implementations differ by a rounding or two. A GRADIENT WITH RESPECT TO A SHARED PARAMETER is
# not that: every element of dL/dcoeff is a sum over the whole batch and the whole sequence, and
# the kernel and the reference sum in different orders. Floating-point addition is not
# associative, so their difference grows with the number of terms -- as a random walk, ~sqrt(n)
# ulp, in the absence of catastrophic cancellation -- and a bound that does not know n cannot
# tell "summed 8,192 terms in a different order" from "computed the wrong thing".
#
# WHAT MADE THIS CONCRETE. mira_higher_order_iir's coefficient-logit gradient, at the largest
# correctness shape (8, 1024, 512): 3.20e-04 against a reference whose largest element is ~30,
# i.e. 89 ulp, where the same tensor is 21 ulp at (1, 33, 7) and 61 at (2, 1024, 128). The
# growth is exactly sqrt(B*T/N) and the kernel is not the loose side of it -- it accumulates its
# per-chunk partials in DOUBLE and reduces them deterministically (see the comment above
# mira_hiir_bwd_point); the reference is a Python loop accumulating T terms in fp32. The flat
# relative bound put the line at 3.0e-04 and the run FAILED by 7%, which is a false failure: it
# was reporting the reference's own summation error and calling it a kernel defect.
#
# THE CONSTANTS ARE HEADROOM OVER THE OBSERVED WALK, NOT A FITTED CURVE. 4*sqrt(n) is about
# three times the 1.4*sqrt(n) measured above, and the +8 keeps a small reduction from being held
# to a tighter standard than a pointwise result. A reduction that is genuinely WRONG -- a missed
# term, a race, a mis-strided partial -- is wrong by a factor, not by three times the rounding
# noise, so this does not buy a broken kernel a pass.
ULP_BASE = 8.0
ULP_PER_ROOT = 4.0


def allowance(mag, fan_in, tol):
    """The largest error this comparison may show, and the ulp figure it was judged by.

    `mag`     |ref|max, the scale the error is relative to
    `fan_in`  how many terms were summed into each element -- 1 for a pointwise result
    returns   (bound, allowed_ulp or None); None means the pointwise rule applied and there is
              nothing extra for the report to say.

    THE POINTWISE RULE IS STILL THE FLOOR. A reduction is allowed the LARGER of the two bounds,
    never the smaller, so no case is judged more loosely than it would have been on a shape
    where its fan-in is one."""
    pointwise = tol * max(1.0, mag)
    if fan_in <= 1:
        return pointwise, None
    ulp = ULP_BASE + ULP_PER_ROOT * math.sqrt(fan_in)
    return max(pointwise, ulp * EPS32 * mag), ulp


def fan_in_of(out_numel, grad_numel):
    """Terms summed per element of a gradient: the output's size over the gradient's.

    EXACT FOR THE CASE THAT MATTERS AND AN OVER-ESTIMATE FOR NO OTHER. A gradient with the same
    shape as its input has fan-in 1 and takes the pointwise rule unchanged; dL/dcoeff at
    (C, N) against an output of (B, T, C) has fan-in B*T/N, which is precisely how many products
    are accumulated into each of its entries."""
    return 1.0 if not grad_numel else max(1.0, out_numel / grad_numel)

# WHERE THE PREVIOUS RUN'S TIMINGS LIVE. Under build/, beside the cubins, because that is
# already the directory this kernel's artefacts occupy and `make clean` removes only *.cubin --
# so a baseline survives the rebuild that `run_profiling_attention.sh` does on every job. It is
# NOT in the repository proper: a timing is a property of the card it was taken on, and
# committing one would invite comparing an L40S against an A100 and calling the difference a
# regression.
BASELINE = "timings.json"


def _baseline_path(family):
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d) and not os.path.isdir(os.path.join(d, "model")):
        d = os.path.dirname(d)
    return os.path.join(d, "build", "kernel", "cuda", family, BASELINE)


def load_baseline(family):
    try:
        with open(_baseline_path(family)) as f:
            return json.load(f)
    except Exception:                                              # noqa: BLE001
        return {}


def save_baseline(family, rows):
    p = _baseline_path(family)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(rows, f, indent=1, sort_keys=True)
        return p
    except Exception as e:                                         # noqa: BLE001
        return f"(not written: {type(e).__name__}: {e})"


def delta(base, key, now):
    """This run against the last, as a percentage. NEGATIVE IS FASTER.

    THE SPEEDUP COLUMN CANNOT ANSWER THIS. `speedup` is ours against the PyTorch reference, and
    optimising the kernel moves that number for two different reasons -- the kernel got faster,
    or the reference got slower on a busier card. When the work is 'make this kernel faster than
    it was yesterday', the quantity is the kernel's own millisecond count against its own
    previous millisecond count, and nothing else on this table reports it."""
    old = base.get(key)
    if not old:
        return "      new"
    d = (now - old) / old * 100.0
    return f"{d:>+8.1f}%" if abs(d) >= 0.5 else "       ~"


def _bar(total, desc):
    """Return a line-oriented counter; CUDA checks are commonly captured in profiler logs."""
    return _PlainBar(total, desc)


class _PlainBar:
    """Minimal progress sink used by noninteractive profiler logs."""

    def __init__(self, total, desc):
        self.total = total
        self.desc = desc
        self.n = 0

    def update(self, count=1):
        self.n += count

    def close(self):
        print(f"  {self.desc}: {self.n}/{self.total} measurements", flush=True)


def _ms(fn, bar, seconds=0.3):
    """Milliseconds per call, on the GPU clock.

    SYNCHRONISED BOTH SIDES. A CUDA launch is asynchronous, so timing without
    torch.cuda.synchronize() measures how fast Python can queue work."""
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    one = time.perf_counter() - t0
    iters = max(3, min(2000, int(seconds / max(one, 1e-6))))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    bar.update(1)
    return (time.perf_counter() - t0) / iters * 1e3


def smallest(shapes):
    """The cheapest of `shapes`, by element count. What `make quicktest` runs, in every family.

    BY SIZE AND NOT BY INDEX, so reordering a family's SHAPES cannot silently change which shape
    the quick test uses. It is a SUBSET of SHAPES and never a shape of its own: a quick test that
    invented a size would be checking something no other target checks, and a disagreement it
    found could not be reproduced by `make unittest`.

    Here rather than in each family's check.py because all three baseline families ask the same
    question of the same tuple, and kernel/cuda/checker.py's header is the record of what a
    second copy of a check costs. kernel/cuda/mira-sdpa/check.py keeps its own one-line expression:
    its shapes are triples, it does not import this module, and smoke() predates this."""
    def size(s):
        n = 1
        for v in (s.values() if isinstance(s, dict) else s):
            n *= v
        return n
    return min(shapes, key=size)


def run(family, cases, shapes, tol=1e-5, describe=(), timing=True):
    """Check every case at every shape, print value and per-gradient error, then time them.

    `cases`   ((name, build), ...) where build(*shape) -> (ref_fn, cuda_fn, inputs)
    `shapes`  ((...), ...) passed straight to build; each family names its own axes
    `describe` modules with a describe() to print, so the log says which cubin was loaded
    `timing`  False stops after the correctness verdict -- `make unittest`

    TIMING IS OPTIONAL AND CORRECTNESS IS NOT, which is the whole difference between the two
    targets that call this. `make check` answers "do they agree, and how much faster?"; `make
    unittest` answers only the first half, because that is the half a job script can gate on. A
    millisecond is a property of the card and of how busy it is; an ULP is not. Nothing above
    this line changes -- the same cases, the same shapes, the same tolerance, the same exit
    status -- so the two targets cannot disagree about whether a kernel is correct."""
    try:
        import torch
    except Exception:                                            # noqa: BLE001
        print("FAIL: no torch -- the kernels are built, but this check cannot run without it.\n")
        print("  Alliance Canada (torch is pip-installed, not a module):")
        print("      module load python/3.12.4")
        print("      python3.12 -m venv $SLURM_TMPDIR/venv/check")
        print("      source $SLURM_TMPDIR/venv/check/bin/activate")
        print("      pip install --no-index torch")
        print("      make check\n")
        print("  MeluXina (torch IS a module):")
        print("      module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0-whl")
        print("      make check\n")
        print("  The cubins are already built and are unaffected by any of this.")
        return 1
    if not torch.cuda.is_available():
        print("FAIL: no CUDA device. salloc --gres=gpu:1 (howto.txt section 1).")
        return 1

    for m in describe:
        print("  " + m.describe())
    print(f"device: {torch.cuda.get_device_name()} "
          f"(sm_{''.join(str(x) for x in torch.cuda.get_device_capability())})\n")

    bad = 0
    for name, build in cases:
        print(f"{name}")
        miss = 0
        for shape in shapes:
            torch.manual_seed(0)
            ref_fn, cu_fn, xs = build(*shape)
            r_in = [x.clone().requires_grad_() for x in xs]
            c_in = [x.clone().requires_grad_() for x in xs]
            ref = ref_fn(*r_in)
            got = cu_fn(*c_in)
            tag = " ".join(f"{k}={v:<5}" for k, v in shape.items()) \
                if isinstance(shape, dict) else " ".join(f"{v:<5}" for v in shape)
            if got is None:
                print(f"  {tag} kernel unavailable -- build it: make")
                miss = 1
                break
            g = torch.randn_like(ref)
            ref.backward(g)
            got.backward(g)
            # VALUE FIRST, THEN EVERY GRADIENT, each named by its input position -- the same
            # shape of report mira/check.py gives, because a reader comparing two families
            # should not have to learn two formats.
            # ABSOLUTE ERROR AND ULPs. The absolute number is what TOL compares, but on its
            # own it cannot answer "is that large?" -- 1.4e-06 is a catastrophe on a quantity
            # of size 1e-4 and is 12 ULP on a quantity of size 1. So the magnitude of the
            # REFERENCE is divided out and the error reported in units of the last place:
            #
            #     ulp = |ref - got|max / (eps * |ref|max)
            #
            # THE TWO SIDES CANNOT AGREE EXACTLY, BY CONSTRUCTION. pairwise_scan combines in
            # log2(T) tree rounds; the kernel accumulates sequentially within a chunk and then
            # applies a carry. Same maths, different SUMMATION ORDER, and floating-point
            # addition is not associative. A length-T fp32 accumulation lands between
            # sqrt(T)*eps and T*eps -- at T=1024 that is 3.8e-06 to 1.2e-04 -- so a few tens
            # of ULP is the expected answer and ZERO would be the surprising one.
            _eps = torch.finfo(torch.float32).eps
            def _pair(a_t, b_t, fan=1.0):
                e = (a_t - b_t).abs().max().item()
                m = a_t.abs().max().item()
                return e, (e / (_eps * m) if m > 0 else 0.0), m, fan
            d = [("value ", *_pair(ref, got))]
            for i, (a_, b_) in enumerate(zip(r_in, c_in)):
                if a_.grad is not None and b_.grad is not None:
                    d.append((f"grad{i} ",
                              *_pair(a_.grad, b_.grad,
                                     fan_in_of(ref.numel(), a_.grad.numel()))))
            ok = True
            cols = []
            for k, v, u, m, fan in d:
                lim, ulp = allowance(m, fan, tol)
                ok &= v <= lim
                cols.append(f"{k}{v:.2e} ({u:>4.0f}"
                            + (f"/{ulp:.0f}ulp)" if ulp is not None else "ulp)"))
            bad += not ok
            print(f"  {tag} " + "  ".join(cols) + ("   OK" if ok else "   MISMATCH"))
        if miss:
            bad += 1
        print()

    if bad:
        print(f"FAILED: {bad} kernel/shape(s) disagree with the reference by more than "
              f"{tol:g} relative -- or, for a gradient reduced over many terms, by more than "
              f"the fan-in allowance printed after the slash.")
        print(f"DO NOT USE THEM -- the {family} baseline would be reporting a wrong number.")
        return 1

    print(f"PASS: every {family} kernel agrees with its reference to better than {tol:g} "
          f"relative (reduced gradients: within their fan-in allowance), in value and in "
          f"every gradient.\n")
    if not timing:
        return 0
    print("SPEED, milliseconds per call (lower is better)\n")
    base = load_baseline(family)
    rows = {}
    print(f"  {'kernel':<20}{'shape':<24}{'PyTorch':>10}{'ours':>9}{'speedup':>10}"
          f"{'vs prev':>10}  pass")
    print("  " + "-" * 84)
    bar = _bar(len(cases) * len(shapes) * 4, "timing")
    for name, build in cases:
        for shape in shapes:
            torch.manual_seed(0)
            ref_fn, cu_fn, xs = build(*shape)
            r_in = [x.clone().requires_grad_() for x in xs]
            c_in = [x.clone().requires_grad_() for x in xs]
            if cu_fn(*c_in) is None:
                continue

            def fwd_bwd(fn, ins):
                out = fn(*ins)
                out.backward(torch.ones_like(out), retain_graph=False)
                for t in ins:
                    t.grad = None
                return out

            f_ref = _ms(lambda: ref_fn(*[x.detach() for x in r_in]), bar)
            f_cu = _ms(lambda: cu_fn(*[x.detach() for x in c_in]), bar)
            b_ref = _ms(lambda: fwd_bwd(ref_fn, r_in), bar)
            b_cu = _ms(lambda: fwd_bwd(cu_fn, c_in), bar)
            tag = " ".join(f"{v}" for v in shape)
            kf, kb = f"{name}|{tag}|fwd", f"{name}|{tag}|fwd+bwd"
            rows[kf], rows[kb] = f_cu, b_cu
            print(f"  {name:<20}{tag:<24}{f_ref:>10.3f}{f_cu:>9.3f}{f_ref / f_cu:>9.2f}x"
                  f"{delta(base, kf, f_cu):>10}  fwd")
            print(f"  {'':<20}{'':<24}{b_ref:>10.3f}{b_cu:>9.3f}{b_ref / b_cu:>9.2f}x"
                  f"{delta(base, kb, b_cu):>10}  fwd+bwd")
    bar.close()
    print("\n  Timed with torch.cuda.synchronize() on both sides -- a CUDA launch is")
    print("  asynchronous, and timing without it measures Python, not the GPU.")
    # NOT WHILE INSTRUMENTED. ncu serialises and replays every launch, so check.py run
    # under it reports numbers 10x too slow -- measured 0.919 ms against 0.089 for
    # mira_fused_iir on 2026-08-15. Writing those as the baseline makes the NEXT
    # run's `vs prev` read as a 90% improvement that never happened. The profiling script
    # sets this before invoking ncu.
    if os.environ.get("MIRALLM_NO_BASELINE") == "1":
        print("  timings NOT saved: MIRALLM_NO_BASELINE=1 (this run is instrumented).")
        return 0
    where = save_baseline(family, rows)
    if base:
        print(f"  `vs prev` compares OUR millisecond count against the previous run's, from")
        print(f"  {where}. Negative is faster. It is not the speedup column: that one moves when")
        print(f"  the PyTorch reference moves, and a busier card can flatter a kernel that")
        print(f"  did not change.")
    else:
        print(f"  No previous timings, so this run IS the baseline -- written to {where}.")
        print(f"  Run `make check` again after a change and the `vs prev` column fills in.")
    print(f"  ONE CARD PER BASELINE: the file is under build/ and is not committed, because a")
    print(f"  millisecond is a property of the GPU it was measured on.")
    return 0
