"""kernel/cuda/rwkv_v7/check.py -- does our RWKV-7 WKV reproduce the PyTorch reference?

Run by `make check`. THIS FILE DID NOT EXIST until 2026-08-15 while the Makefile's `check:`
target already pointed at it -- see kernel/cuda/checker.py for what that cost.

    w_t[j]    = exp(W_SCALE * sigmoid(wraw_t[j])) ,  W_SCALE = -exp(-1/2)
    sa_t[i]   = sum_j S_{t-1}[i][j] a_t[j]                     the delta-rule read
    S_t[i][j] = w_t[j] S_{t-1}[i][j] + sa_t[i] b_t[j] + k_t[j] v_t[i]
    y_t[i]    = sum_j S_t[i][j] r_t[j]                         the NEW state, unlike v6

TWO THINGS THIS CHECK EXISTS TO CATCH, and both are silent:

  - y reads S_t and NOT S_{t-1}. Off by one step and the forward is still smooth, still
    trains, and is not RWKV-7.
  - the backward carries TWO routes into S_{t-1}, through w_t and through a_t via sa_t.
    Drop the second and only the GRADIENTS are wrong -- which is why this file checks every
    one of the six rather than the value alone.

`w` is passed RAW: the kernel applies exp(W_SCALE * sigmoid(.)) internally, so the reference
must too, or the two are not one function measured twice.
"""
import math
import os
import sys

# THE REPOSITORY ROOT, FOUND BY SEARCHING RATHER THAN BY COUNTING. This was
#     os.path.dirname(os.path.dirname(os.path.dirname(...)))
# -- three levels, correct when this file was kernel/cuda/check.py. It moved one level deeper
# into kernel/cuda/<family>/ and the arithmetic was never updated, so it put kernel/ on the
# path and `import kernel` then failed with ModuleNotFoundError. `make check` was broken from
# that move until 2026-08-15 and nothing noticed, because nothing ran it. Counting directories
# breaks silently when a file moves; looking for the root does not.
_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isdir(os.path.join(_d, "model")):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)

from kernel.cuda.checker import run, smallest                      # noqa: E402

# 60 and 64 retain the historical parameter-matching cases. 73 is the production fixed-depth
# match for the 12-layer MIRA-Newton benchmark (d_model=584 over 8 heads).
SHAPES = ((2, 64, 2, 16), (1, 33, 3, 32), (2, 128, 8, 60),
          (2, 128, 11, 64), (4, 256, 8, 64), (2, 128, 8, 73))
W_SCALE = -math.exp(-0.5)


def _cases(torch):
    from kernel.cuda.rwkv_v7.rwkv_v7 import wkv7

    def wkv(B, T, H, N):
        f = lambda s=0.3: torch.randn(B, T, H, N, device="cuda") * s
        r, wraw, k, v = f(), f(), f(), f()
        # a AND b ARE SCALED BY 1/sqrt(N), AND THAT IS NOT COSMETIC. The delta rule adds
        # sa.b^T, a rank-one term whose gain is ~ ||a||.||b|| ~ N.sigma^2 -- at N=64 and
        # sigma=0.3 that is ~5.8 PER STEP, so the state grows geometrically and reached 514
        # by T=256 on the first hardware run. Nothing was wrong with the kernel; the check
        # was driving the recurrence into a regime the architecture never enters.
        #
        # REAL RWKV-7 BUILDS THESE FROM A NORMALISED KEY -- a = -kappa, b = kappa.alpha with
        # kappa unit-norm -- which is exactly what makes the update contractive. 1/sqrt(N)
        # reproduces that scaling without importing the model's parameterisation, and keeps
        # |y| ~ 2 instead of ~500, which is the regime the kernel has to be right in.
        a, b = f(0.3 / math.sqrt(N)), f(0.3 / math.sqrt(N))

        def ref(rr, ww, kk, vv, aa, bb):
            S = torch.zeros(B, H, N, N, device="cuda", dtype=torch.float32)
            out = []
            for t in range(T):
                w = torch.exp(W_SCALE * torch.sigmoid(ww[:, t]))
                sa = (S * aa[:, t].unsqueeze(-2)).sum(-1)             # read S_{t-1}
                S = (S * w.unsqueeze(-2)
                     + sa.unsqueeze(-1) * bb[:, t].unsqueeze(-2)
                     + kk[:, t].unsqueeze(-2) * vv[:, t].unsqueeze(-1))
                out.append((S * rr[:, t].unsqueeze(-2)).sum(-1))      # the NEW state
            return torch.stack(out, 1)

        def cu(rr, ww, kk, vv, aa, bb):
            return wkv7(rr, ww, kk, vv, aa, bb)

        return ref, cu, (r, wraw, k, v, a, b)

    return (("rwkv7 wkv", wkv),)


def main(timing=True, shapes=None):
    """`make check` calls this as a script; `make unittest` imports it and passes timing=False.

    A FUNCTION RATHER THAN A BARE __main__ BLOCK so the correctness half can be asked for
    WITHOUT a second copy of this file's setup living in a Makefile recipe. The body is the
    block that was here, unchanged.

    `shapes` DEFAULTS TO ALL SIX AND IS NARROWED BY ONE CALLER ONLY -- smoke(), below."""
    try:
        import torch
        from kernel.cuda.rwkv_v7 import rwkv_v7 as m
        return run("rwkv_v7", _cases(torch), shapes or SHAPES, describe=(m,), timing=timing)
    except ImportError:
        return run("rwkv_v7", (), shapes or SHAPES, timing=timing)


def smoke():
    """`make quicktest`: EVERY KERNEL, ONE SHAPE, THE PRECISION ERROR AND NOTHING ELSE.

    IT IS THE SAME COMPARISON, ASKED FOR LESS. main() drives checker.run(), which prints the
    absolute error and the ULP count for the value and for every gradient and returns 0 or 1;
    this passes the smallest shape and timing=False, and changes nothing else. There is no
    second copy of the comparison here and there must never be one -- see the header of
    kernel/cuda/checker.py for what that cost the last time.

    timing=False IS WHY THE SPEED TABLE AND build/kernel/cuda/rwkv_v7/timings.json STAY OUT OF
    IT. A millisecond is a property of the card and of how busy it was, so it is not something a
    job script can gate on; and writing the baseline from a quick test would keep rebasing the
    `vs prev` column against whatever node was free. `make check` is where speed is the subject.

    EXIT STATUS IS run()'s, so a disagreement is non-zero and the caller can fail on it."""
    return main(timing=False, shapes=(smallest(SHAPES),))


if __name__ == "__main__":
    sys.exit(main())
