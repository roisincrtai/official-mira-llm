"""kernel/cuda/rwkv_v6/check.py -- does our RWKV-6 WKV reproduce the PyTorch reference?

Run by `make check`. THIS FILE DID NOT EXIST until 2026-08-15 while the Makefile's `check:`
target already pointed at it, so `make check` failed for this family and the kernel reached a
published comparison verified only by numpy transcription on a machine with no GPU.

    S_t[i][j] = w_t[j] S_{t-1}[i][j] + k_t[j] v_t[i]
    y_t[i]    = sum_j r_t[j] ( u[j] k_t[j] v_t[i] + S_{t-1}[i][j] )

THE REFERENCE IS THE DEFINITION, written here as a step-by-step loop rather than borrowed from
profiler/rwkv_v6.py. The profiler's fallback uses pairwise_scan over a broadcast (H, dh, dh)
memory, which is the same function by a different route -- and checking a kernel against another
optimisation checks the two against each other, not either against the maths.
"""
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

# HEAD DIMS THE KERNEL IS ACTUALLY BUILT FOR. 70 and 72 are the widths matched_width picks for
# RWKV-6 against MIRA; 16 is small enough to catch an indexing bug that a round number hides.
# B=1 T=33 is deliberately ragged -- a prime-ish length exercises the tail of every loop.
SHAPES = ((2, 64, 2, 16), (1, 33, 3, 64), (2, 128, 12, 68),
          (2, 128, 8, 70), (4, 256, 8, 72))


def _cases(torch):
    from kernel.cuda.rwkv_v6.rwkv_v6 import wkv6

    def wkv(B, T, H, N):
        f = lambda: torch.randn(B, T, H, N, device="cuda") * 0.3
        r, k, v = f(), f(), f()
        lw = -torch.exp(f())                       # LOG decay, negative: w = exp(lw) in (0,1)
        u = torch.randn(H, N, device="cuda") * 0.3

        def ref(rr, kk, vv, ww, uu):
            """The recurrence exactly as the .cu header states it."""
            S = torch.zeros(B, H, N, N, device="cuda", dtype=torch.float32)
            out = []
            for t in range(T):
                x = kk[:, t].unsqueeze(-2) * vv[:, t].unsqueeze(-1)      # [i][j] = k_j v_i
                out.append((rr[:, t].unsqueeze(-2)
                            * (uu.unsqueeze(0).unsqueeze(-2) * x + S)).sum(-1))
                S = S * ww[:, t].exp().unsqueeze(-2) + x
            return torch.stack(out, 1)

        def cu(rr, kk, vv, ww, uu):
            return wkv6(rr, kk, vv, ww, uu)

        return ref, cu, (r, k, v, lw, u)

    return (("rwkv6 wkv", wkv),)


def main(timing=True, shapes=None):
    """`make check` calls this as a script; `make unittest` imports it and passes timing=False.

    A FUNCTION RATHER THAN A BARE __main__ BLOCK so the correctness half can be asked for
    WITHOUT a second copy of this file's setup living in a Makefile recipe. The body is the
    block that was here, unchanged.

    `shapes` DEFAULTS TO ALL FIVE AND IS NARROWED BY ONE CALLER ONLY -- smoke(), below."""
    try:
        import torch
        from kernel.cuda.rwkv_v6 import rwkv_v6 as m
        return run("rwkv_v6", _cases(torch), shapes or SHAPES, describe=(m,), timing=timing)
    except ImportError:
        return run("rwkv_v6", (), shapes or SHAPES, timing=timing)


def smoke():
    """`make quicktest`: EVERY KERNEL, ONE SHAPE, THE PRECISION ERROR AND NOTHING ELSE.

    IT IS THE SAME COMPARISON, ASKED FOR LESS. main() drives checker.run(), which prints the
    absolute error and the ULP count for the value and for every gradient and returns 0 or 1;
    this passes the smallest shape and timing=False, and changes nothing else. There is no
    second copy of the comparison here and there must never be one -- see the header of
    kernel/cuda/checker.py for what that cost the last time.

    timing=False IS WHY THE SPEED TABLE AND build/kernel/cuda/rwkv_v6/timings.json STAY OUT OF
    IT. A millisecond is a property of the card and of how busy it was, so it is not something a
    job script can gate on; and writing the baseline from a quick test would keep rebasing the
    `vs prev` column against whatever node was free. `make check` is where speed is the subject.

    EXIT STATUS IS run()'s, so a disagreement is non-zero and the caller can fail on it."""
    return main(timing=False, shapes=(smallest(SHAPES),))


if __name__ == "__main__":
    sys.exit(main())
