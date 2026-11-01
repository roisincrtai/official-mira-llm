"""kernel/cuda/mamba_s6/check.py -- does our selective scan reproduce the PyTorch reference?

Run by `make check`. THIS FILE DID NOT EXIST until 2026-08-15 while the Makefile's `check:`
target already pointed at it -- see kernel/cuda/checker.py for what that cost.

    dA_t = exp(delta_t A)                      (d_in, N) per step
    h_t  = dA_t . h_{t-1} + delta_t B_t u_t    h in R^{d_in x N}
    y_t  = sum_n h_t[:,n] C_t[n] + D u_t
    out  = y . SiLU(z)

THE ONE THAT BIT BEFORE, and it is why every gradient is checked rather than the value: d/du
and d/dB were both WRONG on the first derivation, each off by a factor of u, because the write
term delta*B*u has each factor exactly once and I had counted u twice. The forward was correct
throughout -- only the gradients were not, and nothing in a forward-only check would have seen it.

LAYOUT, matching mamba_ssm.ops.selective_scan_interface.selective_scan_fn:
    u, delta, z, out   (B, d_in, T)      A (d_in, N)      D (d_in)      B_, C_ (B, N, T)
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

# N = 16 is Mamba's default d_state and the only one the profile uses; 8 and 32 exercise the
# other built instantiations. d_in is NOT a multiple of the 256-thread block in the third case,
# which is what catches an indexing bug that round numbers hide.
SHAPES = ((2, 64, 32, 16), (1, 33, 15, 8), (2, 256, 128, 16), (4, 512, 256, 32))


def _cases(torch):
    import torch.nn.functional as F
    from kernel.cuda.mamba_s6.mamba_s6 import selective_scan

    def scan(B, T, d, N):
        u = torch.randn(B, d, T, device="cuda") * 0.3
        dlt = F.softplus(torch.randn(B, d, T, device="cuda") * 0.3)   # as the mixer feeds it
        A = -torch.exp(torch.randn(d, N, device="cuda") * 0.3)
        Bm = torch.randn(B, N, T, device="cuda") * 0.3
        Cm = torch.randn(B, N, T, device="cuda") * 0.3
        Dv = torch.randn(d, device="cuda") * 0.3
        z = torch.randn(B, d, T, device="cuda") * 0.3

        def ref(uu, dd, AA, BB, CC, DD, zz):
            h = torch.zeros(B, d, N, device="cuda", dtype=torch.float32)
            out = []
            for t in range(T):
                dA = torch.exp(dd[:, :, t].unsqueeze(-1) * AA.unsqueeze(0))
                h = dA * h + (dd[:, :, t] * uu[:, :, t]).unsqueeze(-1) * BB[:, :, t].unsqueeze(1)
                y = (h * CC[:, :, t].unsqueeze(1)).sum(-1) + DD.unsqueeze(0) * uu[:, :, t]
                out.append(y * F.silu(zz[:, :, t]))
            return torch.stack(out, -1)

        def cu(uu, dd, AA, BB, CC, DD, zz):
            return selective_scan(uu, dd, AA, BB, CC, DD, zz)

        return ref, cu, (u, dlt, A, Bm, Cm, Dv, z)

    return (("ms6 selective_scan", scan),)


def main(timing=True, shapes=None):
    """`make check` calls this as a script; `make unittest` imports it and passes timing=False.

    A FUNCTION RATHER THAN A BARE __main__ BLOCK so the correctness half can be asked for
    WITHOUT a second copy of this file's setup living in a Makefile recipe. The body is the
    block that was here, unchanged.

    `shapes` DEFAULTS TO ALL FOUR AND IS NARROWED BY ONE CALLER ONLY -- smoke(), below."""
    try:
        import torch
        from kernel.cuda.mamba_s6 import mamba_s6 as m
        return run("mamba_s6", _cases(torch), shapes or SHAPES, describe=(m,), timing=timing)
    except ImportError:
        return run("mamba_s6", (), shapes or SHAPES, timing=timing)


def smoke():
    """`make quicktest`: EVERY KERNEL, ONE SHAPE, THE PRECISION ERROR AND NOTHING ELSE.

    IT IS THE SAME COMPARISON, ASKED FOR LESS. main() drives checker.run(), which prints the
    absolute error and the ULP count for the value and for every gradient and returns 0 or 1;
    this passes the smallest shape and timing=False, and changes nothing else. There is no
    second copy of the comparison here and there must never be one -- see the header of
    kernel/cuda/checker.py for what that cost the last time.

    timing=False IS WHY THE SPEED TABLE AND build/kernel/cuda/mamba_s6/timings.json STAY OUT OF
    IT. A millisecond is a property of the card and of how busy it was, so it is not something a
    job script can gate on; and writing the baseline from a quick test would keep rebasing the
    `vs prev` column against whatever node was free. `make check` is where speed is the subject.

    EXIT STATUS IS run()'s, so a disagreement is non-zero and the caller can fail on it."""
    return main(timing=False, shapes=(smallest(SHAPES),))


if __name__ == "__main__":
    sys.exit(main())
