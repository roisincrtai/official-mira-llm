"""kernel/cuda/rwkv_v8/check.py -- does UPSTREAM's RWKV-8 kernel reproduce the recurrence?

Run by `make check`; `make unittest` imports main(timing=False). Same harness, same table and
same exit status as every other family -- kernel/cuda/checker.py.

WHAT IS BEING CHECKED HERE IS NOT OURS, AND THAT CHANGES WHAT THE CHECK MEANS. For mira,
mamba_s6, rwkv_v6 and rwkv_v7 the .cu is written in this repository and the check is how we
find out whether we got it right. Here the kernel is BlinkDL's, vendored unmodified, so this
check answers a different and still necessary question: WAS IT COMPILED AND WIRED CORRECTLY ON
THIS CARD -- right -D_C_, right chunk length, right dtype, right argument order, and a
backward that matches the forward. A wiring mistake looks exactly like a wrong kernel from the
loss curve.

THE RECURRENCE, read off upstream/cuda/wkv7_cuda.cu's forward_kernel and not off a paper:

    w_t[j]    = exp(-exp(w_raw_t[j]))                       upstream's line 21, exactly
    sa_t[i]   = sum_j a_t[j] S_{t-1}[i][j]                  the delta-rule read
    S_t[i][j] = w_t[j] S_{t-1}[i][j] + sa_t[i] b_t[j] + k_t[j] v_t[i]
    y_t[i]    = sum_j S_t[i][j] q_t[j]                      the NEW state

Two things worth saying about that, because both are silent when wrong:

  - y reads S_t, not S_{t-1}. Off by one step the forward is still smooth and still trains.
  - the decay is exp(-exp(w)), NOT the exp(W_SCALE * sigmoid(w)) that OUR kernel/cuda/rwkv_v7
    takes. Those are two different parameterisations of the same idea and the reference has to
    use upstream's, or the two sides are not one function measured twice.

EVERYTHING IS fp32. Upstream's WindBackstepping asserts float32 on every input and their .cu's
scalar type is `float`; RWKV-8's kernel is RWKV-7's with the bf16 typedef replaced.

A MISSING KERNEL IS NOT A FAILURE. Upstream compiles at import with cpp_extension.load, which
needs nvcc and a host compiler on THIS node; when that is not available the cases list is
empty and the status is 0, exactly as kernel/cuda/rwkv_v7/check.py does when its module cannot
be imported. `make unittest` in a job script must not fail because a login node has no CUDA.
"""
import os
import sys

# THE REPOSITORY ROOT, FOUND BY SEARCHING RATHER THAN BY COUNTING -- see the same block in
# kernel/cuda/rwkv_v7/check.py for what counting directories cost the last time a file moved.
_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isdir(os.path.join(_d, "model")):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)

from kernel.cuda.checker import run                                # noqa: E402

# HEAD SIZE IS 16 IN EVERY SHAPE, because upstream compiles -D_C_=16 and reshapes with a
# literal 16. T IS ALWAYS A MULTIPLE OF 16, because upstream asserts T % CHUNK_LEN == 0.
SHAPES = ((2, 64, 2, 16), (1, 32, 4, 16), (2, 128, 8, 16), (4, 256, 16, 16))


def _cases(torch):
    from kernel.cuda.rwkv_v8 import rwkv_v8_upstream as K

    def wkv(B, T, H, N):
        f = lambda s=0.3: torch.randn(B, T, H, N, device="cuda", dtype=torch.float32) * s
        q, wraw, k, v = f(), f(), f(), f()
        # a AND b ARE SCALED BY 1/sqrt(N) FOR THE SAME REASON THEY ARE IN rwkv_v7/check.py:
        # the delta rule adds a rank-one term of gain ~ ||a||.||b||, so unscaled at N=16 the
        # state grows geometrically and the check ends up measuring a recurrence the
        # architecture never enters. Real RWKV-7/8 builds these from a unit-norm key --
        # a = -kappa, b = kappa*alpha -- which is what makes the update contractive.
        a, b = f(0.3 / (N ** 0.5)), f(0.3 / (N ** 0.5))

        def ref(qq, ww, kk, vv, aa, bb):
            S = torch.zeros(B, H, N, N, device="cuda", dtype=torch.float32)
            out = []
            for t in range(T):
                w = torch.exp(-torch.exp(ww[:, t]))                # upstream's line 21
                sa = (S * aa[:, t].unsqueeze(-2)).sum(-1)          # read S_{t-1}
                S = (S * w.unsqueeze(-2)
                     + sa.unsqueeze(-1) * bb[:, t].unsqueeze(-2)
                     + kk[:, t].unsqueeze(-2) * vv[:, t].unsqueeze(-1))
                out.append((S * qq[:, t].unsqueeze(-2)).sum(-1))   # the NEW state
            return torch.stack(out, 1)

        def cu(qq, ww, kk, vv, aa, bb):
            # UPSTREAM TAKES (B, T, H*C) AND SPLITS THE HEADS ITSELF, with a literal 16. The
            # reshape here is the wrapper's job, not theirs, and it is differentiable, so the
            # gradients the harness compares are the gradients of the same six inputs.
            flat = [x.reshape(B, T, H * N) for x in (qq, ww, kk, vv, aa, bb)]
            return K.run_cuda_rwkv7g(*flat).reshape(B, T, H, N)

        return ref, cu, (q, wraw, k, v, a, b)

    return (("rwkv8 wkv (upstream)", wkv),)


def main(timing=True):
    """`make check` runs this as a script; `make unittest` imports it with timing=False."""
    try:
        import torch                                               # noqa: F401
        from kernel.cuda.rwkv_v8 import rwkv_v8_upstream as m
    except ImportError:
        return run("rwkv_v8", (), SHAPES, timing=timing)
    if m.build() is None:
        # NOT A FAILURE. upstream_extension has already printed the reason once.
        print(f"  {m.describe()}")
        print("  nothing to check: upstream's kernel could not be built on this node.")
        print("  build it where nvcc and a C++ compiler are: make -C kernel/cuda/rwkv_v8\n")
        return run("rwkv_v8", (), SHAPES, timing=timing)
    return run("rwkv_v8", _cases(torch), SHAPES, describe=(m,), timing=timing)


if __name__ == "__main__":
    sys.exit(main())
