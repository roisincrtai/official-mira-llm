"""kernel/cuda/rwkv_v7/rwkv_v7.py -- OUR RWKV-7 WKV. No flash-linear-attention.

    from kernel.cuda.rwkv_v7.rwkv_v7 import wkv7
    y = wkv7(r, w, k, v, a, b)          # (B, T, H, N) each. None when unavailable

THE STATE IS SAVED EVERY `CHUNK` STEPS AND THE BACKWARD INVERTS THE RECURRENCE BETWEEN THEM,
so the memory the backward holds is (B, H, T/CHUNK, N, N) rather than (B, T, H, N, N) -- 1.45
GiB rather than 23.14 GiB at the benchmark's geometry. Nothing about the forward, the model or
the checkpoint changes; see rwkv_v7.cu for the inversion and its error bound.

SIX INPUTS, NOT FIVE, AND NO BONUS. RWKV-7 replaced v6's `u` term with the delta rule: the
state is corrected by its own read before it is written, through the (a, b) pair.

    sa_t[i] = sum_j S_{t-1}[i][j] a_t[j]
    S_t[i][j] = w_t[j] S_{t-1}[i][j] + sa_t[i] b_t[j] + k_t[j] v_t[i]
    y_t[i]   = sum_j S_t[i][j] r_t[j]                       <- the NEW state, unlike v6

`w` is passed RAW: the kernel applies w = exp(W_SCALE * sigmoid(w_raw)) itself, as the authors'
kernel does, so the gradient of the sigmoid stays inside and the caller cannot get it wrong.

Layout is (B, T, H, N) throughout, matching fla.ops.rwkv7 with head_first=False, so
profiler/baselines.py calls one or the other with no reshaping.

See rwkv_v7.cu for the maths, the row-per-thread state and the verification.
"""
from .. import driver as D

CUBIN = "rwkv_v7/rwkv_v7.cubin"
# ONE ENTRY POINT PER HEAD SIZE: N is a template parameter in the .cu, so the
# state row can live in registers. A head dim not in this tuple is refused and
# reported -- never truncated.
SIZES = (16, 32, 48, 60, 64, 68, 70, 72, 73, 80, 96, 128)
# HOW OFTEN THE STATE IS SAVED FOR THE BACKWARD, in steps. The backward recovers every state in
# between by inverting the recurrence, so this is a pure memory/arithmetic trade and nothing
# about the model or the gradient depends on it.
#
# 16 IS THE AUTHORS' VALUE (upstream/op_x070.py:57), AND IT IS ALSO WHERE THE TRADE STOPS BEING
# FREE. The error a backstep amplifies is bounded by 1/w < 1.834 per step and re-anchored at
# every chunk boundary, so the bound grows as 1.834^C -- not linearly. Measured against a float64
# reference, worst relative gradient error at T=64, N=16, in float32:
#
#     C =  4    2.3e-07      4x less state
#     C = 16    5.4e-06     16x less state       <- here
#     C = 24    7.7e-05     21x
#     C = 32    4.0e-04     32x
#     C = 48    1.1e-01     32x                  <- already meaningless
#     C = 64    2.0e+01     64x                  <- gradients destroyed
#
# So 16 buys the memory back at float32 noise and 32 is the last value anyone should consider.
# It is a runtime argument rather than a template parameter, so sweeping it is a change here and
# not a rebuild -- but the table above is the reason not to.
CHUNK = 16
FNS = tuple(f"rwkv7_forward_n{n}" for n in SIZES) + \
      tuple(f"rwkv7_backward_n{n}" for n in SIZES)
_fn = None


def status(device=None):
    try:
        import torch
        if not torch.cuda.is_available():
            return False, D.why()
        i = torch.cuda.current_device() if device is None else int(device)
        m = D.module_for(CUBIN, i, FNS)
    except Exception:                                          # noqa: BLE001
        return False, D.why()
    return (True, m["__path__"]) if m else (False, D.why())


def _make():
    import torch

    class WKV7(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, wraw, k, v, a, b, keep):
            i = r.device.index
            m = D.module_for(CUBIN, i, FNS)
            if m is None:
                raise RuntimeError("no module")
            B, T, H, N = r.shape
            dev = r.device
            y = torch.empty_like(r)
            # (B, H, ceil(T/CHUNK), N, N) -- S AT EVERY CHUNK BOUNDARY, not at every step.
            #
            # It was (B, T, H, N, N), and that is the cost of an outer-product state written
            # once per token: at the benchmark's own geometry -- B 18, T 512, H 10, N 72, 13
            # layers -- 1.780 GiB per layer and 23.14 GiB across the model, which is why this
            # family alone needed 41 GB where the others needed 13 to 16, and why the shared
            # batch of 18 is what it is. Chunked it is 1.45 GiB. The backward inverts the
            # recurrence between boundaries; see rwkv_v7.cu.
            #
            # NOT ALLOCATED, AND NOT WRITTEN, WHEN NOBODY WILL READ IT. `keep` is
            # torch.is_grad_enabled() from OUTSIDE -- grad mode is off inside Function.forward,
            # so it cannot be read here. Under torch.no_grad() -- how the profiler times every
            # forward, and how every evaluation runs -- the .cu takes a null pointer and writes
            # nothing at all, so the forward is untouched by any of this.
            chunks = (T + CHUNK - 1) // CHUNK
            ss = (torch.empty(B, H, chunks, N, N, device=dev, dtype=torch.float32)
                  if keep else None)
            # sa_t, the delta-rule read. Recomputing it in the backward would mean replaying
            # the whole forward; N values per (b,t,h) is the cheap side of that trade.
            sa = (torch.empty(B, T, H, N, device=dev, dtype=torch.float32)
                  if keep else None)
            ok, why = D.launch_dims(m[f"rwkv7_forward_n{N}"], m["__ctx__"], i, B * H, N,
                               [r.data_ptr(), wraw.data_ptr(), k.data_ptr(), v.data_ptr(),
                                a.data_ptr(), b.data_ptr(), y.data_ptr(),
                                (ss.data_ptr() if keep else 0), (sa.data_ptr() if keep else 0)],
                               [B, T, H, N, CHUNK])
            if not ok:
                raise RuntimeError(f"rwkv7_forward {why}")
            ctx.save_for_backward(r, wraw, k, v, a, b, ss, sa)
            ctx.meta = (B, T, H, N, keep, CHUNK)

            return y

        @staticmethod
        def backward(ctx, gy):
            r, wraw, k, v, a, b, ss, sa = ctx.saved_tensors
            B, T, H, N, keep, chunk = ctx.meta
            if not keep:
                # Unreachable while `keep` is torch.is_grad_enabled().
                raise RuntimeError("rwkv7_backward: forward saved no state")
            i = r.device.index
            m = D.module_for(CUBIN, i, FNS)
            gy = gy.contiguous()
            gv = torch.empty_like(v)
            # ZEROED: everything summed over the i axis is finished with atomicAdd.
            gr, gw = torch.zeros_like(r), torch.zeros_like(wraw)
            gk = torch.zeros_like(k)
            ga, gb = torch.zeros_like(a), torch.zeros_like(b)
            ok, why = D.launch_dims(m[f"rwkv7_backward_n{N}"], m["__ctx__"], i, B * H, N,
                               [r.data_ptr(), wraw.data_ptr(), k.data_ptr(), v.data_ptr(),
                                a.data_ptr(), b.data_ptr(), ss.data_ptr(), sa.data_ptr(),
                                gy.data_ptr(), gr.data_ptr(), gw.data_ptr(), gk.data_ptr(),
                                gv.data_ptr(), ga.data_ptr(), gb.data_ptr()],
                               [B, T, H, N, chunk])
            if not ok:
                raise RuntimeError(f"rwkv7_backward {why}")
            return gr, gw, gk, gv, ga, gb, None      # None for `keep`

    return WKV7


def wkv7(r, wraw, k, v, a, b):
    """RWKV-7's WKV over (B, T, H, N), or None if this path is unavailable."""
    global _fn
    try:
        import torch
    except Exception:                                          # noqa: BLE001
        return None
    if not (getattr(r, "is_cuda", False) and r.dim() == 4):
        return None
    if r.shape[-1] not in SIZES:
        # ONE KERNEL PER HEAD SIZE, so this is membership and not a ceiling. It
        # returned None in silence before, and at RWKV-6's matched width -- 576/8
        # = 72 -- that meant the baseline ran on PyTorch while every table in the
        # profile said "fused (ours)". A limit the caller cannot see reads exactly
        # like a kernel that works.
        return D.fail("rwkv_v7", f"head dim {r.shape[-1]} has no kernel; built for {SIZES}")
    if not status(r.device.index)[0]:
        return None
    if _fn is None:
        _fn = _make()
    f = lambda t: t.contiguous().float()
    try:
        return _fn.apply(f(r), f(wraw), f(k), f(v), f(a), f(b), torch.is_grad_enabled())
    except Exception as e:                                     # noqa: BLE001
        return D.fail("rwkv_v7", f"kernel failed ({type(e).__name__}: {e})")


def describe():
    ok, detail = status()
    return f"rwkv_v7: {detail}" if ok else f"rwkv_v7: not loaded ({detail})"
