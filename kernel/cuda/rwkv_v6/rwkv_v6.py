"""kernel/cuda/rwkv_v6/rwkv_v6.py -- OUR RWKV-6 WKV. No flash-linear-attention.

    from kernel.cuda.rwkv_v6.rwkv_v6 import wkv6
    y = wkv6(r, k, v, logw, u)          # (B, T, H, N) each; u is (H, N). None when unavailable

Layout matches fla.ops.rwkv6.chunk_rwkv6 with head_first=False, so profiler/baselines.py calls
one or the other with no reshaping. `logw` is the LOG decay, as fla takes it.

See rwkv_v6.cu for the maths, the row-per-thread state and the verification.
"""
from .. import driver as D

CUBIN = "rwkv_v6/rwkv_v6.cubin"
# ONE ENTRY POINT PER HEAD SIZE: N is a template parameter in the .cu, so the
# state row can live in registers. A head dim not in this tuple is refused and
# reported -- never truncated.
SIZES = (16, 32, 48, 60, 64, 68, 70, 72, 80, 96, 128)
FNS = tuple(f"rwkv6_forward_n{n}" for n in SIZES) + \
      tuple(f"rwkv6_backward_n{n}" for n in SIZES)
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

    class WKV6(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, k, v, lw, u, keep):
            i = r.device.index
            m = D.module_for(CUBIN, i, FNS)
            if m is None:
                raise RuntimeError("no module")
            B, T, H, N = r.shape
            dev = r.device
            y = torch.empty_like(r)
            # (B, T, H, N, N) -- S_{t-1} at every step. THIS IS THE COST OF THE OUTER-PRODUCT
            # STATE: N times what MIRA's backward keeps, and the reason the chunked formulation
            # exists. A first version is allowed to be honest about that rather than wrong.
                        # NOT ALLOCATED, AND NOT WRITTEN, WHEN NOBODY WILL READ IT. `keep` is
            # torch.is_grad_enabled() from OUTSIDE -- grad mode is off inside Function.forward,
            # so it cannot be read here. At B=2, T=4,096, H=8, N=72 this state is 1.36 GB, and
            # the kernel writes every byte of it with a stride of N between neighbouring
            # threads. Under torch.no_grad() -- how the profiler times every forward -- all of
            # it was written, never read, and thrown away. The .cu has always taken a null
            # pointer here; the wrapper simply never passed one.
            ss = (torch.empty(B, T, H, N, N, device=dev, dtype=torch.float32)
                  if keep else None)
            ok, why = D.launch_dims(m[f"rwkv6_forward_n{N}"], m["__ctx__"], i, B * H, N,
                               [r.data_ptr(), k.data_ptr(), v.data_ptr(), lw.data_ptr(),
                                u.data_ptr(), y.data_ptr(), (ss.data_ptr() if keep else 0)], [B, T, H, N])
            if not ok:
                raise RuntimeError(f"rwkv6_forward {why}")
            ctx.save_for_backward(r, k, v, lw, u, ss)
            ctx.meta = (B, T, H, N, keep)

            return y

        @staticmethod
        def backward(ctx, gy):
            r, k, v, lw, u, ss = ctx.saved_tensors
            B, T, H, N, keep = ctx.meta
            if not keep:
                # Unreachable while `keep` is torch.is_grad_enabled().
                raise RuntimeError("rwkv6_backward: forward saved no state")
            i = r.device.index
            m = D.module_for(CUBIN, i, FNS)
            gy = gy.contiguous()
            gv = torch.empty_like(v)
            # ZEROED: gr, gk, glw and gu are finished with atomicAdd across the i axis.
            gr, gk, glw = torch.zeros_like(r), torch.zeros_like(k), torch.zeros_like(lw)
            gu = torch.zeros_like(u)
            ok, why = D.launch_dims(m[f"rwkv6_backward_n{N}"], m["__ctx__"], i, B * H, N,
                               [r.data_ptr(), k.data_ptr(), v.data_ptr(), lw.data_ptr(),
                                u.data_ptr(), ss.data_ptr(), gy.data_ptr(), gr.data_ptr(),
                                gk.data_ptr(), gv.data_ptr(), glw.data_ptr(), gu.data_ptr()],
                               [B, T, H, N])
            if not ok:
                raise RuntimeError(f"rwkv6_backward {why}")
            return gr, gk, gv, glw, gu, None      # None for `keep`

    return WKV6


def wkv6(r, k, v, logw, u):
    """RWKV-6's WKV over (B, T, H, N), or None if this path is unavailable."""
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
        return D.fail("rwkv_v6", f"head dim {r.shape[-1]} has no kernel; built for {SIZES}")
    if not status(r.device.index)[0]:
        return None
    if _fn is None:
        _fn = _make()
    f = lambda t: t.contiguous().float()
    try:
        return _fn.apply(f(r), f(k), f(v), f(logw), f(u), torch.is_grad_enabled())
    except Exception as e:                                     # noqa: BLE001
        return D.fail("rwkv_v6", f"kernel failed ({type(e).__name__}: {e})")


def describe():
    ok, detail = status()
    return f"rwkv_v6: {detail}" if ok else f"rwkv_v6: not loaded ({detail})"
