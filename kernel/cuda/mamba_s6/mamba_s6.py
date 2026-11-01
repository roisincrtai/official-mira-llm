"""kernel/cuda/mamba_s6/mamba_s6.py -- OUR selective scan. No mamba-ssm, no causal-conv1d.

    from kernel.cuda.mamba_s6.mamba_s6 import selective_scan
    y = selective_scan(u, delta, A, B, C, D, z)      # None when unavailable

Signature and layout match mamba_ssm.ops.selective_scan_interface.selective_scan_fn, so
profiler/baselines.py calls one or the other with no reshaping:

    u, delta, z, y   (B, d_in, T)      A (d_in, N)      D (d_in)      B_, C_ (B, N, T)

See mamba_s6.cu for the maths, the register-resident N-wide state, and the verification.
"""
from .. import driver as D

CUBIN = "mamba_s6/mamba_s6.cubin"
# ONE ENTRY POINT PER STATE WIDTH: N is a template parameter in the .cu so h[] and
# av[] can live in registers instead of spilling to local memory.
SIZES = (8, 16, 32, 64)
FNS = tuple(f"ms6_forward_n{n}" for n in SIZES) + \
      tuple(f"ms6_backward_n{n}" for n in SIZES)
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

    class SelectiveScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, dlt, A, Bm, Cm, Dv, z, keep):
            i = u.device.index
            m = D.module_for(CUBIN, i, FNS)
            if m is None:
                raise RuntimeError("no module")
            Bt, d, T = u.shape
            N = A.shape[1]
            dev, ctxp = u.device, m["__ctx__"]
            out = torch.empty_like(u)
            # (B, T, d, N) -- the state at every step, which the BACKWARD needs. Recomputing it
            # would mean re-running the whole forward inside the backward, and Mamba's own kernel
            # makes the same trade.
            #
            # NOT ALLOCATED, AND NOT WRITTEN, WHEN NOBODY WILL READ IT. `keep` is
            # torch.is_grad_enabled() from OUTSIDE this function -- grad mode is off inside a
            # Function.forward, so it cannot be read here. At B=2, T=4,096, d_in=1,024, N=16
            # this tensor is 537 MB, and the kernel writes every byte of it with a stride of N
            # between neighbouring threads. Under torch.no_grad() -- which is how the profiler
            # times every forward -- all of that was written, never read, and thrown away.
            # The .cu has always taken a null pointer here; the wrapper simply never passed one.
            hs = (torch.empty(Bt, T, d, N, device=dev, dtype=torch.float32) if keep else None)
            ok, why = D.launch(m[f"ms6_forward_n{N}"], ctxp, i, Bt * d,
                               [u.data_ptr(), dlt.data_ptr(), A.data_ptr(), Bm.data_ptr(),
                                Cm.data_ptr(), 0 if Dv is None else Dv.data_ptr(),
                                z.data_ptr(), out.data_ptr(),
                                hs.data_ptr() if keep else 0],
                               [Bt, d, T, N])
            if not ok:
                raise RuntimeError(f"ms6_forward {why}")
            ctx.save_for_backward(u, dlt, A, Bm, Cm,
                                  Dv if Dv is not None else u.new_zeros(0), z,
                                  hs if keep else u.new_zeros(0))
            ctx.meta = (Bt, d, T, N, Dv is not None, keep)
            return out

        @staticmethod
        def backward(ctx, go):
            u, dlt, A, Bm, Cm, Dv, z, hs = ctx.saved_tensors
            Bt, d, T, N, has_D, keep = ctx.meta
            if not keep:
                # Unreachable while `keep` is torch.is_grad_enabled(): autograd cannot call a
                # backward it did not record. Explicit because the alternative is silent
                # garbage read out of a zero-length tensor.
                raise RuntimeError("ms6_backward: the forward ran without saving the state")
            i = u.device.index
            m = D.module_for(CUBIN, i, FNS)
            go = go.contiguous()
            gu, gdlt, gz = torch.empty_like(u), torch.empty_like(dlt), torch.empty_like(z)
            # ZEROED: the kernel finishes these with atomicAdd across (b, t) or (d).
            gA, gBm, gCm = torch.zeros_like(A), torch.zeros_like(Bm), torch.zeros_like(Cm)
            gD = torch.zeros_like(Dv) if has_D else None
            ok, why = D.launch(m[f"ms6_backward_n{N}"], m["__ctx__"], i, Bt * d,
                               [u.data_ptr(), dlt.data_ptr(), A.data_ptr(), Bm.data_ptr(),
                                Cm.data_ptr(), Dv.data_ptr() if has_D else 0, z.data_ptr(),
                                hs.data_ptr(), go.data_ptr(), gu.data_ptr(), gdlt.data_ptr(),
                                gA.data_ptr(), gBm.data_ptr(), gCm.data_ptr(),
                                gD.data_ptr() if has_D else 0, gz.data_ptr()],
                               [Bt, d, T, N])
            if not ok:
                raise RuntimeError(f"ms6_backward {why}")
            return gu, gdlt, gA, gBm, gCm, gD, gz, None      # None for `keep`

    return SelectiveScan


def selective_scan(u, dlt, A, Bm, Cm, Dv, z):
    """y = SiLU(z) * (C.h + D u) with h the selective state, or None if unavailable."""
    global _fn
    try:
        import torch
    except Exception:                                          # noqa: BLE001
        return None
    if not (getattr(u, "is_cuda", False) and u.dim() == 3 and A.dim() == 2):
        return None
    if A.shape[1] not in SIZES:
        return D.fail("mamba_s6", f"d_state {A.shape[1]} has no kernel; built for {SIZES}")
    if not status(u.device.index)[0]:
        return None
    if _fn is None:
        _fn = _make()
    f = lambda t: None if t is None else t.contiguous().float()
    # READ HERE, WHERE GRAD MODE IS STILL VISIBLE. Inside Function.forward it is always off.
    keep = torch.is_grad_enabled()
    try:
        return _fn.apply(f(u), f(dlt), f(A), f(Bm), f(Cm), f(Dv), f(z), keep)
    except Exception as e:                                     # noqa: BLE001
        return D.fail("mamba_s6", f"kernel failed ({type(e).__name__}: {e})")


def describe():
    ok, detail = status()
    return f"mamba_s6: {detail}" if ok else f"mamba_s6: not loaded ({detail})"
