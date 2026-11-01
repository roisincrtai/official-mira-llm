"""kernel/cuda/rwkv_v6/upstream/op_x060.py -- UPSTREAM RWKV-6's OWN CUDA OP, unmodified.

Everything between the VERBATIM SLICE markers is RWKV-LM's `RWKV-v5/src/model.py`, lines
77-181, byte for byte: the `elif 'x060' ...:` branch of upstream's CUDA section, holding both
the `states` sub-branch and the ordinary one -- the cpp_extension.load of
cuda/wkv6_{op.cpp,cuda.cu}, `class WKV_6` and `def RUN_CUDA_RWKV6`.

RWKV-v5/ IS WHERE RWKV-6 LIVES. The archive's RWKV-v6/ holds one README whose only line says
to use RWKV-v5 with --my_testing "x060". That is upstream's layout, not a guess of ours.

THE `if 0: pass` ABOVE THE SLICE IS THERE BECAUSE THE SLICE STARTS WITH `elif`. Upstream's
line 77 continues the `if 'x070'` above it, and taking that branch on its own needs something
for it to be the else-branch OF. Two lines of ours; not one line of theirs touched. The
alternative was dedenting their block by four columns, which changes every line in it.

WHAT THE PRELUDE SUPPLIES, all of them names the slice reads from its module scope:

    load             upstream imports it at its line 39, outside the slice
    HEAD_SIZE        upstream sets it at line 42 from RWKV_HEAD_SIZE_A; 64 is RWKV-6's default
                     head_size_a and is what -D_N_ is compiled with
    RWKV_MY_TESTING  'x060', so this is the branch taken
    RWKV_TRAIN_TYPE  '', so the `states` sub-branch -- a different kernel, wkv6state -- is not
    RWKV_CTXLEN      the context length. THE KERNEL IS COMPILED FOR ONE T: wkv6_cuda.cu
                     declares `float sbbbb[_T_-2]` per thread in the backward, so -D_T_ is not
                     advisory. rwkv_v6_upstream.py sets this from the real sequence length
                     before importing this module, which is why the import happens on the first
                     forward and not at package import.

NOTHING HERE IS BUILT AT PACKAGE IMPORT: the load below invokes nvcc.
"""
import os

import torch                                                            # noqa: F401  the slice uses it
# WIRED TO THE PROJECT'S NINJA-FREE BUILDER. Upstream calls
# torch.utils.cpp_extension.load(), whose only build backend is Ninja and whose scratch
# directory defaults to ~/.cache/torch_extensions, outside the project. The kernel, its
# sources and its TORCH_LIBRARY registrations are unchanged; one nvcc invocation produces
# the same shared object under build/kernel/cuda/<family>/upstream/.
def load(name, sources, extra_cuda_cflags=(), **_ignored):
    import os
    from kernel.cuda.upstream_extension import build_torch_library
    return build_torch_library("rwkv_v6", name, list(sources), os.getcwd(),
                               tuple(extra_cuda_cflags))

os.environ["RWKV_MY_TESTING"] = "x060"
os.environ["RWKV_TRAIN_TYPE"] = ""
os.environ.setdefault("RWKV_CTXLEN", "512")
HEAD_SIZE = int(os.environ.get("RWKV_HEAD_SIZE_A", "64"))

if 0:
    pass

####################################################################################################
# VERBATIM SLICE BEGINS -- RWKV-v5/src/model.py lines 77-181, byte for byte.
# Nothing from here to the END marker was written in this repository. Re-extract
#     sed -n '77,181p' <archive>/RWKV-v5/src/model.py
# and diff it against this block; the diff must be empty.
####################################################################################################
elif 'x060' in os.environ["RWKV_MY_TESTING"]:
    if os.environ["RWKV_TRAIN_TYPE"] == 'states':
        wkv6state_cuda = load(name="wkv6state", sources=["cuda/wkv6state_op.cpp", f"cuda/wkv6state_cuda.cu"],
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
            
        class WKV_6STATE(torch.autograd.Function):
            @staticmethod
            def forward(ctx, B, T, C, H, r, k, v, w, u, s):
                with torch.no_grad():
                    assert r.dtype == torch.bfloat16
                    assert k.dtype == torch.bfloat16
                    assert v.dtype == torch.bfloat16
                    assert w.dtype == torch.bfloat16
                    assert u.dtype == torch.bfloat16
                    assert s.dtype == torch.bfloat16
                    assert HEAD_SIZE == C // H
                    ctx.B = B
                    ctx.T = T
                    ctx.C = C
                    ctx.H = H
                    assert r.is_contiguous()
                    assert k.is_contiguous()
                    assert v.is_contiguous()
                    assert w.is_contiguous()
                    assert u.is_contiguous()
                    assert s.is_contiguous()
                    ctx.save_for_backward(r, k, v, w, u, s)
                    y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    wkv6state_cuda.forward(B, T, C, H, r, k, v, w, u, s, y)
                    return y

            @staticmethod
            def backward(ctx, gy):
                with torch.no_grad():
                    assert gy.dtype == torch.bfloat16
                    B = ctx.B
                    T = ctx.T
                    C = ctx.C
                    H = ctx.H
                    assert gy.is_contiguous()
                    r, k, v, w, u, s = ctx.saved_tensors
                    gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    wkv6state_cuda.backward(B, T, C, H, r, k, v, w, u, s, gy, gr, gk, gv, gw, gu, gs)
                    gu = torch.sum(gu, 0).view(H, C//H)
                    gs = torch.sum(gs, 0).view(H, C//H, C//H)
                    return (None, None, None, None, gr, gk, gv, gw, gu, gs)

        def RUN_CUDA_RWKV6_STATE(B, T, C, H, r, k, v, w, u, s):
            return WKV_6STATE.apply(B, T, C, H, r, k, v, w, u, s)
    else:
        load(name="wkv6", sources=["cuda/wkv6_op.cpp", f"cuda/wkv6_cuda.cu"], is_python_module=False,
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
            
        class WKV_6(torch.autograd.Function):
            @staticmethod
            def forward(ctx, r, k, v, w, u):
                with torch.no_grad():
                    B, T, C = r.size()
                    H = C // HEAD_SIZE
                    assert C % HEAD_SIZE == 0
                    assert r.dtype == torch.bfloat16
                    assert k.dtype == torch.bfloat16
                    assert v.dtype == torch.bfloat16
                    assert w.dtype == torch.bfloat16
                    assert u.dtype == torch.bfloat16
                    ctx.B = B
                    ctx.T = T
                    ctx.C = C
                    ctx.H = H
                    assert r.is_contiguous()
                    assert k.is_contiguous()
                    assert v.is_contiguous()
                    assert w.is_contiguous()
                    assert u.is_contiguous()
                    ctx.save_for_backward(r, k, v, w, u)
                    y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    torch.ops.wkv6.forward(B, T, C, H, r, k, v, w, u, y)
                    return y

            @staticmethod
            def backward(ctx, gy):
                with torch.no_grad():
                    assert gy.dtype == torch.bfloat16
                    B = ctx.B
                    T = ctx.T
                    C = ctx.C
                    H = ctx.H
                    assert gy.is_contiguous()
                    r, k, v, w, u = ctx.saved_tensors
                    gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                    torch.ops.wkv6.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                    gu = torch.sum(gu, 0).view(H, C//H)
                    return (gr, gk, gv, gw, gu)

        def RUN_CUDA_RWKV6(r, k, v, w, u):
            return WKV_6.apply(r, k, v, w, u)
####################################################################################################
# VERBATIM SLICE ENDS.
####################################################################################################
