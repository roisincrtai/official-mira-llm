"""kernel/cuda/rwkv_v7/upstream/op_x070.py -- UPSTREAM RWKV-7's OWN CUDA OP, unmodified.

WHAT IS OURS IN THIS FILE AND WHAT IS NOT. Everything above the VERBATIM SLICE marker is a
prelude written here; everything between the markers is RWKV-LM's `RWKV-v5/src/model.py`,
lines 44-75, copied byte for byte. That block is the `if 'x070' ...:` branch of upstream's
CUDA section: the cpp_extension.load of cuda/wkv7_{cuda.cu,op.cpp}, `class WindBackstepping`
and `def RUN_CUDA_RWKV7g`. Nothing in it was reformatted, renamed or reordered.

WHY A SLICE AND NOT THE WHOLE FILE. The upstream file is 1,460 lines, imports pytorch_lightning
and deepspeed at module scope, defines a LightningModule, and branches on five RWKV_*
environment variables. Importing it would drag a training framework this repository does not
use into a benchmark that needs one mixer. The whole file is vendored UNCHANGED beside this
one as baselines/rwkv_v7/upstream/model.py, so the slice can be diffed against its source
without leaving the repository.

WHAT THE PRELUDE HAS TO SUPPLY, and it is only names the slice reads from ITS module scope:

    load        upstream imports this at its line 39, outside the slice
    HEAD_SIZE   upstream sets it at line 42 from RWKV_HEAD_SIZE_A. 64, because upstream's own
                RUN_CUDA_RWKV7g hard-codes `HC//64, 64` two lines below the load -- that is
                upstream's constraint on this branch, not one added here
    os          for the environment read on the slice's first line

THE RELATIVE SOURCE PATHS ARE UPSTREAM'S AND ARE LEFT ALONE. `sources=['cuda/wkv7_cuda.cu',
'cuda/wkv7_op.cpp']` resolves against the working directory, so the vendored sources sit at
upstream/cuda/ -- upstream's own layout -- and rwkv_v7_upstream.py chdir's here for the
duration of the import. Editing the paths would have been the smaller-looking change and the
one that stops the file diffing against its source.

NOTHING HERE IS BUILT AT PACKAGE IMPORT. This module is imported by rwkv_v7_upstream.py on the
first forward, not when baselines/ is imported, because the load below invokes nvcc.
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
    return build_torch_library("rwkv_v7", name, list(sources), os.getcwd(),
                               tuple(extra_cuda_cflags))

os.environ["RWKV_MY_TESTING"] = "x070"
HEAD_SIZE = 64

####################################################################################################
# VERBATIM SLICE BEGINS -- RWKV-v5/src/model.py lines 44-75, byte for byte.
# Nothing from here to the END marker was written in this repository. Re-extract
#     sed -n '44,75p' <archive>/RWKV-v5/src/model.py
# and diff it against this block; the diff must be empty.
####################################################################################################
if 'x070' in os.environ["RWKV_MY_TESTING"]:
    CHUNK_LEN = 16

    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b):
            B,T,H,C = w.shape 
            assert T%CHUNK_LEN == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            return y
        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            return dw,dq,dk,dv,dz,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
        return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)
####################################################################################################
# VERBATIM SLICE ENDS.
####################################################################################################
