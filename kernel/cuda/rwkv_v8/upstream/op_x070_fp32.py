"""kernel/cuda/rwkv_v8/upstream/op_x070_fp32.py -- UPSTREAM RWKV-8's OWN CUDA OP, unmodified.

Everything between the VERBATIM SLICE markers is RWKV-LM's `RWKV-v8/251105_reverse_run.py`,
lines 113-142, byte for byte: the cpp_extension.load of cuda/wkv7_{cuda.cu,op.cpp},
`class WindBackstepping` and `def RUN_CUDA_RWKV7g`.

THIS IS THE fp32 WIND-BACKSTEPPING KERNEL, NOT A NEW ONE. RWKV-8's cuda/ is RWKV-7's with
`using bf = float` in place of `using bf = __nv_bfloat16`; upstream/PROVENANCE has the
four-line diff. What is new in RWKV-8 is ROSA, and ROSA has no CUDA anywhere in the archive.

HEAD SIZE 16 AND fp32 ARE UPSTREAM'S. The slice sets HEAD_SIZE = 16 itself, its
RUN_CUDA_RWKV7g hard-codes `HC//16, 16`, and WindBackstepping asserts float32 on every input.
Neither was changed. baselines/rwkv_v8 enforces the head dimension through head_ok and gives
the sub-layer fp32 parameters, which is where an adaptation belongs.

THE SLICE IMPORTS `load` ITSELF on its first line, so unlike the RWKV-6 and RWKV-7 preludes
this one does not supply it. The relative source paths are upstream's and are left alone;
rwkv_v8_upstream.py chdir's to this directory, whose cuda/ holds the vendored sources in
upstream's own layout.

TORCH_LIBRARY NAME CLASH, AND IT IS FIXED IN THE .cpp. This op and RWKV-7's both registered
the namespace `wind_backstepping`, and torch permits ONE registration of a namespace per
process. That was not a "run them separately" inconvenience: the benchmark walks every family
in ONE process and FAMILIES puts RWKV-7 first, so RWKV-8 was refused its kernel on every run
that reached it. cuda/wkv7_op.cpp here now registers `wind_backstepping_fp32` and the two call
sites below follow it -- three lines, in this family's copy only. RWKV-7's archive is untouched.

They were never the same op: this one carries its state in fp32, RWKV-7's in bf16.
kernel/cuda/upstream_extension.py still refuses a second claimant on a namespace, which is now
a guard against some future collision rather than a certainty on every run.
"""
import os                                                               # noqa: F401
import torch                                                            # noqa: F401  the slice uses it

####################################################################################################
# VERBATIM SLICE BEGINS -- RWKV-v8/251105_reverse_run.py lines 113-142, byte for byte.
# Nothing from here to the END marker was written in this repository. Re-extract
#     sed -n '113,142p' <archive>/RWKV-v8/251105_reverse_run.py
# and diff it against this block; the diff must be empty.
####################################################################################################
# WIRED TO THE PROJECT'S NINJA-FREE BUILDER. Upstream calls
# torch.utils.cpp_extension.load(), whose only build backend is Ninja and whose scratch
# directory defaults to ~/.cache/torch_extensions, outside the project. The kernel, its
# sources and its TORCH_LIBRARY registrations are unchanged; one nvcc invocation produces
# the same shared object under build/kernel/cuda/<family>/upstream/.
def load(name, sources, extra_cuda_cflags=(), **_ignored):
    import os
    from kernel.cuda.upstream_extension import build_torch_library
    return build_torch_library("rwkv_v8", name, list(sources), os.getcwd(),
                               tuple(extra_cuda_cflags))
HEAD_SIZE = 16
CHUNK_LEN = 16
flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
load(name="wind_backstepping_fp32", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=False, extra_cuda_cflags=flags)
class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        B,T,H,C = w.shape
        assert T%CHUNK_LEN == 0 # if T%CHUNK_LEN != 0: pad your input to T%CHUNK_LEN == 0, or change CHUNK_LEN (will be slower)
        assert all(i.dtype==torch.float32 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping_fp32.forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.float32 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping_fp32.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db
def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//16,16) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)
####################################################################################################
# VERBATIM SLICE ENDS.
####################################################################################################
