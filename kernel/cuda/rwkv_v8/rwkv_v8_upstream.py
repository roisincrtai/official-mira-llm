"""kernel/cuda/rwkv_v8/rwkv_v8_upstream.py -- UPSTREAM RWKV-8's wind_backstepping, fp32.

    from kernel.cuda.rwkv_v8 import rwkv_v8_upstream as K
    K.build()                                     # compiles, once, upstream's way
    y = K.run_cuda_rwkv7g(q, w, k, v, a, b)       # (B, T, C) fp32 in and out

fp32, NOT bf16, AND THE ASSERT IS UPSTREAM'S. `assert all(i.dtype==torch.float32 ...)` is on
their forward and their backward, and the kernel's scalar type is `float`. baselines/rwkv_v8
gives the sub-layer fp32 parameters and casts at the boundary so that assert is satisfied by
the wrapper rather than by editing their line.

HEAD SIZE 16, AND IT IS UPSTREAM'S. Their RUN_CUDA_RWKV7g reshapes with `HC//16, 16` as a
literal and their HEAD_SIZE is 16, which is what -D_C_ is compiled from.
baselines/rwkv_v8.head_ok refuses any other head dimension, so the width search never proposes
a width this kernel was not built for. It also caps the width: mira_llm's match_width searches
head counts up to 32, so 32 x 16 = 512 is the widest RWKV-8 can be matched at.

T MUST BE A MULTIPLE OF 16 -- `assert T%CHUNK_LEN == 0`, upstream's, in their forward.

THE NAMESPACE WAS SHARED WITH RWKV-7 AND IS NOT ANY MORE. Both upstream archives register
TORCH_LIBRARY(wind_backstepping), torch allows ONE holder per namespace per process, and the
benchmark walks every family inside ONE process -- so the shared name did not mean "run them
separately", it meant RWKV-8 NEVER GOT A KERNEL, because FAMILIES puts RWKV-7 first and RWKV-7
always claimed it. This family's vendored wkv7_op.cpp now registers `wind_backstepping_fp32`,
which is the only edit made to it. The two really are different kernels -- this one keeps its
state in fp32, RWKV-7's in bf16 -- so they were never entitled to the same name.

upstream_extension.py still refuses a second claimant on any namespace, which is now a guard
against a future collision rather than a thing that happens on every run.
"""
import os

from .. import upstream_extension as U

KEY = "rwkv_v8_upstream"
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream")
DOTTED = "kernel.cuda.rwkv_v8.upstream.op_x070_fp32"
NAMESPACE = "wind_backstepping_fp32"
HEAD_SIZE = 16
CHUNK_LEN = 16


def build():
    """Compile upstream's kernel, once. The module, or None."""
    return U.load(KEY, HERE, DOTTED, namespace=NAMESPACE)


def run_cuda_rwkv7g(q, w, k, v, a, b):
    """Upstream's RUN_CUDA_RWKV7g. build() must have been called; the wrapper does it.

    RAISES rather than falling back: upstream's RWKV_Tmix_x070 calls this and nothing else,
    and a substitute written here would be our arithmetic under their name."""
    m = U._MODS.get(KEY)
    if m is None:
        raise RuntimeError(
            f"upstream RWKV-8 kernel unavailable: {U.why(KEY)}. There is no PyTorch fallback "
            f"in upstream's file.")
    return m.RUN_CUDA_RWKV7g(q, w, k, v, a, b)


def status():
    return U.status(KEY)


def describe():
    ok, detail = status()
    return f"rwkv_v8 (upstream): {detail}" if ok else f"rwkv_v8 (upstream): not loaded ({detail})"
