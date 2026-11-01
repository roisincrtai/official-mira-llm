"""kernel/cuda/rwkv_v6/rwkv_v6_upstream.py -- UPSTREAM RWKV-6's wkv6, wired, not rewritten.

    from kernel.cuda.rwkv_v6 import rwkv_v6_upstream as K
    K.build(ctxlen=512)                     # compiles, once, upstream's way
    y = K.run_cuda_rwkv6(r, k, v, w, u)     # (B, T, C) bf16 in, (B, T, C) bf16 out

THIS IS THE SECOND RWKV-6 KERNEL IN THIS DIRECTORY AND THE OTHER ONE STAYS. rwkv_v6.py is
ours -- a cubin, one entry point per head size, loaded through the driver API. This one is
BlinkDL's, compiled from the .cu vendored at upstream/cuda/ by upstream's own
cpp_extension.load. They are both here because the benchmark's claim is about ARCHITECTURES,
and a claim about architectures cannot rest on a kernel we wrote for someone else. Nothing was
deleted to make room.

WHAT IS UPSTREAM'S AND WHAT IS OURS. Everything that computes anything is upstream's:
upstream/cuda/wkv6_{cuda.cu,op.cpp} byte for byte, and upstream/op_x060.py, whose body is
lines 77-181 of RWKV-v5/src/model.py -- their load() call, their `class WKV_6`, their
`RUN_CUDA_RWKV6`. This file only decides WHEN that module is imported and says what went
wrong when it cannot be.

WHY THE IMPORT IS DEFERRED TO A CALL AND NOT DONE AT `import kernel.cuda.rwkv_v6`. Two
reasons, and the second is the binding one:

    the compile is real. upstream's load() runs nvcc. Doing that when `baselines` is imported
    would make a CPU-only box, a --dry-run, or a width search fail at import time for a
    kernel nothing was going to call.

    THE KERNEL IS COMPILED FOR ONE CONTEXT LENGTH. wkv6_cuda.cu's backward declares
    `float sbbbb[_T_-2]` as a per-thread array, so -D_T_ is not a hint, and _T_ is read from
    RWKV_CTXLEN at the moment upstream's load() line is executed. The sequence length is
    known at the first forward and not before, so the import happens there.

ONE HEAD SIZE, AND IT IS 64. -D_N_ comes from upstream's HEAD_SIZE, which their file takes
from RWKV_HEAD_SIZE_A; 64 is RWKV-6's default head_size_a. baselines/rwkv_v6.head_ok refuses
any other head dimension rather than letting the width search propose one the kernel was not
built for -- the same discipline rwkv_v6.SIZES enforces for our own cubin, and for the same
reason: a limit the caller cannot see reads exactly like a kernel that works.

ONE CONTEXT LENGTH PER PROCESS, TOO. The first forward fixes _T_ for the life of the
interpreter. A second sequence length is refused here in one sentence rather than read out of
a buffer that is the wrong size.
"""
import os

from .. import upstream_extension as U

KEY = "rwkv_v6_upstream"
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream")
DOTTED = "kernel.cuda.rwkv_v6.upstream.op_x060"
# THE TORCH_LIBRARY NAMESPACE upstream/cuda/wkv6_op.cpp registers. Recorded so that two
# families cannot silently fight over it; RWKV-6's is its own and clashes with nothing.
NAMESPACE = "wkv6"
HEAD_SIZE = 64

_ctxlen = None
_head_size = None


def build(ctxlen, head_size=HEAD_SIZE):
    """Compile upstream's kernel for a context length at least this long, once.

    A SHORTER SEQUENCE THAN THE ONE IT WAS BUILT FOR IS FINE, and reading the kernel is what
    settles it. `_T_` occurs in exactly one place in upstream/cuda/wkv6_cuda.cu:

        223:  float saaaa[_N_] = {0}, sbbbb[_T_-2] = {0}, scccc[_N_] = {0};

    inside kernel_backward_201, and nowhere else -- kernel_forward does not mention it. That
    array is written at sbbbb[(t-t_2)/C] over t in [t_2, t_T_1] and read at sbbbb[(t-t_1)/C]
    over t in [t_2, t_T_1), so every index it touches lies in [0, T-3] for the RUNTIME T, while
    the array holds _T_-2 of them. Building at _T_ and running at T <= _T_ therefore touches
    only a prefix: correct, with the tail merely unused.

        already built for T=257 and asked for T=256

    ASK FOR THE LONGEST FIRST. A larger T than the compiled one is still refused, because that
    one really would run off the end of the array, and the message says what to do about it.
    """
    global _ctxlen, _head_size
    want = int(ctxlen)
    requested_head_size = int(head_size)
    if _head_size is not None and requested_head_size != _head_size:
        return U._say(KEY, f"built for head size {_head_size} and asked for {requested_head_size}")
    if _ctxlen is not None and want > _ctxlen:
        return U._say(KEY, f"built for T={_ctxlen} and asked for T={want}; upstream's "
                           f"wkv6_cuda.cu sizes a per-thread array with -D_T_, so a process can "
                           f"serve any length UP TO the one it compiled. Build the longest "
                           f"context first, or run one process per length")
    if _ctxlen is not None:
        return U._MODS.get(KEY)
    m = U.load(KEY, HERE, DOTTED, namespace=NAMESPACE,
               env={"RWKV_CTXLEN": int(ctxlen), "RWKV_HEAD_SIZE_A": requested_head_size})
    if m is not None and _ctxlen is None:
        _ctxlen = int(ctxlen)
        _head_size = requested_head_size
    return m


def run_cuda_rwkv6(r, k, v, w, u):
    """Upstream's RUN_CUDA_RWKV6. build() must have been called; the wrapper does it.

    RAISES rather than falling back, and that is the honest answer. Upstream ships no PyTorch
    path for this recurrence -- `RUN_CUDA_RWKV6` is the only implementation in their file --
    so a fallback here would be one WE wrote, which is the thing this whole directory exists
    to avoid. baselines/rwkv_v6 reports the backend at build time, so a run that would end up
    here has already said so in the log."""
    m = U._MODS.get(KEY)
    if m is None:
        raise RuntimeError(
            f"upstream RWKV-6 kernel unavailable: {U.why(KEY)}. There is no PyTorch fallback: "
            f"upstream's RWKV_Tmix_x060 calls RUN_CUDA_RWKV6 and nothing else, and writing a "
            f"substitute here would put our arithmetic under their name.")
    return m.RUN_CUDA_RWKV6(r, k, v, w, u)


def status():
    return U.status(KEY)


def describe():
    ok, detail = status()
    return f"rwkv_v6 (upstream): {detail}" if ok else f"rwkv_v6 (upstream): not loaded ({detail})"
