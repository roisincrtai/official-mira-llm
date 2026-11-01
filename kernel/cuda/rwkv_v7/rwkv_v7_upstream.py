"""kernel/cuda/rwkv_v7/rwkv_v7_upstream.py -- UPSTREAM RWKV-7's wind_backstepping, wired.

    from kernel.cuda.rwkv_v7 import rwkv_v7_upstream as K
    K.build()                                     # compiles, once, upstream's way
    y = K.run_cuda_rwkv7g(q, w, k, v, a, b)       # (B, T, C) bf16 in and out

THIS IS THE SECOND RWKV-7 KERNEL IN THIS DIRECTORY AND THE OTHER ONE STAYS. rwkv_v7.py is
ours: a cubin with one entry point per head size, a step-serial state row in registers, loaded
through the driver API. This one is BlinkDL's "wind backstepping" -- a CHUNKED formulation,
CHUNK_LEN 16, which is a different algorithm for the same recurrence and is the one RWKV-7 is
actually trained with. Keeping both is the point: a speed number against our transcription of
their maths is a number about us.

WHAT IS UPSTREAM'S. upstream/cuda/wkv7_{cuda.cu,op.cpp}, byte for byte, and upstream/op_x070.py
whose body is lines 44-75 of RWKV-v5/src/model.py -- their load(), their `WindBackstepping`,
their `RUN_CUDA_RWKV7g`. This file decides when that is imported and reports failures.

HEAD SIZE 64, AND IT IS UPSTREAM'S CONSTRAINT NOT OURS. Their RUN_CUDA_RWKV7g reshapes with
`HC//64, 64` as a literal -- the copy in RWKV-v7/train_temp/src/model.py carries the comment
"have to hard-code the number here, or pytorch will complain" -- and -D_C_ is compiled from
the same 64. baselines/rwkv_v7/upstream_mixer.head_ok refuses anything else so the width
search never proposes a width this kernel cannot run.

T MUST BE A MULTIPLE OF 16. `assert T%CHUNK_LEN == 0` is upstream's, in their forward.
Upstream's advice on the same line is to pad; the wrapper does not pad, it checks, because a
padded sequence changes what the benchmark measured.

THE NAMESPACE IS SHARED WITH RWKV-8 AND ONLY ONE FAMILY CAN HOLD IT. Both this and
kernel/cuda/rwkv_v8 register TORCH_LIBRARY(wind_backstepping) -- RWKV-8's kernel is this one
with `using bf = float` -- and torch allows a namespace one holder per process. Whichever
family builds first keeps it; the second is told so in a sentence by
kernel/cuda/upstream_extension.py. Run them in separate invocations:

    python -m benchmark.pretrain_benchmark --only rwkv_v7_upstream
    python -m benchmark.pretrain_benchmark --only rwkv_v8
"""
import os

from .. import upstream_extension as U

KEY = "rwkv_v7_upstream"
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream")
DOTTED = "kernel.cuda.rwkv_v7.upstream.op_x070"
NAMESPACE = "wind_backstepping"
HEAD_SIZE = 64
CHUNK_LEN = 16


def build():
    """Compile upstream's kernel, once. The module, or None."""
    return U.load(KEY, HERE, DOTTED, namespace=NAMESPACE,
                  env={"RWKV_HEAD_SIZE_A": HEAD_SIZE})


def run_cuda_rwkv7g(q, w, k, v, a, b):
    """Upstream's RUN_CUDA_RWKV7g. build() must have been called; the wrapper does it.

    RAISES rather than falling back. Upstream's RWKV_Tmix_x070 calls this and nothing else,
    and a PyTorch substitute written here would be our arithmetic under their name. The
    family's backend is printed at build time, so this is never the first anyone hears of it."""
    m = U._MODS.get(KEY)
    if m is None:
        raise RuntimeError(
            f"upstream RWKV-7 kernel unavailable: {U.why(KEY)}. There is no PyTorch fallback "
            f"in upstream's file. The reimplementation with one -- family 'rwkv_v7' -- is a "
            f"different family and is still registered.")
    return m.RUN_CUDA_RWKV7g(q, w, k, v, a, b)


def status():
    return U.status(KEY)


def describe():
    ok, detail = status()
    return f"rwkv_v7 (upstream): {detail}" if ok else f"rwkv_v7 (upstream): not loaded ({detail})"
