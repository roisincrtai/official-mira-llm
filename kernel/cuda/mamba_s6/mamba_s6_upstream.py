"""kernel/cuda/mamba_s6/mamba_s6_upstream.py -- UPSTREAM MAMBA's selective_scan_cuda, wired.

    from kernel.cuda.mamba_s6 import mamba_s6_upstream as K
    K.build()          # compiles upstream/selective_scan/*, once, and installs it under the
                       # name upstream imports it by

THIS IS THE SECOND S6 KERNEL IN THIS DIRECTORY AND THE OTHER ONE STAYS. mamba_s6.py is ours --
a cubin, templated on the state width, loaded through the driver API. This one is Tri Dao and
Albert Gu's, from the ten source files vendored at upstream/selective_scan/. Both are here
because the benchmark's claim is about architectures, and a claim about architectures must not
rest on our transcription of someone else's kernel. Nothing was deleted.

HOW UPSTREAM SHIPS IT, AND WHY WE DO NOT DO THAT. Their setup.py declares a CUDAExtension named
`selective_scan_cuda` and `pip install mamba-ssm` builds it. That install is not available
here: requirements.txt records that mamba-ssm declares torch, that pip therefore resolved it,
and that it UNINSTALLED a cluster's torch build in favour of a generic PyPI wheel. Vendored
source compiled in place installs nothing and touches no environment.

WHAT IS THEIRS AND WHAT IS OURS IN THE BUILD. The ten source files are theirs, unmodified. The
source list, the include directory and the nvcc flags below are theirs too, transcribed from
their setup.py -- the flag list is `extra_compile_args["nvcc"]` at setup.py lines 225-244 and
the sources are the `name="selective_scan_cuda"` CUDAExtension. What is ours is the decision to
run that through torch.utils.cpp_extension.load instead of through setup.py, and the two lines
that put the result into sys.modules under the name their code imports it by:

    mamba_ssm/ops/selective_scan_interface.py:      import selective_scan_cuda

cpp_extension.load returns a module object and does not register it, so without that their
import would find nothing and their own `_check_selective_scan_cuda` would raise the
"reinstall with pip" message. Injecting it is an adaptation at the edge; their file is
untouched.

-gencode IS NOT TRANSCRIBED. setup.py emits a fixed list of architectures; cpp_extension.load
compiles for the card it finds, through TORCH_CUDA_ARCH_LIST, which is what every other build
on this cluster already honours and is right for a scheduler that hands out whatever is free.

A MISSING KERNEL IS NOT AN ERROR. build() returns None and says why, once.
"""
import os
import sys
import importlib

from .. import upstream_extension as U

KEY = "mamba_s6_upstream"
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream")
SRC = os.path.join(HERE, "selective_scan")
# THE MODULE NAME IS NOT A CHOICE: mamba_ssm imports `selective_scan_cuda` by that name.
EXT_NAME = "selective_scan_cuda"

# Upstream's setup.py, the name="selective_scan_cuda" CUDAExtension, in its order.
SOURCES = ("selective_scan.cpp",
           "selective_scan_fwd_fp32.cu",
           "selective_scan_fwd_fp16.cu",
           "selective_scan_fwd_bf16.cu",
           "selective_scan_bwd_fp32_real.cu",
           "selective_scan_bwd_fp32_complex.cu",
           "selective_scan_bwd_fp16_real.cu",
           "selective_scan_bwd_fp16_complex.cu",
           "selective_scan_bwd_bf16_real.cu",
           "selective_scan_bwd_bf16_complex.cu")

# Upstream's setup.py, extra_compile_args, the non-HIP branch, minus its -gencode list.
CXX_FLAGS = ("-O3", "-std=c++17")
NVCC_FLAGS = ("-O3", "-std=c++17",
              "-U__CUDA_NO_HALF_OPERATORS__",
              "-U__CUDA_NO_HALF_CONVERSIONS__",
              "-U__CUDA_NO_BFLOAT16_OPERATORS__",
              "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
              "-U__CUDA_NO_BFLOAT162_OPERATORS__",
              "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
              "--expt-relaxed-constexpr",
              "--expt-extended-lambda",
              "--use_fast_math",
              "--ptxas-options=-v",
              "-lineinfo")


def build():
    """Compile upstream's selective scan and install it under its own name. Module or None."""
    if KEY in U._MODS:
        return U._MODS[KEY]
    with U._LOCK:
        if KEY in U._MODS:
            return U._MODS[KEY]
        try:
            got = importlib.import_module(EXT_NAME)
        except Exception:                                         # noqa: BLE001
            got = None
        if got is not None:
            # ALREADY THERE, and that is a legitimate outcome: a user who did install
            # mamba-ssm in their own script has the real one, and using it is more faithful
            # than shadowing it with ours.
            U._MODS[KEY] = got
            U._WHY[KEY] = f"{EXT_NAME} was already importable; using it rather than rebuilding"
            return got
        try:
            import torch                                           # noqa: F401
        except Exception as e:                                     # noqa: BLE001
            U._MODS[KEY] = None
            return U._say(KEY, f"no torch ({type(e).__name__}: {e}); upstream S6 not built")
        if not os.path.isdir(SRC):
            U._MODS[KEY] = None
            return U._say(KEY, f"vendored sources missing at {SRC}")
        try:
            m = U.build_torch_library(
                "mamba_s6", EXT_NAME, [os.path.join(SRC, s) for s in SOURCES], SRC,
                NVCC_FLAGS, python_module=True)
        except Exception as e:                                     # noqa: BLE001
            U._MODS[KEY] = None
            return U._say(KEY, f"upstream's selective scan did not build "
                               f"({type(e).__name__}: {e}). It needs nvcc AND a C++ compiler "
                               f"on THIS node and a writable project build directory; ten "
                               f"source files take several minutes the first time.")
        sys.modules[EXT_NAME] = m
        U._MODS[KEY] = m
        U._WHY[KEY] = f"upstream kernel built from {SRC}"
        return m


def selective_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                   delta_softplus=False):
    import torch
    if torch.is_grad_enabled():
        raise RuntimeError("mamba_s6_upstream.selective_scan is a forward-only profiler path")
    module = build()
    if module is None:
        raise RuntimeError(f"upstream Mamba kernel unavailable: {U.why(KEY)}")
    u = u.contiguous()
    delta = delta.contiguous()
    A = A.contiguous()
    B = B.contiguous().unsqueeze(1) if B.dim() == 3 else B.contiguous()
    C = C.contiguous().unsqueeze(1) if C.dim() == 3 else C.contiguous()
    D = None if D is None else D.contiguous()
    z = None if z is None else z.contiguous()
    delta_bias = None if delta_bias is None else delta_bias.contiguous()
    out, _, *rest = module.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
    return rest[0] if z is not None else out


def status():
    return U.status(KEY)


def describe():
    ok, detail = status()
    return f"mamba_s6 (upstream): {detail}" if ok else f"mamba_s6 (upstream): not loaded ({detail})"
