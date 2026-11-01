from __future__ import annotations

import importlib.util
from pathlib import Path


BUILD = Path(__file__).resolve().parents[3] / "build" / "kernel" / "metal" / "mira_dspv2"
METALLIB = BUILD / "mira_iir.metallib"
EXTENSION = BUILD / "mira_metal.so"
ORDERS = (1, 2, 4, 8)
MEMORY_DIMENSIONS = (8, 16, 32, 64)
ORDER_MEMORIES = {order: MEMORY_DIMENSIONS for order in ORDERS}
RADII = ("cauchy", "gaussian", "expexp", "expsoftplus", "hill")
ROTATIONS = ("mobius", "hyperbolic", "identity")
ARCH_STAGE = {"cauchy_x_mobius": 3, "cauchy_x_hyperbolic": 6, "cauchy_x_identity": 7,
              "gaussian_x_mobius": 8, "gaussian_x_hyperbolic": 9, "gaussian_x_identity": 10,
              "expexp_x_mobius": 11, "expexp_x_hyperbolic": 12, "expexp_x_identity": 13,
              "expsoftplus_x_mobius": 14, "expsoftplus_x_hyperbolic": 15,
              "expsoftplus_x_identity": 16,
              "hill_x_mobius": 17, "hill_x_hyperbolic": 18, "hill_x_identity": 19}
ARCH_MAPS = {f"{radius}_x_{rotation}": (radius_kind, rotation_kind)
             for radius_kind, radius in enumerate(RADII)
             for rotation_kind, rotation in enumerate(ROTATIONS)}
VALUE_TYPES = {"torch.float32": "float", "torch.float16": "half", "torch.bfloat16": "bfloat"}
CHUNK = 32
SIMD = 32
FORWARD_GROUP = 256
BACKWARD_GROUP = 1024
SHARED_FLOATS = 4096
HAS_STATE = 1
HAS_FINAL = 2
HAS_VALID = 4
SAVE_CHECKPOINT = 8
USED = {"forward": 0, "backward": 0}
MODULE = {}
LIBRARIES = {}
FAILURES = {}
REPORTED = set()
FUNCTION = None


class _Function:
    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.max_threads_per_threadgroup = int(module.max_threads(name))

    def __call__(self, buffers, params, threads, group_size):
        self.module.launch(self.name, list(buffers), [int(value) for value in params],
                           int(threads), int(group_size))


class _Library:
    def __init__(self, module, order, memory, value):
        suffix = f"_k{order}_m{memory}_{value}"
        self.mira_forward = _Function(module, "mira_forward" + suffix)
        self.mira_backward = _Function(module, "mira_backward" + suffix)


def _module():
    if "module" in MODULE:
        return MODULE["module"]
    if "failure" in MODULE:
        raise RuntimeError(MODULE["failure"])
    try:
        for path in (EXTENSION, METALLIB):
            if not path.is_file():
                raise RuntimeError(f"{path} is missing; build it with "
                                   f"make -f kernel/metal/mira_dspv2/Makefile build")
        spec = importlib.util.spec_from_file_location("mira_metal", EXTENSION)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.load(str(METALLIB))
    except Exception as error:
        MODULE["failure"] = f"mira_dspv2 Metal build in {BUILD}: {error}"
        raise RuntimeError(MODULE["failure"]) from error
    MODULE["module"] = module
    return module


def _library(order, memory, value):
    key = (order, memory, value)
    if key in FAILURES:
        raise RuntimeError(FAILURES[key])
    if key not in LIBRARIES:
        try:
            LIBRARIES[key] = _Library(_module(), order, memory, value)
        except Exception as error:
            FAILURES[key] = f"mira_dspv2 Metal kernels for K={order}, d_m={memory}, {value}: {error}"
            raise RuntimeError(FAILURES[key]) from error
    return LIBRARIES[key]


def _on_device(tensor):
    return tensor.device.type == "mps"


def status(device=None):
    try:
        import torch
    except Exception as error:
        return False, str(error)
    backend = getattr(torch.backends, "mps", None)
    if backend is None or not backend.is_available():
        return False, "MPS is unavailable"
    try:
        _library(2, 16, "float")
    except Exception as error:
        return False, str(error)
    return True, str(METALLIB)


def usage():
    return dict(USED)


def report(index=None):
    if "mps" in REPORTED:
        return False
    REPORTED.add("mps")
    ok, where = status(index)
    print(f"[mira_dspv2] metal {'loaded ' + where if ok else 'unavailable: ' + where}", flush=True)
    return ok


def _lanes(memory):
    return min(memory, SIMD)


def _simdgroups(order, memory):
    return min(SIMD, SHARED_FLOATS // ((order + 2) * memory))


def _group(function, target):
    limit = int(getattr(function, "max_threads_per_threadgroup", target) or target)
    return max(SIMD, min(target, limit) // SIMD * SIMD)


def _row_stride(tensor, tokens):
    if tensor.stride(2) != 1:
        return None
    row = int(tensor.stride(1))
    if int(tensor.stride(0)) != tokens * row:
        return None
    return row


def dense_tail_row_stride(tensor, tokens):
    expected = 1
    for axis in range(tensor.dim() - 1, 1, -1):
        if int(tensor.stride(axis)) != expected:
            return None
        expected *= int(tensor.shape[axis])
    row = int(tensor.stride(1))
    if int(tensor.stride(0)) != tokens * row:
        return None
    return row


def _rows(tensor, tokens):
    row = dense_tail_row_stride(tensor, tokens)
    if row is None:
        tensor = tensor.contiguous()
        row = dense_tail_row_stride(tensor, tokens)
    return tensor, row


def _launch(function, buffers, params, groups, group):
    function(buffers, params, threads=groups * group, group_size=group)


def _layout(tensors, tokens):
    placed, rows = [], []
    for tensor in tensors:
        tensor, row = _rows(tensor, tokens)
        placed.append(tensor)
        rows.append(row)
    return placed, rows


def _forward(x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read, valid, state,
             arch, final, save):
    import torch

    batch, tokens, channels = x.shape
    order = int(raw_alpha.shape[-1])
    memory = int(memory_write.shape[-1])
    function = _library(order, memory, VALUE_TYPES[str(x.dtype)]).mira_forward
    lanes = _lanes(memory)
    group = _group(function, FORWARD_GROUP)
    tiles = -(-channels // (group // lanes))
    chunks = -(-tokens // CHUNK)
    (x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read), rows = _layout(
        (x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read), tokens)
    row_x, row_alpha, row_beta, row_rho, row_gamma, row_write, row_read = rows
    device = x.device
    spare = torch.zeros(1, dtype=torch.float32, device=device)
    output = torch.empty(batch, tokens, channels, dtype=x.dtype, device=device)
    final_state = (torch.empty(batch, order, channels, memory, dtype=torch.float32, device=device)
                   if final else spare)
    checkpoint = (torch.empty(batch, chunks, order, channels, memory, dtype=torch.float32,
                              device=device) if save else spare)
    flags = ((HAS_STATE if state is not None else 0) | (HAS_FINAL if final else 0)
             | (HAS_VALID if valid is not None else 0) | (SAVE_CHECKPOINT if save else 0))
    radius, rotation = ARCH_MAPS[arch]
    params = [batch, tokens, channels, tiles, group, radius, rotation, CHUNK, chunks, flags,
              row_x, row_gamma, row_alpha, row_beta, row_rho, row_write, row_read]
    mask = (valid.contiguous() if valid is not None
            else torch.ones(1, dtype=torch.bool, device=device))
    initial = state.contiguous() if state is not None else spare
    _launch(function, [x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read, mask,
                       initial, output, final_state, checkpoint], params, batch * tiles, group)
    USED["forward"] += 1
    return output, (final_state if final else None), (checkpoint if save else None)


def _backward(x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read, checkpoint,
              valid, arch, grad_output):
    import torch

    batch, tokens, channels = x.shape
    order = int(raw_alpha.shape[-1])
    memory = int(memory_write.shape[-1])
    function = _library(order, memory, VALUE_TYPES[str(x.dtype)]).mira_backward
    lanes = _lanes(memory)
    group = _group(function, min(BACKWARD_GROUP, _simdgroups(order, memory) * SIMD))
    tiles = -(-channels // (group // lanes))
    chunks = -(-tokens // CHUNK)
    threads = batch * tiles * group
    (x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read), rows = _layout(
        (x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read), tokens)
    row_x, row_alpha, row_beta, row_rho, row_gamma, row_write, row_read = rows
    device = x.device
    dtype = x.dtype
    grad_output = grad_output.to(dtype).contiguous()
    scratch = torch.empty(CHUNK * order * (memory // lanes) * threads, dtype=torch.float32,
                          device=device)
    grad_x = torch.empty(batch, tokens, channels, dtype=dtype, device=device)
    grad_alpha = torch.empty(batch, tokens, channels, order, dtype=dtype, device=device)
    grad_beta = torch.empty(batch, tokens, channels, order, dtype=dtype, device=device)
    grad_rho = torch.empty(batch, tokens, channels, order + 1, dtype=dtype, device=device)
    grad_gamma = torch.empty(batch, tokens, channels, dtype=dtype, device=device)
    partial = torch.empty(batch, tokens, tiles, order + 2, memory, dtype=torch.float32,
                          device=device)
    flags = HAS_VALID if valid is not None else 0
    radius, rotation = ARCH_MAPS[arch]
    params = [batch, tokens, channels, tiles, group, radius, rotation, CHUNK, chunks, flags,
              row_x, row_gamma, row_alpha, row_beta, row_rho, row_write, row_read]
    mask = (valid.contiguous() if valid is not None
            else torch.ones(1, dtype=torch.bool, device=device))
    _launch(function, [x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read, mask,
                       checkpoint, grad_output, scratch, grad_x, grad_alpha, grad_beta, grad_rho,
                       grad_gamma, partial], params, batch * tiles, group)
    reduced = partial.sum(dim=2)
    grad_read = reduced[:, :, :order + 1].to(dtype)
    grad_write = reduced[:, :, order + 1].to(dtype)
    USED["backward"] += 1
    return grad_x, grad_alpha, grad_beta, grad_rho, grad_gamma, grad_write, grad_read


def make_function():
    import torch
    from torch.autograd.function import once_differentiable

    class MatrixLatticeIIR(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read,
                    valid, arch):
            output, unused, checkpoint = _forward(
                x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read, valid, None,
                arch, final=False, save=True)
            del unused
            ctx.save_for_backward(x, raw_alpha, raw_beta, raw_rho, gamma, memory_write,
                                  memory_read, checkpoint)
            ctx.valid = valid
            ctx.arch = arch
            return output

        @staticmethod
        @once_differentiable
        def backward(ctx, grad_output):
            grads = _backward(*ctx.saved_tensors, ctx.valid, ctx.arch, grad_output)
            return (*grads, None, None)

    return MatrixLatticeIIR


def _ready(order, memory, dtype):
    value = VALUE_TYPES.get(str(dtype))
    if value is None or order not in ORDERS or memory not in ORDER_MEMORIES[order]:
        return False
    try:
        _library(order, memory, value)
    except Exception:
        return False
    return True


def matrix_lattice_iir(
    x, raw_alpha, raw_rho, gamma, memory_write, memory_read, raw_beta=None,
    valid=None, state=None, return_state=False, packed=None, arch="cauchy_x_hyperbolic",
):
    global FUNCTION
    if arch not in ARCH_STAGE:
        raise ValueError(f"arch must be one of {tuple(ARCH_STAGE)}, not {arch!r}")
    try:
        import torch
    except Exception:
        return None
    if not (x.dim() == 3 and gamma.dim() == 3 and raw_alpha.dim() == 4
            and raw_rho.dim() == 4 and memory_write.dim() == 3
            and memory_read.dim() == 4):
        return None
    batch, tokens, channels = x.shape
    if not batch or not tokens or not channels:
        raise ValueError("mira-dsp requires nonempty batch, token and channel axes")
    order = int(raw_alpha.shape[-1])
    memory = int(memory_write.shape[-1])
    if raw_alpha.shape != (batch, tokens, channels, order):
        return None
    if raw_rho.shape != (batch, tokens, channels, order + 1):
        return None
    if gamma.shape != x.shape:
        return None
    if memory_write.shape != (batch, tokens, memory):
        return None
    if memory_read.shape != (batch, tokens, order + 1, memory):
        return None
    if not (raw_beta is not None and raw_beta.shape == raw_alpha.shape):
        return None
    tensors = (x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read)
    if not all(_on_device(tensor) for tensor in tensors):
        return None
    if len({tensor.device for tensor in tensors}) != 1:
        return None
    if len({tensor.dtype for tensor in tensors}) != 1:
        return None
    if not _ready(order, memory, x.dtype):
        return None
    if valid is not None:
        if valid.shape != (batch, tokens) or valid.device != x.device or valid.dtype != torch.bool:
            raise ValueError("mira-dsp mask must be a boolean [batch, tokens] MPS tensor")
        valid = valid.contiguous()
    if state is not None:
        if (state.shape != (batch, order, channels, memory) or state.device != x.device
                or state.dtype != torch.float32):
            raise ValueError("mira-dsp state must be FP32 [batch, order, channels, memory]")
        if state.requires_grad:
            raise ValueError("mira-dsp cached state is inference-only")
        if not return_state:
            raise ValueError("mira-dsp initial state requires return_state=True")
        state = state.contiguous()
    if return_state:
        with torch.no_grad():
            output, final, unused = _forward(
                x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read, valid, state,
                arch, final=True, save=False)
        return output, final
    keep = torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors)
    if not keep:
        return _forward(x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read,
                        valid, None, arch, final=False, save=False)[0]
    if FUNCTION is None:
        FUNCTION = make_function()
    return FUNCTION.apply(x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read,
                          valid, arch)


def matrix_lattice_iir_control(
    x, control, coefficient_weight, coefficient_bias, spans, gamma, memory_write,
    valid=None, state=None, return_state=False, arch="cauchy_x_hyperbolic",
):
    import torch.nn.functional as F

    batch, tokens, channels = x.shape
    alpha_span = spans[0]
    order = (alpha_span[1] - alpha_span[0]) // channels
    memory = int(memory_write.shape[-1])
    coefficients = F.linear(control, coefficient_weight, coefficient_bias)
    raw_alpha, raw_beta, raw_rho, memory_read = coefficients.split(
        tuple(hi - lo for lo, hi in spans), dim=-1)
    return matrix_lattice_iir(
        x, raw_alpha.view(batch, tokens, channels, order),
        raw_rho.view(batch, tokens, channels, order + 1), gamma, memory_write,
        memory_read.view(batch, tokens, order + 1, memory),
        raw_beta=raw_beta.view(batch, tokens, channels, order), valid=valid, state=state,
        return_state=return_state, arch=arch)


def fused_scan_timing(x, control, coefficient_weight, coefficient_bias, spans, gamma,
                      memory_write, arch="cauchy_x_hyperbolic"):
    return None
