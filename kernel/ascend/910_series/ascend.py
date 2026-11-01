from __future__ import annotations

import importlib.util
from pathlib import Path
import re

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.autograd.function import once_differentiable


ORDERS = (1, 2, 4, 8)
MEMORY_DIMENSIONS = (8, 16, 32, 64)
ARCHES = tuple(f"{radius}_x_{rotation}"
               for radius in ("cauchy", "gaussian", "expexp", "expsoftplus", "hill")
               for rotation in ("mobius", "hyperbolic", "identity"))
BUILD = Path(__file__).resolve().parents[3] / "build/kernel/ascend/910_series"
MODULES = {}


def processor(device=None):
    import torch_npu

    name = torch_npu.npu.get_device_name(device)
    if not re.fullmatch(r"Ascend910[A-Za-z0-9_]*", name):
        raise RuntimeError(f"unsupported Ascend processor: {name!r}")
    return name


def library(device=None):
    return BUILD / processor(device) / "mira_ascend.so"


def status(device=None):
    try:
        import torch_npu
    except ImportError:
        return False, "torch_npu is unavailable"
    if not torch_npu.npu.is_available():
        return False, "Ascend NPU is unavailable"
    try:
        path = library(device)
    except RuntimeError as error:
        return False, str(error)
    if not path.is_file():
        return False, f"Ascend kernel is missing: {path}"
    return True, str(path)


def load(device=None):
    name = processor(device)
    if name not in MODULES:
        ready, reason = status(device)
        if not ready:
            raise RuntimeError(reason)
        spec = importlib.util.spec_from_file_location("mira_ascend", library(device))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.soc_version() != name:
            raise RuntimeError(f"Ascend kernel target {module.soc_version()!r} does not match {name!r}")
        MODULES[name] = module
    return MODULES[name]


def native_scan(matrix, carry):
    import torch_npu

    order = matrix.shape[-1]
    tokens = matrix.shape[1]
    matrix = torch_npu.npu_format_cast(F.pad(matrix, (0, 8 - order)).contiguous(), 2)
    carry = torch_npu.npu_format_cast(carry.contiguous(), 2)
    module = load(matrix.device)
    distance = 1
    while distance < tokens:
        next_matrix = torch.empty_like(matrix)
        next_carry = torch.empty_like(carry)
        module.affine(matrix, carry, next_matrix, next_carry, distance)
        matrix, carry = next_matrix, next_carry
        distance *= 2
    return matrix[..., :order], carry


def apply_matrix(matrix, state):
    return (matrix.unsqueeze(-1) * state.unsqueeze(-3)).sum(dim=-2)


class AffineScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, carry, initial):
        prefix, values = native_scan(matrix, carry)
        values = values + apply_matrix(prefix, initial.unsqueeze(1))
        ctx.save_for_backward(matrix, values, initial)
        return values

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        matrix, values, initial = ctx.saved_tensors
        zero = torch.zeros_like(matrix[:, :1])
        reverse_matrix = torch.cat((matrix[:, 1:].transpose(-1, -2), zero), dim=1).flip(1)
        unused, adjoint = native_scan(reverse_matrix, gradient.flip(1))
        del unused
        adjoint = adjoint.flip(1)
        previous = torch.cat((initial.unsqueeze(1), values[:, :-1]), dim=1)
        matrix_gradient = (adjoint.unsqueeze(-2) * previous.unsqueeze(-3)).sum(dim=-1)
        initial_gradient = apply_matrix(matrix[:, 0].transpose(-1, -2), adjoint[:, 0])
        return matrix_gradient, adjoint, initial_gradient


def stage_maps(alpha, beta, arch):
    radius_name, rotation = arch.split("_x_")
    if rotation == "mobius":
        small = alpha.abs() <= 1.0
        bounded = torch.where(small, alpha, torch.zeros_like(alpha))
        inverse = torch.where(small, torch.ones_like(alpha), alpha).reciprocal()
        square, inverse_square = bounded.square(), inverse.square()
        cosine = torch.where(small, 2.0 * bounded / (1.0 + square),
                             2.0 * inverse / (1.0 + inverse_square))
        sine = torch.where(small, (1.0 - square) / (1.0 + square),
                           (inverse_square - 1.0) / (1.0 + inverse_square))
    elif rotation == "hyperbolic":
        decay = torch.exp(-alpha.abs())
        cosine, sine = 2.0 * decay / (1.0 + decay.square()), torch.tanh(alpha)
    else:
        cosine, sine = torch.cos(alpha), torch.sin(alpha)
    if radius_name == "cauchy":
        radius = (1.0 + beta.square()).reciprocal()
    elif radius_name == "gaussian":
        radius = torch.exp(-beta.square())
    elif radius_name == "expexp":
        radius = torch.exp(-torch.exp(beta.clamp(max=80.0)))
    elif radius_name == "expsoftplus":
        radius = torch.sigmoid(-beta)
    else:
        square = beta.square()
        large = square > 1.0
        bounded = torch.where(large, torch.zeros_like(square), square)
        radius = torch.where(large, 1.0 - (1.0 + square).reciprocal(),
                             bounded / (1.0 + bounded))
    return radius * cosine, radius * sine


def lattice(x, alpha, beta, rho, gamma, write, read, valid, initial, arch):
    batch, tokens, channels = x.shape
    order, memory = alpha.shape[-1], write.shape[-1]
    cosine, sine = stage_maps(alpha, beta, arch)
    basis = torch.eye(order + 1, dtype=torch.float32, device=x.device)
    wave = basis[0].expand(batch, tokens, channels, order + 1)
    levels = [None] * (order + 1)
    for stage in range(order, 0, -1):
        delayed = basis[stage]
        c = cosine[..., stage - 1:stage]
        s = sine[..., stage - 1:stage]
        levels[stage] = s * wave + c * delayed
        wave = c * wave - s * delayed
    levels[0] = wave
    maps = torch.stack(levels, dim=-2)
    matrix = maps[..., :order, 1:]
    injection = x.unsqueeze(-1) * write.unsqueeze(-2)
    carry = maps[..., :order, :1] * injection.unsqueeze(-2)
    if valid is not None:
        keep = valid[:, :, None, None, None]
        identity = torch.eye(order, dtype=torch.float32, device=x.device)
        matrix = torch.where(keep, matrix, identity)
        carry = torch.where(keep, carry, torch.zeros_like(carry))
    values = AffineScan.apply(matrix.contiguous(), carry.contiguous(), initial)
    previous = torch.cat((initial.unsqueeze(1), values[:, :-1]), dim=1)
    all_levels = apply_matrix(maps[..., 1:], previous) + maps[..., :1] * injection.unsqueeze(-2)
    compressed = (all_levels * read.unsqueeze(2)).sum(dim=-1)
    output = (compressed * rho).sum(dim=-1) * F.silu(gamma)
    if valid is not None:
        output = torch.where(valid.unsqueeze(-1), output, torch.zeros_like(output))
    return output, values[:, -1].permute(0, 2, 1, 3).contiguous()


def matrix_lattice_iir(
    x, raw_alpha, raw_rho, gamma, memory_write, memory_read, raw_beta=None,
    valid=None, state=None, return_state=False, packed=None, arch="cauchy_x_hyperbolic",
):
    if arch not in ARCHES:
        raise ValueError(f"unsupported Ascend lattice arch: {arch!r}")
    if x.ndim != 3 or raw_alpha.ndim != 4 or memory_write.ndim != 3:
        raise ValueError("Ascend lattice input ranks are incompatible")
    batch, tokens, channels = x.shape
    order, memory = raw_alpha.shape[-1], memory_write.shape[-1]
    if min(batch, tokens, channels) < 1:
        raise ValueError("Ascend lattice requires nonempty batch, token and channel axes")
    if order not in ORDERS or memory not in MEMORY_DIMENSIONS:
        raise ValueError(f"unsupported Ascend lattice geometry: K={order}, memory={memory}")
    expected = ((raw_alpha, (batch, tokens, channels, order)),
                (raw_beta, (batch, tokens, channels, order)),
                (raw_rho, (batch, tokens, channels, order + 1)),
                (gamma, x.shape), (memory_write, (batch, tokens, memory)),
                (memory_read, (batch, tokens, order + 1, memory)))
    tensors = (x,) + tuple(value for value, shape in expected)
    for value, shape in expected:
        if value is None or value.shape != shape:
            raise ValueError(f"Ascend lattice tensor must have shape {tuple(shape)}")
    if x.device.type != "npu" or any(value.device != x.device for value in tensors):
        raise ValueError("Ascend lattice tensors must be on one NPU")
    if any(value.dtype != x.dtype for value in tensors) or x.dtype not in (
            torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Ascend lattice inputs must share FP32, FP16 or BF16 dtype")
    if x.dtype == torch.bfloat16:
        import torch_npu
        with torch_npu.npu.device(x.device):
            if not torch_npu.npu.is_bf16_supported():
                raise ValueError("BF16 is unsupported by the selected Ascend processor")
    if valid is not None and (valid.shape != (batch, tokens) or valid.device != x.device
                              or valid.dtype != torch.bool):
        raise ValueError("Ascend lattice mask must be boolean [B,T] on the input NPU")
    if state is not None:
        if state.shape != (batch, order, channels, memory) or state.device != x.device or state.dtype != torch.float32:
            raise ValueError("Ascend lattice state must be FP32 [B,K,C,M] on the input NPU")
        if state.requires_grad or not return_state:
            raise ValueError("Ascend lattice cached state requires inference and return_state=True")
    if packed is not None:
        if len(packed) != 2 or packed[0].shape != (batch, tokens, channels + memory) or packed[1].shape != (
                batch, tokens, channels * (3 * order + 1) + (order + 1) * memory):
            raise ValueError("Ascend lattice packed projection shapes are incompatible")
        if any(value.device != x.device or value.dtype != x.dtype for value in packed):
            raise ValueError("Ascend lattice packed projections must share input device and dtype")
    load(x.device)
    initial = (torch.zeros(batch, channels, order, memory, dtype=torch.float32, device=x.device)
               if state is None else state.permute(0, 2, 1, 3).contiguous())
    with torch.autocast(device_type="npu", enabled=False):
        outputs, states = [], []
        for first in range(0, channels, 128):
            last = min(first + 128, channels)
            arguments = (x[:, :, first:last].float(), raw_alpha[:, :, first:last].float(),
                         raw_beta[:, :, first:last].float(), raw_rho[:, :, first:last].float(),
                         gamma[:, :, first:last].float(), memory_write.float(), memory_read.float(),
                         valid, initial[:, first:last], arch)
            if return_state:
                with torch.no_grad():
                    output, final = lattice(*arguments)
                states.append(final)
            elif torch.is_grad_enabled() and any(value.requires_grad for value in tensors):
                output, unused = checkpoint(lattice, *arguments, use_reentrant=False)
            else:
                output, unused = lattice(*arguments)
            outputs.append(output.to(x.dtype))
        mixed = torch.cat(outputs, dim=-1)
        return (mixed, torch.cat(states, dim=2)) if return_state else mixed
