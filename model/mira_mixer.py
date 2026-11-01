from __future__ import annotations

import importlib
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

# alpha IS AN ANGLE AND beta A RADIUS, so neither carries an activation and neither needs a
# clamp: alpha enters through the Cayley rotation, whose cos and sin are bounded for every real
# argument, and beta through r = 1 / (1 + beta^2), which lies in (0, 1] for every real argument.
#     (U_t^(k-1), V_t^(k)) = r_t^(k) R(theta_t^(k)) (U_t^(k), V_{t-1}^(k-1)),   k = K..1
#     sin(theta) = (1 - alpha^2) / (1 + alpha^2),  cos(theta) = 2 alpha / (1 + alpha^2)
#     r = 1 / (1 + beta^2)
# under arch="cauchy_x_mobius", and the other cayley_parameterizations <radius>_x_<rotation> swap in
#     rotation hyperbolic  sin(theta) = tanh(alpha),  cos(theta) = sech(alpha)
#     rotation identity    theta = alpha: sin(theta) = sin(alpha),  cos(theta) = cos(alpha)
#     radius   gaussian    r = exp(-beta^2)
#     radius   expexp      r = exp(-exp(beta))
#     radius   expsoftplus r = exp(-exp(s_r^(c)) softplus(beta)), s_r^(c) per channel
#     radius   hill        r = beta^2 / (1 + beta^2) = 1 - 1 / (1 + beta^2)
#     U_t^(K) = x-tilde_t s_t^T,  V_t^(0) = U_t^(0),  persistent state V^(0..K-1)
#     y_t = W_out ( ( sum_{k=0}^{K} rho_t^(k) . (V_t^(k) nu_t^(k)) ) . gamma_t )
#
# The readout carries no activation. The gate keeps its SiLU.


import default_config as config


def _configured(key):
    """The configured value for one geometry key. This module holds no copy of the numbers."""
    import default_config as config
    return int(config.MODEL[key])


# THE INITIALIZERS: model/mira_initializer.py registers them and calls the mixer's, each lives in
# model/initializers/<name>.py and their shared parts in model/initializers/_common.py. These names
# are re-exported so every existing `from model.mira_mixer import ...` of them still resolves.
mira_initializer = importlib.import_module("model.mira_initializer")
_initializer_common = importlib.import_module("model.initializers._common")
parse_lattice_initializer = mira_initializer.parse_lattice_initializer
parse_lattice_init_horizon = mira_initializer.parse_lattice_init_horizon
_multiply_polynomials = _initializer_common.multiply_polynomials
_initial_reflection_coefficients = _initializer_common.initial_reflection_coefficients

# THE DIAGNOSTICS (the dynamics profile) live in model/mira_profiling.py; the mixer only hands it
# its tensors through the record_* calls below, and does nothing else when the profile is off.
mira_profiling = importlib.import_module("model.mira_profiling")


def cayley_rotation(alpha):
    """cos and sin of theta = pi/2 - 2 arctan(alpha), without trigonometry.

        sin = (1 - alpha^2) / (1 + alpha^2),   cos = 2 alpha / (1 + alpha^2).

    This is the Cayley rotation shifted by pi/2. It is orthogonal for every real alpha and puts
    the K=1 recurrent eigenvalue -r sin(theta) near -r under a zero-centred nonzero alpha
    initialization instead of putting that eigenvalue near zero.

    THE RECIPROCAL BRANCH IS FOR MAGNITUDE, NOT FOR CORRECTNESS. Both branches evaluate the same
    rational map; at |alpha| > 1 it is written in 1/alpha so that alpha^2 never overflows."""
    alpha = alpha.float()
    small = alpha.abs() <= 1.0
    bounded = torch.where(small, alpha, torch.zeros_like(alpha))
    reciprocal = torch.where(small, torch.ones_like(alpha), alpha).reciprocal()
    square = bounded.square()
    inverse_square = reciprocal.square()
    cosine = torch.where(small, 2.0 * bounded / (1.0 + square),
                         2.0 * reciprocal / (1.0 + inverse_square))
    sine = torch.where(small, (1.0 - square) / (1.0 + square),
                       (inverse_square - 1.0) / (1.0 + inverse_square))
    return cosine, sine


def hyperbolic_rotation(alpha):
    """cos and sin of the hyperbolic stage: sech(alpha) and tanh(alpha).

        sech = 2 e / (1 + e^2),   e = exp(-|alpha|).

    Orthogonal for every real alpha, since tanh^2 + sech^2 = 1."""
    alpha = alpha.float()
    decay = torch.exp(-alpha.abs())
    cosine = 2.0 * decay / (1.0 + decay * decay)
    return cosine, torch.tanh(alpha)


def identity_rotation(alpha):
    """cos and sin of the identity stage: theta = alpha, so cos(alpha) and sin(alpha).

    Orthogonal for every real alpha, since sin^2 + cos^2 = 1, and d theta / d alpha = 1."""
    alpha = alpha.float()
    return torch.cos(alpha), torch.sin(alpha)


def cayley_radius(beta):
    """r = 1 / (1 + beta^2)."""
    return torch.reciprocal(1.0 + beta * beta)


def gaussian_radius(beta):
    """r = exp(-beta^2)."""
    return torch.exp(-(beta * beta))


EXPEXP_BETA_CEILING = 80.0


def expexp_radius(beta):
    """r = exp(-exp(beta)), so log(-log r) = beta.

    beta is capped at EXPEXP_BETA_CEILING before the inner exponential: above it r is already 0 in
    float32 (exp(-exp(80)) underflows) and so is dr/dbeta = -exp(beta) r, while exp(beta) itself
    would reach inf past 88.7 and turn the zero gradient into inf * 0. The CUDA kernel caps it the
    same way."""
    return torch.exp(-torch.exp(beta.clamp(max=EXPEXP_BETA_CEILING)))


def expsoftplus_radius(beta):
    """Unscaled helper: r = exp(-softplus(beta)) = sigmoid(-beta).

    The stage horizon is H = -1 / log r = 1 / softplus(beta), so beta = log(expm1(1 / H)), and
    |d log H / d beta| = sigmoid(beta) / softplus(beta) = H (1 - exp(-1 / H)) <= 1 for every H.
    The mixer applies its per-channel exp(s_r) scale separately. This helper is also used by
    callers that do not provide an s_r tensor."""
    return torch.sigmoid(-beta)


def hill_radius(beta):
    """r = q / (1 + q), q = beta^2: the Hill function of coefficient 2, 1 - r = 1 / (1 + q).

    dr/dbeta = 2 beta / (1 + q)^2. Above q = 1, r is 1 - (1 - r) with 1 - r = 1 / (1 + q) formed
    directly; at and below it, r is q / (1 + q), evaluated on q only where q <= 1, so neither
    branch overflows."""
    square = beta * beta
    large = square > 1.0
    bounded = torch.where(large, torch.zeros_like(square), square)
    return torch.where(large, 1.0 - torch.reciprocal(1.0 + square), bounded / (1.0 + bounded))


def stage_maps(arch):
    """(rotation, radius) for `arch`: the functions from alpha to (cos, sin) and from beta to r."""
    rotations = {"mobius": cayley_rotation, "hyperbolic": hyperbolic_rotation,
                 "identity": identity_rotation}
    radii = {"cauchy": cayley_radius, "gaussian": gaussian_radius, "expexp": expexp_radius,
             "expsoftplus": expsoftplus_radius, "hill": hill_radius}
    maps = {f"{radius}_x_{rotation}": (rotations[rotation], radii[radius])
            for radius in radii for rotation in rotations}
    if arch not in maps:
        raise ValueError(f"arch must be one of {tuple(maps)}, not {arch!r}")
    return maps[arch]


def _declined(family, status, signal, order, memory_dimension):
    """The message a declined kernel raises. MIRA does not run on the PyTorch reference."""
    batch, tokens, channels = signal.shape
    detail = status[1] if isinstance(status, tuple) else status
    return (
        f"the {family} kernel declined [B,T,C]=[{batch},{tokens},{channels}] "
        f"d_m={memory_dimension} K={order} dtype={signal.dtype}: {detail}. "
        f"MIRA does not fall back to the PyTorch reference -- build the cubin with "
        f"make -C kernel/cuda/{family} universal, then make -C kernel/cuda/{family} check.")


def stage_system(alpha, beta, arch):
    """(A, b, C, d) of the assembled lattice for coefficients alpha, beta [..., K]: A [..., K, K],
    b [..., K], C [..., K+1, K], d [..., K+1], formed by pushing the K+1 basis inputs (the write, then
    V_{t-1}^(0..K-1)) through the same stage code the reference readout runs."""
    rotation, radius_map = stage_maps(arch)
    order = alpha.shape[-1]
    eye = torch.eye(order + 1, dtype=alpha.dtype, device=alpha.device)
    lead = alpha.shape[:-1]
    wave = eye[0].expand(*lead, order + 1)
    delayed = [eye[1 + j].expand(*lead, order + 1) for j in range(order)]
    levels = [None] * (order + 1)
    for stage in range(order, 0, -1):
        cosine, sine = rotation(alpha[..., stage - 1:stage])
        radius = radius_map(beta[..., stage - 1:stage])
        cosine, sine, radius = cosine.to(alpha.dtype), sine.to(alpha.dtype), radius.to(alpha.dtype)
        levels[stage] = radius * (sine * wave + cosine * delayed[stage - 1])
        wave = radius * (cosine * wave - sine * delayed[stage - 1])
    levels[0] = wave
    maps = torch.stack(levels, dim=-2)
    return maps[..., :order, 1:], maps[..., :order, 0], maps[..., :, 1:], maps[..., :, 0]


# THE MIRA CUDA BACKEND (2026-10-12): kernel/cuda/mira_dsp (named mira_v7_stable before; the
# coefficient-forming inference scan with the next-token x / gamma prefetch fixed) is the only MIRA
# CUDA package and the default for training, evaluation, the benchmark, the ablation and the
# profiler. mira_v1 .. mira_v6 and mira_v5_stable were removed on 2026-10-12 (git history keeps
# them). The package exposes matrix_lattice_iir, matrix_lattice_iir_control, status, usage and
# report, and the mixer reads whichever module this name holds at call time.
MIRA_CUDA = os.environ.get("MIRA_CUDA_BIN", config.MIRA_CUDA_BIN)
MIRA_CUDA_VERSIONS = config.MIRA_CUDA_BINS


def _import_mira_cuda(version):
    try:
        return importlib.import_module(f"kernel.cuda.{version}.mira")
    except Exception:
        return None


matrix_lattice_cuda = _import_mira_cuda(MIRA_CUDA)


def use_mira_cuda(version):
    """Select the MIRA CUDA package (one of MIRA_CUDA_VERSIONS) for every later call in this process."""
    global MIRA_CUDA, matrix_lattice_cuda
    if version not in MIRA_CUDA_VERSIONS:
        raise ValueError(f"MIRA CUDA version must be one of {MIRA_CUDA_VERSIONS}, not {version!r}")
    MIRA_CUDA = version
    matrix_lattice_cuda = _import_mira_cuda(version)
    return matrix_lattice_cuda

# THE INFERENCE STATE, AND ONLY ITS SWITCH AND ITS CEILING ARE READ HERE. The container itself
# is built by the caller that decodes -- mirachat.py -- and reaches the mixer as `cache`.
#
# THROUGH importlib, AS THE KERNEL ABOVE IS. This module is imported from model/__init__.py by
# way of model/mira_llm.py, so `model` is in sys.modules but only partly built: `from model
# import prefix_cache` would be an attribute lookup on that half-finished package and raise.
# import_module loads the submodule and returns it directly, which is defined at any point in
# the parent's own import.
prefix_cache = importlib.import_module("model.prefix_cache")


def available(device=None):
    from kernel.hardware import resolve
    selected = resolve(os.environ.get("GPU", "auto")) if device is None else torch.device(device)
    if selected.type == "npu":
        return importlib.import_module("kernel.ascend.910_series.ascend").status(selected)
    if matrix_lattice_cuda is None:
        return False, f"kernel.cuda.{MIRA_CUDA} did not import"
    return matrix_lattice_cuda.status()


def usage():
    return (matrix_lattice_cuda.usage() if matrix_lattice_cuda is not None else
            {"forward": 0, "backward": 0})


def matrix_lattice_iir(
    x: torch.Tensor,
    raw_alpha: torch.Tensor,
    raw_rho: torch.Tensor,
    gamma: torch.Tensor,
    memory_write: torch.Tensor,
    memory_read: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    chunk: int = 32,
    raw_beta: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    return_state: bool = False,
    arch: str = "cauchy_x_hyperbolic",
):
    """The canonical ladder, with the K stages of each token solved at once (see the comment above
    `segment`); matrix_lattice_iir_sequential runs the same stages one at a time and is its reference.

        U_t^(k-1) = r_t^(k) ( cos(theta_t^(k)) U_t^(k) - sin(theta_t^(k)) V_{t-1}^(k-1) )
        V_t^(k)   = r_t^(k) ( sin(theta_t^(k)) U_t^(k) + cos(theta_t^(k)) V_{t-1}^(k-1) )

    for k = K..1 from U_t^(K) = x-tilde_t s_t^T, closing with V_t^(0) = U_t^(0) and carrying the
    persistent state V^(0..K-1). Nothing the stages compute is left unread.

    `state` IS V_{-1}^(0..K-1), AND THE DEFAULT IS THE ZEROS EVERY TRAINING WINDOW STARTS FROM.
    Given one, the scan begins from it instead; `return_state=True` additionally hands back
    V_{T-1}^(0..K-1), so a call over tokens [0,n) followed by a call over [n,T) carrying that
    state produces what one call over [0,T) produces. This is not a new mechanism -- the chunk
    loop below already threads exactly this tensor from chunk to chunk, and this makes its two
    ends reachable by a caller. Both are [B, K, C, M], which is model/prefix_cache.py's shape.

    NEITHER ARGUMENT IS USED IN TRAINING. model.lm.train calls the model without a cache, so
    nothing reaches these and the scan is byte-for-byte the one it always ran.
    """
    if raw_beta is None:
        raise ValueError("the canonical stage needs raw_beta, the radius coefficient")
    if x.ndim != 3:
        raise ValueError("x must have shape [B,T,C]")
    batch, tokens, channels = x.shape
    if raw_alpha.ndim != 4:
        raise ValueError("raw_alpha must have shape [B,T,C,K]")
    order = raw_alpha.shape[3]
    if raw_alpha.shape != (batch, tokens, channels, order):
        raise ValueError("raw_alpha has an incompatible shape")
    if raw_rho.shape != (batch, tokens, channels, order + 1):
        raise ValueError("raw_rho must have shape [B,T,C,K+1]")
    if gamma.shape != x.shape:
        raise ValueError("gamma must have the same shape as x")
    if memory_write.ndim != 3 or memory_write.shape[:2] != (batch, tokens):
        raise ValueError("memory_write must have shape [B,T,M]")
    if memory_read.shape != (batch, tokens, order + 1, memory_write.shape[-1]):
        raise ValueError("memory_read must have shape [B,T,K+1,M]")
    if valid is not None and valid.shape != (batch, tokens):
        raise ValueError("valid must have shape [B,T]")

    rotation, radius_map = stage_maps(arch)
    signal = x.float()
    alpha = raw_alpha.float()
    beta = None if raw_beta is None else raw_beta.float()
    rho = raw_rho.float()
    gate = F.silu(gamma.float())
    write = memory_write.float()
    read = memory_read.float()
    memory = write.shape[-1]
    if state is None:
        state = torch.zeros(
            batch, order, channels, memory, device=x.device, dtype=torch.float32)
    else:
        if state.shape != (batch, order, channels, memory):
            raise ValueError(
                f"state must have shape [B,K,C,M] = "
                f"{(batch, order, channels, memory)}, got {tuple(state.shape)}")
        state = state.to(device=x.device, dtype=torch.float32)

    # ONE CHUNK AT A TIME, RECOMPUTED IN BACKWARD. The scan writes order + 1 tensors of
    # [B, C, M] per token, so a 512-token sequence holds T x (order + 1) of them for the
    # backward pass; at the benchmark shape that is what exhausts the card. Each chunk keeps
    # its own activations and the boundary state, and is recomputed rather than stored.
    #
    # THE K STAGES ARE SOLVED AT ONCE, NOT ONE AFTER ANOTHER. With D^(k) = r^(k) cos(theta^(k))
    # and E^(k) = r^(k) sin(theta^(k)) (stage k at index k - 1), the stage recursion
    #     U_t^(k-1) = D^(k) U_t^(k) - E^(k) V_{t-1}^(k-1),   V_t^(k) = E^(k) U_t^(k) + D^(k) V_{t-1}^(k-1)
    # unrolls to
    #     U_t^(k) = G[k, K] U_t^(K) - sum_{j >= k} G[k, j] E^(j+1) V_{t-1}^(j),
    #     G[k, i] = D^(k+1) ... D^(i) for i >= k (G[k, k] = 1), else 0,
    # so V_t^(0..K) = Phi_t V_{t-1}^(0..K-1) + S_t with S_t = S_coef_t U_t^(K): one contraction per
    # token over every stage, the product version of the manuscript's pseudo code. Rows 0..K-1 of
    # Phi_t are the state transition and row K gives V_t^(K).
    upper = torch.ones(order + 1, order + 1, dtype=torch.bool, device=x.device).triu()
    strictly_upper = upper.triu(1)

    def stage_operators(alpha, beta):
        """(Phi [B,c,C,K+1,K], S_coef [B,c,C,K+1]) of every token and stage of a chunk."""
        cosine, sine = rotation(alpha)
        radius = radius_map(beta)
        stage_d = radius * cosine
        stage_e = radius * sine
        one = torch.ones_like(stage_d[..., :1])
        d_with_one = torch.cat([one, stage_d], dim=-1)
        e_with_one = torch.cat([one, stage_e], dim=-1)
        products = torch.where(strictly_upper, d_with_one[..., None, :], 1.0).cumprod(-1) * upper
        coefficient = e_with_one * products[..., order]
        transition = (-e_with_one[..., None] * products[..., :order] * stage_e[..., None, :]
                      + torch.diag_embed(stage_d, offset=-1)[..., :order])
        return transition, coefficient

    def segment(signal, alpha, beta, rho, gate, write, read, valid, state):
        tokens = signal.shape[1]
        transition, coefficient = stage_operators(alpha, beta)
        outputs = []
        for token in range(tokens):
            U = signal[:, token].unsqueeze(-1) * write[:, token].unsqueeze(-2)
            all_levels = (torch.einsum("bckj,bjcm->bkcm", transition[:, token], state)
                          + coefficient[:, token].transpose(1, 2).unsqueeze(-1) * U.unsqueeze(1))
            next_state = all_levels[:, :order]
            compressed = torch.einsum(
                "bkcm,bkm->bkc", all_levels, read[:, token])
            summed = (rho[:, token].transpose(1, 2) * compressed).sum(dim=1)
            mixed = summed * gate[:, token]
            if valid is None:
                state = next_state
            else:
                keep = valid[:, token]
                state = torch.where(keep.view(batch, 1, 1, 1), next_state, state)
                mixed = torch.where(keep.view(batch, 1), mixed, torch.zeros_like(mixed))
            outputs.append(mixed)
        return torch.stack(outputs, dim=1), state

    pieces = []
    for first in range(0, tokens, chunk):
        last = min(first + chunk, tokens)
        arguments = (signal[:, first:last], alpha[:, first:last],
                     None if beta is None else beta[:, first:last], rho[:, first:last],
                     gate[:, first:last], write[:, first:last], read[:, first:last],
                     None if valid is None else valid[:, first:last], state)
        if torch.is_grad_enabled() and any(
                torch.is_tensor(value) and value.requires_grad for value in arguments):
            piece, state = checkpoint.checkpoint(segment, *arguments, use_reentrant=False)
        else:
            piece, state = segment(*arguments)
        pieces.append(piece)
    mixed = torch.cat(pieces, dim=1).to(x.dtype)
    return (mixed, state) if return_state else mixed


def matrix_lattice_iir_sequential(
    x: torch.Tensor,
    raw_alpha: torch.Tensor,
    raw_rho: torch.Tensor,
    gamma: torch.Tensor,
    memory_write: torch.Tensor,
    memory_read: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    chunk: int = 32,
    raw_beta: Optional[torch.Tensor] = None,
    state: Optional[torch.Tensor] = None,
    return_state: bool = False,
    arch: str = "cauchy_x_hyperbolic",
):
    """The canonical ladder, the stage loop k = K..1 run one stage at a time: the sequential reference
    matrix_lattice_iir is verified against (same arguments, same results to float rounding).

        U_t^(k-1) = r_t^(k) ( cos(theta_t^(k)) U_t^(k) - sin(theta_t^(k)) V_{t-1}^(k-1) )
        V_t^(k)   = r_t^(k) ( sin(theta_t^(k)) U_t^(k) + cos(theta_t^(k)) V_{t-1}^(k-1) )

    for k = K..1 from U_t^(K) = x-tilde_t s_t^T, closing with V_t^(0) = U_t^(0) and carrying the
    persistent state V^(0..K-1). Nothing the stages compute is left unread.

    `state` IS V_{-1}^(0..K-1), AND THE DEFAULT IS THE ZEROS EVERY TRAINING WINDOW STARTS FROM.
    Given one, the scan begins from it instead; `return_state=True` additionally hands back
    V_{T-1}^(0..K-1), so a call over tokens [0,n) followed by a call over [n,T) carrying that
    state produces what one call over [0,T) produces. This is not a new mechanism -- the chunk
    loop below already threads exactly this tensor from chunk to chunk, and this makes its two
    ends reachable by a caller. Both are [B, K, C, M], which is model/prefix_cache.py's shape.

    NEITHER ARGUMENT IS USED IN TRAINING. model.lm.train calls the model without a cache, so
    nothing reaches these and the scan is byte-for-byte the one it always ran.
    """
    if raw_beta is None:
        raise ValueError("the canonical stage needs raw_beta, the radius coefficient")
    if x.ndim != 3:
        raise ValueError("x must have shape [B,T,C]")
    batch, tokens, channels = x.shape
    if raw_alpha.ndim != 4:
        raise ValueError("raw_alpha must have shape [B,T,C,K]")
    order = raw_alpha.shape[3]
    if raw_alpha.shape != (batch, tokens, channels, order):
        raise ValueError("raw_alpha has an incompatible shape")
    if raw_rho.shape != (batch, tokens, channels, order + 1):
        raise ValueError("raw_rho must have shape [B,T,C,K+1]")
    if gamma.shape != x.shape:
        raise ValueError("gamma must have the same shape as x")
    if memory_write.ndim != 3 or memory_write.shape[:2] != (batch, tokens):
        raise ValueError("memory_write must have shape [B,T,M]")
    if memory_read.shape != (batch, tokens, order + 1, memory_write.shape[-1]):
        raise ValueError("memory_read must have shape [B,T,K+1,M]")
    if valid is not None and valid.shape != (batch, tokens):
        raise ValueError("valid must have shape [B,T]")

    rotation, radius_map = stage_maps(arch)
    signal = x.float()
    alpha = raw_alpha.float()
    beta = None if raw_beta is None else raw_beta.float()
    rho = raw_rho.float()
    gate = F.silu(gamma.float())
    write = memory_write.float()
    read = memory_read.float()
    memory = write.shape[-1]
    if state is None:
        state = torch.zeros(
            batch, order, channels, memory, device=x.device, dtype=torch.float32)
    else:
        if state.shape != (batch, order, channels, memory):
            raise ValueError(
                f"state must have shape [B,K,C,M] = "
                f"{(batch, order, channels, memory)}, got {tuple(state.shape)}")
        state = state.to(device=x.device, dtype=torch.float32)

    # ONE CHUNK AT A TIME, RECOMPUTED IN BACKWARD. The scan writes order + 1 tensors of
    # [B, C, M] per token, so a 512-token sequence holds T x (order + 1) of them for the
    # backward pass; at the benchmark shape that is what exhausts the card. Each chunk keeps
    # its own activations and the boundary state, and is recomputed rather than stored.
    def segment(signal, alpha, beta, rho, gate, write, read, valid, state):
        tokens = signal.shape[1]
        outputs = []
        for token in range(tokens):
            U = signal[:, token].unsqueeze(-1) * write[:, token].unsqueeze(-2)
            levels = [None] * (order + 1)
            for stage in range(order, 0, -1):
                delayed = state[:, stage - 1]
                coefficient = alpha[:, token, :, stage - 1].unsqueeze(-1)
                damping = beta[:, token, :, stage - 1].unsqueeze(-1)
                # NO TRIGONOMETRY. The shifted Cayley map forms
                #     sin = (1 - alpha^2) / (1 + alpha^2),
                #     cos = 2 alpha / (1 + alpha^2),
                # and the hyperbolic map forms sin = tanh(alpha), cos = sech(alpha),
                # directly from the coefficient, so the rotation stays bounded and orthogonal
                # for every real alpha without evaluating a periodic function. The identity map
                # (<radius>_x_identity) is the one exception: theta = alpha, sin(alpha), cos(alpha).
                cosine, sine = rotation(coefficient)
                radius = radius_map(damping)
                levels[stage] = radius * (sine * U + cosine * delayed)
                U = radius * (cosine * U - sine * delayed)
            levels[0] = U
            all_levels = torch.stack(levels, dim=1)
            next_state = all_levels[:, :order]
            compressed = torch.einsum(
                "bkcm,bkm->bkc", all_levels, read[:, token])
            summed = (rho[:, token].transpose(1, 2) * compressed).sum(dim=1)
            mixed = summed * gate[:, token]
            if valid is None:
                state = next_state
            else:
                keep = valid[:, token]
                state = torch.where(keep.view(batch, 1, 1, 1), next_state, state)
                mixed = torch.where(keep.view(batch, 1), mixed, torch.zeros_like(mixed))
            outputs.append(mixed)
        return torch.stack(outputs, dim=1), state

    pieces = []
    for first in range(0, tokens, chunk):
        last = min(first + chunk, tokens)
        arguments = (signal[:, first:last], alpha[:, first:last],
                     None if beta is None else beta[:, first:last], rho[:, first:last],
                     gate[:, first:last], write[:, first:last], read[:, first:last],
                     None if valid is None else valid[:, first:last], state)
        if torch.is_grad_enabled() and any(
                torch.is_tensor(value) and value.requires_grad for value in arguments):
            piece, state = checkpoint.checkpoint(segment, *arguments, use_reentrant=False)
        else:
            piece, state = segment(*arguments)
        pieces.append(piece)
    mixed = torch.cat(pieces, dim=1).to(x.dtype)
    return (mixed, state) if return_state else mixed


def _drop_write_bias(module, state_dict, prefix, local_metadata, strict,
                     missing_keys, unexpected_keys, error_msgs):
    """s_t = SiLU(raw_t) has no write bias. A state dict written between 437191c and the restore (2026-10-03) carries
    `write_bias`; it is dropped here so such a checkpoint still loads. A nonzero b_s cannot be represented, so the
    loaded model then differs from the one that was trained, and a warning says so."""
    key = prefix + "write_bias"
    if key in state_dict:
        value = state_dict.pop(key)
        if bool((value != 0).any()):
            import warnings
            warnings.warn(f"{key}: dropped a nonzero write bias (max |b_s| {float(value.abs().max()):.4g}); "
                          "s_t = SiLU(raw_t) has no write bias, so this checkpoint's outputs change")


def load_input_bias(module, state_dict, prefix, local_metadata, strict,
                    missing_keys, unexpected_keys, error_msgs):
    removed = tuple(tuple(state_dict[prefix + name].shape)
                    for name in ("signal_bias", "write_bias") if prefix + name in state_dict)
    module.input_projection.weight._mira_removed_input_bias_shapes = removed
    for name in ("input_projection.bias", "signal_bias", "write_bias"):
        state_dict.pop(prefix + name, None)


class MiraMixer(nn.Module):
    def __init__(
        self,
        width: int,
        lattice_order: int = None,
        d_e: int = None,
        coefficient_bottleneck: int = None,
        memory_dimension: int = None,
        dropout: float = 0.0,
        positive_poles: bool = False,
        lattice_initializer: str = "W_uniform_b_jeffreys",
        lattice_init_horizon=None,
        arch: str = "cauchy_x_hyperbolic",
    ):
        super().__init__()
        import default_config as config
        arch = config.lattice_arch(arch) or arch
        stage_maps(arch)
        if lattice_order is None:
            lattice_order = _configured("lattice_order")
        if d_e is None:
            d_e = _configured("lattice_d_e")
        if coefficient_bottleneck is None:
            coefficient_bottleneck = _configured("coefficient_bottleneck")
        if memory_dimension is None:
            memory_dimension = _configured("memory_dimension")
        if not 1 <= int(lattice_order) <= 16:
            raise ValueError("lattice_order must be in [1,16]")
        self.width = int(width)
        self.order = int(lattice_order)
        self.lattice_order = self.order
        self.inner = int(d_e)
        self.coefficient_bottleneck = int(coefficient_bottleneck)
        self.memory_dimension = int(memory_dimension)
        self.positive_poles = bool(positive_poles)
        self.arch = arch
        self.lattice_initializer = parse_lattice_initializer(lattice_initializer)
        # ONE RANGE PER MIXER. MiraLLM resolves a per-layer mapping to this layer's range before the
        # block is built; a mixer built on its own from a mapping takes the mapping's default range.
        _horizon = parse_lattice_init_horizon(lattice_init_horizon)
        self.lattice_init_horizon = _horizon["default"] if isinstance(_horizon, dict) else _horizon
        if self.positive_poles and self.order != 1:
            raise ValueError("positive_poles requires lattice_order=1")
        if self.positive_poles and self.arch != "cauchy_x_mobius":
            raise ValueError("positive_poles requires arch='cauchy_x_mobius'")
        if min(self.inner, self.coefficient_bottleneck, self.memory_dimension) < 1:
            raise ValueError("d_e, coefficient_bottleneck, and memory_dimension must be positive")
        channels, order = self.inner, self.order
        # THREE PROJECTIONS, NOT ONE FUSED SPLIT. The signal is read from the convolved
        # representation, and the bottleneck and the gate from the normalized one before the
        # convolution; a single in_proj cannot take two different sources.
        #
        # THE GATE READS z, NOT x. text.tex:257 defines gamma_t = SiLU(W_gamma z_t + b_gamma) on
        # the RMSNormed input, as text.tex:154 and :171 do for s_t and c_t.
        self.input_projection = nn.Linear(
            self.width, channels + self.memory_dimension, bias=False)
        self.to_control = nn.Linear(self.width, self.coefficient_bottleneck)
        self.to_gate = nn.Linear(self.width, channels)
        alpha_size = channels * order
        beta_size = channels * order
        rho_size = channels * (order + 1)
        read_size = (order + 1) * self.memory_dimension
        self.coefficient_projection = nn.Linear(
            self.coefficient_bottleneck,
            alpha_size + beta_size + rho_size + read_size)
        alpha_end = alpha_size
        beta_end = alpha_end + beta_size
        rho_end = beta_end + rho_size
        # WHERE EACH QUANTITY SITS IN THE TWO FUSED PROJECTIONS, WRITTEN DOWN ONCE. The
        # projections are fused because one GEMM is faster than several, so `signal` and
        # `memory_write` are row ranges of input_projection and alpha, beta, rho and
        # memory_read are row ranges of coefficient_projection. The forward and the
        # initialiser both read these spans, so the offsets exist in exactly one place and
        # cannot drift apart.
        self.signal_span = (0, channels)
        self.write_span = (channels, channels + self.memory_dimension)
        self.alpha_span = (0, alpha_end)
        self.beta_span = (alpha_end, beta_end)
        self.rho_span = (beta_end, rho_end)
        self.read_span = (rho_end, rho_end + read_size)
        self.out_proj = nn.Linear(channels, self.width, bias=False)
        self.dropout = nn.Dropout(dropout)
        self._register_load_state_dict_pre_hook(load_input_bias, with_module=True)
        self.initialize_lattice_parameters()

    def _project(self, projection, span, x):
        """`x` against the rows [span) of `projection`, bias included when it has one.

        THE ROWS ARE A VIEW OF THE FUSED PARAMETER, so this is one slice and one F.linear and
        no copy of the weight is made. It is what the six per-quantity projections have always
        computed; they were `LinearView` objects until 2026-09-05 and are these two lines now,
        because a plain object holding a slice is invisible to anything that walks
        `named_modules()` -- which is every adapter and every quantizer."""
        lo, hi = span
        bias = None if projection.bias is None else projection.bias[lo:hi]
        return F.linear(x, projection.weight[lo:hi], bias)

    def initialize_lattice_parameters(self):
        """The coefficient projection's initial values: model/mira_initializer.py."""
        return mira_initializer.initialize_lattice_parameters(self)

    def state_values(self) -> int:
        return self.order * self.inner * self.memory_dimension

    def residual_out(self):
        return self.out_proj

    def project_input(self, normalized_x):
        return self.input_projection(normalized_x)

    def project_coefficients(self, control):
        return self.coefficient_projection(control)

    def project_recurrence_inputs(self, normalized_x, tokens, coefficients=True):
        """(input projection, coefficient projection, gamma); with coefficients=False the second
        is the control SiLU(to_control(z)) the coefficient projection would have been applied to
        (mira_v4's matrix_lattice_iir_control projects it, or forms it inside the scan)."""
        # ONE STREAM. The three projections are issued in order. They contend for the same SMs,
        # so overlapping them on separate streams costs more than it saves.
        projections = (self.input_projection, self.to_gate, self.to_control)
        if (not self.training and not torch.is_grad_enabled()
                and type(normalized_x) is torch.Tensor
                and not nn.modules.module._global_forward_pre_hooks
                and not nn.modules.module._global_forward_hooks
                and all(type(projection) is nn.Linear
                        and not projection._forward_pre_hooks
                        and not projection._forward_hooks
                        and "forward" not in projection.__dict__
                        for projection in projections)):
            parameters = tuple(parameter for projection in projections
                               for parameter in (projection.weight, projection.bias)
                               if parameter is not None)
            try:
                signature = tuple(
                    (id(parameter), parameter.data_ptr(), parameter._version,
                     parameter.device, parameter.dtype, tuple(parameter.shape),
                     tuple(parameter.stride()))
                    for parameter in parameters
                ) if all(type(parameter) is nn.Parameter for parameter in parameters) else None
            except RuntimeError:
                signature = None
            if signature is not None:
                stream = (torch.cuda.current_stream(normalized_x.device).cuda_stream
                          if normalized_x.is_cuda else None)
                signature = (signature, stream)
                packed = getattr(self, "_packed_input_projection", None)
                if packed is None or packed[0] != signature:
                    weight = torch.cat(tuple(projection.weight for projection in projections))
                    bias = torch.cat(tuple(
                        projection.bias if projection.bias is not None
                        else projection.weight.new_zeros(projection.out_features)
                        for projection in projections))
                    packed = (signature, weight, bias)
                    self._packed_input_projection = packed
                projected = F.linear(normalized_x, packed[1], packed[2])
                projected_input, gamma, raw_control = projected.split(
                    tuple(projection.out_features for projection in projections), dim=-1)
                control = F.silu(raw_control)
                mira_profiling.record_control(self, control)
                return (projected_input,
                        self.project_coefficients(control) if coefficients else control, gamma)
        control = F.silu(self.to_control(normalized_x))
        mira_profiling.record_control(self, control)
        gamma = self.to_gate(normalized_x)
        return (
            self.project_input(normalized_x),
            self.project_coefficients(control) if coefficients else control,
            gamma,
        )

    def forward(self, normalized_x, attn_mask=None, cache=None, layer=None):
        # THE PREFIX CACHE IS INFERENCE-ONLY, AND THESE ARE THE CONDITIONS THAT MAKE IT ONE.
        # Carrying state across calls in TRAINING is truncated backpropagation through time: it
        # makes the loss depend on the order the batches arrive in and is a different objective
        # from the one the benchmark reports. `self.training` is what says which this is, and it
        # is the module's own mode -- model.lm.train calls model.train() and every inference
        # entry point calls model.eval().
        #
        # THE TEST IS self.training AND NOT torch.is_grad_enabled(): a decode loop may leave the
        # grad graph on, and nothing is kept across steps either way, because PrefixCache.put
        # detaches. `layer` names the slot and a cache without one cannot be indexed.
        # INFERENCE.prefix_cache=0 switches the whole path off.
        if cache is not None and not (
                prefix_cache.enabled() and layer is not None
                and not self.training and hasattr(cache, "get")):
            raise ValueError("streaming cache is not wired into mira-dsp")
        batch, tokens, unused = normalized_x.shape
        del unused
        channels, order = self.inner, self.order
        alpha_size = channels * order
        beta_size = channels * order
        rho_size = channels * (order + 1)
        read_size = (order + 1) * self.memory_dimension
        # mira_v4: THE SCAN FORMS alpha, beta AND rho FROM THE CONTROL. When the selected MIRA CUDA
        # package provides matrix_lattice_iir_control, an uncached, gradient-free forward hands it
        # the control and coefficient_projection instead of the 10,800-wide projected coefficients;
        # it either forms them inside the scan or projects them itself exactly as below. Training,
        # the prefix cache, positive_poles, the diagnostics profile and an adapted or hooked
        # coefficient_projection keep the path below.
        projection = self.coefficient_projection
        from_control = getattr(matrix_lattice_cuda, "matrix_lattice_iir_control", None)
        if (from_control is not None and normalized_x.is_cuda and cache is None
                and not self.positive_poles and not mira_profiling.enabled()
                and not torch.is_grad_enabled()
                and type(projection) is nn.Linear and not projection._forward_pre_hooks
                and not projection._forward_hooks and "forward" not in projection.__dict__):
            input_projection, control, gamma = self.project_recurrence_inputs(
                normalized_x, tokens, coefficients=False)
            signal, raw_memory_write = input_projection.split(
                (channels, self.memory_dimension), dim=-1)
            memory_write = F.silu(raw_memory_write)
            valid = None if attn_mask is None else attn_mask[:, -tokens:].bool()
            mixed = from_control(
                signal, control, projection.weight, projection.bias,
                (self.alpha_span, self.beta_span, self.rho_span, self.read_span),
                gamma, memory_write.contiguous(), valid=valid, arch=self.arch)
            if mixed is None:
                raise RuntimeError(_declined("mira-dsp", available(), signal, order,
                                             self.memory_dimension))
            output = self.out_proj(self.dropout(mixed))
            if valid is not None:
                output = torch.where(valid.unsqueeze(-1), output, torch.zeros_like(output))
            return output
        input_projection, coefficient_projection, gamma = self.project_recurrence_inputs(
            normalized_x, tokens)
        signal, raw_memory_write = input_projection.split(
            (channels, self.memory_dimension), dim=-1)
        raw_alpha, raw_beta, raw_rho, memory_read = coefficient_projection.split(
            (alpha_size, beta_size, rho_size, read_size), dim=-1)
        if self.positive_poles:
            # The shifted sine changes sign, but not magnitude, under alpha -> 1/alpha. Mapping
            # every |alpha| < 1 to its reciprocal therefore makes sin(theta) non-positive and
            # the K=1 pole -r sin(theta) non-negative while preserving its initialized radius.
            value = raw_alpha.float()
            safe = torch.where(value == 0.0, torch.full_like(value, 1.0e-6), value)
            raw_alpha = torch.where(value.abs() < 1.0, safe.reciprocal(), value).to(raw_alpha.dtype)
            if torch.is_grad_enabled() and cache is None:
                coefficient_projection = torch.cat(
                    (raw_alpha, raw_beta, raw_rho, memory_read), dim=-1)
                raw_alpha, raw_beta, raw_rho, memory_read = coefficient_projection.split(
                    (alpha_size, beta_size, rho_size, read_size), dim=-1)
        memory_write = F.silu(raw_memory_write)
        raw_alpha = raw_alpha.view(batch, tokens, channels, order)
        raw_beta = raw_beta.view(batch, tokens, channels, order)
        raw_rho = raw_rho.view(batch, tokens, channels, order + 1)
        memory_read = memory_read.view(
            batch, tokens, order + 1, self.memory_dimension)
        if mira_profiling.enabled():
            mira_profiling.record_coefficients(
                self, normalized_x, raw_alpha, raw_beta, raw_rho, memory_read, gamma, signal,
                raw_memory_write)
        valid = None if attn_mask is None else attn_mask[:, -tokens:].bool()
        mixed = None
        # A CACHED STEP GOES THROUGH THE REFERENCE, WHICH IS THE ONLY PATH THAT TAKES A STATE.
        # matrix_lattice_iir scans every chunk from a zero carry and returns the output alone --
        # it has nowhere to put V_{-1} in and nowhere to hand V_{T-1} back -- so a cached call
        # cannot use it and is not offered to it. The kernel keeps every path it had: training
        # and every uncached forward reach it exactly as before, because `cache` is None there.
        if signal.is_cuda or signal.device.type == "npu":
            backend = (matrix_lattice_cuda if signal.is_cuda else
                       importlib.import_module("kernel.ascend.910_series.ascend"))
            if backend is None:
                raise RuntimeError("mira-dsp CUDA backend is unavailable")
            # THE STRIDED VIEWS GO STRAIGHT IN. Without a gradient the two fused projections are
            # split, so signal, raw_alpha, raw_beta, raw_rho and memory_read are views whose rows
            # step by the projection's width rather than their own; .contiguous() copied all five.
            # At B=2, T=8,192, d_e=1536, K=4 the coefficient projection alone is 20,048 values per
            # token, so those copies read and wrote 1.41 GB per layer per forward. The kernel takes
            # a row stride: matrix_lattice_iir measures it with dense_tail_row_stride, refuses any
            # layout it cannot serve, and every phase already indexes rows through that stride.
            result = backend.matrix_lattice_iir(
                signal, raw_alpha, raw_rho,
                gamma, memory_write.contiguous(), memory_read,
                raw_beta=raw_beta, valid=valid,
                state=None if cache is None else cache.get(layer),
                return_state=cache is not None, arch=self.arch,
                packed=(input_projection, coefficient_projection)
                if torch.is_grad_enabled() and cache is None else None)
            if result is None:
                raise RuntimeError(_declined("mira-dsp", available(), signal, order,
                                             self.memory_dimension))
            if cache is None:
                mixed = result
            else:
                mixed, final = result
        if cache is not None:
            # V_{-1}^(0..K-1) FROM THE SLOT, V_{T-1}^(0..K-1) BACK INTO IT. A layer that has not
            # run yet holds None and the scan starts from zeros, which is the same V_{-1} = 0
            # every training window starts from. `put` detaches, so no graph survives the call.
            if mixed is None:
                mixed, final = matrix_lattice_iir(
                    signal, raw_alpha, raw_rho, gamma, memory_write, memory_read, valid,
                    raw_beta=raw_beta,
                    state=cache.get(layer), return_state=True, arch=self.arch)
            cache.put(layer, final)
            # THE CEILING IS CHECKED AFTER THE FIRST LAYER HAS PUT SOMETHING THERE, because that
            # is the first moment the geometry is a measured number rather than a prediction. It
            # refuses rather than trimming: this state is a summary of every token so far folded
            # into K matrices, so there is no oldest part of it to drop -- a budget that does not
            # fit is a smaller batch, not a shorter memory.
            fits, why = cache.within_budget()
            if not fits:
                raise ValueError(why)
        elif mixed is None:
            mixed = matrix_lattice_iir(
                signal, raw_alpha, raw_rho, gamma, memory_write, memory_read, valid,
                raw_beta=raw_beta, arch=self.arch)
        output = self.out_proj(self.dropout(mixed))
        if valid is not None:
            output = torch.where(valid.unsqueeze(-1), output, torch.zeros_like(output))
        if mira_profiling.enabled():
            mira_profiling.record_output(self, output)
        return output
