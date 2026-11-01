"""The MIRA dynamics profile: the diagnostics measured on a forward pass while it is switched on.

Kept apart from the model. model/mira_mixer.py's mixer calls record_control, record_coefficients and
record_output with tensors it has already formed, and only when enabled(); model.lm switches the
profile on for the steps it records (collect_stats) and reads it back (layer_stats for history.json,
full_stats for dynamics.jsonl, see helpers/dynamics.py). Nothing here changes what the mixer computes.
"""
from __future__ import annotations

import importlib
import math

import torch
import torch.nn.functional as F


def _mira_mixer():
    """model.mira_mixer, imported when first needed (it imports this module)."""
    return importlib.import_module("model.mira_mixer")


_PROFILE = {"on": False}


def _channel_reductions(split):
    """(cat, avg) for statistics over the lattice channels. Under fsdp-channel `cat` joins each
    rank's channels and `avg` averages a per-rank mean over the ranks (every rank holds the same
    number of channels); otherwise both return their argument."""
    if not split:
        return (lambda tensor, dim: tensor), (lambda tensor: tensor)
    channel = importlib.import_module("model.distributed.distributed_fsdp_channel")
    return channel.channel_cat, channel.channel_mean


@torch.no_grad()
def _summarize(raw_alpha, raw_rho, gamma, raw_beta=None, arch="cauchy_x_hyperbolic", split=False):
    """Scalar diagnostics of one forward pass. No coefficient tensor is retained.

    The stage rotation uses theta = pi/2 - 2 arctan(alpha), modulo 2 pi, and
    r = 1 / (1 + beta^2). For K=1 the recurrent eigenvalue is -r sin(theta), so its magnitude
    and horizon are r |sin(theta)| and -1 / log(r |sin(theta)|), not r and -1 / log(r).
    For K>1 these statistics describe each stage's direct recurrent factor; the eigenvalues of
    the assembled K by K transition depend on all stages. The angle statistics come from the
    rational sine and cosine that the stage already forms.
    """
    cat, avg = _channel_reductions(split)
    alpha = raw_alpha.float()
    rotation, radius_map = _mira_mixer().stage_maps(arch)
    cosine, sine = rotation(alpha)
    if raw_beta is None:
        radius = cosine
    else:
        beta = raw_beta.float()
        radius = radius_map(beta)
    channel = cat((radius * sine.abs()).mean(dim=(0, 1)), 0).clamp(max=1.0 - 1.0e-7)
    horizon = -1.0 / torch.log(channel.clamp(min=1.0e-7))
    quantiles = torch.tensor([0.1, 0.5, 0.9], device=alpha.device)
    horizon_q = torch.quantile(horizon.flatten(), quantiles)
    radius_q = torch.quantile(channel.flatten(), quantiles)
    # THE ANGLE, WITHOUT arctan. |sin| is the angle's own magnitude near zero and saturates at 1,
    # so it reports how far a lattice stage has turned; the period in tokens follows from the angle itself.
    angle = torch.atan2(sine.abs(), cosine)
    angle_channel = cat(angle.mean(dim=(0, 1)), 0)
    angle_q = torch.quantile(angle_channel.flatten(), quantiles)
    turning = (angle_channel > 1.0e-6).float()
    period = (2.0 * math.pi) / angle_channel.clamp(min=1.0e-6)
    return {
        "horizon_p10": float(horizon_q[0]), "horizon_p50": float(horizon_q[1]),
        "horizon_p90": float(horizon_q[2]), "horizon_max": float(horizon.amax()),
        "radius_p10": float(radius_q[0]), "radius_p50": float(radius_q[1]),
        "radius_p90": float(radius_q[2]), "radius_max": float(channel.amax()),
        "horizon_slowest_p50": float(
            torch.quantile(horizon.amax(dim=-1), torch.tensor(0.5, device=alpha.device))),
        # COMPLEX STATISTICS. angle is |arg lambda|; complex_frac is the share of lattice stages that have
        # turned off the real axis at all; period is 2 pi / angle in tokens; and cycles is how many
        # periods a lattice stage completes within its own horizon, which is what decides whether
        # an oscillation is observable before the lattice stage has forgotten.
        "angle_p10": float(angle_q[0]), "angle_p50": float(angle_q[1]),
        "angle_p90": float(angle_q[2]), "angle_max": float(angle_channel.amax()),
        "complex_frac": float(turning.mean()),
        "period_p50": float(torch.quantile(period.flatten(), quantiles)[1]),
        "cycles_p50": float(torch.quantile(
            (horizon / period).flatten(), quantiles)[1]),
        "rotation_abs_mean": float(avg(sine.abs().mean())),
        "alpha_abs_mean": float(avg(alpha.abs().mean())),
        "alpha_negative_frac": float(avg((alpha < 0).float().mean())),
        "alpha_token_std": float(avg(alpha.std(dim=(0, 1)).mean())),
        "alpha_channel_std": float(cat(alpha.mean(dim=(0, 1)), 0).std()),
        "rho_abs_mean": [float(v) for v in avg(raw_rho.float().abs().mean(dim=(0, 1, 2)))],
        "gate_mean": float(avg(F.silu(gamma.float()).mean())),
    }


# THE EXTENDED DIAGNOSTICS: dynamics.jsonl only. history.json keeps exactly _summarize's keys (it is
# rewritten whole at every checkpoint, so it stays small); full_stats adds the keys below to each layer's
# record in dynamics.jsonl. Every value is a scalar, or a list of K+1 per-level scalars, so a layer adds
# about seventy numbers per profile whatever the batch, the context or the width. A record written
# before these keys existed simply lacks them, and every reader treats a missing key as not measured.
#
# THE ASSEMBLED SYSTEM PER (token, channel). The K stages compose into one linear map from
# (write, V_{t-1}^(0..K-1)) to the K+1 levels V_t^(0..K):
#     V_t^(0..K-1) = A_t V_{t-1}^(0..K-1) + b_t u_t,    levels_t = C_t V_{t-1}^(0..K-1) + d_t u_t,
# with u_t one entry of the write x~_t s_t^T. Frozen at one token's coefficients, the channel's
# transfer from the write to its rho-weighted level sum sum_k rho^(k) level_k is
#     G(z) = c^T (z I - A)^{-1} b + D,   c = C^T rho,  D = rho^T d,
# whose POLES are eig(A) (the true pole pair, not a single stage's r sin(theta)) and whose ZEROS are
# eig(A - b c^T / D) when D != 0. They are measured on PROFILE_TOKENS evenly spaced tokens of the first
# sequence, every channel.
PROFILE_TOKENS = 512


def _eigenvalues(matrices):
    """Eigenvalues of a batch of K x K real matrices [..., K, K], complex [..., K]. K = 2 is solved in
    closed form on the matrices' own device, lambda = (tr +- sqrt(tr^2 - 4 det)) / 2; other orders go
    through torch.linalg.eigvals on the CPU, where the general solver runs anyway."""
    if matrices.shape[-1] == 2:
        trace = matrices[..., 0, 0] + matrices[..., 1, 1]
        det = matrices[..., 0, 0] * matrices[..., 1, 1] - matrices[..., 0, 1] * matrices[..., 1, 0]
        root = torch.sqrt((trace * trace - 4.0 * det).to(torch.complex128))
        half = (trace / 2.0).to(torch.complex128)
        return torch.stack((half + root / 2.0, half - root / 2.0), dim=-1)
    return torch.linalg.eigvals(matrices.cpu()).to(matrices.device)


def _q(values, prefix, quantiles=(0.1, 0.5, 0.9)):
    """{prefix_p10, prefix_p50, prefix_p90} of the finite entries of `values` (None when there are none)."""
    flat = values.detach().flatten().double()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {f"{prefix}_p{int(round(100 * q))}": None for q in quantiles}
    if flat.numel() > 1 << 20:
        flat = flat[torch.linspace(0, flat.numel() - 1, 1 << 20, device=flat.device).long()]
    levels = torch.quantile(flat, torch.tensor(quantiles, dtype=flat.dtype, device=flat.device))
    return {f"{prefix}_p{int(round(100 * q))}": float(v) for q, v in zip(quantiles, levels)}


def _rms(tensor):
    return float(tensor.detach().float().square().mean().sqrt()) if tensor is not None and tensor.numel() else None


@torch.no_grad()
def _summarize_ext(mixer, raw_alpha, raw_beta, raw_rho, memory_read, gamma, signal, raw_write,
                   control=None):
    """The extended per-layer diagnostics (see the block comment above). Never raises into training:
    the caller wraps it."""
    out = {}
    split = hasattr(mixer, "_channel")
    cat, avg = _channel_reductions(split)

    def rms(tensor, part=True):
        if tensor is None or not tensor.numel():
            return None
        mean_square = tensor.detach().float().square().mean()
        return float((avg(mean_square) if part else mean_square).sqrt())

    order = raw_alpha.shape[-1]
    tokens = raw_alpha.shape[1]
    index = torch.linspace(0, tokens - 1, min(tokens, PROFILE_TOKENS), device=raw_alpha.device).long()
    alpha = cat(raw_alpha[0, index], 1).double()
    beta = cat(raw_beta[0, index], 1).double()
    rho = cat(raw_rho[0, index], 1).double()
    A, b, C, d = _mira_mixer().stage_system(alpha, beta, mixer.arch)
    # POLES of the assembled pair, per sampled (token, channel).
    poles = _eigenvalues(A)                                                 # [S, C, K] complex
    magnitude = poles.abs().clamp(max=1.0 - 1.0e-12)
    horizon = -1.0 / torch.log(magnitude.clamp(min=1.0e-300))
    out.update(_q(horizon, "pole_horizon"))
    out.update(_q(magnitude, "pole_abs"))
    out["pole_horizon_slowest_p50"] = _q(horizon.amax(dim=-1), "s")["s_p50"]
    angle = poles.angle().abs()
    out["pole_angle_p50"] = _q(angle, "a")["a_p50"]
    out["pole_complex_frac"] = float((poles.imag.abs() > 1.0e-9 * poles.abs().clamp(min=1.0e-300)
                                      ).double().mean())
    # ZEROS of G(z) = c^T (zI - A)^{-1} b + D, c = C^T rho, D = rho^T d.
    c = (C * rho.unsqueeze(-1)).sum(dim=-2)                                 # [S, C, K]
    direct = (rho * d).sum(dim=-1)                                          # [S, C]
    defined = direct.abs() > 1.0e-12
    out["zero_defined_frac"] = float(defined.double().mean())
    if bool(defined.any()):
        Ad, bd, cd, Dd = A[defined], b[defined], c[defined], direct[defined]
        zeros = _eigenvalues(Ad - bd.unsqueeze(-1) * cd.unsqueeze(-2) / Dd[:, None, None])
        out.update(_q(zeros.abs(), "zero_abs"))
        out["zero_outside_frac"] = float((zeros.abs() > 1.0).double().mean())
        gap = (zeros.unsqueeze(-1) - poles[defined].unsqueeze(-2)).abs().amin(dim=-1)
        out["zero_pole_gap_p50"] = _q(gap, "g")["g_p50"]
    else:
        out.update({"zero_abs_p10": None, "zero_abs_p50": None, "zero_abs_p90": None,
                    "zero_outside_frac": None, "zero_pole_gap_p50": None})
    # GAINS of the channel transfer at z = 1 (slowly varying input) and z = -1 (alternating input).
    eye = torch.eye(order, dtype=A.dtype, device=A.device)
    for name, point in (("gain_dc_p50", 1.0), ("gain_nyquist_p50", -1.0)):
        try:
            solved = torch.linalg.solve(point * eye - A, b.unsqueeze(-1)).squeeze(-1)
            gain = ((c * solved).sum(dim=-1) + direct).abs()
            out[name] = _q(gain, "g")["g_p50"]
        except RuntimeError:
            out[name] = None
    # RADIUS PARAMETER: beta's token spread is what sets the radius decay near the fold beta = 0.
    beta_all = raw_beta.detach().float()
    out["beta_abs_mean"] = float(avg(beta_all.abs().mean()))
    out["beta_sq_mean"] = float(avg(beta_all.square().mean()))
    out["beta_token_std"] = (float(avg(beta_all.std(dim=(0, 1)).mean()))
                             if beta_all.shape[1] > 1 else None)
    out["rho_token_std"] = (float(avg(raw_rho.detach().float().std(dim=(0, 1)).mean()))
                            if raw_rho.shape[1] > 1 else None)
    # READOUT AND WRITE SCALES.
    out["nu_rms"] = [float(v) for v in memory_read.detach().float().square().mean(dim=(0, 1, 3)).sqrt()]
    out["signal_rms"] = rms(signal)
    out["write_rms"] = _rms(F.silu(raw_write.float()))
    gate = F.silu(gamma.detach().float())
    out["gate_abs_mean"] = float(avg(gate.abs().mean()))
    out["gate_negative_frac"] = float(avg((gate < 0).float().mean()))
    # COMMON CONTROL c_t: scale, token spread and how many directions it uses (participation ratio).
    if control is not None:
        flat = control.detach().double().reshape(-1, control.shape[-1])
        out["control_rms"] = float(flat.square().mean().sqrt())
        centred = flat - flat.mean(dim=0, keepdim=True)
        spectrum = torch.linalg.eigvalsh(centred.T @ centred / max(flat.shape[0] - 1, 1)).clamp(min=0.0)
        total = spectrum.sum()
        out["control_participation"] = float(total.square() / spectrum.square().sum()) if total > 0 else None
        out["control_token_std"] = float(centred.std(dim=0).mean())
    # PARAMETERS, as RMS per group (the bias spectrum b_alpha, b_beta and the random rows that mask it).
    weight = mixer.coefficient_projection.weight.detach().float()
    bias = mixer.coefficient_projection.bias.detach().float()
    for name, (lo, hi) in (("alpha", mixer.alpha_span), ("beta", mixer.beta_span),
                           ("rho", mixer.rho_span), ("nu", mixer.read_span)):
        out[f"W_{name}_rms"] = rms(weight[lo:hi], part=name != "nu")
        out[f"b_{name}_rms"] = rms(bias[lo:hi], part=name != "nu")
    projection = mixer.input_projection.weight.detach().float()
    out["W_in_rms"] = rms(projection[:mixer.inner])
    out["W_s_rms"] = _rms(projection[mixer.inner:])
    out["W_c_rms"] = _rms(mixer.to_control.weight)
    out["b_c_mean"] = float(mixer.to_control.bias.detach().float().mean()) if mixer.to_control.bias is not None else None
    out["W_gamma_rms"] = rms(mixer.to_gate.weight)
    out["b_gamma_mean"] = (float(avg(mixer.to_gate.bias.detach().float().mean()))
                           if mixer.to_gate.bias is not None else None)
    out["W_out_rms"] = rms(mixer.out_proj.weight)
    return out


def collect_stats(model, on=True):
    """Switch the diagnostics on or off; return how many lattice mixers were switched."""
    _PROFILE["on"] = bool(on)
    return sum(1 for m in model.modules() if isinstance(m, _mira_mixer().MiraMixer))


def layer_stats(model):
    """The scalars measured by the most recent forward pass, one entry per lattice mixer."""
    return [dict(m._profile) for m in model.modules()
            if isinstance(m, _mira_mixer().MiraMixer) and getattr(m, "_profile", None)]


def full_stats(model):
    """Everything measured this step, in the shape helpers/dynamics.py records: each layer's
    _summarize keys (the ones history.json also keeps) plus the extended keys of _summarize_ext."""
    layers = [{**m._profile, **(getattr(m, "_profile_ext", None) or {})}
              for m in model.modules()
              if isinstance(m, _mira_mixer().MiraMixer) and getattr(m, "_profile", None)]
    return {"ssm": layers, "attn": []}


def enabled():
    """Whether the profile is switched on."""
    return _PROFILE["on"]


def record_control(mixer, control):
    """Keep the common control c_t = SiLU(to_control(z)) of this forward for summarize_ext."""
    if _PROFILE["on"]:
        mixer._profile_control = control.detach()


def record_coefficients(mixer, normalized_x, raw_alpha, raw_beta, raw_rho, memory_read, gamma,
                        signal, raw_memory_write):
    """The per-layer profile of this forward: mixer._profile (_summarize, also kept in history.json)
    and mixer._profile_ext (_summarize_ext, dynamics.jsonl only)."""
    mixer._profile = _summarize(raw_alpha, raw_rho, gamma, raw_beta, mixer.arch,
                                split=hasattr(mixer, "_channel"))
    # A DIAGNOSTIC MUST NOT END A RUN: a failure here leaves the extended keys out of this
    # profile and nothing else.
    try:
        mixer._profile_ext = _summarize_ext(
            mixer, raw_alpha, raw_beta, raw_rho, memory_read, gamma, signal,
            raw_memory_write, getattr(mixer, "_profile_control", None))
    except Exception as error:                                            # noqa: BLE001
        mixer._profile_ext = {"ext_error": str(error)[:200]}
    mixer._profile_control = None
    mixer._profile_input_rms = _rms(normalized_x)


def record_output(mixer, output):
    """THE MIXER'S CONTRIBUTION against its own normalized input, both RMS over the batch."""
    if isinstance(getattr(mixer, "_profile_ext", None), dict):
        mixer._profile_ext["mixer_in_rms"] = getattr(mixer, "_profile_input_rms", None)
        mixer._profile_ext["mixer_out_rms"] = _rms(output)
