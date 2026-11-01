"""The parts the MIRA lattice initializers share: model/initializers/<name>.py builds each
lattice_initializer from these, and model/mira_initializer.py registers and calls it.

Every initializer sets the initial weights W and biases b of the coefficient projection
(W_alpha, W_beta, W_rho, W_nu; b_alpha, b_beta, b_rho, b_nu), d_r its input width. Those that place
poles form, per channel, a stable degree-K polynomial (conjugate pole pairs, one real pole when K is
odd), convert it into reflection coefficients kappa_k by the Schur step-down recursion, take
    r_k = 1 - 2^-13 (1 - |kappa_k|),   sin(theta_k) = kappa_k / r_k,
and invert these stage factors through the arch's own maps, so every arch starts from the same r and
theta. The arch is <radius>_x_<rotation>, and each part inverts on its own:

    mobius rotation      |alpha| = sqrt((1 - sin)/(1 + sin))
    hyperbolic rotation  alpha = artanh(sin)
    identity rotation    alpha = arcsin(sin)   (theta = alpha, the branch with cos >= 0, as hyperbolic)
    cauchy radius        |beta| = sqrt(1/r - 1)
    gaussian radius      |beta| = sqrt(-log r)
    expexp radius        beta = log(-log r)   (one branch, so beta_sign does not enter it)
    expsoftplus radius   beta = log(expm1((-log r) / exp(s_r^(c))))   (one branch as well)
    hill radius          |beta| = sqrt(r/(1 - r))

At r = 0.99 the cauchy |beta| is 0.1005, the gaussian |beta| 0.1003 and the expexp beta -4.600, the
hill |beta| 9.950; expsoftplus beta depends on the sampled s_r^(c).

The functions under THE STEPS below are the stages an initializer is assembled from, in the order
they draw from the random generator: projection_start, a horizon (jeffreys_horizon,
nested_jeffreys_horizon or uniform_horizon), place_poles, then uniform_weight_rows.
"""
from __future__ import annotations

import math

import torch


def multiply_polynomials(left, right):
    product = left.new_zeros(left.shape[0], left.shape[1] + right.shape[1] - 1)
    for index in range(right.shape[1]):
        product[:, index:index + left.shape[1]] += left * right[:, index:index + 1]
    return product


# W_uniform_b_jeffreys: the phase-bank angle, except that a pole pair whose horizon is at least
# HORIZON_TIE_MINIMUM tokens takes the angle min(phase-bank angle, HORIZON_TIE_KAPPA / horizon).
HORIZON_TIE_MINIMUM = 100.0
HORIZON_TIE_KAPPA = 2.0


# W_uniform_b_jeffreys: the standard deviation of the uniform coefficient-projection weights.
UNIFORM_WEIGHT_STD = 0.02


# W_uniform_b_jeffreys: the lag window T of the Fisher information I(lambda).
JEFFREYS_CONTEXT = 256
JEFFREYS_GRID = 200001


def jeffreys_horizons(channels, minimum, maximum, context=JEFFREYS_CONTEXT):
    """The channels' horizons, sorted by decay rate, as the quantiles (i + 1/2)/channels of the
    density sqrt(I(lambda)) lambda over log lambda on [1/maximum, 1/minimum], where
    I(lambda) = sum_{t=1}^{T} t q^t = q (1 - (T + 1) q^T + T q^(T + 1)) / (1 - q)^2, q = exp(-2 lambda)."""
    levels = (torch.arange(channels, dtype=torch.float64) + 0.5) / channels
    return jeffreys_quantiles(levels, minimum, maximum, context)


def jeffreys_quantiles(levels, minimum, maximum, context=JEFFREYS_CONTEXT):
    """The horizons at quantile `levels` (float64, in (0, 1), counted from the fastest decay
    rate) of the density jeffreys_horizons describes."""
    log_rate = torch.linspace(math.log(1.0 / maximum), math.log(1.0 / minimum), JEFFREYS_GRID,
                              dtype=torch.float64)
    q = torch.exp(-2.0 * torch.exp(log_rate))
    information = q * (1.0 - (context + 1) * q.pow(context) + context * q.pow(context + 1)) \
        / (1.0 - q).square()
    density = information.sqrt() * torch.exp(log_rate)
    cumulative = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(
        0.5 * (density[1:] + density[:-1]) * (log_rate[1:] - log_rate[:-1]), 0)))
    cumulative = cumulative / cumulative[-1]
    index = torch.searchsorted(cumulative, levels).clamp(1, JEFFREYS_GRID - 1)
    weight = (levels - cumulative[index - 1]) / (cumulative[index] - cumulative[index - 1])
    rate = log_rate[index - 1] + weight * (log_rate[index] - log_rate[index - 1])
    return torch.exp(-rate)


def van_der_corput(index):
    """The base-2 van der Corput value of `index`: 0, 1/2, 1/4, 3/4, 1/8, 5/8, ..."""
    value, digit = 0.0, 0.5
    while index:
        value += digit * (index & 1)
        index >>= 1
        digit *= 0.5
    return value


def nested_jeffreys_horizons(permutation, order, minimum, maximum):
    """[channels, ceil(order/2)] horizons of W_uniform_b_nested_jeffreys.

    Column 0 is jeffreys_horizons(channels, minimum, maximum)[permutation], the horizon jeffreys
    gives each channel; column m is the quantile at level (l_c + phi_m) mod 1 of the same density,
    l_c = (permutation[c] + 1/2)/channels, phi_m = van_der_corput(m)."""
    channels = permutation.shape[0]
    first = jeffreys_horizons(channels, minimum, maximum)[permutation]
    levels = (permutation.double() + 0.5) / channels
    columns = [first]
    for mode in range(1, (int(order) + 1) // 2):
        columns.append(jeffreys_quantiles(
            torch.remainder(levels + van_der_corput(mode), 1.0), minimum, maximum))
    return torch.stack(columns, dim=-1)


def decimal_reflection(factors, order):
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 100
        polynomial = [Decimal(1)]
        for factor in factors:
            terms = [Decimal(float(value)) for value in factor]
            product = [Decimal(0)] * (len(polynomial) + len(terms) - 1)
            for left, a in enumerate(polynomial):
                for right, b in enumerate(terms):
                    product[left + right] += a * b
            polynomial = product
        reflection = [0.0] * order
        current = polynomial
        for degree in range(order, 0, -1):
            coefficient = current[-1]
            reflection[degree - 1] = float(coefficient)
            denominator = 1 - coefficient * coefficient
            current = [(current[index] - coefficient * current[len(current) - 1 - index])
                       / denominator for index in range(len(current) - 1)]
    return reflection


def initial_reflection_coefficients(horizon, order, right_side=False, phase_bank=False,
                                    exact=False, horizon_tied=False):
    """Reflection coefficients [channels, order] of the stable polynomial the initializers build.

    `horizon` [channels] is one horizon per channel: the real pole has radius exp(-1/H) and pair j
    radius exp(-(1+j/J)/H). `horizon` [channels, ceil(order/2)] is one horizon per pole
    (W_uniform_b_nested_jeffreys): pair j has radius exp(-1/H[:, j]) and the
    real pole of an odd order radius exp(-1/H[:, order // 2])."""
    per_pole = horizon.ndim == 2
    pole_targets = torch.exp(-torch.reciprocal(horizon.double()))
    target = pole_targets[:, 0] if per_pole else pole_targets
    polynomial = torch.ones(target.shape[0], 1, dtype=torch.float64)
    factors = []
    if order % 2:
        real_target = pole_targets[:, order // 2] if per_pole else target
        real_coefficient = -real_target if right_side else real_target
        factors.append(torch.stack((torch.ones_like(real_target), real_coefficient), dim=-1))
        polynomial = multiply_polynomials(polynomial, factors[-1])
    frequencies = target.new_tensor(
        ((0.25, 0.125, 0.375, 0.0625, 0.1875, 0.3125, 0.4375, 0.03125)
         if right_side else
         (0.25, 0.5, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625)))
    pairs = order // 2
    for index in range(pairs):
        pole_radius = (pole_targets[:, index] if per_pole
                       else target.pow(1.0 + index / max(1, pairs)))
        if phase_bank:
            periods = target.new_tensor((5, 7, 11, 17, 23, 31, 43, 61))
            channel = torch.arange(target.shape[0], device=target.device)
            shift = 2.0 * math.pi / periods[channel.remainder(len(periods))] - math.pi / 4.0
            angle = torch.acos(torch.cos(math.pi * frequencies[index] + shift))
            if horizon_tied:
                pair_horizon = -torch.reciprocal(torch.log(pole_radius))
                tied = torch.minimum(angle, HORIZON_TIE_KAPPA / pair_horizon)
                angle = torch.where(pair_horizon >= HORIZON_TIE_MINIMUM, tied, angle)
        else:
            angle = math.pi * frequencies[index]
        factor = torch.stack((
            torch.ones_like(target),
            -2.0 * pole_radius * torch.cos(angle),
            pole_radius.square(),
        ), dim=-1)
        factors.append(factor)
        polynomial = multiply_polynomials(polynomial, factor)
    reflection = torch.empty(target.shape[0], order, dtype=torch.float64)
    current = polynomial
    for degree in range(order, 0, -1):
        coefficient = current[:, -1]
        reflection[:, degree - 1] = coefficient
        denominator = 1.0 - coefficient.square()
        current = (
            current - coefficient.unsqueeze(-1) * current.flip(-1)
        )[:, :-1] / denominator.unsqueeze(-1)
    unstable = ~(torch.isfinite(reflection).all(-1) & (reflection.abs() < 1.0).all(-1))
    for row in torch.nonzero(unstable).flatten().tolist():
        reflection[row] = reflection.new_tensor(
            decimal_reflection([factor[row].tolist() for factor in factors], order))
    if not (torch.isfinite(reflection).all() and (reflection.abs() < 1.0).all()):
        raise ValueError(f"order-{order} initial reflection coefficients are not inside (-1, 1)")
    return reflection if exact else reflection.float()


def nonzero_uniform(reference, lower, upper):
    magnitude = torch.empty_like(reference, device="cpu").uniform_(lower, upper)
    sign = torch.empty_like(reference, device="cpu").bernoulli_(0.5).mul_(2.0).sub_(1.0)
    return magnitude * sign


def arch_maps(arch):
    """(radius, rotation) of a cayley_parameterization <radius>_x_<rotation>."""
    radius, _, rotation = str(arch).partition("_x_")
    if (radius not in ("cauchy", "gaussian", "expexp", "expsoftplus", "hill")
            or rotation not in ("mobius", "hyperbolic", "identity")):
        raise ValueError("arch must be <cauchy|gaussian|expexp|expsoftplus|hill>_x_"
                         f"<mobius|hyperbolic|identity>, not {arch!r}")
    return radius, rotation


def softplus_inverse(decay):
    """beta with softplus(beta) = decay > 0: log(expm1(decay)) = decay + log(-expm1(-decay)).

    The second form is the same number; it keeps full precision for small decay (long horizons)
    and never overflows for large decay, where expm1(decay) would."""
    return decay + torch.log(-torch.expm1(-decay))


def stage_coefficients(arch, reflection, exact_reflection, alpha_sign, beta_sign):
    """(alpha, beta) of every stage from its reflection coefficient, through `arch`'s maps.

    Each stage takes the radius r = 1 - 2^-13 (1 - |kappa|) and the sine sin = kappa / r. The
    mobius alpha and the cauchy beta are the float32 arithmetic they always were, with the signs
    drawn; the hyperbolic and identity alphas are taken in float64 from `exact_reflection`, because
    r is within 2^-13 of 1 and artanh(kappa / r) and arcsin(kappa / r) need that distance. Both
    take the branch cos(theta) >= 0, so alpha_sign does not enter them. The gaussian and expexp
    betas are taken in float64 from the gap 1 - r = 2^-13 (1 - |kappa|), as -log r = -log1p(-gap);
    the expexp beta = log(-log r) has one branch, so beta_sign does not enter it. So does the
    expsoftplus beta = log(expm1(s)), s = -log r / exp(s_r^(c)) = 1 / (H exp(s_r^(c))),
    evaluated as s + log(-expm1(-s)), which is
    the same number and stays finite for large s, where expm1(s) would overflow. The hill
    |beta| = sqrt(r/(1 - r)) is taken in float64 from the same gap, as sqrt((1 - gap)/gap), with the
    sign beta_sign."""
    radius_map, rotation_map = arch_maps(arch)
    radius = 1.0 - (1.0 - reflection.abs()) / float(1 << 13)
    stage_sine = reflection / radius
    if exact_reflection is not None:
        exact_radius = 1.0 - (1.0 - exact_reflection.abs()) / float(1 << 13)
        exact_sine = exact_reflection / exact_radius
    if rotation_map == "mobius":
        alpha_magnitude = torch.sqrt((1.0 - stage_sine) / (1.0 + stage_sine))
        initial_alpha = alpha_sign * alpha_magnitude
    elif rotation_map == "identity":
        initial_alpha = torch.asin(exact_sine).float()
    else:
        initial_alpha = torch.atanh(exact_sine).float()
    if radius_map in ("gaussian", "expexp", "expsoftplus", "hill"):
        gap = (1.0 - (exact_reflection if exact_reflection is not None else reflection.double()).abs()
               ) / float(1 << 13)
        if radius_map == "gaussian":
            initial_beta = beta_sign * torch.sqrt(-torch.log1p(-gap)).float()
        elif radius_map == "expexp":
            initial_beta = torch.log(-torch.log1p(-gap)).float()
        elif radius_map == "hill":
            initial_beta = beta_sign * torch.sqrt((1.0 - gap) / gap).float()
        else:
            decay = -torch.log1p(-gap)
            initial_beta = softplus_inverse(decay).float()
    else:
        initial_beta = beta_sign * torch.sqrt(torch.reciprocal(radius) - 1.0)
    return initial_alpha, initial_beta


def row_scale_ratio(arch, mobius_alpha, mobius_beta, alpha, beta):
    """The factor on each structured weight row under `arch`, against the cauchy_x_mobius row.

    A weight row moves its coefficient per token; the stage feels that move through
    d theta / d alpha and d log(-log r) / d beta. The factor is the cauchy_x_mobius derivative
    over the `arch` derivative at the initial coefficients, so a token moves every stage by the
    same first-order amount under every arch:

        mobius rotation      d theta / d alpha = 2 / (1 + alpha^2)
        hyperbolic rotation  d theta / d alpha = sech(alpha)
        identity rotation    d theta / d alpha = 1
        cauchy radius        d log(-log r) / d beta = 2 |beta| / ((1 + beta^2) log(1 + beta^2))
        gaussian radius      d log(-log r) / d beta = 2 / |beta|
        expexp radius        d log(-log r) / d beta = 1
        expsoftplus radius   d log(-log r) / d beta = sigmoid(beta) / softplus(beta)
        hill radius          d log(-log r) / d beta = 2 / (|beta| (1 + beta^2) log(1 + 1/beta^2))"""
    radius_map, rotation_map = arch_maps(arch)
    if (radius_map, rotation_map) == ("cauchy", "mobius"):
        return None
    mobius_alpha = mobius_alpha.double()
    mobius_beta = mobius_beta.double()
    if rotation_map == "mobius":
        rotation = torch.ones_like(mobius_alpha)
    elif rotation_map == "identity":
        rotation = 2.0 / (1.0 + mobius_alpha.square())
    else:
        rotation = 2.0 / (1.0 + mobius_alpha.square()) * torch.cosh(alpha.double())
    if radius_map == "gaussian":
        mobius_size = mobius_beta.abs()
        size = beta.double().abs()
        decay = torch.where(
            mobius_size > 0.0,
            mobius_size * size / ((1.0 + mobius_size.square()) * torch.log1p(mobius_size.square())
                                  ).clamp_min(1.0e-300),
            torch.ones_like(mobius_size))
    elif radius_map == "expexp":
        mobius_size = mobius_beta.abs()
        decay = torch.where(
            mobius_size > 0.0,
            2.0 * mobius_size / ((1.0 + mobius_size.square()) * torch.log1p(mobius_size.square())
                                 ).clamp_min(1.0e-300),
            torch.ones_like(mobius_size))
    elif radius_map in ("expsoftplus", "hill"):
        mobius_size = mobius_beta.abs()
        own = beta.double()
        if radius_map == "hill":
            size = own.abs().clamp_min(1.0e-8)
            own_rate = 2.0 / (size * (1.0 + size.square()) * torch.log1p(size.square().reciprocal()))
        else:
            own_rate = torch.sigmoid(own) / torch.nn.functional.softplus(own)
        decay = torch.where(
            mobius_size > 0.0,
            2.0 * mobius_size / ((1.0 + mobius_size.square()) * torch.log1p(mobius_size.square())
                                 * own_rate).clamp_min(1.0e-300),
            torch.ones_like(mobius_size))
    else:
        decay = torch.ones_like(mobius_beta)
    return torch.cat((rotation.reshape(-1), decay.reshape(-1))).float()


def uniform_coefficients(arch, draw_alpha, draw_beta):
    """alpha_beta_uniform's (alpha, beta) for `arch`, from the uniform draws cauchy_x_mobius uses directly.

    The draws are read as cauchy_x_mobius coefficients -- sin = (1 - a^2)/(1 + a^2),
    r = 1/(1 + b^2) -- and a hyperbolic or identity rotation inverts that stage factor through its
    own map (artanh or arcsin of the sine), and a gaussian or expexp radius inverts r through its
    own map (sqrt(-log r), log(-log r) or log(expm1(-log r))), and the hill radius takes beta = 1/b,
    which gives beta^2 = 1/b^2 = r/(1 - r); so the method starts every arch from the same poles."""
    radius_map, rotation_map = arch_maps(arch)
    new_alpha, new_beta = draw_alpha, draw_beta
    if rotation_map in ("hyperbolic", "identity"):
        alpha = draw_alpha.double()
        sine = ((1.0 - alpha.square()) / (1.0 + alpha.square())).clamp(
            -1.0 + 1.0e-12, 1.0 - 1.0e-12)
        inverse = torch.atanh if rotation_map == "hyperbolic" else torch.asin
        new_alpha = inverse(sine).to(draw_alpha.dtype)
    if radius_map == "gaussian":
        beta = draw_beta.double()
        new_beta = (torch.sign(beta) * torch.sqrt(torch.log1p(beta.square()))).to(draw_beta.dtype)
    elif radius_map == "expexp":
        beta = draw_beta.double()
        new_beta = torch.log(torch.log1p(beta.square()).clamp_min(1.0e-12)).to(draw_beta.dtype)
    elif radius_map == "expsoftplus":
        beta = draw_beta.double()
        decay = torch.log1p(beta.square()).clamp_min(1.0e-12)
        new_beta = softplus_inverse(decay).to(draw_beta.dtype)
    elif radius_map == "hill":
        beta = draw_beta.double()
        new_beta = (torch.where(beta < 0.0, -1.0, 1.0) / beta.abs().clamp_min(1.0e-8)
                    ).to(draw_beta.dtype)
    return new_alpha, new_beta


def control_mean(mixer):
    """E[c_j] for c = silu(to_control(z)) with the mixer's current to_control, per bottleneck unit.

    z is the RMS-normalized mixer input, taken as isotropic with unit variance per coordinate, so
    to_control's pre-activation j is Gaussian with mean b_j and variance ||W_j||^2."""
    layer = mixer.to_control
    mean = (layer.bias.detach().double() if layer.bias is not None
            else torch.zeros(layer.out_features, dtype=torch.float64))
    spread = layer.weight.detach().double().norm(dim=1)
    nodes = torch.linspace(-12.0, 12.0, 4801, dtype=torch.float64)
    density = torch.exp(-0.5 * nodes.square())
    density = density / density.sum()
    points = mean.unsqueeze(-1) + spread.unsqueeze(-1) * nodes
    return (torch.nn.functional.silu(points) * density).sum(-1)


# --------------------------------------------------------------------------------------------------
# THE STEPS

def projection(mixer):
    """(weight, bias, (alpha_lo, alpha_hi), (beta_lo, beta_hi), bound, readout_scale) of the coefficient
    projection: bound = 1/sqrt(d_r), readout_scale = (K+1)^(-1/4)."""
    return (mixer.coefficient_projection.weight, mixer.coefficient_projection.bias,
            mixer.alpha_span, mixer.beta_span, 1.0 / math.sqrt(mixer.coefficient_bottleneck),
            (mixer.order + 1.0) ** -0.25)


def projection_start(mixer):
    """Every weight and bias of the coefficient projection: random sign, magnitude from U(s/4, s),
    s = (K+1)^(-1/4) / sqrt(d_r). The rows an initializer does not set keep these."""
    weight, bias, _, _, bound, readout_scale = projection(mixer)
    initial_weight = nonzero_uniform(
        weight, bound * readout_scale / 4.0, bound * readout_scale)
    initial_bias = nonzero_uniform(
        bias, bound * readout_scale / 4.0, bound * readout_scale)
    weight.copy_(initial_weight)
    bias.copy_(initial_bias)


def jeffreys_horizon(mixer):
    """[d_e] horizons: the Jeffreys quantiles (c - 1/2)/d_e on the layer's range, randomly permuted."""
    lo, hi = mixer.lattice_init_horizon
    return jeffreys_horizons(mixer.inner, lo, hi)[torch.randperm(mixer.inner)].float()


def nested_jeffreys_horizon(mixer):
    """[d_e, ceil(K/2)] horizons, one per pole; column 0 is jeffreys_horizon's of the same permutation."""
    lo, hi = mixer.lattice_init_horizon
    return nested_jeffreys_horizons(torch.randperm(mixer.inner), mixer.order, lo, hi).float()


def uniform_horizon(mixer):
    """[d_e] horizons from U(lo, hi) of the layer's range."""
    lo, hi = mixer.lattice_init_horizon
    return torch.empty(mixer.inner, dtype=torch.float32).uniform_(lo, hi)


def place_poles(mixer, horizon, right_side=False, phase_bank=False):
    """Put the poles of `horizon` into b_alpha, b_beta (the stage coefficients, random signs) and
    scale the W_alpha, W_beta rows to them: random sign, magnitude from
    U(1/4, 1) min(|b|/(4 sqrt(d_r)), s), times row_scale_ratio under an arch other than
    cauchy_x_mobius. phase_bank: the phase-bank angles 2 pi / P, P cycling 5, 7, ..., 61, tied to
    min(angle, 2/H) when H >= 100 (the jeffreys initializers); otherwise the fixed angles pi f_j.
    Returns the stage coefficients (alpha, beta) [d_e, K] written into the biases."""
    weight, bias, (alpha_lo, alpha_hi), (beta_lo, beta_hi), bound, readout_scale = projection(mixer)
    horizon_tied = phase_bank
    reflection = initial_reflection_coefficients(
        horizon, mixer.order, right_side=right_side, phase_bank=phase_bank,
        horizon_tied=horizon_tied)
    exact_reflection = (initial_reflection_coefficients(
        horizon, mixer.order, right_side=right_side, phase_bank=phase_bank, exact=True,
        horizon_tied=horizon_tied)
        if mixer.arch != "cauchy_x_mobius" else None)
    alpha_sign = torch.empty(
        mixer.inner, mixer.order, dtype=torch.float32).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    beta_sign = torch.empty(
        mixer.inner, mixer.order, dtype=torch.float32).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    initial_alpha, initial_beta = stage_coefficients(
        mixer.arch, reflection, exact_reflection, alpha_sign, beta_sign)
    mobius_alpha, mobius_beta = (
        (initial_alpha, initial_beta) if mixer.arch == "cauchy_x_mobius" else
        stage_coefficients("cauchy_x_mobius", reflection, None, alpha_sign, beta_sign))
    structured_bias = torch.cat((mobius_alpha.reshape(-1), mobius_beta.reshape(-1)))
    structured_rows = alpha_hi - alpha_lo + beta_hi - beta_lo
    row_scale = structured_bias.abs().clamp_min(1.0e-6).div(
        4.0 * math.sqrt(mixer.coefficient_bottleneck)).clamp_max(
            bound * readout_scale)
    ratio = row_scale_ratio(
        mixer.arch, mobius_alpha, mobius_beta, initial_alpha, initial_beta)
    if ratio is not None:
        row_scale = row_scale * ratio
    structured_weight = nonzero_uniform(
        weight[:structured_rows], 0.25, 1.0) * row_scale.unsqueeze(-1)
    weight[alpha_lo:beta_hi].copy_(structured_weight)
    bias[alpha_lo:alpha_hi].copy_(initial_alpha.reshape(-1))
    bias[beta_lo:beta_hi].copy_(initial_beta.reshape(-1))
    return initial_alpha, initial_beta


def uniform_weight_rows(mixer, nested=False):
    """W_alpha, W_beta, W_rho and W_nu: every weight row from U(-sqrt(3) s, sqrt(3) s),
    s = UNIFORM_WEIGHT_STD; the biases are kept. nested: W_rho and W_nu then scaled by
    (3/(K+1))^(1/4), so their part of sum_k rho_k (level_k . nu_k), a sum over K+1 equal-power
    levels, does not grow with K (the factor is 1 at K = 2)."""
    weight = mixer.coefficient_projection.weight
    limit = math.sqrt(3.0) * UNIFORM_WEIGHT_STD
    weight.uniform_(-limit, limit)
    if nested:
        scale = (3.0 / (mixer.order + 1.0)) ** 0.25
        weight[mixer.rho_span[0]:mixer.rho_span[1]].mul_(scale)
        weight[mixer.read_span[0]:mixer.read_span[1]].mul_(scale)
