"""W_jeffreys_b_uniform, "$W$ jeffreys w/ $b$ uniform": the control for the jeffreys bias.

W_uniform_b_jeffreys, then
    b_alpha, b_beta from U(-1/sqrt(d_r), 1/sqrt(d_r))
    every entry of alpha/beta weight row i set to (b*_i - b_i) / sum_j E[c_j], b*_i the W_uniform_b_jeffreys
      bias of the row, c = silu(to_control(z)) the d_r bottleneck units, E[c_j] with the RMS-normalized
      input z taken as N(0, I), so E[alpha], E[beta] at the start equal those of W_uniform_b_jeffreys
    W_rho, W_nu, b_rho, b_nu as W_uniform_b_jeffreys
"""
import math

import torch

from model.initializers import _common
from model.initializers.W_uniform_b_jeffreys import build as jeffreys
from model.mira_initializer import register


def place_in_weight_rows(mixer, coefficients, draw_bias):
    """The jeffreys stage coefficients carried by the alpha/beta weight rows instead of the biases:
    draw_bias(b) redraws b_alpha, b_beta in place, and every entry of alpha/beta row i becomes
    (b*_i - b_i) / sum_j E[c_j], b*_i the W_uniform_b_jeffreys bias of the row and E[c_j] the mean of
    c = silu(to_control(z)) with z taken as N(0, I) (_common.control_mean), so the coefficient averaged
    over tokens is the jeffreys one."""
    weight = mixer.coefficient_projection.weight
    bias = mixer.coefficient_projection.bias
    alpha_lo, beta_hi = mixer.alpha_span[0], mixer.beta_span[1]
    draw_bias(bias[alpha_lo:beta_hi])
    response = _common.control_mean(mixer).sum()
    if not torch.isfinite(response) or response.abs() < 1.0e-12:
        raise ValueError(f"{mixer.lattice_initializer} needs a nonzero mean control response")
    initial_alpha, initial_beta = coefficients
    target = torch.cat((initial_alpha.reshape(-1), initial_beta.reshape(-1))).double()
    rows = (target - bias[alpha_lo:beta_hi].double()) / response
    weight[alpha_lo:beta_hi].copy_(rows.unsqueeze(-1).expand(-1, weight.shape[1]).float())


@register("W_jeffreys_b_uniform")
def initialize(mixer):
    bound = 1.0 / math.sqrt(mixer.coefficient_bottleneck)
    place_in_weight_rows(mixer, jeffreys(mixer), lambda bias: bias.uniform_(-bound, bound))
