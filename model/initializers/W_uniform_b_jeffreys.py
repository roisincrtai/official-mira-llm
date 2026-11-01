"""W_uniform_b_jeffreys (formerly uniform_weight_jeffreys_horizon_bias_init), "$W$ uniform w/ $b$ jeffreys".

    horizons H_c, c = 1..d_e: the quantiles (c - 1/2)/d_e of the Jeffreys prior p(log lambda) ~ sqrt(I(lambda)),
      lambda = 1/H, I(lambda) = sum_{t=1}^{256} t exp(-2 lambda t), on the layer's lattice_init_horizon,
      randomly permuted over the channels
    pole pair exp(-1/H_c) exp(+-i w_c), w_c = 2 pi / P_c, P_c cycling 5, 7, 11, 17, 23, 31, 43, 61 over the
      channels; w_c = min(w_c, 2/H_c) when H_c >= 100
    b_alpha, b_beta: the stage coefficients of those poles (_common.place_poles)
    every weight (W_alpha, W_beta, W_rho, W_nu) from U(-0.02 sqrt(3), 0.02 sqrt(3))
    b_rho, b_nu: random sign, magnitude from U(s/4, s), s = (K+1)^(-1/4) / sqrt(d_r)
"""
from model.initializers import _common
from model.mira_initializer import register


def build(mixer):
    """Initialize `mixer` as W_uniform_b_jeffreys; return the stage coefficients (alpha, beta) [d_e, K]
    written into b_alpha, b_beta."""
    _common.projection_start(mixer)
    coefficients = _common.place_poles(mixer, _common.jeffreys_horizon(mixer), phase_bank=True)
    _common.uniform_weight_rows(mixer)
    return coefficients


@register("W_uniform_b_jeffreys")
def initialize(mixer):
    build(mixer)
