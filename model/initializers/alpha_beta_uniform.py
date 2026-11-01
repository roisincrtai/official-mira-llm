"""alpha_beta_uniform (formerly uniform_init), "$\\alpha,\\beta$ uniform".

    W_alpha, W_beta, b_alpha, b_beta from U(-1/sqrt(d_r), 1/sqrt(d_r)); b_alpha, b_beta read as
      sin(theta) = (1 - b_alpha^2)/(1 + b_alpha^2), r = 1/(1 + b_beta^2) and mapped to the same poles under
      the arch's modulus and argument maps (_common.uniform_coefficients)
    W_rho, W_nu, b_rho, b_nu: random sign, magnitude from U(s/4, s), s = (K+1)^(-1/4) / sqrt(d_r)
"""
from model.initializers import _common
from model.mira_initializer import register


@register("alpha_beta_uniform")
def initialize(mixer):
    _common.projection_start(mixer)
    weight, bias, (alpha_lo, alpha_hi), (beta_lo, beta_hi), bound, _ = _common.projection(mixer)
    weight[alpha_lo:beta_hi].uniform_(-bound, bound)
    bias[alpha_lo:beta_hi].uniform_(-bound, bound)
    draw_alpha, draw_beta = _common.uniform_coefficients(
        mixer.arch, bias[alpha_lo:alpha_hi].clone(), bias[beta_lo:beta_hi].clone())
    bias[alpha_lo:alpha_hi].copy_(draw_alpha)
    bias[beta_lo:beta_hi].copy_(draw_beta)
