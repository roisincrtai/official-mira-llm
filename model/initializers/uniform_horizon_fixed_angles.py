"""uniform_horizon_fixed_angles (formerly schur_init_both_side), "uniform horizon w/ fixed angles".

    H_c from U(lo, hi) of the layer's lattice_init_horizon
    pole pair j = 0..J-1 (J = floor(K/2)): radius exp(-(1 + j/J)/H_c), angle pi f_j,
      f = 1/4, 1/2, 3/4, 1/8, 3/8, 5/8, 7/8, 1/16; real pole (odd K) at -exp(-1/H_c)
    b_alpha, b_beta: the stage coefficients of those poles (_common.place_poles)
    W_alpha, W_beta: random sign, magnitude from U(1/4, 1) min(|b|/(4 sqrt(d_r)), s), b the bias of the row
    W_rho, W_nu, b_rho, b_nu: random sign, magnitude from U(s/4, s), s = (K+1)^(-1/4) / sqrt(d_r)
"""
from model.initializers import _common
from model.mira_initializer import register


@register("uniform_horizon_fixed_angles")
def initialize(mixer):
    _common.projection_start(mixer)
    _common.place_poles(mixer, _common.uniform_horizon(mixer))
