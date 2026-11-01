"""W_uniform_b_nested_jeffreys (formerly uniform_weight_nested_jeffreys_horizon_bias_init),
"$W$ uniform w/ $b$ nested jeffreys".

W_uniform_b_jeffreys with one horizon PER POLE rather than one per channel. Channel c keeps its
Jeffreys quantile level l_c = (i_c + 1/2)/d_e (the same random permutation); its pole m = 0 .. ceil(K/2) - 1
-- pair m for m < floor(K/2), the real pole at m = floor(K/2) when K is odd -- takes the horizon
H_(c,m) = F^-1((l_c + phi_m) mod 1), F the same Jeffreys distribution and phi_m = 0, 1/2, 1/4, 3/4, 1/8, ...
the base-2 van der Corput sequence. Pair m has radius exp(-1/H_(c,m)) and the phase-bank angle with
frequency index m, tied to min(angle, 2/H_(c,m)) when H_(c,m) >= 100. Pole m does not depend on K, so the
order-K poles are a subset of the order-(K+2) poles; K = 2 is built exactly as W_uniform_b_jeffreys.
Every weight from U(-0.02 sqrt(3), 0.02 sqrt(3)), then W_rho and W_nu scaled by (3/(K+1))^(1/4).
"""
from model.initializers import _common
from model.mira_initializer import register


@register("W_uniform_b_nested_jeffreys")
def initialize(mixer):
    _common.projection_start(mixer)
    _common.place_poles(mixer, _common.nested_jeffreys_horizon(mixer), phase_bank=True)
    _common.uniform_weight_rows(mixer, nested=True)
