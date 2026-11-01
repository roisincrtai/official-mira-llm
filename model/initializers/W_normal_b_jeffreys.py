"""W_normal_b_jeffreys, "$W$ normal w/ $b$ jeffreys".

W_uniform_b_jeffreys, then every weight (W_alpha, W_beta, W_rho, W_nu) from N(0, 0.02^2); the biases are kept.
"""
from model.initializers import _common
from model.initializers.W_uniform_b_jeffreys import build as jeffreys
from model.mira_initializer import register


@register("W_normal_b_jeffreys")
def initialize(mixer):
    jeffreys(mixer)
    mixer.coefficient_projection.weight.normal_(0.0, _common.UNIFORM_WEIGHT_STD)
