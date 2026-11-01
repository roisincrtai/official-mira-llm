"""W_b_normal, "$W,b$ pytorch-normal".

W_uniform_b_jeffreys, then every weight and bias of the coefficient projection from N(0, 1/d_r).
"""
import math

from model.initializers.W_uniform_b_jeffreys import build as jeffreys
from model.mira_initializer import register


@register("W_b_normal")
def initialize(mixer):
    jeffreys(mixer)
    weight = mixer.coefficient_projection.weight
    scale = 1.0 / math.sqrt(weight.shape[1])
    weight.normal_(0.0, scale)
    mixer.coefficient_projection.bias.normal_(0.0, scale)
