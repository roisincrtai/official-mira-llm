"""W_jeffreys_b_normal, "$W$ jeffreys w/ $b$ normal": the control for the jeffreys bias.

W_jeffreys_b_uniform with b_alpha, b_beta from N(0, 1/d_r).
"""
import math

from model.initializers.W_jeffreys_b_uniform import place_in_weight_rows
from model.initializers.W_uniform_b_jeffreys import build as jeffreys
from model.mira_initializer import register


@register("W_jeffreys_b_normal")
def initialize(mixer):
    bound = 1.0 / math.sqrt(mixer.coefficient_bottleneck)
    place_in_weight_rows(mixer, jeffreys(mixer), lambda bias: bias.normal_(0.0, bound))
