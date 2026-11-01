"""W_b_uniform, "$W,b$ pytorch-uniform".

W_uniform_b_jeffreys, then every weight and bias of the coefficient projection from
U(-1/sqrt(d_r), 1/sqrt(d_r)), the PyTorch default of a linear layer.
"""
import math

from model.initializers.W_uniform_b_jeffreys import build as jeffreys
from model.mira_initializer import register


@register("W_b_uniform")
def initialize(mixer):
    jeffreys(mixer)
    weight = mixer.coefficient_projection.weight
    limit = 1.0 / math.sqrt(weight.shape[1])
    weight.uniform_(-limit, limit)
    mixer.coefficient_projection.bias.uniform_(-limit, limit)
