"""uniform_horizon_fixed_angles_right (formerly schur_init_right_side),
"uniform horizon w/ fixed angles, right half-plane".

uniform_horizon_fixed_angles with every pole in the right half-plane: angles pi f_j,
f = 1/4, 1/8, 3/8, 1/16, 3/16, 5/16, 7/16, 1/32, and the real pole at +exp(-1/H_c);
the same as uniform_horizon_fixed_angles at K = 2.
"""
from model.initializers import _common
from model.mira_initializer import register


@register("uniform_horizon_fixed_angles_right")
def initialize(mixer):
    _common.projection_start(mixer)
    _common.place_poles(mixer, _common.uniform_horizon(mixer), right_side=True)
