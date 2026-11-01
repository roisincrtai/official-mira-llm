"""The MIRA lattice initializers: the registry and the wrapper the mixer calls.

Each lattice_initializer is one file, model/initializers/<name>.py, whose function
`initialize(mixer)` is decorated with @register("<name>") and sets the initial weights W and
biases b of the coefficient projection (its docstring states how). The parts they share are in
model/initializers/_common.py. The registered names must be exactly
default_config.LATTICE_INITIALIZERS, the canonical names configs and checkpoints carry; former
names are read through default_config.canonical_lattice_initializer.
"""
from __future__ import annotations

import importlib

import torch

INITIALIZERS = {}
_LOADED = []


def register(name):
    """Decorator: register `initialize(mixer)` as lattice_initializer `name`."""
    import default_config as config
    if name not in config.LATTICE_INITIALIZERS:
        raise ValueError(f"{name!r} is not in default_config.LATTICE_INITIALIZERS")

    def wrap(initialize):
        if INITIALIZERS.get(name, initialize) is not initialize:
            raise ValueError(f"lattice_initializer {name!r} is registered twice")
        INITIALIZERS[name] = initialize
        return initialize
    return wrap


def registered():
    """{name: initialize} of every lattice_initializer, the package imported on first use."""
    if not _LOADED:
        importlib.import_module("model.initializers")
        import default_config as config
        missing = sorted(set(config.LATTICE_INITIALIZERS) - set(INITIALIZERS))
        if missing:
            raise RuntimeError(f"no model/initializers/<name>.py registers {missing}")
        _LOADED.append(True)
    return INITIALIZERS


def parse_lattice_initializer(value):
    """The canonical lattice_initializer name of `value` (a former name is accepted)."""
    import default_config as config
    return config.canonical_lattice_initializer(value)


def parse_lattice_init_horizon(value):
    """One (min, max) range, or {"default": (min, max), "L<n>": (min, max), ...} for per-layer
    ranges (default_config.canonical_lattice_init_horizon). None is the configured value."""
    import default_config as config
    if value is None:
        value = config.MODEL["lattice_init_horizon"]
    return config.canonical_lattice_init_horizon(value)


def layer_lattice_init_horizon(value, layer):
    """The (min, max) range of MIRA layer `layer` (0-based): its L<layer> entry, else `default`."""
    import default_config as config
    if value is None:
        value = config.MODEL["lattice_init_horizon"]
    return config.layer_lattice_init_horizon(value, layer)


def initialize_lattice_parameters(mixer):
    """Initialize `mixer`'s coefficient projection with its lattice_initializer; return `mixer`."""
    initializers = registered()
    name = mixer.lattice_initializer
    if name not in initializers:
        raise ValueError(f"unknown lattice_initializer {name!r}; one of {tuple(initializers)}")
    with torch.no_grad():
        initializers[name](mixer)
    return mixer
