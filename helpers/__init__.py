"""
helpers -- everything the stages share, and nothing any single stage owns.

    helpers.utils            device resolution, tqdm progress, the cosine schedule, table
                             printing, held-out probes, the mixed-precision optimizer, the
                             checkpoint/history IO that resume and the figures are built on,
                             and the on-disk token cache
    helpers.common           the setup every stage runs: arguments, tokenizer, model factory,
                             monitor and generation previews
    helpers.dataset_helpers     every preference and instruction dataset behind one interface
    helpers.visualization    every figure, drawn as PDF under outputs/plots/<stage>/

`utils` is re-exported at package level, so the form every trainer already uses --

    from helpers import progress, save_ckpt, CosineLR, MasterAdamW

-- keeps working. The other four are imported explicitly:

    from helpers import common
    from model import lm       # the length-normalised training loop, moved 2026-09-05
    from helpers import dataset_helpers as dsets
    from helpers import visualization as viz

They are NOT imported here. helpers.common imports mirachat at module scope, so pulling common
in at package-initialisation time would make that a circular import. mirachat itself imports
only helpers.utils and helpers.kv_cache, never helpers.common. Importing a submodule by name is enough to load it, so nothing is lost.
"""
from .utils import *          # noqa: F401,F403
from . import utils           # noqa: F401
