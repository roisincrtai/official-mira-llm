"""CUDA kernel packages.

MIRA variant directories use their canonical hyphenated architecture names.  Python import
statements cannot contain hyphens."""

from __future__ import annotations

# import importlib
# import sys
#
#
# for _directory, _source_name in (
#         ("mira-sdpa", "mira_sdpa"),
#         ("mira-newton", "mira_newton")):
#     _module = importlib.import_module(f"{__name__}.{_directory}")
#     sys.modules[f"{__name__}.{_source_name}"] = _module
#
# del _directory, _source_name, _module
