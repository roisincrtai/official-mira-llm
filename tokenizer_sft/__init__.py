"""Vocabulary adaptation: EXTEND a trained BPE with merges learned on a new corpus."""
from .run import build, extend, extended_path, remap_specials, tokenizer_name

__all__ = ["build", "extend", "extended_path", "remap_specials", "tokenizer_name"]
