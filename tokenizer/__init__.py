"""Stage 3: the byte-level BPE tokenizer every stage shares."""
#
# `run` is NOT imported here. `python -m tokenizer.run` makes runpy import this package
# first, and if the module it is about to execute is already in sys.modules it runs a
# SECOND time as __main__ -- two module objects, two copies of every module-level value,
# and a RuntimeWarning. Import it explicitly where it is needed:
#
#     from tokenizer import run as train_bpe
# THE C++ TOKENIZER WHEN ITS LIBRARY IS THERE, the Python one otherwise: LIBBPE is set by the
# stage scripts from the yaml key `libbpe`, and left unset when that key is commented out or the
# library failed to build.
from .cpp import bpe as _cpp_bpe
from .py.bpe import BPETokenizer as _PythonBPETokenizer

BPETokenizer = (_cpp_bpe.BPETokenizer if _cpp_bpe.load_library() is not None
                else _PythonBPETokenizer)
