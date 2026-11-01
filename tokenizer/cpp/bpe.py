"""tokenizer/cpp/bpe.py -- tokenizer/py/bpe.py's BPETokenizer with the work done in C++.

The library is the file the yaml key `libbpe` names, built from the sources in tokenizer/cpp/ by
the stage scripts and exported to the trainers as the environment variable LIBBPE. encode,
encode_ordinary, decode and build run in it; every other method, every id and the file format are
the Python class's own, and the C++ gives the same ids, the same text, and the same merges,
history and corpus signature.
"""
import array
import ctypes
import json
import os

from ..py import bpe as _py

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIBRARY_ENV = "LIBBPE"

_LOG = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p)
_PROGRESS = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64)
_JSON = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p)
_lib = None
_memfd = None


def library_path():
    """The library LIBBPE names, relative paths read from the repository root; "" when unset."""
    path = os.environ.get(LIBRARY_ENV, "").strip()
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_library():
    """The loaded library, or None when LIBBPE is unset or its file cannot be loaded."""
    global _lib
    if _lib is not None:
        return _lib
    path = library_path()
    if not path:
        return None
    try:
        lib = _open(path)
    except OSError as error:
        print(f"[tokenizer] {path} cannot be loaded ({error}); using the Python tokenizer",
              flush=True)
        return None
    i64, p = ctypes.c_int64, ctypes.c_void_p
    p_i64 = ctypes.POINTER(i64)
    lib.bpe_create.argtypes = [ctypes.c_char_p, p_i64, i64, ctypes.c_char_p, p_i64, i64]
    lib.bpe_create.restype = p
    lib.bpe_destroy.argtypes = [p]
    lib.bpe_destroy.restype = None
    lib.bpe_encode.argtypes = [p, ctypes.c_char_p, i64, ctypes.c_int,
                               ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))]
    lib.bpe_encode.restype = i64
    lib.bpe_decode.argtypes = [p, p_i64, i64, ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
    lib.bpe_decode.restype = i64
    lib.bpe_free.argtypes = [p]
    lib.bpe_free.restype = None
    lib.bpe_last_error.argtypes = []
    lib.bpe_last_error.restype = ctypes.c_char_p
    lib.bpe_counter_create.argtypes = []
    lib.bpe_counter_create.restype = p
    lib.bpe_counter_add.argtypes = [p, ctypes.c_char_p, i64]
    lib.bpe_counter_add.restype = ctypes.c_int
    lib.bpe_counter_destroy.argtypes = [p]
    lib.bpe_counter_destroy.restype = None
    lib.bpe_build.argtypes = [p, i64, ctypes.c_char_p, i64, i64, ctypes.c_char_p, p_i64, i64,
                              ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, i64,
                              _LOG, _PROGRESS, _JSON, _JSON]
    lib.bpe_build.restype = p
    lib.bpe_result_merges.argtypes = [p, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),
                                      ctypes.POINTER(p_i64)]
    lib.bpe_result_merges.restype = i64
    lib.bpe_result_history.argtypes = [p]
    lib.bpe_result_history.restype = ctypes.c_char_p
    lib.bpe_result_sig.argtypes = [p]
    lib.bpe_result_sig.restype = ctypes.c_char_p
    lib.bpe_result_destroy.argtypes = [p]
    lib.bpe_result_destroy.restype = None
    _lib = lib
    return lib


def _open(path):
    """The library at `path`, loaded from memory when its filesystem is mounted noexec.

    A shared object is mapped executable when it is loaded, which a noexec mount refuses with
    "failed to map segment from shared object" (as /data on gpu2 is). The file stays where it was
    built; its bytes are copied into an anonymous in-memory file, held open for the process, and
    loaded through /proc/self/fd/N, as kernel/cuda/upstream_extension.py does for the upstream
    kernels."""
    global _memfd
    try:
        return ctypes.CDLL(path)
    except OSError as error:
        if "failed to map segment" not in str(error) or not hasattr(os, "memfd_create"):
            raise
    handle = os.memfd_create(os.path.basename(path), 0)
    with open(path, "rb") as source:
        while True:
            block = source.read(1 << 20)
            if not block:
                break
            os.write(handle, block)
    lib = ctypes.CDLL(f"/proc/self/fd/{handle}")
    _memfd = handle
    print(f"[tokenizer] {os.path.dirname(path)} is on a filesystem that refuses to map a shared "
          f"object executable (a noexec mount); {os.path.basename(path)} stays there and is "
          f"loaded from memory instead", flush=True)
    return lib


def _required():
    lib = load_library()
    if lib is None:
        raise RuntimeError(f"[tokenizer] the C++ tokenizer needs its library: set {LIBRARY_ENV} to "
                           f"the libbpe.so built by make -f tokenizer/cpp/Makefile")
    return lib


def _error(lib):
    return RuntimeError(f"[tokenizer] {lib.bpe_last_error().decode('utf-8', 'replace')}")


def _packed(strings):
    """(bytes laid end to end, their lengths as int64) for the C interface."""
    lengths = array.array("q", (len(s) for s in strings))
    pointer = (ctypes.c_int64 * len(lengths)).from_buffer(lengths) if lengths else None
    return b"".join(strings), lengths, pointer


class BPETokenizer(_py.BPETokenizer):
    def _rebuild(self):
        super()._rebuild()
        lib = _required()
        merge_bytes, merge_lengths, merge_pointer = _packed(
            [bytes(symbol) for pair in self.merges for symbol in pair])
        special_bytes, special_lengths, special_pointer = _packed(
            [token.encode("utf-8") for token in self.specials])
        handle = lib.bpe_create(merge_bytes, merge_pointer, len(self.merges),
                                special_bytes, special_pointer, len(self.specials))
        if not handle:
            raise _error(lib)
        previous = getattr(self, "_handle", None)
        self._handle = handle
        if previous:
            lib.bpe_destroy(previous)

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle and _lib is not None:
            try:
                _lib.bpe_destroy(handle)
            except Exception:                                      # noqa: BLE001
                pass
            self._handle = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_handle", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._handle = None
        self._rebuild()

    def _encode(self, text, split_specials):
        data = text.encode("utf-8")
        out = ctypes.POINTER(ctypes.c_int32)()
        n = _lib.bpe_encode(self._handle, data, len(data), 1 if split_specials else 0,
                            ctypes.byref(out))
        if n < 0:
            raise _error(_lib)
        try:
            ids = array.array("i")
            if n:
                ids.frombytes(ctypes.string_at(out, 4 * n))
            return ids.tolist()
        finally:
            _lib.bpe_free(out)

    def encode_ordinary(self, text):
        return self._encode(text, False)

    def encode(self, text, split_specials=True):
        if split_specials and ("<|endoftext|>" in text or "<|unk|>" in text):
            return super().encode(text, split_specials=True)
        return self._encode(text, split_specials)

    def decode(self, ids, skip_special_tokens=True):
        values = array.array("q", (int(i) for i in ids))
        pointer = (ctypes.c_int64 * len(values)).from_buffer(values) if values else None
        out = ctypes.POINTER(ctypes.c_char)()
        length = _lib.bpe_decode(self._handle, pointer, len(values),
                                 1 if skip_special_tokens else 0, ctypes.byref(out))
        if length < 0:
            raise _error(_lib)
        try:
            return ctypes.string_at(out, length).decode("utf-8")
        finally:
            _lib.bpe_free(out)

    @classmethod
    def build(cls, texts, num_merges=10000, checkpoint_path=None, checkpoint_every=250,
              log=print, monitor=None, plot_every=200, init_merges=None, init_history=None,
              init_sig=None, min_freq=1, init_across_corpora=False):
        """tokenizer/py/bpe.py's BPETokenizer.build, run by the library: the same merges in the
        same order, the same history and corpus signature, the same <checkpoint_path>.partial and
        the same progress bars."""
        from tqdm import tqdm

        lib = _required()
        failures, bars, history = [], {}, []
        phases = {1: ("BPE: indexing pairs", "word"), 2: ("BPE: replaying checkpoint", "merge"),
                  3: ("BPE: learning merges", "merge")}

        def guarded(function):
            def call(*args):
                try:
                    function(*args)
                    return 1
                except BaseException as error:                    # noqa: BLE001
                    failures.append(error)
                    return 0
            return call

        def on_log(line):
            log(line.decode("utf-8"))

        def on_progress(phase, done, total):
            bar = bars.get(phase)
            if bar is None:
                for other in bars.values():
                    other.close()
                desc, unit = phases[phase]
                bar = tqdm(desc=desc, unit=unit, total=total, dynamic_ncols=True,
                           **({"initial": done} if phase == 3 else {}))
                bars[phase] = bar
            if done > bar.n:
                bar.update(done - bar.n)
            if phase != 3 and done >= total:
                bar.close()

        def on_resume(text):
            history[:] = json.loads(text)

        def on_merge(text):
            record = json.loads(text)
            history.append(record)
            bar = bars.get(3)
            if bar is not None and hasattr(bar, "set_postfix"):
                bar.set_postfix(pair_freq=int(record["pair_freq"]), vocab=record["vocab_size"],
                                bytes_per_tok=f"{record['bytes_per_token']:.2f}")
            merge = record["merge"]
            if monitor and plot_every > 0 and merge % plot_every == 0:
                try:
                    monitor(history, merge)                      # live bpe_dynamics.pdf
                except Exception as e:                           # plotting must never kill a long build
                    log(f"[plot] bpe: {e}")

        callbacks = (_LOG(guarded(on_log)), _PROGRESS(guarded(on_progress)),
                     _JSON(guarded(on_resume)), _JSON(guarded(on_merge)))
        counter = lib.bpe_counter_create()
        if not counter:
            raise _error(lib)
        try:
            for t in tqdm(list(texts), desc="BPE: counting words", unit="seg", dynamic_ncols=True):
                data = t.encode("utf-8")
                if not lib.bpe_counter_add(counter, data, len(data)):
                    raise _error(lib)
            init = [bytes(symbol) for pair in (init_merges or []) for symbol in pair]
            init_bytes, init_lengths, init_pointer = _packed(init)
            result = lib.bpe_build(
                counter, int(num_merges),
                checkpoint_path.encode("utf-8") if checkpoint_path else None,
                int(checkpoint_every), int(min_freq), init_bytes, init_pointer, len(init) // 2,
                json.dumps(list(init_history or [])).encode("utf-8"),
                init_sig.encode("utf-8") if init_sig is not None else None,
                1 if init_across_corpora else 0, len(cls.SPECIALS), *callbacks)
            if not result:
                if failures:
                    raise failures[0]
                raise _error(lib)
            try:
                data, lengths = ctypes.POINTER(ctypes.c_char)(), ctypes.POINTER(ctypes.c_int64)()
                n = lib.bpe_result_merges(result, ctypes.byref(data), ctypes.byref(lengths))
                sizes = lengths[:2 * n]
                blob = ctypes.string_at(data, sum(sizes)) if n else b""
                symbols, offset = [], 0
                for size in sizes:
                    symbols.append(tuple(blob[offset:offset + size]))
                    offset += size
                merges = list(zip(symbols[0::2], symbols[1::2]))
                built_history = json.loads(lib.bpe_result_history(result).decode("utf-8"))
                sig = lib.bpe_result_sig(result).decode("utf-8")
            finally:
                lib.bpe_result_destroy(result)
        finally:
            lib.bpe_counter_destroy(counter)
            for bar in bars.values():
                bar.close()
        tok = cls(merges, build_history=built_history)
        tok.corpus_sig = sig                         # lets a later run verify and EXTEND it
        return tok
