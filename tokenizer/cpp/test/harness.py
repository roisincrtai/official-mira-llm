"""Tests of tokenizer/cpp/json.cpp against JSONTestSuite and against Python's json module.

The oracle is Python's json made strict: the text must be well-formed UTF-8 (no byte order mark),
NaN/Infinity/-Infinity are rejected (unless python mode), and a string holding a lone surrogate
is rejected (a UTF-8 string cannot hold one). When both accept, the C++ dump_json must equal
json.dumps of what Python read, byte for byte, under the same options.
"""
import json
import math
import os
import random
import struct
import subprocess
import sys
from decimal import Decimal, getcontext

sys.set_int_max_str_digits(0)
HERE = os.path.dirname(os.path.abspath(__file__))
# the binaries and JSONTestSuite (github.com/nst/JSONTestSuite), from build.sh
BIN = os.path.normpath(os.path.join(HERE, "..", "..", "..", "build", "tokenizer", "cpp", "test"))
DRIVER = os.path.join(BIN, os.environ.get("DRIVER", "driver"))
SUITE = os.environ.get("JSONTESTSUITE", os.path.join(BIN, "JSONTestSuite", "test_parsing"))
REJECT = object()
FAILURES = []


def check(condition, what):
    if not condition:
        FAILURES.append(what)
        if len(FAILURES) <= 30:
            print("FAIL:", what[:600])


def run(cases, mode="json", python=0, max_depth=10000, indent=-1, ensure=1, sort=0, allow_nan=1,
        locale=None, driver=None):
    feed = b"".join(struct.pack("<I", len(c)) + c for c in cases)
    argv = [driver or DRIVER, mode, str(python), str(max_depth), str(indent), str(ensure),
            str(sort), str(allow_nan)] + ([locale] if locale else [])
    done = subprocess.run(argv, input=feed, capture_output=True)
    if done.returncode != 0:
        raise SystemExit(f"driver exited {done.returncode}: {done.stderr.decode(errors='replace')[-3000:]}")
    out, i, results = done.stdout, 0, []
    while i < len(out):
        j = out.index(b"\n", i)
        head = out[i:j].split()
        n = int(head[-1])
        body = out[j + 1:j + 1 + n]
        i = j + 1 + n
        if head[0] == b"OK":
            results.append(("OK", body))
        elif head[0] == b"ERR":
            results.append(("ERR", (int(head[1]), int(head[2]), int(head[3])), body.decode()))
        else:
            results.append(("DUMPERR", body.decode()))
    assert len(results) == len(cases), (len(results), len(cases))
    return results


def has_surrogate(v):
    stack = [v]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            if any(0xD800 <= ord(ch) <= 0xDFFF for ch in x):
                return True
        elif isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, dict):
            stack.extend(x.keys())
            stack.extend(x.values())
    return False


def _no_constant(name):
    raise ValueError(f"constant {name}")


class _Surrogate(ValueError):
    pass


def _pairs(pairs):
    # every key and value Python read, before a repeated key discards the earlier ones
    for key, value in pairs:
        if has_surrogate(key) or has_surrogate(value):
            raise _Surrogate()
    return dict(pairs)


def oracle(data, python=False):
    try:
        s = data.decode("utf-8")
    except UnicodeDecodeError:
        return REJECT
    try:
        v = json.loads(s, parse_constant=None if python else _no_constant, object_pairs_hook=_pairs)
    except (ValueError, RecursionError):
        return REJECT
    if has_surrogate(v):
        return REJECT
    return v


def expected_dump(v, indent=-1, ensure=1, sort=0, allow_nan=1):
    try:
        return ("OK", json.dumps(v, indent=None if indent < 0 else indent, ensure_ascii=bool(ensure),
                                 sort_keys=bool(sort), allow_nan=bool(allow_nan)).encode())
    except ValueError as e:
        return ("DUMPERR", str(e))


def compare(cases, python=0, label="", **options):
    """Every case against the oracle: the same verdict, and when accepted the same dump."""
    results = run(cases, python=python, **options)
    opts = {k: v for k, v in options.items() if k in ("indent", "ensure", "sort", "allow_nan")}
    agree = 0
    for case, got in zip(cases, results):
        want = oracle(case, bool(python))
        if want is REJECT:
            check(got[0] == "ERR", f"{label}: accepted what Python rejects: {case[:300]!r} -> {got}")
        else:
            exp = expected_dump(want, **opts)
            check(got[0] == exp[0] and (exp[0] == "DUMPERR" or got[1] == exp[1]),
                  f"{label}: {case[:300]!r}: got {str(got)[:300]} want {str(exp)[:300]}")
        agree += 1
    return agree


# ------------------------------------------------------------------------------------------------
# generators
# ------------------------------------------------------------------------------------------------

SPECIAL_CHARS = ['"', "\\", "/", "\b", "\f", "\n", "\r", "\t", "\x00", "\x1f", "\x7f", "\x80",
                 " ", " ", "﻿", "￿", "�", "\U0001F600", "\U0010FFFF",
                 "é", "日", " "]


def random_char(rng):
    r = rng.random()
    if r < 0.4:
        return chr(rng.randrange(0x20, 0x7F))
    if r < 0.6:
        return rng.choice(SPECIAL_CHARS)
    if r < 0.7:
        return chr(rng.randrange(0, 0x20))
    if r < 0.85:
        cp = rng.randrange(0x80, 0x10000)
        return chr(cp) if not 0xD800 <= cp <= 0xDFFF else "x"
    return chr(rng.randrange(0x10000, 0x110000))


def random_string(rng, longest=12):
    return "".join(random_char(rng) for _ in range(rng.randrange(0, longest)))


def random_float(rng, python):
    r = rng.random()
    if r < 0.4:
        x = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
    elif r < 0.6:
        x = rng.choice([0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, 1.7976931348623157e308,
                        0.1, 1e16, 1e15, 9007199254740993.0, 1e-4, 1e-5, 123456789012345680.0,
                        0.30000000000000004, 1e22, 1e23])
    else:
        x = rng.uniform(-1e6, 1e6) * 10.0 ** rng.randrange(-30, 30)
    if (math.isnan(x) or math.isinf(x)) and not python:
        x = 1.5
    return x


def random_int(rng):
    r = rng.random()
    if r < 0.5:
        return rng.randrange(-1000, 1000)
    if r < 0.8:
        return rng.choice([2**63 - 1, -2**63, 2**63, -2**63 - 1, 2**64, -2**64, 2**31, 0])
    return rng.randrange(-10**rng.randrange(1, 300), 10**rng.randrange(1, 300))


def random_value(rng, depth=0, python=False):
    r = rng.random()
    if depth > 5 or r < 0.5:
        k = rng.randrange(6)
        return [None, True, False][rng.randrange(3)] if k == 0 else (
            random_int(rng) if k in (1, 2) else random_float(rng, python) if k == 3 else random_string(rng))
    if r < 0.75:
        return [random_value(rng, depth + 1, python) for _ in range(rng.randrange(0, 6))]
    return {random_string(rng, 6): random_value(rng, depth + 1, python) for _ in range(rng.randrange(0, 6))}


WS = [" ", "\t", "\n", "\r"]


def ws(rng):
    return "".join(rng.choice(WS) for _ in range(rng.choice([0, 0, 0, 1, 2])))


def write_string(rng, s):
    out = ['"']
    for ch in s:
        cp = ord(ch)
        r = rng.random()
        if ch == '"' or ch == "\\" or cp < 0x20 or r < 0.15:
            short = {'"': '\\"', "\\": "\\\\", "/": "\\/", "\b": "\\b", "\f": "\\f", "\n": "\\n",
                     "\r": "\\r", "\t": "\\t"}
            if ch in short and rng.random() < 0.6:
                out.append(short[ch])
            elif cp < 0x10000:
                out.append(("\\u%04x" if rng.random() < 0.5 else "\\u%04X") % cp)
            else:
                c = cp - 0x10000
                out.append("\\u%04x\\u%04x" % (0xD800 | (c >> 10), 0xDC00 | (c & 0x3FF)))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def write_float(rng, x):
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    r = rng.random()
    if r < 0.5:
        return repr(x) if "e" in repr(x) or "." in repr(x) else repr(x) + ".0"
    p = rng.randrange(0, 25)
    text = ("%." + str(p) + "e") % x
    if rng.random() < 0.3:
        text = text.replace("e", "E")
    if rng.random() < 0.3:
        text = text.replace("e+", "e")
    return text


def write(rng, v, dups=True):
    """JSON text of v with random whitespace, escapes and number forms; sometimes a repeated key."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return write_float(rng, v)
    if isinstance(v, str):
        return write_string(rng, v)
    if isinstance(v, list):
        return "[" + ws(rng) + ("," + ws(rng)).join(write(rng, x, dups) + ws(rng) for x in v) + "]"
    items = list(v.items())
    if dups and items and rng.random() < 0.3:
        k, _ = rng.choice(items)
        items.insert(rng.randrange(len(items) + 1), (k, random_value(rng, 4)))
    return "{" + ws(rng) + ("," + ws(rng)).join(
        write_string(rng, k) + ws(rng) + ":" + ws(rng) + write(rng, x, dups) + ws(rng) for k, x in items) + "}"


def mutate(rng, data):
    data = bytearray(data)
    for _ in range(rng.randrange(1, 4)):
        op = rng.randrange(3)
        at = rng.randrange(len(data) + 1)
        b = rng.choice([rng.randrange(256), ord(rng.choice('[]{}",:\\/-+.eE0123456789 \t\n\rtfnu')),
                        0xC0, 0xED, 0xF4, 0xF5, 0x80, 0xBF, 0xEF, 0xFF])
        if op == 0:
            data.insert(at, b)
        elif op == 1 and data:
            del data[min(at, len(data) - 1)]
        elif data:
            data[min(at, len(data) - 1)] = b
    return bytes(data)
