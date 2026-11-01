"""Every test of tokenizer/cpp/json.cpp; prints one line per section and exits 1 on any failure.
DRIVER=driver_asan runs the same under AddressSanitizer and UndefinedBehaviorSanitizer,
DRIVER=driver_clang with the clang build.
SCALE multiplies the random cases (default 1)."""
import json
import math
import os
import random
import struct
import sys
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (BIN, FAILURES, REJECT, SUITE, check, compare, expected_dump, mutate, oracle,  # noqa
                     random_float, random_value, run, write)

SCALE = float(os.environ.get("SCALE", "1"))
only = set(sys.argv[1:])


def section(name):
    return not only or name in only


def report(name, count):
    print(f"{name:34s} {count:>10,} cases   failures so far: {len(FAILURES)}", flush=True)


# 1. JSONTestSuite: every y_ accepted, every n_ rejected, i_ as documented ----------------------
if section("suite"):
    names = sorted(os.listdir(SUITE))
    cases = [open(os.path.join(SUITE, n), "rb").read() for n in names]
    results = run(cases)
    implementation = {}
    for name, case, got in zip(names, cases, results):
        if name.startswith("y_"):
            check(got[0] == "OK", f"suite {name} must be accepted: {got}")
            want = oracle(case)
            if want is not REJECT:
                check(got[1] == expected_dump(want)[1], f"suite {name}: {got[1][:200]} vs Python")
        elif name.startswith("n_"):
            check(got[0] == "ERR", f"suite {name} must be rejected: {got[:2]}")
        else:
            implementation[name] = "accepted" if got[0] == "OK" else "rejected"
    # the implementation-defined cases: what this parser does, checked against its specification
    for name, verdict in implementation.items():
        want = oracle(open(os.path.join(SUITE, name), "rb").read())
        check((want is not REJECT) == (verdict == "accepted"),
              f"suite {name}: {verdict}, the strict oracle says {'accept' if want is not REJECT else 'reject'}")
    with open(os.path.join(BIN, "implementation_defined.txt"), "w") as f:
        for name, verdict in implementation.items():
            f.write(f"{verdict:9s} {name}\n")
    report("JSONTestSuite y_/n_/i_", len(cases))
    # the same files in python mode: NaN/Infinity accepted as Python does
    results = run(cases, python=1)
    for name, case, got in zip(names, cases, results):
        want = oracle(case, python=True)
        check((want is REJECT) == (got[0] == "ERR"), f"suite (python mode) {name}: {got[0]}")
    report("JSONTestSuite, python mode", len(cases))

# 2. generated documents, every writer option ---------------------------------------------------
if section("generated"):
    rng = random.Random(1)
    n = int(4000 * SCALE)
    total = 0
    for python in (0, 1):
        values = [random_value(rng, python=bool(python)) for _ in range(n)]
        texts = [write(rng, v).encode() for v in values]
        for indent in (-1, 0, 2, 4):
            for ensure in (1, 0):
                for sort in (0, 1):
                    for allow_nan in (1, 0):
                        total += compare(texts, python=python, label=f"generated p{python} i{indent} e{ensure} s{sort} a{allow_nan}",
                                         indent=indent, ensure=ensure, sort=sort, allow_nan=allow_nan)
    report("generated documents x 32 options", total)

# 3. mutated documents: the same verdict as the oracle -----------------------------------------
if section("mutated"):
    rng = random.Random(2)
    n = int(60000 * SCALE)
    seeds = [write(rng, random_value(rng)).encode() for _ in range(2000)]
    cases = [mutate(rng, rng.choice(seeds)) for _ in range(n)]
    cases += [mutate(rng, s[: rng.randrange(len(s) + 1)]) for s in seeds]
    total = compare(cases, label="mutated")
    total += compare(cases, python=1, label="mutated python mode")
    report("mutated documents", total)

# 4. numbers --------------------------------------------------------------------------------------
if section("numbers"):
    rng = random.Random(3)
    getcontext_prec = 60
    from decimal import getcontext
    getcontext().prec = 800
    texts = []
    # every double kind, written as Python's repr
    for _ in range(int(200 * SCALE)):
        xs = [random_float(rng, False) for _ in range(500)]
        texts.append(("[" + ",".join(repr(x) for x in xs) + "]").encode())
    # long decimal strings, extreme exponents
    for _ in range(int(200 * SCALE)):
        items = []
        for _ in range(500):
            digits = "".join(rng.choice("0123456789") for _ in range(rng.randrange(1, 40)))
            digits = digits.lstrip("0") or "0"
            frac = "".join(rng.choice("0123456789") for _ in range(rng.randrange(0, 30)))
            exp = rng.choice(["", "e%d" % rng.randrange(-400, 400), "E+%d" % rng.randrange(0, 400),
                              "e-%d" % rng.randrange(300, 360)])
            items.append(rng.choice(["", "-"]) + digits + ("." + frac if frac else "") + exp)
        texts.append(("[" + ",".join(items) + "]").encode())
    # halfway cases: the exact midpoint of two adjacent doubles, and a hair either side
    for _ in range(int(100 * SCALE)):
        items = []
        for _ in range(300):
            x = abs(random_float(rng, False)) or 1.0
            if math.isinf(x):
                continue
            mid = (Decimal(x) + Decimal(math.nextafter(x, math.inf))) / 2
            for d in (mid, mid + Decimal(10) ** (mid.adjusted() - 700), mid - Decimal(10) ** (mid.adjusted() - 700)):
                items.append(format(d, "e"))
        texts.append(("[" + ",".join(items) + "]").encode())
    # every power of two (the only doubles whose rounding interval is not symmetric), its two
    # neighbours, and their negatives: exhaustive for the second candidate of python_float
    xs = []
    for k in range(-1074, 1024):
        x = math.ldexp(1.0, k)
        xs += [x, math.nextafter(x, 0), math.nextafter(x, math.inf), -x, -math.nextafter(x, 0)]
    xs = [x for x in xs if math.isfinite(x)]
    texts.append(("[" + ",".join(repr(x) for x in xs) + "]").encode())
    # integers: int64 boundaries and big integers up to 5,000 digits
    ints = [0, -0, 1, -1, 2**63 - 1, -2**63, 2**63, -2**63 - 1, 2**64 - 1, 2**64, -2**64, 10**18, 10**19]
    ints += [rng.randrange(10**(k - 1), 10**k) * rng.choice([1, -1]) for k in range(1, 400)]
    ints += [rng.randrange(10**4999, 10**5000)]
    texts.append(("[" + ",".join(str(i) for i in ints) + ", -0, -0.0, 0e0, 0E-0, 1e400, -1e400, 1e-400]").encode())
    total = 0
    for t in texts:
        total += t.count(b",") + 1
    compare(texts, label="numbers")
    report("numbers (doubles, decimals, ints)", total)

# 5. depth: the limit exactly, and depth beyond any call stack ----------------------------------
if section("depth"):
    cases = [b"[" * 10000 + b"]" * 10000, b"[" * 10001 + b"]" * 10001,
             b'{"a":' * 10000 + b"1" + b"}" * 10000, b'{"a":' * 10001 + b"1" + b"}" * 10001]
    got = run(cases)
    check(got[0][0] == "OK" and got[0][1] == b"[" * 10000 + b"]" * 10000, "depth 10000 array")
    check(got[1][0] == "ERR" and "max_depth 10000" in got[1][2] and got[1][1][0] == 10000, f"depth 10001 array {got[1][:2]}")
    check(got[2][0] == "OK" and got[2][1] == b'{"a": ' * 10000 + b"1" + b"}" * 10000, "depth 10000 object")
    check(got[3][0] == "ERR" and got[3][1][0] == 50000 and "max_depth 10000" in got[3][2], f"depth 10001 object {got[3][1]}")
    deep = 1_000_000
    got = run([b"[" * deep + b"]" * deep, b'{"":' * deep + b"[]" + b"}" * deep, b"[" * deep], max_depth=deep + 1)
    check(got[0][0] == "OK" and got[0][1] == b"[" * deep + b"]" * deep, "1,000,000 deep array round trip")
    check(got[1][0] == "OK" and got[1][1] == b'{"": ' * deep + b"[]" + b"}" * deep, "1,000,000 deep object round trip")
    check(got[2][0] == "ERR" and got[2][1][0] == deep, f"1,000,000 unclosed {got[2][:2]}")
    d = 3000
    want = "".join("[\n" + " " * (2 * (k + 1)) for k in range(d - 1)) + "[]" + "".join(
        "\n" + " " * (2 * k) + "]" for k in range(d - 2, -1, -1))
    got = run([b"[" * d + b"]" * d], indent=2)
    check(got[0] == ("OK", want.encode()), "3,000 deep, indented")
    report("depth", 9)

# 6. errors: the message, the offset, the line and the column -----------------------------------
if section("errors"):
    table = [
        (b"", 0, 1, 1, "expected a value, found the end of the text"),
        (b"  \n ", 4, 2, 2, "expected a value, found the end of the text"),
        (b"[1,]", 3, 1, 4, "expected a value, found ']'"),
        (b'{"a":1,}', 7, 1, 8, "expected a key, found '}'"),
        (b'{"a" 1}', 5, 1, 6, "expected ':', found '1'"),
        (b"[1 2]", 3, 1, 4, "expected ',' or ']', found '2'"),
        (b'{"a":1 "b":2}', 7, 1, 8, "expected ',' or '}', found '\"'"),
        (b"[\n1,\n  01]", 8, 3, 4, "expected ',' or ']', found '1'"),
        (b"01", 1, 1, 2, "expected the end of the text, found '1'"),
        (b"-", 1, 1, 2, "expected a digit, found the end of the text"),
        (b"-a", 1, 1, 2, "expected a digit, found 'a'"),
        (b"1.", 2, 1, 3, "expected a digit after '.', found the end of the text"),
        (b"1.e5", 2, 1, 3, "expected a digit after '.', found 'e'"),
        (b"1e", 2, 1, 3, "expected a digit in the exponent, found the end of the text"),
        (b"1e+", 3, 1, 4, "expected a digit in the exponent, found the end of the text"),
        (b"1e+x", 3, 1, 4, "expected a digit in the exponent, found 'x'"),
        (b"+1", 0, 1, 1, "expected a value, found '+'"),
        (b".5", 0, 1, 1, "expected a value, found '.'"),
        (b"tru", 0, 1, 1, "expected a value, found 't'"),
        (b"nul", 0, 1, 1, "expected a value, found 'n'"),
        (b"truex", 4, 1, 5, "expected the end of the text, found 'x'"),
        (b"NaN", 0, 1, 1, "expected a value, found 'N'"),
        (b"-Infinity", 1, 1, 2, "expected a digit, found 'I'"),
        (b'"abc', 4, 1, 5, "a string is not closed"),
        (b'"\\', 2, 1, 3, "a string is not closed"),
        (b'"a\tb"', 2, 1, 3, "a control character (byte 0x09) in a string must be escaped"),
        (b'"\x00"', 1, 1, 2, "a control character (byte 0x00) in a string must be escaped"),
        (b'"\\x"', 1, 1, 2, "an invalid escape, '\\' followed by 'x'"),
        (b'"\\\x01"', 1, 1, 2, "an invalid escape, '\\' followed by byte 0x01"),
        (b'"\\\xc3\xa9"', 1, 1, 2, "an invalid escape, '\\' followed by byte 0xC3"),
        (b'"\\ud800\\ue000"', 1, 1, 2, "\\u escape of a high surrogate not followed by a low one"),
        (b'"\\ud800\\udbff"', 1, 1, 2, "\\u escape of a high surrogate not followed by a low one"),
        (b'"\\u12"', 3, 1, 4, "expected four hexadecimal digits after \\u"),
        (b'"\\u12g4"', 5, 1, 6, "expected a hexadecimal digit after \\u, found 'g'"),
        (b'"\\udc00"', 1, 1, 2, "\\u escape of a lone low surrogate"),
        (b'"\\ud800"', 1, 1, 2, "\\u escape of a high surrogate not followed by a low one"),
        (b'"\\ud800\\u0041"', 1, 1, 2, "\\u escape of a high surrogate not followed by a low one"),
        (b'"\xc0\x80"', 1, 1, 2, "a string is not well-formed UTF-8 (byte 0xC0)"),
        (b'"\xed\xa0\x80"', 1, 1, 2, "a string is not well-formed UTF-8 (byte 0xED)"),
        (b'"\xf4\x90\x80\x80"', 1, 1, 2, "a string is not well-formed UTF-8 (byte 0xF4)"),
        (b'"\xe2\x82"', 1, 1, 2, "a string is not well-formed UTF-8 (byte 0xE2)"),
        (b"\xef\xbb\xbf{}", 0, 1, 1, "expected a value, found byte 0xEF"),
        (b"[1]\xff", 3, 1, 4, "expected the end of the text, found byte 0xFF"),
        (b"[1] // c", 4, 1, 5, "expected the end of the text, found '/'"),
        (b"\x0b[]", 0, 1, 1, "expected a value, found byte 0x0B"),
        (b"[", 1, 1, 2, "expected a value, found the end of the text"),
        (b"[1", 2, 1, 3, "expected ',' or ']', found the end of the text"),
        (b'{"a":1', 6, 1, 7, "expected ',' or '}', found the end of the text"),
        (b'{"a"', 4, 1, 5, "expected ':', found the end of the text"),
        (b"{", 1, 1, 2, "expected a key, found the end of the text"),
        (b"{1:2}", 1, 1, 2, "expected a key, found '1'"),
        (b"[1,", 3, 1, 4, "expected a value, found the end of the text"),
        (b"\r\n\r\n x", 5, 3, 2, "expected a value, found 'x'"),
    ]
    got = run([t[0] for t in table])
    for (text, offset, line, column, message), result in zip(table, got):
        check(result[0] == "ERR", f"error {text!r}: accepted")
        if result[0] == "ERR":
            check(result[1] == (offset, line, column), f"error {text!r}: position {result[1]} want {(offset, line, column)}")
            check(result[2] == f"JSON: {message} at line {line} column {column} (byte {offset})",
                  f"error {text!r}: message {result[2]!r} want {message!r}")
    got = run([b"NaN", b"-NaN", b"Infinity", b"-Infinity", b"[NaN,Infinity,-Infinity]", b"nan", b"-infinity"], python=1)
    check([g[0] for g in got] == ["OK", "ERR", "OK", "OK", "OK", "ERR", "ERR"], f"python constants {got}")
    check(got[4][1] == b"[NaN, Infinity, -Infinity]", "python constants dump")
    report("errors and positions", len(table) + 7)

# 7. the writer ----------------------------------------------------------------------------------
if section("writer"):
    got = run([b"[NaN]", b"Infinity", b"-Infinity", b"[1.5]"], python=1, allow_nan=0)
    check([g[0] for g in got] == ["DUMPERR", "DUMPERR", "DUMPERR", "OK"], f"allow_nan=0 {got}")
    check("out of range float values are not JSON compliant" in got[0][1], "allow_nan message")
    cases = [b'{"b":1,"a":{"d":[],"c":{}},"\\u00e9":2,"\\ud83d\\ude00":3,"\\uffff":4,"A":5,"":6}',
             b'["\\u007f\\u0080\\u2028\\u0000\\u001f\\ufeff\\/"]', b"[]", b"{}", b'[[], {}, [[]], {"a": {}}]',
             b'"\\ud83d\\ude00 \xf0\x9f\x98\x80 \xc3\xa9"', b"[1, [2, [3, {\"x\": [4]}]]]"]
    total = 0
    for indent in (-1, 0, 1, 3, 8):
        for ensure in (0, 1):
            for sort in (0, 1):
                total += compare(cases, label=f"writer i{indent} e{ensure} s{sort}", indent=indent, ensure=ensure, sort=sort)
    report("writer options", total)

# 8. JSON Lines ----------------------------------------------------------------------------------
if section("jsonl"):
    def python_jsonl(data):
        lines = data.split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()
        values = []
        for number, line in enumerate(lines, 1):
            if line.strip(b" \t\r") == b"":
                return ("ERR", number)
            v = oracle(line)
            if v is REJECT:
                return ("ERR", number)
            values.append(v)
        return ("OK", b"".join(json.dumps(v).encode() + b"\n" for v in values))
    rng = random.Random(5)
    cases = [b"", b"1", b"1\n", b"1\n2", b"1\r\n2\r\n", b"\n", b"1\n\n2\n", b"1\n \n", b"[1,\n2]\n",
             b'{"a":1}\n[2]\n"x"\nnull\n', b"1\n2\n3\n\n", b" 1 \n\t2\t\n", b"1\n\r\n2", b"1\n\t\n"]
    for _ in range(int(3000 * SCALE)):
        rows = [write(rng, random_value(rng)).replace("\n", " ").replace("\r", " ").encode() for _ in range(rng.randrange(1, 6))]
        text = b"\n".join(rows) + rng.choice([b"", b"\n", b"\r\n"])
        if rng.random() < 0.3:
            text = mutate(rng, text)
        cases.append(text)
    got = run(cases, mode="jsonl")
    for case, result in zip(cases, got):
        want = python_jsonl(case)
        if want[0] == "OK":
            check(result == ("OK", want[1]), f"jsonl {case[:200]!r}: {str(result)[:200]}")
        else:
            check(result[0] == "ERR" and result[1][1] == want[1], f"jsonl {case[:200]!r}: line {result[1] if result[0]=='ERR' else result} want {want}")
    check(got[0] == ("OK", b""), "jsonl empty text is no values")
    report("JSON Lines", len(cases))

# 9. a locale whose decimal point is ',' --------------------------------------------------------
if section("locale"):
    texts = [b"[1.5, 2.25e3, -0.000123, 1e400, 123456789.987654321]"]
    a = run(texts)
    b = run(texts, locale="de_DE.UTF-8")
    check(a == b and a[0][0] == "OK", f"locale de_DE: {a} vs {b}")
    report("locale de_DE.UTF-8", 1)

print("FAILURES:", len(FAILURES))
sys.exit(1 if FAILURES else 0)
