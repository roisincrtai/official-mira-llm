"""Each mutation is one deliberate bug in json.cpp; the tests must fail for every one."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "build", "tokenizer", "cpp", "test"))
base = open(f"{SRC}/json.cpp").read()
M = [
 ("leading zeros accepted", "if (pos_ < n && text_[pos_] == '0') {\n            ++pos_;\n        } else if", "if (false) {\n        } else if"),
 ("raw control characters accepted", "while (run < n && s[run] >= 0x20 &&", "while (run < n && s[run] >= 0x01 &&"),
 ("repeated key kept twice", "if (found != frame.slots.end()) {", "if (false && found != frame.slots.end()) {"),
 ("repeated key keeps its first value", "node.items[found->second] = Json();\n            return &node.items[found->second];", "node.items.emplace_back();\n            return &node.items.back();"),
 ("trailing comma accepted", "                if (d == ',') {\n                    ++pos_;", "                if (d == ',') {\n                    ++pos_;\n                    skip();\n                    if (pos_ < text_.size() && text_[pos_] == (object ? '}' : ']')) continue;"),
 ("lone low surrogate accepted", "if (cp >= 0xDC00 && cp <= 0xDFFF)\n                        fail_at", "if (false)\n                        fail_at"),
 ("vertical tab is whitespace", "if (c != ' ' && c != '\\t' && c != '\\n' && c != '\\r') return;", "if (c != ' ' && c != '\\t' && c != '\\n' && c != '\\r' && c != '\\v') return;"),
 ("DEL not escaped", "else if (cp < 0x7F) out.push_back", "else if (cp <= 0x7F) out.push_back"),
 ("sort_keys by signed char", "return static_cast<unsigned char>(x) <\n                                                static_cast<unsigned char>(y);", "return x < y;"),
 ("decimal point not swapped", "if (point != '.') std::replace", "if (false) std::replace"),
 ("no second candidate in repr", "if (back < magnitude) {\n            std::string above", "if (false) {\n            std::string above"),
 ("max_depth off by one", "if (stack.size() >= options_.max_depth)", "if (stack.size() > options_.max_depth)"),
 ("column off by one", "std::size_t column = at - start + 1,", "std::size_t column = at - start,"),
 ("blank JSON Lines line skipped", "        if (blank)\n            throw JsonError", "        if (blank) { start = end + 1; ++line; continue; }\n        if (false)\n            throw JsonError"),
 ("big integers become floats", "            v.type = Json::BigInt;\n            v.text = std::string(lexeme);", "            v.type = Json::Float;\n            v.text = std::string();"),
 ("recursive destructor", "Json::~Json() {\n    if (items.empty()) return;", "Json::~Json() {\n    return;"),
 ("ill-formed UTF-8 accepted", "if (k < 0) fail(\"a string is not well-formed UTF-8 (\" + shown(c) + \")\");", "if (k < 0) k = 1;"),
 ("exponent without digits accepted", "            if (pos_ >= n || !is_digit(text_[pos_]))\n                fail(\"expected a digit in the exponent", "            if (false)\n                fail(\"expected a digit in the exponent"),
]
caught = 0
for name, old, new in M:
    assert base.count(old) == 1, name
    src = base.replace(old, new)
    if name == "recursive destructor":
        # a destructor that leaks would not crash; make it the plain recursive one instead
        src = src.replace("Json::~Json() {\n    return;", "Json::~Json() {\n    if (true) return;")
        src = src.replace("Json::~Json() {\n    if (true) return;", "Json::~Json() = default;\nvoid unused_destructor_body() {\n    std::vector<Json> items;\n    if (items.empty()) return;", 1)
    d = f"{OUT}/mutant"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/json.cpp", "w").write(src)
    for f in ("json.hpp", "utf8.hpp", "utf8.cpp"):
        open(f"{d}/{f}", "w").write(open(f"{SRC}/{f}").read())
    hpp = open(f"{d}/json.hpp").read()
    if name == "recursive destructor":
        hpp = hpp.replace("    ~Json();", "    ~Json() = default;")
        src2 = open(f"{d}/json.cpp").read().replace("Json::~Json() = default;\n", "")
        open(f"{d}/json.cpp", "w").write(src2)
    open(f"{d}/json.hpp", "w").write(hpp)
    b = subprocess.run(f"g++ -std=c++17 -O2 -I{d} {HERE}/driver.cpp {d}/json.cpp {d}/utf8.cpp -o {d}/driver && "
                       f"g++ -std=c++17 -O2 -I{d} {HERE}/unit.cpp {d}/json.cpp {d}/utf8.cpp -o {d}/unit",
                       shell=True, capture_output=True, text=True)
    if b.returncode:
        print(f"{name:40s} DID NOT BUILD: {b.stderr[-500:]}")
        continue
    u = subprocess.run([f"{d}/unit"], capture_output=True)
    t = subprocess.run([sys.executable, os.path.join(HERE, "tests.py")], capture_output=True, text=True,
                       env={**os.environ, "DRIVER": os.path.join("mutant", "driver"), "SCALE": "0.1"})
    fails = [l for l in t.stdout.splitlines() if l.startswith("FAIL")]
    hit = t.returncode != 0 or u.returncode != 0
    caught += hit
    why = (fails[0][:110] if fails else (t.stderr.strip().splitlines() or ["unit failed"])[-1][:110]) if hit else ""
    print(f"{name:40s} {'CAUGHT' if hit else 'MISSED'}  unit={'fail' if u.returncode else 'pass'}  {why}", flush=True)
print(f"caught {caught} of {len(M)}")
