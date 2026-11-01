// Unit tests of tokenizer/cpp/json.hpp's C++ interface: what the text driver cannot reach.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "json.hpp"

using namespace mira_bpe;

static int failures = 0, checks = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        ++checks;                                                                   \
        if (!(cond)) {                                                              \
            ++failures;                                                             \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond);            \
        }                                                                           \
    } while (0)

template <class F>
static std::string thrown(F f) {
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

static Json str(const std::string& s) {
    Json v;
    v.type = Json::String;
    v.text = s;
    return v;
}

int main() {
    // values: copy, assignment, self-assignment, move, get, json_int
    Json doc = parse_json(R"({"a": [1, {"b": "x"}], "c": null, "d": 1.5})");
    Json copy = doc;
    CHECK(dump_json(copy) == dump_json(doc));
    Json assigned;
    assigned = doc;
    CHECK(dump_json(assigned) == dump_json(doc));
    Json& alias = assigned;
    assigned = alias;                                   // self-assignment
    CHECK(dump_json(assigned) == dump_json(doc));
    Json moved = std::move(copy);
    CHECK(dump_json(moved) == dump_json(doc));
    CHECK(copy.items.empty());
    Json move_assigned;
    move_assigned = std::move(moved);
    CHECK(dump_json(move_assigned) == dump_json(doc));
    CHECK(doc.get("a") != nullptr && doc.get("a")->type == Json::Array);
    CHECK(doc.get("c") != nullptr && doc.get("c")->type == Json::Null);
    CHECK(doc.get("missing") == nullptr);
    CHECK(Json().get("a") == nullptr);
    Json i = json_int(-42);
    CHECK(i.type == Json::Int && i.integer == -42 && dump_json(i) == "-42");
    CHECK(dump_json(json_int(std::numeric_limits<std::int64_t>::min())) == "-9223372036854775808");

    // a tree built in code, 2,000,000 deep: copied, written, destroyed without recursion
    {
        Json root;
        Json* at = &root;
        const int depth = 2000000;
        for (int k = 0; k < depth; ++k) {
            at->type = Json::Array;
            at->items.emplace_back();
            at = &at->items.back();
        }
        Json twin = root;
        std::string text = dump_json(twin);
        CHECK(text.size() == 2u * depth + 4u && text.compare(0, 3, "[[[") == 0);
        Json other;
        other = root;
        CHECK(dump_json(other) == text);
    }

    // values read: types and fields
    Json all = parse_json(R"([null, true, false, 0, -0, 9223372036854775807, -9223372036854775808,
        9223372036854775808, -9223372036854775809, 1.0, -0.0, 1e400, "sé😀"])");
    CHECK(all.items.size() == 13);
    CHECK(all.items[0].type == Json::Null);
    CHECK(all.items[1].type == Json::Bool && all.items[1].boolean);
    CHECK(all.items[2].type == Json::Bool && !all.items[2].boolean);
    CHECK(all.items[3].type == Json::Int && all.items[3].integer == 0);
    CHECK(all.items[4].type == Json::Int && all.items[4].integer == 0);
    CHECK(all.items[5].type == Json::Int && all.items[5].integer == std::numeric_limits<std::int64_t>::max());
    CHECK(all.items[6].type == Json::Int && all.items[6].integer == std::numeric_limits<std::int64_t>::min());
    CHECK(all.items[7].type == Json::BigInt && all.items[7].text == "9223372036854775808" &&
          all.items[7].real == 9223372036854775808.0);
    CHECK(all.items[8].type == Json::BigInt && all.items[8].text == "-9223372036854775809");
    CHECK(all.items[9].type == Json::Float && all.items[9].real == 1.0);
    CHECK(all.items[10].type == Json::Float && std::signbit(all.items[10].real));
    CHECK(all.items[11].type == Json::Float && std::isinf(all.items[11].real));
    CHECK(all.items[12].type == Json::String && all.items[12].text == "s\xC3\xA9\xF0\x9F\x98\x80");

    // a repeated key: the first position, the last value
    Json dup = parse_json(R"({"k": 1, "x": 2, "k": {"deep": [3]}})");
    CHECK(dup.keys.size() == 2 && dup.keys[0] == "k" && dup.keys[1] == "x");
    CHECK(dump_json(dup) == R"({"k": {"deep": [3]}, "x": 2})");

    // options
    ParseOptions none;
    none.max_depth = 0;
    CHECK(thrown([&] { parse_json("[]", none); }).find("max_depth 0") != std::string::npos);
    CHECK(dump_json(parse_json("7", none)) == "7");
    CHECK(ParseOptions::python().allow_nan_infinity && !ParseOptions().allow_nan_infinity);
    CHECK(ParseOptions().max_depth == 10000);
    Json nan = parse_json("NaN", ParseOptions::python());
    CHECK(nan.type == Json::Float && std::isnan(nan.real));

    // JsonError fields
    try {
        parse_json("[1,\n  x]");
        CHECK(false);
    } catch (const JsonError& e) {
        CHECK(e.offset == 6 && e.line == 2 && e.column == 3);
    }
    try {
        parse_jsonl("1\n2\n[x]\n");
        CHECK(false);
    } catch (const JsonError& e) {
        CHECK(e.offset == 5 && e.line == 3 && e.column == 2);
    }
    try {
        parse_jsonl("1\n\n2");
        CHECK(false);
    } catch (const JsonError& e) {
        CHECK(e.offset == 2 && e.line == 2 && e.column == 1);
    }
    CHECK(parse_jsonl("").empty());
    CHECK(parse_jsonl("1\n2\n").size() == 2);

    // the writer: what only a value built in code can hold
    CHECK(thrown([] { dump_json(str("ok\xFF")); }).find("not well-formed UTF-8") != std::string::npos);
    CHECK(thrown([] { dump_json(str("\xED\xA0\x80")); }).find("not well-formed UTF-8") != std::string::npos);
    {
        Json o;
        o.type = Json::Object;
        o.keys.push_back("bad\xC0");
        o.items.push_back(json_int(1));
        CHECK(thrown([&] { dump_json(o); }).find("not well-formed UTF-8") != std::string::npos);
        o.keys.push_back("extra");
        CHECK(thrown([&] { dump_json(o); }).find("keys and values differ") != std::string::npos);
    }
    for (const char* bad : {"", "-", "12a", "01", "-01", "+1", "1.0", " 1"}) {
        Json b;
        b.type = Json::BigInt;
        b.text = bad;
        CHECK(thrown([&] { dump_json(b); }).find("BigInt") != std::string::npos);
    }
    for (const char* good : {"0", "-0", "12345678901234567890123", "-9"}) {
        Json b;
        b.type = Json::BigInt;
        b.text = good;
        CHECK(dump_json(b) == good);
    }
    {
        Json f;
        f.type = Json::Float;
        f.real = std::numeric_limits<double>::infinity();
        DumpOptions strict;
        strict.allow_nan = false;
        CHECK(thrown([&] { dump_json(f, strict); }).find("not JSON compliant") != std::string::npos);
        CHECK(dump_json(f) == "Infinity");
        f.real = 2.5;
        CHECK(dump_json(f, strict) == "2.5");
    }
    {
        // dump_jsonl: one line each, the indent ignored, the options kept
        std::vector<Json> rows{parse_json(R"({"b": 1, "a": [1, 2]})"), str("\xC3\xA9")};
        DumpOptions o;
        o.indent = 4;
        o.sort_keys = true;
        o.ensure_ascii = false;
        CHECK(dump_jsonl(rows, o) == "{\"a\": [1, 2], \"b\": 1}\n\"\xC3\xA9\"\n");
        CHECK(dump_jsonl({}) == "");
    }

    std::printf("unit: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
