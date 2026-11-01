// tokenizer/cpp/json.hpp -- a strict, complete JSON reader and writer (RFC 8259, ECMA-404), and
// JSON Lines.
//
// Standard C++17 only. The design follows nlohmann/json:
//   - a lexer that scans the RFC 8259 grammar exactly: numbers by its state machine
//     (-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?), strings with every byte checked against
//     the well-formed UTF-8 table (Unicode Table 3-7), control characters U+0000..U+001F rejected
//     unless escaped, \u escapes with surrogate pairs required to pair;
//   - a parser that keeps its open arrays and objects on an explicit stack, not the call stack, so
//     no input can overflow the call stack; nesting deeper than ParseOptions::max_depth is an
//     error;
//   - numbers converted independently of the C locale (the decimal point is swapped for the
//     locale's before strtod, as nlohmann/json's lexer does), correctly rounded by strtod;
//   - values copied, written and destroyed without recursion.
//
// The values and the writer follow Python's json module, so that what Python writes reads back
// unchanged and what is written here is what Python writes:
//   - an integer that fits in 64 bits is Int; a longer one is BigInt and keeps its digits exactly
//     (Python's int); a number with a fraction or an exponent is Float (Python's float), and
//     one beyond the double range is +-inf, as Python reads it;
//   - a repeated key in an object keeps its first position and takes its last value (Python's
//     dict);
//   - dump_json writes as Python's json.dumps: the same separators, ensure_ascii, indent,
//     sort_keys and allow_nan, floats as Python's repr.
//
// Strict reading rejects what RFC 8259 does not define: NaN and Infinity (allowed with
// ParseOptions::allow_nan_infinity, which is ParseOptions::python()), a byte order mark,
// comments, trailing commas, and a \u escape of a lone surrogate (it is no character, and a
// string here is UTF-8).
//
// JSON Lines (jsonlines.org): one JSON value per line, lines separated by '\n'; a '\r' before
// the '\n' is whitespace; the last line may end with '\n'; an empty line is an error.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mira_bpe {

// One JSON value; an object keeps its keys in the order they were first written.
struct Json {
    enum Type { Null, Bool, Int, BigInt, Float, String, Array, Object } type = Null;
    bool boolean = false;
    std::int64_t integer = 0;          // Int
    double real = 0.0;                 // Float; for BigInt the nearest double
    std::string text;                  // String (UTF-8); BigInt: its decimal digits, '-' first
    std::vector<Json> items;           // Array elements, or Object values
    std::vector<std::string> keys;     // Object keys, parallel to items

    Json() = default;
    Json(const Json& other);
    Json(Json&& other) noexcept = default;
    Json& operator=(const Json& other);
    Json& operator=(Json&& other) noexcept = default;
    ~Json();

    // The value of `key` in an Object, or null.
    const Json* get(const std::string& key) const;
};

// A malformed text: what was wrong, and where (offset counts bytes from 0; line and column
// count from 1, the column in bytes).
struct JsonError : std::runtime_error {
    JsonError(const std::string& what, std::size_t offset, std::size_t line, std::size_t column);
    std::size_t offset, line, column;
};

struct ParseOptions {
    bool allow_nan_infinity = false;   // read NaN, Infinity and -Infinity, as Python's json does
    std::size_t max_depth = 10000;     // arrays and objects open at once
    // Python's json.loads: NaN, Infinity and -Infinity are read.
    static ParseOptions python() {
        ParseOptions options;
        options.allow_nan_infinity = true;
        return options;
    }
};

// The JSON value of `text`; throws JsonError when `text` is not one JSON value.
Json parse_json(std::string_view text, const ParseOptions& options = ParseOptions());
// The values of the lines of `text`; throws JsonError naming the line of the first bad one.
std::vector<Json> parse_jsonl(std::string_view text, const ParseOptions& options = ParseOptions());

// Python's json.dumps and its defaults.
struct DumpOptions {
    int indent = -1;                   // < 0: one line; >= 0: one item per line, indented
    bool ensure_ascii = true;          // escape every character beyond ASCII as \uXXXX
    bool sort_keys = false;            // write an object's keys in code point order
    bool allow_nan = true;             // write NaN, Infinity and -Infinity; else throw
};

// The text of `value`; throws std::invalid_argument for a NaN or an infinity without allow_nan,
// for a string or key that is not well-formed UTF-8, and for a BigInt that is not an integer.
std::string dump_json(const Json& value, const DumpOptions& options = DumpOptions());
// One line per value, each ended by '\n'; options.indent is ignored.
std::string dump_jsonl(const std::vector<Json>& values, const DumpOptions& options = DumpOptions());

// An Int value.
Json json_int(std::int64_t value);

}  // namespace mira_bpe
