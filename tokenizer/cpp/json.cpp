// tokenizer/cpp/json.cpp -- see json.hpp. Standard C++17 only.
#include "json.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "utf8.hpp"

namespace mira_bpe {

// ---------------------------------------------------------------------------------------------
// values: copied and destroyed without recursion
// ---------------------------------------------------------------------------------------------

namespace {

// `source` without its elements.
Json shallow(const Json& source) {
    Json copy;
    copy.type = source.type;
    copy.boolean = source.boolean;
    copy.integer = source.integer;
    copy.real = source.real;
    copy.text = source.text;
    copy.keys = source.keys;
    return copy;
}

}  // namespace

Json::Json(const Json& other)
    : type(other.type), boolean(other.boolean), integer(other.integer), real(other.real),
      text(other.text), keys(other.keys) {
    std::vector<std::pair<const Json*, Json*>> work{{&other, this}};
    while (!work.empty()) {
        auto [source, target] = work.back();
        work.pop_back();
        target->items.reserve(source->items.size());       // the pointers below stay valid
        for (const Json& item : source->items) {
            target->items.push_back(shallow(item));
            if (!item.items.empty()) work.emplace_back(&item, &target->items.back());
        }
    }
}

Json& Json::operator=(const Json& other) {
    if (this != &other) {
        Json copy(other);
        *this = std::move(copy);
    }
    return *this;
}

// The elements are moved to a stack and taken apart one level at a time, so the destructor of
// any element it runs finds no elements below it.
Json::~Json() {
    if (items.empty()) return;
    std::vector<Json> pending = std::move(items);
    while (!pending.empty()) {
        Json current = std::move(pending.back());
        pending.pop_back();
        for (Json& item : current.items)
            if (!item.items.empty()) pending.push_back(std::move(item));
    }
}

const Json* Json::get(const std::string& key) const {
    for (std::size_t i = 0; i < keys.size(); ++i)
        if (keys[i] == key) return &items[i];
    return nullptr;
}

JsonError::JsonError(const std::string& what, std::size_t offset_, std::size_t line_,
                     std::size_t column_)
    : std::runtime_error(what), offset(offset_), line(line_), column(column_) {}

Json json_int(std::int64_t value) {
    Json v;
    v.type = Json::Int;
    v.integer = value;
    return v;
}

// ---------------------------------------------------------------------------------------------
// reading
// ---------------------------------------------------------------------------------------------

namespace {

bool is_digit(char c) { return c >= '0' && c <= '9'; }

// A byte for a message: itself when printable ASCII, else its hexadecimal value.
std::string shown(unsigned char c) {
    static const char* HEX = "0123456789ABCDEF";
    if (c >= 0x21 && c < 0x7F) return std::string("'") + static_cast<char>(c) + "'";
    return std::string("byte 0x") + HEX[c >> 4] + HEX[c & 0xF];
}

double to_double(std::string_view lexeme) {
    // strtod reads the decimal point of the C locale, so '.' is swapped for it (nlohmann/json).
    std::string buffer(lexeme);
    const char point = *std::localeconv()->decimal_point;
    if (point != '.') std::replace(buffer.begin(), buffer.end(), '.', point);
    errno = 0;
    return std::strtod(buffer.c_str(), nullptr);   // +-HUGE_VAL beyond the range, as Python
}

class Parser {
public:
    Parser(std::string_view text, const ParseOptions& options, std::size_t base, std::size_t line)
        : text_(text), options_(options), base_(base), first_line_(line) {}

    Json document() {
        Json root;
        Json* target = &root;
        std::vector<Frame> stack;
        while (true) {
            // a value into *target
            skip();
            if (pos_ >= text_.size()) fail("expected a value, " + found());
            char c = text_[pos_];
            bool opened = false;
            if (c == '[' || c == '{') {
                if (stack.size() >= options_.max_depth)
                    fail("nesting deeper than max_depth " + std::to_string(options_.max_depth));
                ++pos_;
                skip();
                target->type = c == '[' ? Json::Array : Json::Object;
                char close = c == '[' ? ']' : '}';
                if (pos_ < text_.size() && text_[pos_] == close) {
                    ++pos_;
                } else {
                    stack.push_back(Frame{target, {}});
                    opened = true;
                    if (c == '[') {
                        target->items.emplace_back();
                        target = &target->items.back();
                    } else {
                        target = member(stack.back());
                    }
                }
            } else if (c == '"') {
                target->type = Json::String;
                target->text = string();
            } else if (c == '-' || is_digit(c)) {
                number(*target);
            } else if (literal("true")) {
                target->type = Json::Bool;
                target->boolean = true;
            } else if (literal("false")) {
                target->type = Json::Bool;
            } else if (literal("null")) {
                target->type = Json::Null;
            } else if (options_.allow_nan_infinity && literal("NaN")) {
                target->type = Json::Float;
                target->real = std::numeric_limits<double>::quiet_NaN();
            } else if (options_.allow_nan_infinity && literal("Infinity")) {
                target->type = Json::Float;
                target->real = std::numeric_limits<double>::infinity();
            } else {
                fail("expected a value, " + found());
            }
            if (opened) continue;
            // after a value: the next one of the innermost open array or object, or its end
            while (true) {
                if (stack.empty()) {
                    skip();
                    if (pos_ != text_.size()) fail("expected the end of the text, " + found());
                    return root;
                }
                Frame& frame = stack.back();
                bool object = frame.node->type == Json::Object;
                skip();
                char d = pos_ < text_.size() ? text_[pos_] : '\0';
                if (d == ',') {
                    ++pos_;
                    if (object) {
                        target = member(frame);
                    } else {
                        frame.node->items.emplace_back();
                        target = &frame.node->items.back();
                    }
                    break;
                }
                if (d == (object ? '}' : ']')) {
                    ++pos_;
                    stack.pop_back();
                    continue;
                }
                fail(std::string("expected ',' or '") + (object ? '}' : ']') + "', " + found());
            }
        }
    }

private:
    // An open array or object, and for an object the slot of each key read so far.
    struct Frame {
        Json* node;
        std::unordered_map<std::string, std::size_t> slots;
    };

    [[noreturn]] void fail(const std::string& what) const { fail_at(what, pos_); }

    // What is at pos_, for a message.
    std::string found() const {
        return pos_ < text_.size() ? "found " + shown(static_cast<unsigned char>(text_[pos_]))
                                   : std::string("found the end of the text");
    }

    [[noreturn]] void fail_at(const std::string& what, std::size_t at) const {
        std::size_t line = first_line_, start = 0;
        for (std::size_t i = 0; i < at; ++i)                 // at <= text_.size()
            if (text_[i] == '\n') {
                ++line;
                start = i + 1;
            }
        std::size_t column = at - start + 1, offset = base_ + at;
        throw JsonError("JSON: " + what + " at line " + std::to_string(line) + " column " +
                            std::to_string(column) + " (byte " + std::to_string(offset) + ")",
                        offset, line, column);
    }

    void skip() {
        while (pos_ < text_.size()) {
            char c = text_[pos_];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') return;
            ++pos_;
        }
    }

    bool literal(const char* word) {
        std::size_t length = std::strlen(word);
        if (text_.compare(pos_, length, word) != 0) return false;
        pos_ += length;
        return true;
    }

    // A key, its ':', and the slot of its value: a new one, or the earlier one of a repeated key.
    Json* member(Frame& frame) {
        skip();
        if (pos_ >= text_.size() || text_[pos_] != '"') fail("expected a key, " + found());
        std::string key = string();
        skip();
        if (pos_ >= text_.size() || text_[pos_] != ':') fail("expected ':', " + found());
        ++pos_;
        Json& node = *frame.node;
        auto found = frame.slots.find(key);
        if (found != frame.slots.end()) {
            node.items[found->second] = Json();
            return &node.items[found->second];
        }
        frame.slots.emplace(key, node.items.size());
        node.keys.push_back(std::move(key));
        node.items.emplace_back();
        return &node.items.back();
    }

    unsigned hex4() {
        if (pos_ + 4 > text_.size()) fail("expected four hexadecimal digits after \\u");
        unsigned value = 0;
        for (int k = 0; k < 4; ++k) {
            char c = text_[pos_];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<unsigned>(c - 'A' + 10);
            else fail("expected a hexadecimal digit after \\u, " + found());
            ++pos_;
        }
        return value;
    }

    // The string at pos_ (its opening quote), UTF-8.
    std::string string() {
        ++pos_;
        std::string out;
        const unsigned char* s = reinterpret_cast<const unsigned char*>(text_.data());
        const std::size_t n = text_.size();
        while (true) {
            std::size_t run = pos_;
            while (run < n && s[run] >= 0x20 && s[run] < 0x80 && s[run] != '"' && s[run] != '\\')
                ++run;
            out.append(text_.data() + pos_, run - pos_);
            pos_ = run;
            if (pos_ >= n) fail("a string is not closed");
            unsigned char c = s[pos_];
            if (c == '"') {
                ++pos_;
                return out;
            }
            if (c < 0x20)
                fail("a control character (" + shown(c) + ") in a string must be escaped");
            if (c >= 0x80) {
                std::uint32_t cp = 0;
                int k = utf8_sequence(s + pos_, n - pos_, &cp);
                if (k < 0) fail("a string is not well-formed UTF-8 (" + shown(c) + ")");
                out.append(text_.data() + pos_, static_cast<std::size_t>(k));
                pos_ += static_cast<std::size_t>(k);
                continue;
            }
            // an escape
            std::size_t start = pos_;
            ++pos_;
            if (pos_ >= n) fail("a string is not closed");
            char e = text_[pos_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    std::uint32_t cp = hex4();
                    if (cp >= 0xDC00 && cp <= 0xDFFF)
                        fail_at("\\u escape of a lone low surrogate", start);
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (text_.compare(pos_, 2, "\\u") != 0)
                            fail_at("\\u escape of a high surrogate not followed by a low one",
                                    start);
                        pos_ += 2;
                        std::uint32_t low = hex4();
                        if (low < 0xDC00 || low > 0xDFFF)
                            fail_at("\\u escape of a high surrogate not followed by a low one",
                                    start);
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    }
                    append_utf8(out, cp);
                } break;
                default:
                    pos_ = start;
                    fail("an invalid escape, '\\' followed by " +
                         shown(static_cast<unsigned char>(e)));
            }
        }
    }

    // The number at pos_, by the RFC 8259 grammar
    // -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?, or -Infinity.
    void number(Json& v) {
        const std::size_t start = pos_, n = text_.size();
        bool integer = true;
        if (text_[pos_] == '-') {
            ++pos_;
            if (options_.allow_nan_infinity && literal("Infinity")) {
                v.type = Json::Float;
                v.real = -std::numeric_limits<double>::infinity();
                return;
            }
        }
        if (pos_ < n && text_[pos_] == '0') {
            ++pos_;
        } else if (pos_ < n && is_digit(text_[pos_])) {
            while (pos_ < n && is_digit(text_[pos_])) ++pos_;
        } else {
            fail("expected a digit, " + found());
        }
        if (pos_ < n && text_[pos_] == '.') {
            integer = false;
            ++pos_;
            if (pos_ >= n || !is_digit(text_[pos_])) fail("expected a digit after '.', " + found());
            while (pos_ < n && is_digit(text_[pos_])) ++pos_;
        }
        if (pos_ < n && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            integer = false;
            ++pos_;
            if (pos_ < n && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            if (pos_ >= n || !is_digit(text_[pos_]))
                fail("expected a digit in the exponent, " + found());
            while (pos_ < n && is_digit(text_[pos_])) ++pos_;
        }
        std::string_view lexeme = text_.substr(start, pos_ - start);
        if (integer) {
            std::int64_t value = 0;
            // the lexeme is all digits, so from_chars fails only when it is out of range
            const char* end = lexeme.data() + lexeme.size();
            if (std::from_chars(lexeme.data(), end, value).ec == std::errc()) {
                v.type = Json::Int;
                v.integer = value;
                return;
            }
            v.type = Json::BigInt;
            v.text = std::string(lexeme);
            v.real = to_double(lexeme);
            return;
        }
        v.type = Json::Float;
        v.real = to_double(lexeme);
    }

    std::string_view text_;
    const ParseOptions& options_;
    std::size_t base_, first_line_;
    std::size_t pos_ = 0;
};

}  // namespace

Json parse_json(std::string_view text, const ParseOptions& options) {
    return Parser(text, options, 0, 1).document();
}

std::vector<Json> parse_jsonl(std::string_view text, const ParseOptions& options) {
    std::vector<Json> values;
    std::size_t start = 0, line = 1;
    while (start < text.size()) {
        std::size_t end = text.find('\n', start);
        if (end == std::string_view::npos) end = text.size();
        std::string_view row = text.substr(start, end - start);
        bool blank = std::all_of(row.begin(), row.end(), [](char c) {
            return c == ' ' || c == '\t' || c == '\r';
        });
        if (blank)
            throw JsonError("JSON: line " + std::to_string(line) + " of a JSON Lines text is empty"
                            " (byte " + std::to_string(start) + ")",
                            start, line, 1);
        values.push_back(Parser(row, options, start, line).document());
        start = end + 1;
        ++line;
    }
    return values;
}

// ---------------------------------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------------------------------

namespace {

// `text` ("d.ddde+XX" of a positive number, from std::scientific) as its digits and its decimal
// exponent.
void split_scientific(const std::string& text, std::string& digits, int& exponent) {
    std::size_t e = text.find('e');
    exponent = std::atoi(text.c_str() + e + 1);
    digits.clear();
    for (std::size_t i = 0; i < e; ++i)
        if (text[i] != '.') digits.push_back(text[i]);
}

std::string join_scientific(const std::string& digits, int exponent) {
    std::string text = digits.substr(0, 1);
    if (digits.size() > 1) text += "." + digits.substr(1);
    return text + "e" + std::to_string(exponent);
}

// The digits of D + 1 at the scale of D, D having digits.size() digits and the first at
// 10^exponent.
void increment_digits(std::string& digits, int& exponent) {
    std::size_t i = digits.size();
    while (i > 0 && digits[i - 1] == '9') digits[--i] = '0';
    if (i == 0) {
        digits.insert(digits.begin(), '1');
        digits.pop_back();
        ++exponent;
    } else {
        ++digits[i - 1];
    }
}

// Python's repr of a float (David Gay's shortest mode): the fewest significant digits that read
// back as x, the closest to x when two such have as few; written in fixed notation for decimal
// exponents -4 to 15 and in scientific notation otherwise. At each precision the correctly
// rounded digits are tried, and when they fall below x the digits one unit above them: when x is
// a power of two its rounding interval is twice as wide above as below, so those can read back
// as x when the nearer ones below do not (the other way round never: below, the interval is the
// narrower). 17 significant digits always read back. Reading back uses strtod, which gives +-inf
// beyond the range (an istream gives the largest double there).
std::string python_float(double x) {
    if (std::isnan(x)) return "NaN";
    if (std::isinf(x)) return x > 0 ? "Infinity" : "-Infinity";
    if (x == 0.0) return std::signbit(x) ? "-0.0" : "0.0";
    const bool negative = x < 0;
    const double magnitude = std::fabs(x);
    std::string digits;
    int exponent = 0;
    for (int precision = 1;; ++precision) {
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << std::scientific << std::setprecision(precision - 1) << magnitude;
        split_scientific(out.str(), digits, exponent);
        if (precision == 17) break;
        double back = to_double(join_scientific(digits, exponent));
        if (back == magnitude) break;
        if (back < magnitude) {
            std::string above = digits;
            int above_exponent = exponent;
            increment_digits(above, above_exponent);
            if (to_double(join_scientific(above, above_exponent)) == magnitude) {
                digits = above;
                exponent = above_exponent;
                break;
            }
        }
    }
    // The digits never end in 0: such digits would be a value with one digit fewer, which would
    // have read back at the precision before.
    int point = exponent + 1;
    std::string out;
    if (exponent >= -4 && exponent < 16) {
        if (point <= 0) out = "0." + std::string(static_cast<std::size_t>(-point), '0') + digits;
        else if (static_cast<std::size_t>(point) >= digits.size())
            out = digits + std::string(static_cast<std::size_t>(point) - digits.size(), '0') + ".0";
        else out = digits.substr(0, static_cast<std::size_t>(point)) + "." +
                   digits.substr(static_cast<std::size_t>(point));
    } else {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        std::string power = std::to_string(exponent < 0 ? -exponent : exponent);
        if (power.size() < 2) power = "0" + power;
        out += std::string("e") + (exponent < 0 ? "-" : "+") + power;
    }
    return negative ? "-" + out : out;
}

void dump_string(const std::string& text, bool ensure_ascii, std::string& out) {
    static const char* HEX = "0123456789abcdef";
    auto escape = [&](std::uint32_t unit) {
        out += "\\u";
        for (int shift = 12; shift >= 0; shift -= 4) out.push_back(HEX[(unit >> shift) & 0xF]);
    };
    out.push_back('"');
    const unsigned char* s = reinterpret_cast<const unsigned char*>(text.data());
    std::size_t i = 0, n = text.size();
    while (i < n) {
        std::uint32_t cp = 0;
        int k = utf8_sequence(s + i, n - i, &cp);
        if (k < 0)
            throw std::invalid_argument("JSON: a string is not well-formed UTF-8 (byte " +
                                        std::to_string(i) + " of it)");
        std::size_t at = i;
        i += static_cast<std::size_t>(k);
        switch (cp) {
            case '"': out += "\\\""; continue;
            case '\\': out += "\\\\"; continue;
            case '\n': out += "\\n"; continue;
            case '\r': out += "\\r"; continue;
            case '\t': out += "\\t"; continue;
            case '\b': out += "\\b"; continue;
            case '\f': out += "\\f"; continue;
            default: break;
        }
        if (cp < 0x20) escape(cp);
        else if (cp < 0x7F) out.push_back(static_cast<char>(cp));
        else if (!ensure_ascii) out.append(text, at, static_cast<std::size_t>(k));
        else if (cp < 0x10000) escape(cp);
        else {
            cp -= 0x10000;
            escape(0xD800 | (cp >> 10));
            escape(0xDC00 | (cp & 0x3FF));
        }
    }
    out.push_back('"');
}

bool is_integer_text(const std::string& text) {
    std::size_t i = text.size() > 0 && text[0] == '-' ? 1 : 0;
    if (i >= text.size()) return false;
    if (text[i] == '0') return i + 1 == text.size();
    return std::all_of(text.begin() + static_cast<std::ptrdiff_t>(i), text.end(), is_digit);
}

// A value that is not an Array or an Object.
void dump_scalar(const Json& value, const DumpOptions& options, std::string& out) {
    if (value.type == Json::Null) {
        out += "null";
    } else if (value.type == Json::Bool) {
        out += value.boolean ? "true" : "false";
    } else if (value.type == Json::Int) {
        out += std::to_string(value.integer);
    } else if (value.type == Json::BigInt) {
        if (!is_integer_text(value.text))
            throw std::invalid_argument("JSON: a BigInt whose text is not an integer");
        out += value.text;
    } else if (value.type == Json::Float) {
        if (!options.allow_nan && !std::isfinite(value.real))
            throw std::invalid_argument("JSON: out of range float values are not JSON compliant");
        out += python_float(value.real);
    } else {
        dump_string(value.text, options.ensure_ascii, out);
    }
}

void dump_into(const Json& root, const DumpOptions& options, std::string& out) {
    const bool lines = options.indent >= 0;
    const char* item_separator = lines ? "," : ", ";
    auto newline = [&](std::size_t depth) {
        out.push_back('\n');
        out.append(depth * static_cast<std::size_t>(options.indent), ' ');
    };
    struct Frame {
        const Json* node;
        std::size_t next;
        std::vector<std::size_t> order;   // the items in writing order
    };
    std::vector<Frame> stack;
    // writes `value`, or opens it
    auto start = [&](const Json& value) {
        if (value.type != Json::Array && value.type != Json::Object) {
            dump_scalar(value, options, out);
            return;
        }
        bool object = value.type == Json::Object;
        if (object && value.keys.size() != value.items.size())
            throw std::invalid_argument("JSON: an object whose keys and values differ in number");
        if (value.items.empty()) {
            out += object ? "{}" : "[]";
            return;
        }
        out.push_back(object ? '{' : '[');
        Frame frame{&value, 0, std::vector<std::size_t>(value.items.size())};
        for (std::size_t i = 0; i < frame.order.size(); ++i) frame.order[i] = i;
        if (object && options.sort_keys)
            std::stable_sort(frame.order.begin(), frame.order.end(),
                             [&](std::size_t a, std::size_t b) {
                                 // UTF-8 byte order is code point order
                                 return std::lexicographical_compare(
                                     value.keys[a].begin(), value.keys[a].end(),
                                     value.keys[b].begin(), value.keys[b].end(),
                                     [](char x, char y) {
                                         return static_cast<unsigned char>(x) <
                                                static_cast<unsigned char>(y);
                                     });
                             });
        stack.push_back(std::move(frame));
    };
    start(root);
    while (!stack.empty()) {
        Frame& frame = stack.back();
        const Json& node = *frame.node;
        bool object = node.type == Json::Object;
        if (frame.next == node.items.size()) {
            stack.pop_back();
            if (lines) newline(stack.size());
            out.push_back(object ? '}' : ']');
            continue;
        }
        if (frame.next > 0) out += item_separator;
        if (lines) newline(stack.size());
        std::size_t index = frame.order[frame.next++];
        if (object) {
            dump_string(node.keys[index], options.ensure_ascii, out);
            out += ": ";
        }
        start(node.items[index]);    // may push, after which `frame` is not used again
    }
}

}  // namespace

std::string dump_json(const Json& value, const DumpOptions& options) {
    std::string out;
    dump_into(value, options, out);
    return out;
}

std::string dump_jsonl(const std::vector<Json>& values, const DumpOptions& options) {
    DumpOptions line = options;
    line.indent = -1;
    std::string out;
    for (const Json& value : values) {
        dump_into(value, line, out);
        out.push_back('\n');
    }
    return out;
}

}  // namespace mira_bpe
