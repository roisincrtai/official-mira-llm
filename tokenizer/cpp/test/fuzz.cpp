// libFuzzer target for tokenizer/cpp/json.cpp, built with AddressSanitizer and
// UndefinedBehaviorSanitizer. For every input, in strict and in Python mode:
//   - parse_json either throws JsonError or gives a value;
//   - a value's text (compact, and indented with sorted keys and raw UTF-8) reads back to a value
//     whose text is the same, byte for byte;
//   - a value written without NaN and Infinity reads back strictly to the same text;
//   - parse_jsonl either throws JsonError or gives values whose dump_jsonl reads back the same.
// Anything else (another exception, a crash, a sanitizer report, a mismatch) stops the fuzzer.
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

#include "json.hpp"

using namespace mira_bpe;

static void require(bool condition) {
    if (!condition) std::abort();
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    std::string_view text(reinterpret_cast<const char*>(data), size);
    for (int python = 0; python < 2; ++python) {
        ParseOptions read;
        read.allow_nan_infinity = python != 0;
        read.max_depth = 256;
        ParseOptions again = ParseOptions::python();
        again.max_depth = 256;
        bool parsed = false;
        Json value;
        try {
            value = parse_json(text, read);
            parsed = true;
        } catch (const JsonError&) {
        }
        if (parsed) {
            std::string compact = dump_json(value);
            require(dump_json(parse_json(compact, again)) == compact);
            // without NaN and Infinity (a number beyond the double range reads as one), the
            // text reads back strictly
            ParseOptions strict;
            strict.max_depth = 256;
            DumpOptions finite;
            finite.allow_nan = false;
            std::string compliant;
            bool written = true;
            try {
                compliant = dump_json(value, finite);
            } catch (const std::invalid_argument&) {
                written = false;
            }
            if (written) require(dump_json(parse_json(compliant, strict), finite) == compliant);
            DumpOptions pretty;
            pretty.indent = 2;
            pretty.sort_keys = true;
            pretty.ensure_ascii = false;
            std::string indented = dump_json(value, pretty);
            require(dump_json(parse_json(indented, again), pretty) == indented);
            DumpOptions sorted;
            sorted.sort_keys = true;
            require(dump_json(parse_json(indented, again), sorted) == dump_json(value, sorted));
        }
        try {
            std::string lines = dump_jsonl(parse_jsonl(text, read));
            require(dump_jsonl(parse_jsonl(lines, again)) == lines);
        } catch (const JsonError&) {
        }
    }
    return 0;
}
