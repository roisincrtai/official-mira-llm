// Test driver for tokenizer/cpp/json.cpp. Reads cases from stdin, each a 4-byte little-endian
// length and that many bytes, and writes for each:
//     "OK <n>\n" and n bytes of dump_json (or dump_jsonl) output, or
//     "ERR <offset> <line> <column> <n>\n" and n bytes of the message, or
//     "DUMPERR <n>\n" and the message when writing failed.
// argv: mode (json|jsonl) python(0|1) max_depth indent ensure_ascii sort_keys allow_nan [locale]
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include "json.hpp"
using namespace mira_bpe;
int main(int argc, char** argv) {
    if (argc < 8) return 64;
    if (argc > 8) std::setlocale(LC_ALL, argv[8]);
    std::string mode = argv[1];
    ParseOptions p;
    p.allow_nan_infinity = std::atoi(argv[2]) != 0;
    p.max_depth = std::strtoull(argv[3], nullptr, 10);
    DumpOptions d;
    d.indent = std::atoi(argv[4]);
    d.ensure_ascii = std::atoi(argv[5]) != 0;
    d.sort_keys = std::atoi(argv[6]) != 0;
    d.allow_nan = std::atoi(argv[7]) != 0;
    std::ios::sync_with_stdio(false);
    while (true) {
        unsigned char h[4];
        if (std::fread(h, 1, 4, stdin) != 4) break;
        std::uint32_t n = std::uint32_t(h[0]) | (std::uint32_t(h[1]) << 8) | (std::uint32_t(h[2]) << 16) | (std::uint32_t(h[3]) << 24);
        std::string text(n, '\0');
        if (n && std::fread(&text[0], 1, n, stdin) != n) return 65;
        std::string out;
        try {
            if (mode == "json") {
                Json v = parse_json(text, p);
                Json copy = v;                       // the copy must write the same
                Json moved = std::move(copy);
                try {
                    out = dump_json(v, d);
                    if (dump_json(moved, d) != out) { std::fprintf(stderr, "copy differs\n"); return 66; }
                } catch (const std::exception& e) {
                    std::string m = e.what();
                    std::printf("DUMPERR %zu\n", m.size());
                    std::fwrite(m.data(), 1, m.size(), stdout);
                    continue;
                }
            } else {
                std::vector<Json> v = parse_jsonl(text, p);
                try { out = dump_jsonl(v, d); } catch (const std::exception& e) {
                    std::string m = e.what();
                    std::printf("DUMPERR %zu\n", m.size());
                    std::fwrite(m.data(), 1, m.size(), stdout);
                    continue;
                }
            }
        } catch (const JsonError& e) {
            std::string m = e.what();
            std::printf("ERR %zu %zu %zu %zu\n", e.offset, e.line, e.column, m.size());
            std::fwrite(m.data(), 1, m.size(), stdout);
            continue;
        }
        std::printf("OK %zu\n", out.size());
        std::fwrite(out.data(), 1, out.size(), stdout);
    }
    return 0;
}
