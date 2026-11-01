// tokenizer/cpp/pretokenize.cpp -- see pretokenize.hpp. Standard C++17 only.
#include "pretokenize.hpp"

#include "utf8.hpp"

namespace mira_bpe {

// \s*\d | \s*[^\s\d]+ | \s+ : leading whitespace joins the digit or the run of other characters
// that follows it; whitespace at the end of the text is a chunk of its own.
std::vector<std::pair<std::size_t, std::size_t>> pretokenize(std::string_view text) {
    std::vector<std::pair<std::size_t, std::size_t>> chunks;
    const unsigned char* s = reinterpret_cast<const unsigned char*>(text.data());
    std::size_t n = text.size(), p = 0;
    while (p < n) {
        std::size_t q = p, length = 0;
        CharClass c = OTHER;
        while (q < n) {
            c = class_at(s + q, n - q, &length);
            if (c != SPACE) break;
            q += length;
        }
        if (q >= n) {
            chunks.emplace_back(p, n - p);
            break;
        }
        if (c == DIGIT) {
            chunks.emplace_back(p, q + length - p);
            p = q + length;
            continue;
        }
        std::size_t r = q + length;
        while (r < n) {
            if (class_at(s + r, n - r, &length) != OTHER) break;
            r += length;
        }
        chunks.emplace_back(p, r - p);
        p = r;
    }
    return chunks;
}

}  // namespace mira_bpe
