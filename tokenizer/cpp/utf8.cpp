// tokenizer/cpp/utf8.cpp -- see utf8.hpp. Standard C++17 only.
#include "utf8.hpp"

namespace mira_bpe {
namespace {

// ---------------------------------------------------------------------------------------------
// Unicode classes: Python's re \s and \d on str, which are str.isspace() and str.isdecimal().
// Generated from Python 3.13 (Unicode 15.1.0).
// ---------------------------------------------------------------------------------------------

struct Range {
    std::uint32_t lo, hi;
};

constexpr Range SPACE_RANGES[] = {
    {0x9, 0xD}, {0x1C, 0x20}, {0x85, 0x85}, {0xA0, 0xA0}, {0x1680, 0x1680}, {0x2000, 0x200A},
    {0x2028, 0x2029}, {0x202F, 0x202F}, {0x205F, 0x205F}, {0x3000, 0x3000},
};

constexpr Range DIGIT_RANGES[] = {
    {0x30, 0x39}, {0x660, 0x669}, {0x6F0, 0x6F9}, {0x7C0, 0x7C9}, {0x966, 0x96F}, {0x9E6, 0x9EF},
    {0xA66, 0xA6F}, {0xAE6, 0xAEF}, {0xB66, 0xB6F}, {0xBE6, 0xBEF}, {0xC66, 0xC6F}, {0xCE6, 0xCEF},
    {0xD66, 0xD6F}, {0xDE6, 0xDEF}, {0xE50, 0xE59}, {0xED0, 0xED9}, {0xF20, 0xF29},
    {0x1040, 0x1049}, {0x1090, 0x1099}, {0x17E0, 0x17E9}, {0x1810, 0x1819}, {0x1946, 0x194F},
    {0x19D0, 0x19D9}, {0x1A80, 0x1A89}, {0x1A90, 0x1A99}, {0x1B50, 0x1B59}, {0x1BB0, 0x1BB9},
    {0x1C40, 0x1C49}, {0x1C50, 0x1C59}, {0xA620, 0xA629}, {0xA8D0, 0xA8D9}, {0xA900, 0xA909},
    {0xA9D0, 0xA9D9}, {0xA9F0, 0xA9F9}, {0xAA50, 0xAA59}, {0xABF0, 0xABF9}, {0xFF10, 0xFF19},
    {0x104A0, 0x104A9}, {0x10D30, 0x10D39}, {0x11066, 0x1106F}, {0x110F0, 0x110F9},
    {0x11136, 0x1113F}, {0x111D0, 0x111D9}, {0x112F0, 0x112F9}, {0x11450, 0x11459},
    {0x114D0, 0x114D9}, {0x11650, 0x11659}, {0x116C0, 0x116C9}, {0x11730, 0x11739},
    {0x118E0, 0x118E9}, {0x11950, 0x11959}, {0x11C50, 0x11C59}, {0x11D50, 0x11D59},
    {0x11DA0, 0x11DA9}, {0x11F50, 0x11F59}, {0x16A60, 0x16A69}, {0x16AC0, 0x16AC9},
    {0x16B50, 0x16B59}, {0x1D7CE, 0x1D7FF}, {0x1E140, 0x1E149}, {0x1E2F0, 0x1E2F9},
    {0x1E4F0, 0x1E4F9}, {0x1E950, 0x1E959}, {0x1FBF0, 0x1FBF9},
};

template <std::size_t N>
bool in_ranges(const Range (&ranges)[N], std::uint32_t cp) {
    std::size_t lo = 0, hi = N;
    while (lo < hi) {
        std::size_t mid = (lo + hi) / 2;
        if (cp < ranges[mid].lo) hi = mid;
        else if (cp > ranges[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

struct AsciiClasses {
    unsigned char of[128];
    AsciiClasses() {
        for (int c = 0; c < 128; ++c) {
            std::uint32_t cp = static_cast<std::uint32_t>(c);
            of[c] = in_ranges(SPACE_RANGES, cp) ? SPACE : in_ranges(DIGIT_RANGES, cp) ? DIGIT : OTHER;
        }
    }
};
const AsciiClasses ASCII_CLASSES;

}  // namespace

// The length of the well-formed UTF-8 sequence at p (its code point in *cp), or -k when the
// sequence is ill-formed and its longest well-formed prefix has k >= 1 bytes.
int utf8_sequence(const unsigned char* p, std::size_t n, std::uint32_t* cp) {
    unsigned char lead = p[0];
    if (lead < 0x80) {
        *cp = lead;
        return 1;
    }
    int need;
    std::uint32_t value;
    unsigned char second_lo = 0x80, second_hi = 0xBF;
    if (lead >= 0xC2 && lead <= 0xDF) {
        need = 2;
        value = lead & 0x1F;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
        need = 3;
        value = lead & 0x0F;
        if (lead == 0xE0) second_lo = 0xA0;
        else if (lead == 0xED) second_hi = 0x9F;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
        need = 4;
        value = lead & 0x07;
        if (lead == 0xF0) second_lo = 0x90;
        else if (lead == 0xF4) second_hi = 0x8F;
    } else {
        return -1;
    }
    for (int k = 1; k < need; ++k) {
        if (static_cast<std::size_t>(k) >= n) return -k;
        unsigned char c = p[k];
        unsigned char lo = k == 1 ? second_lo : 0x80, hi = k == 1 ? second_hi : 0xBF;
        if (c < lo || c > hi) return -k;
        value = (value << 6) | (c & 0x3F);
    }
    *cp = value;
    return need;
}

CharClass class_at(const unsigned char* p, std::size_t n, std::size_t* length) {
    if (p[0] < 0x80) {
        *length = 1;
        return static_cast<CharClass>(ASCII_CLASSES.of[p[0]]);
    }
    std::uint32_t cp = 0;
    int k = utf8_sequence(p, n, &cp);
    if (k < 0) {
        *length = 1;
        return OTHER;
    }
    *length = static_cast<std::size_t>(k);
    if (in_ranges(SPACE_RANGES, cp)) return SPACE;
    if (in_ranges(DIGIT_RANGES, cp)) return DIGIT;
    return OTHER;
}

void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string utf8_replace(std::string_view bytes) {
    std::string out;
    out.reserve(bytes.size());
    const unsigned char* s = reinterpret_cast<const unsigned char*>(bytes.data());
    std::size_t i = 0, n = bytes.size();
    while (i < n) {
        if (s[i] < 0x80) {
            out.push_back(static_cast<char>(s[i]));
            ++i;
            continue;
        }
        std::uint32_t cp = 0;
        int k = utf8_sequence(s + i, n - i, &cp);
        if (k > 0) {
            out.append(bytes.data() + i, static_cast<std::size_t>(k));
            i += static_cast<std::size_t>(k);
        } else {
            out += "\xEF\xBF\xBD";
            i += static_cast<std::size_t>(-k);
        }
    }
    return out;
}

}  // namespace mira_bpe
