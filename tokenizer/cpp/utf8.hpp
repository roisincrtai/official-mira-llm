// tokenizer/cpp/utf8.hpp -- UTF-8 and the character classes of Python's re \s and \d.
//
// Standard C++17 only. A byte that does not start a well-formed UTF-8 sequence is one character
// that is neither whitespace nor a digit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace mira_bpe {

enum CharClass : unsigned char { OTHER = 0, SPACE = 1, DIGIT = 2 };

// The length of the well-formed UTF-8 sequence at p (its code point in *cp), or -k when the
// sequence is ill-formed and its longest well-formed prefix has k >= 1 bytes.
int utf8_sequence(const unsigned char* p, std::size_t n, std::uint32_t* cp);

// The class of the character at p (Python's str.isspace() and str.isdecimal()), its byte length
// in *length; an ill-formed byte is one OTHER character.
CharClass class_at(const unsigned char* p, std::size_t n, std::size_t* length);

// The UTF-8 bytes of the code point cp appended to out.
void append_utf8(std::string& out, std::uint32_t cp);

// `bytes` decoded as UTF-8 with Python's errors="replace", written back as UTF-8.
std::string utf8_replace(std::string_view bytes);

}  // namespace mira_bpe
