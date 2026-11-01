// tokenizer/cpp/pretokenize.hpp -- the pre-tokenizer of tokenizer/py/bpe.py.
//
// Standard C++17 only. \s*\d|\s*[^\s\d]+|\s+ with Python's Unicode \s and \d.
#pragma once

#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

namespace mira_bpe {

constexpr int PRETOK_VERSION = 2;

// The pre-token chunks of `text`, as (offset, length) byte spans that concatenate back to `text`.
std::vector<std::pair<std::size_t, std::size_t>> pretokenize(std::string_view text);

}  // namespace mira_bpe
