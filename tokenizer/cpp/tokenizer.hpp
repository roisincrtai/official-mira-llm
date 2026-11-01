// tokenizer/cpp/tokenizer.hpp -- encoding and decoding with learned merges.
//
// Standard C++17 only. The merge rule is the lowest-ranked adjacent pair, leftmost on a tie; the
// specials are recognised longest first; decoding is UTF-8 with errors="replace".
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mira_bpe {

constexpr std::size_t CHUNK_CACHE_MAX = 500000;

// A merge: the bytes of its left symbol and the bytes of its right symbol.
using Merge = std::pair<std::string, std::string>;

// The key of the adjacent pair (a, b) of symbol ids.
inline std::uint64_t pair_key(std::uint32_t a, std::uint32_t b) {
    return (static_cast<std::uint64_t>(a) << 32) | b;
}

class Tokenizer {
public:
    // Ids: the 256 bytes, then one per merge in rank order, then the specials in list order.
    Tokenizer(const std::vector<Merge>& merges, const std::vector<std::string>& specials);

    // Every character is text; a special in `text` encodes as its bytes.
    std::vector<std::int32_t> encode_ordinary(std::string_view text);
    // The specials are recognised first, longest first, and each is one id.
    std::vector<std::int32_t> encode(std::string_view text, bool split_specials = true);
    // A special carries no bytes; with skip_special_tokens false it is written as its text.
    std::string decode(const std::vector<std::int64_t>& ids, bool skip_special_tokens = true) const;
    std::size_t size() const { return static_cast<std::size_t>(n_base_) + specials_.size(); }

private:
    void merge_word(std::vector<std::int32_t>& symbols) const;
    void encode_chunks(std::string_view text, std::vector<std::int32_t>& out);
    const std::pair<std::int32_t, std::int32_t>* pair_of(std::int32_t a, std::int32_t b) const;

    std::int32_t n_base_ = 0;
    std::vector<std::string> id_bytes_;
    std::vector<char> has_bytes_;
    // (left id << 32 | right id) -> (rank, id of the merged symbol)
    std::unordered_map<std::uint64_t, std::pair<std::int32_t, std::int32_t>> pairs_;
    std::vector<std::string> specials_;
    std::vector<std::size_t> special_order_;
    bool special_first_byte_[256] = {};
    std::unordered_map<std::string, std::vector<std::int32_t>> cache_;
    std::mutex mutex_;
};

}  // namespace mira_bpe
