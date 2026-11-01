// tokenizer/cpp/tokenizer.cpp -- see tokenizer.hpp. Standard C++17 only.
#include "tokenizer.hpp"

#include <algorithm>
#include <climits>
#include <queue>

#include "pretokenize.hpp"
#include "utf8.hpp"

namespace mira_bpe {

Tokenizer::Tokenizer(const std::vector<Merge>& merges, const std::vector<std::string>& specials)
    : specials_(specials) {
    // A symbol's id is the id of its bytes; when two merges spell the same bytes the later one
    // owns them, and the earlier id carries no bytes.
    std::unordered_map<std::string, std::int32_t> symbol_ids;
    symbol_ids.reserve(256 + merges.size());
    for (int b = 0; b < 256; ++b) symbol_ids[std::string(1, static_cast<char>(b))] = b;
    std::int32_t next = 256;
    for (const Merge& merge : merges) symbol_ids[merge.first + merge.second] = next++;
    n_base_ = next;
    id_bytes_.assign(static_cast<std::size_t>(n_base_), std::string());
    has_bytes_.assign(static_cast<std::size_t>(n_base_), 0);
    for (const auto& entry : symbol_ids) {
        id_bytes_[static_cast<std::size_t>(entry.second)] = entry.first;
        has_bytes_[static_cast<std::size_t>(entry.second)] = 1;
    }
    pairs_.reserve(merges.size());
    for (std::size_t rank = 0; rank < merges.size(); ++rank) {
        auto left = symbol_ids.find(merges[rank].first);
        auto right = symbol_ids.find(merges[rank].second);
        if (left == symbol_ids.end() || right == symbol_ids.end()) continue;
        std::int32_t merged = symbol_ids.at(merges[rank].first + merges[rank].second);
        pairs_[pair_key(static_cast<std::uint32_t>(left->second),
                        static_cast<std::uint32_t>(right->second))] = {
            static_cast<std::int32_t>(rank), merged};
    }
    special_order_.resize(specials_.size());
    for (std::size_t i = 0; i < specials_.size(); ++i) special_order_[i] = i;
    std::stable_sort(special_order_.begin(), special_order_.end(), [&](std::size_t a, std::size_t b) {
        return specials_[a].size() > specials_[b].size();
    });
    for (const std::string& special : specials_)
        if (!special.empty()) special_first_byte_[static_cast<unsigned char>(special[0])] = true;
}

const std::pair<std::int32_t, std::int32_t>* Tokenizer::pair_of(std::int32_t a, std::int32_t b) const {
    auto it = pairs_.find(pair_key(static_cast<std::uint32_t>(a), static_cast<std::uint32_t>(b)));
    return it == pairs_.end() ? nullptr : &it->second;
}

// The lowest-ranked adjacent pair is merged, the leftmost on a tie, until no adjacent pair is a
// merge. Up to SHORT_WORD symbols the pairs are scanned; longer words keep them in a heap.
void Tokenizer::merge_word(std::vector<std::int32_t>& s) const {
    constexpr std::size_t SHORT_WORD = 48;
    constexpr std::int32_t NONE = INT32_MAX;
    std::size_t n = s.size();
    if (n < 2) return;
    auto rank_of = [&](std::int32_t a, std::int32_t b) {
        const auto* p = pair_of(a, b);
        return p ? p->first : NONE;
    };
    if (n <= SHORT_WORD) {
        std::vector<std::int32_t> rank(n - 1);
        for (std::size_t i = 0; i + 1 < n; ++i) rank[i] = rank_of(s[i], s[i + 1]);
        while (true) {
            std::int32_t best = NONE;
            std::size_t at = rank.size();
            for (std::size_t i = 0; i < rank.size(); ++i)
                if (rank[i] < best) {
                    best = rank[i];
                    at = i;
                }
            if (at == rank.size()) return;
            s[at] = pair_of(s[at], s[at + 1])->second;
            s.erase(s.begin() + static_cast<std::ptrdiff_t>(at) + 1);
            rank.erase(rank.begin() + static_cast<std::ptrdiff_t>(at));
            if (at > 0) rank[at - 1] = rank_of(s[at - 1], s[at]);
            if (at < rank.size()) rank[at] = rank_of(s[at], s[at + 1]);
        }
    }
    struct Entry {
        std::int32_t rank;
        std::int64_t left, right;
        bool operator>(const Entry& other) const {
            return rank != other.rank ? rank > other.rank : left > other.left;
        }
    };
    std::vector<std::int64_t> prev(n), next(n);
    std::vector<char> alive(n, 1);
    for (std::size_t i = 0; i < n; ++i) {
        prev[i] = static_cast<std::int64_t>(i) - 1;
        next[i] = i + 1 < n ? static_cast<std::int64_t>(i) + 1 : -1;
    }
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        std::int32_t r = rank_of(s[i], s[i + 1]);
        if (r != NONE) heap.push({r, static_cast<std::int64_t>(i), static_cast<std::int64_t>(i) + 1});
    }
    while (!heap.empty()) {
        Entry e = heap.top();
        heap.pop();
        if (!alive[e.left] || next[e.left] != e.right) continue;
        const auto* p = pair_of(s[e.left], s[e.right]);
        if (!p || p->first != e.rank) continue;
        s[e.left] = p->second;
        alive[e.right] = 0;
        next[e.left] = next[e.right];
        if (next[e.right] >= 0) prev[next[e.right]] = e.left;
        if (prev[e.left] >= 0) {
            std::int32_t r = rank_of(s[prev[e.left]], s[e.left]);
            if (r != NONE) heap.push({r, prev[e.left], e.left});
        }
        if (next[e.left] >= 0) {
            std::int32_t r = rank_of(s[e.left], s[next[e.left]]);
            if (r != NONE) heap.push({r, e.left, next[e.left]});
        }
    }
    std::vector<std::int32_t> merged;
    for (std::int64_t i = 0; i >= 0; i = next[i]) merged.push_back(s[i]);
    s.swap(merged);
}

void Tokenizer::encode_chunks(std::string_view text, std::vector<std::int32_t>& out) {
    for (const auto& span : pretokenize(text)) {
        std::string chunk(text.substr(span.first, span.second));
        auto hit = cache_.find(chunk);
        if (hit != cache_.end()) {
            out.insert(out.end(), hit->second.begin(), hit->second.end());
            continue;
        }
        std::vector<std::int32_t> ids;
        ids.reserve(chunk.size());
        for (unsigned char b : chunk) ids.push_back(b);
        merge_word(ids);
        out.insert(out.end(), ids.begin(), ids.end());
        if (cache_.size() < CHUNK_CACHE_MAX) cache_.emplace(std::move(chunk), std::move(ids));
    }
}

std::vector<std::int32_t> Tokenizer::encode_ordinary(std::string_view text) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::int32_t> out;
    out.reserve(text.size() / 3 + 1);
    encode_chunks(text, out);
    return out;
}

std::vector<std::int32_t> Tokenizer::encode(std::string_view text, bool split_specials) {
    if (!split_specials || specials_.empty()) return encode_ordinary(text);
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::int32_t> out;
    out.reserve(text.size() / 3 + 1);
    std::size_t pos = 0, i = 0, n = text.size();
    while (i < n) {
        if (special_first_byte_[static_cast<unsigned char>(text[i])]) {
            bool matched = false;
            for (std::size_t index : special_order_) {
                const std::string& special = specials_[index];
                if (n - i >= special.size() && text.compare(i, special.size(), special) == 0) {
                    if (i > pos) encode_chunks(text.substr(pos, i - pos), out);
                    out.push_back(n_base_ + static_cast<std::int32_t>(index));
                    i += special.size();
                    pos = i;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }
        ++i;
    }
    if (pos < n) encode_chunks(text.substr(pos), out);
    return out;
}

std::string Tokenizer::decode(const std::vector<std::int64_t>& ids, bool skip_special_tokens) const {
    std::string out, buffer;
    for (std::int64_t id : ids) {
        if (id >= n_base_) {
            std::int64_t index = id - n_base_;
            if (!skip_special_tokens && index < static_cast<std::int64_t>(specials_.size())) {
                out += utf8_replace(buffer);
                buffer.clear();
                out += specials_[static_cast<std::size_t>(index)];
            }
            continue;
        }
        if (id < 0 || !has_bytes_[static_cast<std::size_t>(id)]) continue;
        buffer += id_bytes_[static_cast<std::size_t>(id)];
    }
    out += utf8_replace(buffer);
    return out;
}

}  // namespace mira_bpe
