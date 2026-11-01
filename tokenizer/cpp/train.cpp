// tokenizer/cpp/train.cpp -- see train.hpp. Standard C++17 only.
#include "train.hpp"

#include <algorithm>
#include <iomanip>
#include <locale>
#include <queue>
#include <sstream>
#include <stdexcept>

#include "checkpoint.hpp"
#include "pretokenize.hpp"
#include "sha1.hpp"

namespace mira_bpe {
namespace {

// An integer with a comma between thousands, as Python's format spec ",".
std::string with_commas(std::int64_t value) {
    std::string digits = std::to_string(value < 0 ? -value : value), out;
    int count = 0;
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
        if (count && count % 3 == 0) out.push_back(',');
        out.push_back(*it);
        ++count;
    }
    if (value < 0) out.push_back('-');
    std::reverse(out.begin(), out.end());
    return out;
}

std::string fixed(double value, int decimals) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::fixed << std::setprecision(decimals) << value;
    return stream.str();
}

using Item = std::pair<const std::string*, std::int64_t>;

bool bytes_less(const std::string& a, const std::string& b) {
    return std::lexicographical_compare(
        a.begin(), a.end(), b.begin(), b.end(),
        [](char x, char y) { return static_cast<unsigned char>(x) < static_cast<unsigned char>(y); });
}

// The word table's identity: its size, the first 20,000 words in sorted order with their
// counts, and the total count.
std::string corpus_signature(const std::vector<Item>& items) {
    Sha1 sha;
    sha.update(std::to_string(items.size()));
    std::vector<Item> sorted(items);
    std::size_t head = std::min<std::size_t>(sorted.size(), 20000);
    auto less = [](const Item& a, const Item& b) {
        if (*a.first != *b.first) return bytes_less(*a.first, *b.first);
        return a.second < b.second;
    };
    std::partial_sort(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(head), sorted.end(), less);
    std::int64_t total = 0;
    for (std::size_t i = 0; i < head; ++i) {
        sha.update(*sorted[i].first);
        sha.update(std::to_string(sorted[i].second));
    }
    for (const Item& item : items) total += item.second;
    sha.update(std::to_string(total));
    return sha.hexdigest();
}

}  // namespace

void WordCounter::add(std::string_view text) {
    for (const auto& span : pretokenize(text)) ++words_[std::string(text.substr(span.first, span.second))];
}

// Pair counts and a pair -> words index are built once and then maintained: a merge touches only
// the words that contain its pair. The best pair is the most frequent, the smallest (left, right)
// symbol ids on a tie, taken from a heap that holds one entry at the current count for every live
// pair, so an entry whose count is no longer current is discarded when it reaches the top.
BuildResult build(const WordCounter& counter, const BuildOptions& o) {
    auto say = [&](const std::string& line) {
        if (o.log && !o.log(line)) throw BuildStopped{};
    };
    auto report = [&](int phase, std::int64_t done, std::int64_t total) {
        if (o.progress && !o.progress(phase, done, total)) throw BuildStopped{};
    };

    std::vector<Item> items;
    items.reserve(counter.words().size());
    for (const auto& entry : counter.words())
        if (!entry.first.empty()) items.emplace_back(&entry.first, entry.second);

    if (o.min_freq > 1) {
        std::vector<Item> kept;
        std::int64_t occurrences = 0, kept_occurrences = 0;
        for (const Item& item : items) {
            occurrences += item.second;
            if (item.second >= static_cast<std::int64_t>(o.min_freq)) {
                kept.push_back(item);
                kept_occurrences += item.second;
            }
        }
        std::string threshold = std::to_string(o.min_freq);
        if (!kept.empty()) {
            double share = static_cast<double>(100 * static_cast<std::int64_t>(kept.size())) /
                           static_cast<double>(std::max<std::size_t>(items.size(), 1));
            double occurrence_share = static_cast<double>(100 * kept_occurrences) /
                                      static_cast<double>(std::max<std::int64_t>(occurrences, 1));
            say("BPE: min_freq=" + threshold + " keeps " + with_commas(static_cast<std::int64_t>(kept.size())) +
                " of " + with_commas(static_cast<std::int64_t>(items.size())) + " word types (" + fixed(share, 1) +
                "%), which are " + fixed(occurrence_share, 2) + "% of all occurrences");
            items.swap(kept);
        } else {
            say("BPE: min_freq=" + threshold + " would discard every word type; ignored");
        }
    }

    const std::string sig = corpus_signature(items);

    std::vector<std::string> symbols(256);
    for (int b = 0; b < 256; ++b) symbols[static_cast<std::size_t>(b)] = std::string(1, static_cast<char>(b));
    std::vector<std::vector<std::uint32_t>> words;
    std::vector<std::int64_t> freqs;
    words.reserve(items.size());
    freqs.reserve(items.size());
    std::int64_t tokens = 0;
    for (const Item& item : items) {
        std::vector<std::uint32_t> s;
        s.reserve(item.first->size());
        for (unsigned char b : *item.first) s.push_back(b);
        tokens += static_cast<std::int64_t>(s.size()) * item.second;
        words.push_back(std::move(s));
        freqs.push_back(item.second);
    }
    const std::int64_t total_bytes = tokens;

    std::unordered_map<std::uint64_t, std::int64_t> counts;
    std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> where;
    const std::int64_t n_words = static_cast<std::int64_t>(words.size());
    report(1, 0, n_words);
    for (std::size_t i = 0; i < words.size(); ++i) {
        const auto& s = words[i];
        for (std::size_t j = 0; j + 1 < s.size(); ++j) {
            std::uint64_t k = pair_key(s[j], s[j + 1]);
            counts[k] += freqs[i];
            auto& list = where[k];
            if (list.empty() || list.back() != i) list.push_back(static_cast<std::uint32_t>(i));
        }
        if ((i + 1) % 4096 == 0 || i + 1 == words.size()) report(1, static_cast<std::int64_t>(i + 1), n_words);
    }
    using HeapEntry = std::pair<std::int64_t, std::uint64_t>;   // (-count, pair)
    std::vector<HeapEntry> initial;
    initial.reserve(counts.size());
    for (const auto& entry : counts)
        if (entry.second > 0) initial.emplace_back(-entry.second, entry.first);
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap(
        std::greater<HeapEntry>(), std::move(initial));

    auto apply_merge = [&](std::uint32_t a, std::uint32_t b, std::uint32_t ab) -> std::int64_t {
        const std::uint64_t merged_key = pair_key(a, b);
        std::int64_t hit = 0;
        auto found = where.find(merged_key);
        if (found == where.end()) return 0;
        std::vector<std::uint32_t> holders = std::move(found->second);
        where.erase(found);
        std::sort(holders.begin(), holders.end());
        holders.erase(std::unique(holders.begin(), holders.end()), holders.end());
        std::vector<std::uint64_t> touched, before, after;
        std::vector<std::uint32_t> out;
        for (std::uint32_t i : holders) {
            auto& s = words[i];
            if (s.size() < 2) continue;
            out.clear();
            std::int64_t n = 0;
            for (std::size_t j = 0; j < s.size();) {
                if (j + 1 < s.size() && s[j] == a && s[j + 1] == b) {
                    out.push_back(ab);
                    j += 2;
                    ++n;
                } else {
                    out.push_back(s[j]);
                    ++j;
                }
            }
            if (!n) continue;
            const std::int64_t c = freqs[i];
            before.clear();
            after.clear();
            for (std::size_t j = 0; j + 1 < s.size(); ++j) {
                std::uint64_t k = pair_key(s[j], s[j + 1]);
                counts[k] -= c;
                touched.push_back(k);
                before.push_back(k);
            }
            for (std::size_t j = 0; j + 1 < out.size(); ++j) {
                std::uint64_t k = pair_key(out[j], out[j + 1]);
                counts[k] += c;
                touched.push_back(k);
                after.push_back(k);
            }
            std::sort(before.begin(), before.end());
            std::sort(after.begin(), after.end());
            after.erase(std::unique(after.begin(), after.end()), after.end());
            for (std::uint64_t k : after)
                if (!std::binary_search(before.begin(), before.end(), k)) where[k].push_back(i);
            s.swap(out);
            hit += n * c;
            tokens -= n * c;
        }
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        for (std::uint64_t k : touched) {
            std::int64_t v = counts[k];
            if (v > 0) heap.emplace(-v, k);
        }
        return hit;
    };
    auto intern = [&](std::uint32_t a, std::uint32_t b) {
        symbols.push_back(symbols[a] + symbols[b]);
        return static_cast<std::uint32_t>(symbols.size() - 1);
    };

    std::vector<Merge> merges;
    std::vector<Json> history;

    auto done = load_partial(o.checkpoint_path, sig);
    if (!o.init_merges.empty()) {
        if (o.init_sig && *o.init_sig != sig && !o.init_across_corpora) {
            say("BPE: the existing vocabulary was trained on a DIFFERENT corpus; starting from scratch "
                "(pass init_across_corpora=True to extend it anyway, which is what language adaptation wants)");
        } else {
            if (!o.init_sig)
                say("BPE: the existing vocabulary carries no corpus signature; assuming it belongs to this "
                    "corpus and extending it");
            else if (*o.init_sig != sig)
                say("BPE: extending a vocabulary trained on a DIFFERENT corpus, as asked. The base merges are "
                    "kept in order, so every id already learned keeps its meaning and only new symbols are "
                    "appended.");
            if (!done || o.init_merges.size() > done->first.size())
                done = std::make_pair(o.init_merges, o.init_history);
        }
    }
    if (done) {
        std::vector<Merge> replay = std::move(done->first);
        if (replay.size() > o.num_merges) replay.resize(o.num_merges);
        history = std::move(done->second);
        if (history.size() > o.num_merges) history.resize(o.num_merges);
        std::unordered_map<std::string, std::uint32_t> symbol_ids;
        for (std::uint32_t b = 0; b < 256; ++b) symbol_ids[symbols[b]] = b;
        const std::int64_t n_replay = static_cast<std::int64_t>(replay.size());
        report(2, 0, n_replay);
        for (std::size_t t = 0; t < replay.size(); ++t) {
            auto left = symbol_ids.find(replay[t].first), right = symbol_ids.find(replay[t].second);
            if (left == symbol_ids.end() || right == symbol_ids.end())
                throw std::runtime_error("a checkpointed merge names a symbol no earlier merge made");
            std::uint32_t a = left->second, b = right->second;
            std::uint32_t ab = intern(a, b);
            symbol_ids[symbols[ab]] = ab;
            apply_merge(a, b, ab);
            merges.push_back(replay[t]);
            if ((t + 1) % 256 == 0 || t + 1 == replay.size()) report(2, static_cast<std::int64_t>(t + 1), n_replay);
        }
        say("BPE: resumed at merge " + with_commas(static_cast<std::int64_t>(merges.size())) + "/" +
            with_commas(static_cast<std::int64_t>(o.num_merges)));
        if (merges.size() >= o.num_merges) {
            merges.resize(o.num_merges);
            if (history.size() > o.num_merges) history.resize(o.num_merges);
            return {std::move(merges), std::move(history), sig};
        }
    }
    if (o.on_resume && !o.on_resume(history)) throw BuildStopped{};

    const std::int64_t target = static_cast<std::int64_t>(o.num_merges);
    report(3, static_cast<std::int64_t>(merges.size()), target);
    for (std::size_t step = merges.size(); step < o.num_merges; ++step) {
        bool found = false;
        std::uint64_t best = 0;
        std::int64_t freq = 0;
        while (!heap.empty()) {
            HeapEntry top = heap.top();
            heap.pop();
            auto current = counts.find(top.second);
            std::int64_t value = current == counts.end() ? 0 : current->second;
            if (value == -top.first && -top.first > 0) {
                best = top.second;
                freq = -top.first;
                found = true;
                break;
            }
        }
        if (!found || freq < 2) break;
        const std::uint32_t a = static_cast<std::uint32_t>(best >> 32);
        const std::uint32_t b = static_cast<std::uint32_t>(best & 0xFFFFFFFFu);
        const std::int64_t distinct_pairs = static_cast<std::int64_t>(heap.size());
        const std::uint32_t ab = intern(a, b);
        if (!apply_merge(a, b, ab)) {
            symbols.pop_back();
            report(3, static_cast<std::int64_t>(step) + 1, target);
            continue;
        }
        merges.emplace_back(symbols[a], symbols[b]);
        Json record;
        record.type = Json::Object;
        auto field = [&](const char* key, Json value) {
            record.keys.push_back(key);
            record.items.push_back(std::move(value));
        };
        field("merge", json_int(static_cast<std::int64_t>(merges.size())));
        field("pair_freq", json_int(freq));
        field("distinct_pairs", json_int(distinct_pairs));
        field("vocab_size", json_int(static_cast<std::int64_t>(o.n_special + 256 + merges.size())));
        field("tokens", json_int(tokens));
        field("new_token_len", json_int(static_cast<std::int64_t>(symbols[ab].size())));
        Json ratio;
        ratio.type = Json::Float;
        ratio.real = static_cast<double>(total_bytes) / static_cast<double>(std::max<std::int64_t>(tokens, 1));
        field("bytes_per_token", ratio);
        history.push_back(record);
        if (!o.checkpoint_path.empty() && o.checkpoint_every > 0 && merges.size() % o.checkpoint_every == 0)
            save_partial(o.checkpoint_path, sig, merges, history, false);
        if (o.on_merge && !o.on_merge(record)) throw BuildStopped{};
        report(3, static_cast<std::int64_t>(step) + 1, target);
    }
    if (!o.checkpoint_path.empty()) save_partial(o.checkpoint_path, sig, merges, history, true);
    return {std::move(merges), std::move(history), sig};
}


}  // namespace mira_bpe
