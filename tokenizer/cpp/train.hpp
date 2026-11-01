// tokenizer/cpp/train.hpp -- learning the merges of tokenizer/py/bpe.py's build().
//
// Standard C++17 only. The same merges in the same order, with the same per-merge history and
// corpus signature, resumed from and checkpointed to the same <checkpoint_path>.partial file.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "tokenizer.hpp"

namespace mira_bpe {

struct BuildOptions {
    std::size_t num_merges = 10000;
    std::string checkpoint_path;                 // empty: no <path>.partial
    std::size_t checkpoint_every = 250;
    std::size_t min_freq = 1;
    std::vector<Merge> init_merges;              // empty: none
    std::vector<Json> init_history;
    std::optional<std::string> init_sig;
    bool init_across_corpora = false;
    std::size_t n_special = 3;                   // counted into each record's vocab_size
    // Each callback returns false to stop the build, which then throws BuildStopped.
    std::function<bool(const std::string& line)> log;
    // phase 1 indexing pairs, 2 replaying a checkpoint, 3 learning merges
    std::function<bool(int phase, std::int64_t done, std::int64_t total)> progress;
    std::function<bool(const std::vector<Json>& history)> on_resume;
    std::function<bool(const Json& record)> on_merge;
};

struct BuildStopped {};

struct BuildResult {
    std::vector<Merge> merges;
    std::vector<Json> history;
    std::string sig;
};

// The pre-token chunks of the training texts and how often each occurs.
class WordCounter {
public:
    void add(std::string_view text);
    const std::unordered_map<std::string, std::int64_t>& words() const { return words_; }
private:
    std::unordered_map<std::string, std::int64_t> words_;
};

BuildResult build(const WordCounter& counter, const BuildOptions& options);

}  // namespace mira_bpe
