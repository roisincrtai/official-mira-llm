// tokenizer/cpp/checkpoint.hpp -- the <checkpoint_path>.partial file of a build, as
// tokenizer/py/bpe.py reads and writes it: {"sig", "done", "n_merges", "merges", "build_history"},
// each merge a pair of byte lists.
//
// Standard C++17 only.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "json.hpp"
#include "tokenizer.hpp"

namespace mira_bpe {

// <path>.partial, written under a temporary name and renamed.
void save_partial(const std::string& path, const std::string& sig, const std::vector<Merge>& merges,
                  const std::vector<Json>& history, bool done);

// The merges and history of <path>.partial when it belongs to the corpus `sig`.
std::optional<std::pair<std::vector<Merge>, std::vector<Json>>> load_partial(const std::string& path,
                                                                             const std::string& sig);

}  // namespace mira_bpe
