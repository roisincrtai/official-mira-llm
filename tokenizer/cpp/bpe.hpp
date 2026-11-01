// tokenizer/cpp/bpe.hpp -- the byte-level BPE tokenizer of tokenizer/py/bpe.py, in C++.
//
// Standard C++17 only. Given the same merges and specials it produces the ids tokenizer/py/bpe.py
// produces: the same pre-tokenizer (\s*\d|\s*[^\s\d]+|\s+, with Python's Unicode \s and \d), the
// same merge rule (the lowest-ranked adjacent pair, leftmost on a tie), the same recognition of
// specials (longest first) and the same decoding (UTF-8 with errors="replace"). build() learns the
// same merges in the same order, with the same per-merge history and corpus signature, and reads
// and writes the same <checkpoint_path>.partial file.
//
// Text is UTF-8. A byte that does not start a well-formed UTF-8 sequence is one character that is
// neither whitespace nor a digit.
//
// The C interface at the end is what tokenizer/cpp/bpe.py loads from build/tokenizer/cpp/libbpe.so.
//
// The sources, one concern each:
//     utf8.hpp/.cpp          UTF-8, and Python's \s and \d
//     pretokenize.hpp/.cpp   the pre-tokenizer
//     tokenizer.hpp/.cpp     encoding and decoding with learned merges
//     json.hpp/.cpp          a strict, complete JSON and JSON Lines reader and writer
//     sha1.hpp               SHA-1, for the corpus signature
//     checkpoint.hpp/.cpp    the <checkpoint_path>.partial file of a build
//     train.hpp/.cpp         learning the merges
//     bpe.hpp/.cpp           this header, and the C interface libbpe.so exports
#pragma once

#include <cstdint>

#include "json.hpp"
#include "pretokenize.hpp"
#include "tokenizer.hpp"
#include "train.hpp"
#include "utf8.hpp"

extern "C" {
// A tokenizer: merges as 2 * n_merges byte strings laid end to end in merge_bytes with their
// lengths in merge_lengths (left, right, left, right, ...); specials the same way.
void* bpe_create(const char* merge_bytes, const std::int64_t* merge_lengths, std::int64_t n_merges,
                 const char* special_bytes, const std::int64_t* special_lengths,
                 std::int64_t n_specials);
void bpe_destroy(void* tokenizer);
// The number of ids, written to a buffer *ids that bpe_free releases; -1 on error.
std::int64_t bpe_encode(void* tokenizer, const char* text, std::int64_t length, int split_specials,
                        std::int32_t** ids);
// The number of bytes, written to a buffer *text that bpe_free releases; -1 on error.
std::int64_t bpe_decode(void* tokenizer, const std::int64_t* ids, std::int64_t n_ids,
                        int skip_special_tokens, char** text);
void bpe_free(void* buffer);
const char* bpe_last_error();

typedef int (*bpe_log_fn)(const char* line);
typedef int (*bpe_progress_fn)(int phase, std::int64_t done, std::int64_t total);
typedef int (*bpe_json_fn)(const char* json);

void* bpe_counter_create();
int bpe_counter_add(void* counter, const char* text, std::int64_t length);
void bpe_counter_destroy(void* counter);
// A build result, or null on error or when a callback returned 0. init_merges as in bpe_create;
// init_history a JSON array; checkpoint_path and init_sig may be null. on_resume receives the
// history replayed from a checkpoint, on_merge each new history record, both as JSON.
void* bpe_build(void* counter, std::int64_t num_merges, const char* checkpoint_path,
                std::int64_t checkpoint_every, std::int64_t min_freq,
                const char* init_merge_bytes, const std::int64_t* init_merge_lengths,
                std::int64_t n_init_merges, const char* init_history, const char* init_sig,
                int init_across_corpora, std::int64_t n_special, bpe_log_fn log,
                bpe_progress_fn progress, bpe_json_fn on_resume, bpe_json_fn on_merge);
std::int64_t bpe_result_merges(void* result, const char** bytes, const std::int64_t** lengths);
const char* bpe_result_history(void* result);
const char* bpe_result_sig(void* result);
void bpe_result_destroy(void* result);
}
