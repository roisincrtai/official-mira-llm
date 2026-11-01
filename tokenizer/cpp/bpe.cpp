// tokenizer/cpp/bpe.cpp -- the C interface of bpe.hpp. Standard C++17 only.
#include "bpe.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace {

thread_local std::string last_error;

std::vector<std::string> unpack(const char* bytes, const std::int64_t* lengths, std::int64_t count) {
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(count));
    std::size_t offset = 0;
    for (std::int64_t i = 0; i < count; ++i) {
        out.emplace_back(bytes + offset, static_cast<std::size_t>(lengths[i]));
        offset += static_cast<std::size_t>(lengths[i]);
    }
    return out;
}

std::vector<mira_bpe::Merge> unpack_merges(const char* bytes, const std::int64_t* lengths, std::int64_t n) {
    std::vector<std::string> symbols = unpack(bytes, lengths, 2 * n);
    std::vector<mira_bpe::Merge> merges;
    merges.reserve(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
        merges.emplace_back(std::move(symbols[2 * i]), std::move(symbols[2 * i + 1]));
    return merges;
}

struct Result {
    std::string bytes;
    std::vector<std::int64_t> lengths;
    std::string history;
    std::string sig;
};

}  // namespace

extern "C" {

void* bpe_create(const char* merge_bytes, const std::int64_t* merge_lengths, std::int64_t n_merges,
                 const char* special_bytes, const std::int64_t* special_lengths, std::int64_t n_specials) {
    try {
        return new mira_bpe::Tokenizer(unpack_merges(merge_bytes, merge_lengths, n_merges),
                                       unpack(special_bytes, special_lengths, n_specials));
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

void bpe_destroy(void* tokenizer) { delete static_cast<mira_bpe::Tokenizer*>(tokenizer); }

std::int64_t bpe_encode(void* tokenizer, const char* text, std::int64_t length, int split_specials,
                        std::int32_t** ids) {
    try {
        std::vector<std::int32_t> out = static_cast<mira_bpe::Tokenizer*>(tokenizer)->encode(
            std::string_view(text, static_cast<std::size_t>(length)), split_specials != 0);
        *ids = static_cast<std::int32_t*>(std::malloc(std::max<std::size_t>(out.size(), 1) * sizeof(std::int32_t)));
        if (!*ids) throw std::bad_alloc();
        if (!out.empty()) std::memcpy(*ids, out.data(), out.size() * sizeof(std::int32_t));
        return static_cast<std::int64_t>(out.size());
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

std::int64_t bpe_decode(void* tokenizer, const std::int64_t* ids, std::int64_t n_ids, int skip_special_tokens,
                        char** text) {
    try {
        std::string out = static_cast<mira_bpe::Tokenizer*>(tokenizer)->decode(
            std::vector<std::int64_t>(ids, ids + n_ids), skip_special_tokens != 0);
        *text = static_cast<char*>(std::malloc(out.size() + 1));
        if (!*text) throw std::bad_alloc();
        std::memcpy(*text, out.data(), out.size());
        (*text)[out.size()] = '\0';
        return static_cast<std::int64_t>(out.size());
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

void bpe_free(void* buffer) { std::free(buffer); }

const char* bpe_last_error() { return last_error.c_str(); }

void* bpe_counter_create() {
    try {
        return new mira_bpe::WordCounter();
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

int bpe_counter_add(void* counter, const char* text, std::int64_t length) {
    try {
        static_cast<mira_bpe::WordCounter*>(counter)->add(std::string_view(text, static_cast<std::size_t>(length)));
        return 1;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

void bpe_counter_destroy(void* counter) { delete static_cast<mira_bpe::WordCounter*>(counter); }

void* bpe_build(void* counter, std::int64_t num_merges, const char* checkpoint_path, std::int64_t checkpoint_every,
                std::int64_t min_freq, const char* init_merge_bytes, const std::int64_t* init_merge_lengths,
                std::int64_t n_init_merges, const char* init_history, const char* init_sig,
                int init_across_corpora, std::int64_t n_special, bpe_log_fn log, bpe_progress_fn progress,
                bpe_json_fn on_resume, bpe_json_fn on_merge) {
    try {
        mira_bpe::BuildOptions options;
        options.num_merges = static_cast<std::size_t>(std::max<std::int64_t>(num_merges, 0));
        if (checkpoint_path) options.checkpoint_path = checkpoint_path;
        options.checkpoint_every = static_cast<std::size_t>(std::max<std::int64_t>(checkpoint_every, 0));
        options.min_freq = static_cast<std::size_t>(std::max<std::int64_t>(min_freq, 0));
        if (n_init_merges > 0)
            options.init_merges = unpack_merges(init_merge_bytes, init_merge_lengths, n_init_merges);
        if (init_history) {
            mira_bpe::Json records =
                mira_bpe::parse_json(init_history, mira_bpe::ParseOptions::python());
            if (records.type == mira_bpe::Json::Array) options.init_history = std::move(records.items);
        }
        if (init_sig) options.init_sig = std::string(init_sig);
        options.init_across_corpora = init_across_corpora != 0;
        options.n_special = static_cast<std::size_t>(std::max<std::int64_t>(n_special, 0));
        if (log) options.log = [log](const std::string& line) { return log(line.c_str()) != 0; };
        if (progress)
            options.progress = [progress](int phase, std::int64_t done, std::int64_t total) {
                return progress(phase, done, total) != 0;
            };
        if (on_resume)
            options.on_resume = [on_resume](const std::vector<mira_bpe::Json>& history) {
                mira_bpe::Json records;
                records.type = mira_bpe::Json::Array;
                records.items = history;
                return on_resume(mira_bpe::dump_json(records).c_str()) != 0;
            };
        if (on_merge)
            options.on_merge = [on_merge](const mira_bpe::Json& record) {
                return on_merge(mira_bpe::dump_json(record).c_str()) != 0;
            };
        mira_bpe::BuildResult built = mira_bpe::build(*static_cast<mira_bpe::WordCounter*>(counter), options);
        auto* result = new Result();
        for (const mira_bpe::Merge& merge : built.merges) {
            result->bytes += merge.first;
            result->bytes += merge.second;
            result->lengths.push_back(static_cast<std::int64_t>(merge.first.size()));
            result->lengths.push_back(static_cast<std::int64_t>(merge.second.size()));
        }
        mira_bpe::Json records;
        records.type = mira_bpe::Json::Array;
        records.items = std::move(built.history);
        result->history = mira_bpe::dump_json(records);
        result->sig = built.sig;
        return result;
    } catch (const mira_bpe::BuildStopped&) {
        last_error = "stopped";
        return nullptr;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

std::int64_t bpe_result_merges(void* result, const char** bytes, const std::int64_t** lengths) {
    auto* r = static_cast<Result*>(result);
    *bytes = r->bytes.data();
    *lengths = r->lengths.data();
    return static_cast<std::int64_t>(r->lengths.size() / 2);
}

const char* bpe_result_history(void* result) { return static_cast<Result*>(result)->history.c_str(); }

const char* bpe_result_sig(void* result) { return static_cast<Result*>(result)->sig.c_str(); }

void bpe_result_destroy(void* result) { delete static_cast<Result*>(result); }

}  // extern "C"
