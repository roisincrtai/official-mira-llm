// tokenizer/cpp/checkpoint.cpp -- see checkpoint.hpp. Standard C++17 only.
#include "checkpoint.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mira_bpe {
namespace {

Json json_bytes(const std::string& bytes) {
    Json v;
    v.type = Json::Array;
    for (unsigned char b : bytes) v.items.push_back(json_int(b));
    return v;
}

std::string bytes_of(const Json& list) {
    if (list.type != Json::Array) throw std::runtime_error("a merge symbol is not a list of bytes");
    std::string out;
    for (const Json& b : list.items) {
        if (b.type != Json::Int || b.integer < 0 || b.integer > 255)
            throw std::runtime_error("a merge symbol holds a value that is not a byte");
        out.push_back(static_cast<char>(b.integer));
    }
    return out;
}

std::string partial_path(const std::string& path) { return path + ".partial"; }

}  // namespace

void save_partial(const std::string& path, const std::string& sig, const std::vector<Merge>& merges,
                  const std::vector<Json>& history, bool done) {
    std::string target = partial_path(path);
    std::filesystem::path parent = std::filesystem::path(target).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    Json document;
    document.type = Json::Object;
    Json v;
    v.type = Json::String;
    v.text = sig;
    document.keys.push_back("sig");
    document.items.push_back(v);
    Json flag;
    flag.type = Json::Bool;
    flag.boolean = done;
    document.keys.push_back("done");
    document.items.push_back(flag);
    document.keys.push_back("n_merges");
    document.items.push_back(json_int(static_cast<std::int64_t>(merges.size())));
    Json list;
    list.type = Json::Array;
    for (const Merge& merge : merges) {
        Json pair;
        pair.type = Json::Array;
        pair.items.push_back(json_bytes(merge.first));
        pair.items.push_back(json_bytes(merge.second));
        list.items.push_back(std::move(pair));
    }
    document.keys.push_back("merges");
    document.items.push_back(std::move(list));
    Json records;
    records.type = Json::Array;
    records.items = history;
    document.keys.push_back("build_history");
    document.items.push_back(std::move(records));
    std::string temporary = target + ".tmp";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) throw std::runtime_error("cannot write " + temporary);
        file << dump_json(document);
        if (!file) throw std::runtime_error("cannot write " + temporary);
    }
    std::filesystem::rename(temporary, target);
}

// The merges and history of <path>.partial when it belongs to the corpus `sig`.
std::optional<std::pair<std::vector<Merge>, std::vector<Json>>> load_partial(const std::string& path,
                                                                             const std::string& sig) {
    if (path.empty()) return std::nullopt;
    std::string target = partial_path(path);
    std::error_code error;
    if (!std::filesystem::is_regular_file(target, error)) return std::nullopt;
    try {
        std::ifstream file(target, std::ios::binary);
        std::stringstream content;
        content << file.rdbuf();
        Json document = parse_json(content.str(), ParseOptions::python());
        const Json* stored = document.get("sig");
        if (!stored || stored->type != Json::String || stored->text != sig) return std::nullopt;
        const Json* merges = document.get("merges");
        if (!merges || merges->type != Json::Array || merges->items.empty()) return std::nullopt;
        std::vector<Merge> out;
        for (const Json& pair : merges->items) {
            if (pair.type != Json::Array || pair.items.size() != 2) return std::nullopt;
            out.emplace_back(bytes_of(pair.items[0]), bytes_of(pair.items[1]));
        }
        std::vector<Json> history;
        const Json* records = document.get("build_history");
        if (records && records->type == Json::Array) history = records->items;
        return std::make_pair(std::move(out), std::move(history));
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

}  // namespace mira_bpe
