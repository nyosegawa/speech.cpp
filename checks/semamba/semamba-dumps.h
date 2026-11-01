#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "json-reader.h"
#include "model-file.h"
#include "npy.h"
#include "reference-dumps.h"

namespace semamba_dumps {

/** What meta.json of a dump of reference/semamba/dump.py says: the rate, the STFT's sizes and the chunks. */
struct Meta {
    int rate = 0, n_fft = 0, hop_size = 0;
    double chunk_size_in_seconds = 0, hop_length_portion = 0;
    int64_t chunk_size = 0, hop_length = 0, samples = 0;
    std::vector<int64_t> starts, zeroed_frames;
};

inline Meta read_meta(const std::filesystem::path & dump) {
    const JsonValue j = parse_json(dump_text(dump / "meta.json"));
    const auto number = [&](const char * name) {
        const JsonValue * v = j.member(name);
        if (!v || v->kind != JsonValue::Kind::Number) throw std::runtime_error((dump / "meta.json").u8string() + " has no number " + name);
        return std::stod(v->text);
    };
    const auto integers = [&](const char * name) {
        const JsonValue * v = j.member(name);
        if (!v || v->kind != JsonValue::Kind::Array) throw std::runtime_error((dump / "meta.json").u8string() + " has no array " + name);
        std::vector<int64_t> out;
        for (const JsonValue & e : v->items) out.push_back(std::stoll(e.text));
        return out;
    };
    Meta m;
    m.rate = (int) number("rate");
    m.n_fft = (int) number("n_fft");
    m.hop_size = (int) number("hop_size");
    m.chunk_size_in_seconds = number("chunk_size_in_seconds");
    m.hop_length_portion = number("hop_length_portion");
    m.chunk_size = (int64_t) number("chunk_size");
    m.hop_length = (int64_t) number("hop_length");
    m.samples = (int64_t) number("samples");
    m.starts = integers("starts");
    m.zeroed_frames = integers("zeroed_frames");
    return m;
}

/** The dumps of the model of `model` under `root`: the folders under <root>/<general.name>/ that hold meta.json, sorted. */
inline std::vector<std::filesystem::path> dumps(const std::string & root, const ModelFile & model) {
    const std::filesystem::path dir = std::filesystem::u8path(root) / std::filesystem::u8path(model.str("general.name"));
    std::vector<std::filesystem::path> out;
    if (std::filesystem::is_directory(dir)) {
        for (const auto & e : std::filesystem::directory_iterator(dir)) {
            if (std::filesystem::is_regular_file(e.path() / "meta.json")) out.push_back(e.path());
        }
    }
    std::sort(out.begin(), out.end());
    if (out.empty()) throw std::runtime_error("no dump of reference/semamba/dump.py is under " + dir.u8string());
    return out;
}

/** The folder of chunk `k` of a dump. */
inline std::filesystem::path chunk(const std::filesystem::path & dump, size_t k) {
    return dump / ("chunk" + std::to_string(k));
}

inline Npy read(const std::filesystem::path & folder, const char * name) {
    return read_npy((folder / (std::string(name) + ".npy")).u8string());
}

}  // namespace semamba_dumps
