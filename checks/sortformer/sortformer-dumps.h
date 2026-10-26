#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "json-reader.h"
#include "npy.h"
#include "reference-dumps.h"
#include "sortformer/cache.h"

// The dumps of reference/sortformer/dump.py: each input's folder, with its audio and features, holds a folder for each
// streaming configuration, with the steps of its diarization and the state after each, as frame indices.

namespace sortformer_dumps {

inline Npy load(const std::filesystem::path & dir, const std::string & name) {
    return read_npy((dir / (name + ".npy")).u8string());
}

inline JsonValue json_of(const std::filesystem::path & file) {
    return parse_json(dump_text(file));
}

inline int64_t integer(const JsonValue & v, const char * name) {
    const JsonValue * m = v.member(name);
    if (!m || !m->is_integer()) throw std::runtime_error(std::string("a dump's meta.json has no integer ") + name);
    return std::stoll(m->text);
}

inline bool flag(const JsonValue & v, const char * name) {
    const JsonValue * m = v.member(name);
    if (!m || m->kind != JsonValue::Kind::Bool) throw std::runtime_error(std::string("a dump's meta.json has no boolean ") + name);
    return m->boolean;
}

/** One step of a dump: its geometry and the state's lengths before and after it. */
struct Step {
    int64_t features = 0, frames = 0, first_frame = 0, chunk_frames = 0, rc_frames = 0, encoder_frames = 0;
    int64_t cache_before = 0, fifo_before = 0, cache_after = 0, fifo_after = 0;
    bool compressed_before = false, compressed_after = false;
};

/** A configuration's dump: its folder, its settings and its steps, with the offsets of each step's rows in its arrays. */
struct Config {
    std::filesystem::path dir;
    std::string name;
    sortformer::Streaming streaming;
    std::vector<Step> steps;
    std::vector<int64_t> encoder_steps;
    int64_t layer_step = -1;
    JsonValue meta;
};

/** The configurations of the input dumped in `input`, sorted by name. */
inline std::vector<Config> configs(const std::filesystem::path & input) {
    std::vector<Config> out;
    for (const auto & e : std::filesystem::directory_iterator(input, std::filesystem::directory_options::follow_directory_symlink)) {
        if (!std::filesystem::is_regular_file(e.path() / "meta.json") || !std::filesystem::is_regular_file(e.path() / "total_preds.npy")) continue;
        Config c;
        c.dir = e.path();
        c.name = e.path().filename().u8string();
        c.meta = json_of(e.path() / "meta.json");
        const JsonValue & s = *c.meta.member("config");
        c.streaming = {(int) integer(s, "spkcache_len"), (int) integer(s, "fifo_len"), (int) integer(s, "chunk_len"),
                       (int) integer(s, "chunk_right_context"), (int) integer(s, "spkcache_update_period")};
        for (const JsonValue & j : c.meta.member("steps")->items) {
            Step st;
            st.features = integer(j, "features");
            st.frames = integer(j, "frames");
            st.first_frame = integer(j, "first_frame");
            st.chunk_frames = integer(j, "chunk_frames");
            st.rc_frames = integer(j, "rc_frames");
            st.encoder_frames = integer(j, "encoder_frames");
            st.cache_before = integer(j, "cache_before");
            st.fifo_before = integer(j, "fifo_before");
            st.cache_after = integer(j, "cache_after");
            st.fifo_after = integer(j, "fifo_after");
            st.compressed_before = flag(j, "compressed_before");
            st.compressed_after = flag(j, "compressed_after");
            c.steps.push_back(st);
        }
        for (const JsonValue & j : c.meta.member("encoder_steps")->items) c.encoder_steps.push_back(std::stoll(j.text));
        c.layer_step = integer(c.meta, "layer_step");
        out.push_back(c);
    }
    std::sort(out.begin(), out.end(), [](const Config & a, const Config & b) { return a.name < b.name; });
    if (out.empty()) throw std::runtime_error("no configuration of reference/sortformer/dump.py is under " + input.u8string());
    return out;
}

/** The arrays of a configuration's dump that hold the state, and the official frames each index names. */
struct States {
    Npy table, rc, cache_index, cache_probs, fifo_index, fifo_probs, step_probs;
    std::vector<int64_t> cache_offset, fifo_offset, step_offset, rc_offset;
    /** For each frame, the first frame whose pre-encoded row is the same bytes, which dump.py names a copy of either. */
    std::vector<int64_t> canonical;

    States(const Config & c, int speakers) {
        table = load(c.dir, "pre_encode");
        rc = load(c.dir, "rc_pre_encode");
        cache_index = load(c.dir, "cache_index");
        cache_probs = load(c.dir, "cache_preds");
        fifo_index = load(c.dir, "fifo_index");
        fifo_probs = load(c.dir, "fifo_preds");
        step_probs = load(c.dir, "step_preds");
        int64_t cache = 0, fifo = 0, step = 0, right = 0;
        for (const Step & s : c.steps) {
            cache_offset.push_back(cache);
            fifo_offset.push_back(fifo);
            step_offset.push_back(step);
            rc_offset.push_back(right);
            cache += s.cache_after;
            fifo += s.fifo_after;
            step += s.encoder_frames;
            right += s.rc_frames;
        }
        const int64_t d = table.shape[1], frames = table.shape[0];
        std::map<std::string, int64_t> first;
        for (int64_t f = 0; f < frames; f++) {
            const std::string bytes((const char *) &table.f32[(size_t) (f * d)], (size_t) d * sizeof(float));
            canonical.push_back(first.emplace(bytes, f).first->second);
        }
        (void) speakers;
    }

    int64_t d() const { return table.shape[1]; }

    /** The state after step `k` as the dump holds it, or the empty state before the first for k = -1. */
    sortformer::State after(const Config & c, int64_t k, const std::vector<float> & silence, int speakers) const {
        sortformer::State s;
        if (k < 0) return s;
        const Step & st = c.steps[(size_t) k];
        const auto rows = [&](const Npy & index, int64_t offset, int64_t n, std::vector<float> & out, std::vector<int64_t> & frames) {
            for (int64_t i = 0; i < n; i++) {
                const int64_t f = index.i32[(size_t) (offset + i)];
                frames.push_back(f);
                const float * row = f < 0 ? silence.data() : &table.f32[(size_t) (f * d())];
                out.insert(out.end(), row, row + d());
            }
        };
        rows(cache_index, cache_offset[(size_t) k], st.cache_after, s.cache, s.cache_frames);
        rows(fifo_index, fifo_offset[(size_t) k], st.fifo_after, s.fifo, s.fifo_frames);
        s.cache_probs.assign(cache_probs.f32.begin() + cache_offset[(size_t) k] * speakers,
                             cache_probs.f32.begin() + (cache_offset[(size_t) k] + st.cache_after) * speakers);
        s.fifo_probs.assign(fifo_probs.f32.begin() + fifo_offset[(size_t) k] * speakers,
                            fifo_probs.f32.begin() + (fifo_offset[(size_t) k] + st.fifo_after) * speakers);
        s.compressed = st.compressed_after;
        return s;
    }
};

}  // namespace sortformer_dumps
