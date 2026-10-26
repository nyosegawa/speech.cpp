// Checks Sortformer's postprocessing against NeMo's predlist_to_timestamps() on the official probabilities of each
// configuration of reference/sortformer/dump.py, with each set of settings in its segments.json: the segments must be the
// official ones to the 10 ms both round to. Then it gives the same probabilities a few frames at a time to the stream of
// segments, which must give no segment before it is final and no start of a segment under way that the end changes,
// must have given every segment that ends before the time it says is settled, and, once the probabilities have ended,
// must have given the whole probabilities' segments. The dumps are those in <reference out dir>/<the file's general.name>/.
//
// usage: sortformer-segments-check <model.gguf> <reference out dir>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <tuple>

#include "args.h"
#include "backend.h"
#include "ggml-cpu.h"
#include "sortformer-dumps.h"
#include "sortformer/diarizer.h"

using namespace sortformer;
using namespace sortformer_dumps;

namespace {

Postprocessing settings_of(const JsonValue & settings, Postprocessing p) {
    for (const auto & [name, value] : settings.members) {
        const double v = std::stod(value.text);
        if (name == "onset") p.onset = v;
        else if (name == "offset") p.offset = v;
        else if (name == "pad_onset") p.pad_onset = v;
        else if (name == "pad_offset") p.pad_offset = v;
        else if (name == "min_duration_on") p.min_duration_on = v;
        else if (name == "min_duration_off") p.min_duration_off = v;
        else throw std::runtime_error("segments.json sets " + name + ", which the check does not know");
    }
    return p;
}

std::vector<std::tuple<int, double, double>> sorted(const std::vector<Segment> & segments) {
    std::vector<std::tuple<int, double, double>> out;
    for (const Segment & s : segments) out.emplace_back(s.speaker, s.start, s.end);
    std::sort(out.begin(), out.end());
    return out;
}

/**
 * What is wrong with the segments a stream gives the probabilities `probs` in pieces of up to `largest` frames: empty
 * when nothing is. `whole` are the whole probabilities' segments.
 */
std::string as_frames_come(const std::vector<float> & probs, int speakers, double frame, const Postprocessing & post,
                           const std::vector<Segment> & whole, int64_t largest, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int64_t> size(1, largest);
    SegmentStream stream(speakers, frame, post);
    const auto final_set = sorted(whole);
    const int64_t frames = (int64_t) probs.size() / speakers;
    for (int64_t at = 0; at < frames;) {
        const int64_t n = std::min(size(rng), frames - at);
        stream.push(&probs[(size_t) (at * speakers)], n);
        at += n;
        const double settled = stream.settled();
        for (const auto & s : sorted(stream.given())) {
            if (!std::binary_search(final_set.begin(), final_set.end(), s)) return "a segment given at frame " + std::to_string(at) + " is not one the end gives";
        }
        for (const Segment & w : whole) {
            const bool given = std::any_of(stream.given().begin(), stream.given().end(),
                                           [&](const Segment & g) { return g.speaker == w.speaker && g.start == w.start && g.end == w.end; });
            if (w.end < settled - 0.0051 && !given) return "a segment ending before the settled time is not given at frame " + std::to_string(at);
        }
        for (int s = 0; s < speakers; s++) {
            const std::optional<double> start = stream.speaking(s);
            if (!start) continue;
            const bool found = std::any_of(whole.begin(), whole.end(), [&](const Segment & w) { return w.speaker == s && w.start == *start && w.end >= settled - 0.0051; });
            if (!found) return "speaker " + std::to_string(s) + " is said to speak at frame " + std::to_string(at) + " in a segment the end does not give";
        }
    }
    stream.end();
    if (sorted(stream.given()) != final_set) return "the segments given differ from the whole probabilities'";
    return "";
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir>\n", args[0].c_str());
        return 2;
    }
    try {
        start_ggml();
        ggml_backend_t backend = ggml_backend_cpu_init();
        bool ok = true;
        {
            Diarizer diarizer(args[1], backend);
            const int speakers = diarizer.speakers();
            for (const auto & input : reference_dumps(args[2], diarizer.model())) {
                for (const Config & c : configs(input)) {
                    const Npy probs = load(c.dir, "total_preds");
                    const JsonValue sets = json_of(c.dir / "segments.json");
                    for (const auto & [name, set] : sets.members) {
                        const Postprocessing post = settings_of(*set.member("settings"), diarizer.postprocessing());
                        std::vector<Segment> want;
                        const JsonValue & per_speaker = *set.member("segments");
                        for (size_t s = 0; s < per_speaker.items.size(); s++) {
                            for (const JsonValue & seg : per_speaker.items[s].items) want.push_back({std::stod(seg.items[0].text), std::stod(seg.items[1].text), (int) s});
                        }
                        const std::vector<Segment> got = segments(probs.f32, speakers, diarizer.frame_seconds(), post);
                        int64_t differ = 0;
                        const auto a = sorted(got), b = sorted(want);
                        for (size_t i = 0; i < std::max(a.size(), b.size()); i++) {
                            if (i >= a.size() || i >= b.size() || std::get<0>(a[i]) != std::get<0>(b[i]) || std::fabs(std::get<1>(a[i]) - std::get<1>(b[i])) > 1e-9 ||
                                std::fabs(std::get<2>(a[i]) - std::get<2>(b[i])) > 1e-9) {
                                differ++;
                            }
                        }
                        std::string stream;
                        for (const int64_t largest : {(int64_t) 1, (int64_t) 37, (int64_t) 720}) {
                            if (stream.empty()) stream = as_frames_come(probs.f32, speakers, diarizer.frame_seconds(), post, got, largest, 3);
                        }
                        std::printf("%s %s %s: %zu segments, %lld differ from the official%s%s\n", input.filename().u8string().c_str(), c.name.c_str(), name.c_str(),
                                    want.size(), (long long) differ, stream.empty() ? "" : "; ", stream.c_str());
                        ok = ok && differ == 0 && stream.empty();
                    }
                }
            }
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
