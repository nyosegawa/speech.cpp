// Checks Sortformer's whole streaming diarization against NeMo's forward() on each input and configuration of
// reference/sortformer/dump.py, from the dump's audio alone: the probabilities every 10 ms, whose error each step's
// cache and queue carry on to the steps after it, and the segments of the default postprocessing, by the share of 10 ms
// frames where a speaker is given as speaking by one and not by the other. The same audio given in pieces must give the
// probabilities of the whole bit for bit, which it checks in the low_latency and offline configurations. The dumps are
// those in <reference out dir>/<the file's general.name>/.
//
// usage: sortformer-stream-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "sortformer-dumps.h"
#include "sortformer/diarizer.h"
#include "sortformer/layout.h"
#include "weight-types.h"

using namespace sortformer;
using namespace sortformer_dumps;

namespace {

std::vector<float> diarize(Diarizer & diarizer, const Streaming & streaming, const std::vector<float> & audio, size_t largest, unsigned seed) {
    ChunkStream stream(diarizer, streaming);
    std::vector<float> probs;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> size(1, largest);
    for (size_t at = 0; at < audio.size();) {
        const size_t n = std::min(size(rng), audio.size() - at);
        stream.push(audio.data() + at, n, probs);
        at += n;
    }
    stream.end(probs);
    return probs;
}

/** The 10 ms frames, of all speakers, inside a segment of one list and not the other, over the frames inside either. */
double disagreement(const std::vector<Segment> & a, const std::vector<Segment> & b, int speakers, int64_t frames) {
    const auto mask = [&](const std::vector<Segment> & segments) {
        std::vector<char> m((size_t) (frames * speakers), 0);
        for (const Segment & s : segments) {
            for (int64_t f = (int64_t) std::llround(s.start * 100); f < std::min(frames, (int64_t) std::llround(s.end * 100)); f++) m[(size_t) (f * speakers + s.speaker)] = 1;
        }
        return m;
    };
    const std::vector<char> x = mask(a), y = mask(b);
    int64_t either = 0, one = 0;
    for (size_t i = 0; i < x.size(); i++) {
        either += x[i] || y[i];
        one += x[i] != y[i];
    }
    return either ? (double) one / (double) either : 0.0;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        if (!takes_weights(args[1], layout, {"F32"}, "the bounds are set from F32's arithmetic")) return 2;
        start_ggml();
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "auto");
        const bool cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        std::printf("device: %s\n", ggml_backend_name(backend));
        bool ok = true;
        {
            Diarizer diarizer(args[1], backend);
            const int speakers = diarizer.speakers();
            for (const auto & input : reference_dumps(args[2], diarizer.model())) {
                const Npy audio = load(input, "audio");
                for (const Config & c : configs(input)) {
                    const Npy want = load(c.dir, "total_preds");
                    const std::vector<float> got = diarize(diarizer, c.streaming, audio.f32, audio.f32.size(), 1);
                    const Diff diff = compare(got, want.f32);
                    const int64_t frames = (int64_t) want.f32.size() / speakers;
                    const double apart = disagreement(segments(got, speakers, diarizer.frame_seconds(), diarizer.postprocessing()),
                                                      segments(want.f32, speakers, diarizer.frame_seconds(), diarizer.postprocessing()), speakers, frames);
                    std::printf("%s %s (%lld frames)\n", input.filename().u8string().c_str(), c.name.c_str(), (long long) frames);
                    print_diff("  probabilities", diff);
                    std::printf("  segments: %.3f%% of the frames of a speaker differ\n", 100 * apart);
                    // Measured on an Apple M5 (2026-10-26), with the cache choosing its rows from the probabilities
                    // computed here: on the CPU in F32 the probabilities lie within 3.2e-5 of the official, at 104 dB
                    // and more, and no frame of a speaker differs. On Metal each step lies within 3e-3 of the official
                    // from the same input (sortformer-network-check), and a row the cache chose otherwise there moved the
                    // low_latency meeting's probabilities by up to 0.12 and 0.41% of its frames of a speaker, the other
                    // configurations' by none. A wrong step, a cache that holds other rows or a chunk cut elsewhere moves
                    // whole segments.
                    ok = ok && got.size() == want.f32.size() && apart < (cpu ? 0.005 : 0.03);
                    // Audio in pieces, for the configuration a stream takes by default and the offline one: the
                    // others cut chunks by the same code with other numbers, and on the CPU each takes minutes.
                    if (c.name != "low_latency" && c.name != "offline") continue;
                    for (const size_t largest : {(size_t) 160, (size_t) 2399}) {
                        const bool same = diarize(diarizer, c.streaming, audio.f32, largest, 5) == got;
                        std::printf("  in pieces of up to %zu samples: %s\n", largest, same ? "the whole audio's bit for bit" : "DIFFERENT");
                        ok = ok && same;
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
