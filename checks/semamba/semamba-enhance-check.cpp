// Checks SEMamba's enhancement of whole inputs against RE-USE's official inference_chunk.py on the dumps of
// reference/semamba/dump.py: the joining of the official chunks' outputs under their windows, from the dump's own chunks,
// each of which must be asked for where the official took it; each input's audio, at its rate and with its chunk
// settings, through the chunks, the network and the windowed sum, against the official output; and the same audio given
// a piece at a time, in pieces of random sizes, whose samples must be the whole audio's bit for bit. The dumps are those
// in <reference out dir>/<the file's general.name>/.
//
// usage: semamba-enhance-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "semamba-dumps.h"
#include "semamba/enhancer.h"
#include "semamba/layout.h"
#include "weight-types.h"

using namespace semamba_dumps;

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        if (!takes_weights(args[1], semamba::layout, {"F32"}, "the bounds are set from F32's arithmetic")) return 2;
        start_ggml();
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "auto");
        std::printf("device: %s\n", ggml_backend_name(backend));
        bool ok = true;
        {
            semamba::Enhancer enhancer(args[1], backend);
            const float window_floor = enhancer.model().f32("semamba.window_floor");
            for (const auto & dump : dumps(args[2], enhancer.model())) {
                const Meta meta = read_meta(dump);
                const Npy audio = read(dump, "audio"), want = read(dump, "output");

                // The official chunks, given in the order the stream asks for them, each checked to be the official's samples.
                size_t asked = 0;
                bool where = true;
                std::vector<float> joined;
                semamba::Stream official(
                    [&](const float * s, size_t n) {
                        const Npy input = read(chunk(dump, asked), "input");
                        where = where && asked < meta.starts.size() && input.f32 == std::vector<float>(s, s + n);
                        return read(chunk(dump, asked++), "output").f32;
                    },
                    window_floor, meta.rate, meta.chunk_size_in_seconds, meta.hop_length_portion, [&](const float * s, size_t n) {
                        joined.insert(joined.end(), s, s + n);
                        return true;
                    });
                official.push(audio.f32.data(), audio.f32.size());
                official.end();
                std::vector<float> got;
                size_t pieces = 0;
                double last = 0;
                const bool done = enhancer.enhance(
                    audio.f32, meta.rate, meta.chunk_size_in_seconds, meta.hop_length_portion,
                    [&](const float * s, size_t n) {
                        got.insert(got.end(), s, s + n);
                        pieces++;
                        return true;
                    },
                    [&](double d) {
                        last = d;
                        return true;
                    });
                const Diff d = compare(got, want.f32);
                std::printf("%s (%d Hz, %.2f s, %zu chunks)\n", dump.filename().u8string().c_str(), meta.rate, (double) meta.samples / meta.rate,
                            meta.starts.size());
                const Diff j = compare(joined, want.f32);
                print_diff("  the official chunks joined", j);
                std::printf("  chunks asked for: %zu of %zu, %s\n", asked, meta.starts.size(), where ? "each the official's samples" : "NOT THE OFFICIAL'S");
                // The window computed in double against torch's float: 140 dB and more on these inputs (2026-11-01).
                ok = ok && where && asked == meta.starts.size() && joined.size() == want.f32.size() && j.snr_db > 120;
                print_diff("  enhanced audio", d);
                std::printf("  passed in %zu pieces, %zu samples of %lld; progress ended at %.3f\n", pieces, got.size(), (long long) want.f32.size(), last);
                // The phase of a bin is atan2 of the phase decoder's two outputs, which are both near zero in some bins of every
                // input, where the network's rounding (78 to 116 dB from the official's) turns the angle: the audio of
                // these inputs lies 15.8 to 68 dB from the official's on an Apple M5 (2026-11-01), the 400 samples of
                // `tiny` the lowest. The stages above check the arithmetic; this bound catches a chunk out of place.
                ok = ok && done && got.size() == want.f32.size() && last == 1.0 && d.snr_db > 10;

                std::mt19937 rng(11);
                std::uniform_int_distribution<size_t> size(1, (size_t) meta.rate / 3);
                std::vector<float> streamed;
                semamba::Stream stream(enhancer, meta.rate, meta.chunk_size_in_seconds, meta.hop_length_portion, [&](const float * s, size_t n) {
                    streamed.insert(streamed.end(), s, s + n);
                    return true;
                });
                for (size_t at = 0; at < audio.f32.size();) {
                    const size_t n = std::min(size(rng), audio.f32.size() - at);
                    stream.push(audio.f32.data() + at, n);
                    at += n;
                }
                stream.end();
                const bool same = streamed == got;
                std::printf("  in pieces of up to %d samples: %s\n", meta.rate / 3, same ? "the whole audio's bit for bit" : "DIFFERENT");
                ok = ok && same;
            }
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
