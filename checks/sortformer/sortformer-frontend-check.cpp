// Checks Sortformer's frontend against NeMo's preprocessor: the log-mel features of each input of
// reference/sortformer/dump.py, computed from the dump's own audio, and the same features computed a frame at a time as
// pieces of the audio arrive, which must be the whole audio's bit for bit. The dumps are those of the model, in
// <reference out dir>/<its general.name>/.
//
// usage: sortformer-frontend-check <model.gguf> <reference out dir>

#include <cstdio>
#include <random>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "nemo-frontend.h"
#include "npy.h"
#include "reference-dumps.h"
#include "sortformer/layout.h"

namespace {

/**
 * The features of `audio` as the frontend computes them for audio that arrives in pieces of up to `largest` samples, a
 * frame once its window's samples are there and the rest at the end.
 */
std::vector<float> in_pieces(const NemoFrontend & frontend, const std::vector<float> & audio, size_t largest, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> size(1, largest);
    const int mels = frontend.mels();
    std::vector<float> x;
    std::vector<double> mel;
    std::vector<float> out;
    int64_t computed = 0, received = 0;
    for (size_t at = 0; at < audio.size();) {
        const size_t n = std::min(size(rng), audio.size() - at);
        x.resize(x.size() + n);
        frontend.preemphasize(audio.data() + at, n, at == 0, at == 0 ? 0.0f : audio[at - 1], x.data() + at);
        at += n;
        received += (int64_t) n;
        const int64_t ready = received >= frontend.window_end(0) ? (received - frontend.window_end(0)) / frontend.hop_length() + 1 : 0;
        if (ready > computed) {
            mel.resize((size_t) ((ready - computed) * mels));
            frontend.log_mel(x.data(), 0, received, computed, ready - computed, mel.data());
            out.insert(out.end(), mel.begin(), mel.end());
            computed = ready;
        }
    }
    const int64_t frames = frontend.frames(audio.size());
    mel.resize((size_t) ((frames - computed) * mels));
    frontend.log_mel(x.data(), 0, received, computed, frames - computed, mel.data());
    out.insert(out.end(), mel.begin(), mel.end());
    return out;
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
        // The frontend runs on the host; the backend only holds its window and filterbank.
        ggml_backend_t backend = ggml_backend_cpu_init();
        bool ok = true;
        {
            const ModelFile model(args[1], backend, sortformer::layout);
            const NemoFrontend frontend(model, "sortformer.", false);
            for (const auto & d : reference_dumps(args[2], model)) {
                const Npy audio = read_npy((d / "audio.npy").u8string());
                const Npy want = read_npy((d / "features.npy").u8string());
                const std::vector<float> got = frontend.features(audio.f32);
                const Diff diff = compare(got, want.f32);
                std::printf("%s (%lld frames)\n", d.filename().u8string().c_str(), (long long) want.shape[0]);
                print_diff("  features", diff);
                // The gap between this double precision and torch.stft()'s float32: 131 to 155 dB on these inputs on an
                // Apple M5 (2026-10-26), as FastConformer's frontend, the same code, gives 94 to 127 dB. A wrong window,
                // padding, filterbank or pre-emphasis falls far below.
                ok = ok && got.size() == want.f32.size() && want.shape[1] == frontend.mels() && diff.snr_db > 90;
                for (const size_t largest : {(size_t) 1, (size_t) 333, (size_t) 4800}) {
                    const std::vector<float> pieces = in_pieces(frontend, audio.f32, largest, 7);
                    const bool same = pieces == got;
                    std::printf("  in pieces of up to %zu samples: %s\n", largest, same ? "the whole audio's bit for bit" : "DIFFERENT");
                    ok = ok && same;
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
