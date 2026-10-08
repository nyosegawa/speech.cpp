// Checks the Qwen3-ASR frontend against transformers' feature extractor: the log-mel features of each dump of
// reference/qwen3-asr/dump.py, computed from the dump's own audio. The dumps are those of the model, in
// <reference out dir>/<its general.name>/.
//
// usage: qwen3-asr-frontend-check <model.gguf> <reference out dir>

#include <cstdio>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "npy.h"
#include "qwen3-asr/frontend.h"
#include "qwen3-asr/layout.h"
#include "reference-dumps.h"

using namespace qwen3_asr;

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
            const ModelFile model(args[1], backend, layout, [](const std::string & name) { return name.rfind("frontend.", 0) == 0; });
            const Frontend frontend(model);
            for (const auto & d : reference_dumps(args[2], model)) {
                const Npy audio = read_npy((d / "audio.npy").u8string());
                const Npy want = read_npy((d / "features.npy").u8string());
                const std::vector<float> got = frontend.features(audio.f32);
                const Diff diff = compare(got, want.f32);
                std::printf("%s (%lld frames)\n", d.filename().u8string().c_str(), (long long) want.shape[0]);
                print_diff("  features", diff);
                // Measured on an Apple M5 on 2026-10-06: 123.5 to 134.4 dB on the nine inputs of both models, the gap
                // between this double precision and the float32 of torch.stft() and the mel product. Reflecting the
                // audio's start one sample off gives 43.9 dB, and a floor 0.001 below the utterance's maximum less 8
                // gives 70.4 dB.
                ok = ok && got.size() == want.f32.size() && want.shape[1] == frontend.mels() && diff.snr_db > 100;
            }
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
