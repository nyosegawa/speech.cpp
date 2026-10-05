// Checks the FastConformer encoder against NeMo's ConformerEncoder on each dump of
// reference/fastconformer/dump.py: the subsampling, every conformer layer and the output, all from the dump's
// own features, so that the encoder's error is its own.
//
// usage: fastconformer-encoder-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <cstdio>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "fastconformer-dumps.h"
#include "fastconformer/encoder.h"
#include "npy.h"

using namespace fastconformer;

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        bool ok = true;
        {
            ModelFile model(args[1], backend);
            const Encoder encoder(model);
            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            for (const auto & d : fastconformer_dumps(args[2])) {
                const Npy features = read_npy((d / "features.npy").u8string());
                const Npy subsampled = read_npy((d / "subsampled.npy").u8string());
                const Npy layers = read_npy((d / "layers.npy").u8string());
                const Npy encoded = read_npy((d / "encoded.npy").u8string());
                const int64_t frames = features.shape[0];
                Graph g;
                EncoderStages stages;
                ggml_tensor * out = encoder.build(g, features.f32, frames, &stages);
                g.output(out);
                g.output(stages.subsampled);
                for (ggml_tensor * t : stages.layers) g.output(t);
                g.compute(backend, allocr);

                const int64_t t = encoder.subsampled_frames(frames);
                const size_t width = (size_t) encoder.d_model(), n = (size_t) t * width;
                std::printf("%s (%lld frames, %lld after subsampling)\n", d.filename().u8string().c_str(), (long long) frames,
                            (long long) t);
                ok = ok && t == subsampled.shape[0] && t == encoded.shape[0] && (size_t) layers.shape[0] == stages.layers.size();
                print_diff("  subsampling", compare(Graph::read(stages.subsampled), subsampled.f32));
                for (size_t l = 0; l < stages.layers.size(); l++) {
                    const Diff dl = compare(Graph::read(stages.layers[l]).data(), &layers.f32[l * n], n);
                    print_diff("  layer " + std::to_string(l), dl);
                }
                const Diff de = compare(Graph::read(out), encoded.f32);
                print_diff("  encoder output", de);
                // Measured on an Apple M5 on 2026-10-05: 114 to 118 dB on the CPU with F32 weights and 55 to 56 dB
                // with F16; 63 to 64 dB on Metal with either, since its matrix kernel rounds its inputs to half
                // precision. A wrong operation in any layer falls far below.
                ok = ok && de.snr_db > 40;
            }
            ggml_gallocr_free(allocr);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
