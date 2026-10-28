// Checks Nemotron 3.5 Transcribe's ASR encoder against NeMo on every window of each dump of
// reference/nemotron-transcribe/dump.py, each from the window's own features as the official code normalized them: the
// stacked frames' projection, the embedding norm's and each layer's output, and the final norm's. A window holds the
// features of its valid frames and, at the end of the audio, the zero frame the preprocessor's padding gives it, which
// the official attention masks out; the check encodes the valid frames alone and compares their 80 ms frames. The dumps
// are those in <reference out dir>/<the file's general.name>/.
//
// usage: nemotron-transcribe-encoder-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <cstdio>
#include <string>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "json-reader.h"
#include "nemotron-transcribe/encoder.h"
#include "nemotron-transcribe/layout.h"
#include "npy.h"
#include "reference-dumps.h"
#include "weight-types.h"

using namespace nemotron_transcribe;

namespace {

/** The rows [0, n) of a dump of rows of `width` values, [rows, width] row-major or [k, rows, width] at `layer`. */
std::vector<float> rows_of(const Npy & a, int64_t n, int64_t width, int64_t layer = -1) {
    const int64_t rows = layer < 0 ? a.shape[0] : a.shape[1];
    const float * from = &a.f32[(size_t) ((layer < 0 ? 0 : layer * rows) * width)];
    return std::vector<float>(from, from + n * width);
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
            const ModelFile model(args[1], backend, layout, [](const std::string & name) { return name.rfind("asr.", 0) == 0; });
            AsrEncoder encoder(model, backend);
            const int64_t d = encoder.d_model(), factor = encoder.factor();
            Diff worst_layer, worst_output;
            for (const auto & dump : reference_dumps(args[2], model)) {
                const JsonValue meta = parse_json(dump_text(dump / "meta.json"));
                const JsonValue * windows = meta.member("windows");
                for (size_t k = 0; k < windows->items.size(); k++) {
                    const JsonValue & w = windows->items[k];
                    const int64_t valid = std::stoll(w.member("length")->text), n = (valid + factor - 1) / factor;
                    const std::filesystem::path dir = dump / "asr" / std::to_string(k);
                    const Npy input = read_npy((dir / "input.npy").u8string());
                    EncoderStages stages;
                    const std::vector<float> out = encoder.encode(input.f32.data(), valid, &stages);
                    std::printf("%s window %zu (%lld frames, %lld of 80 ms)\n", dump.filename().u8string().c_str(), k, (long long) valid,
                                (long long) n);
                    print_diff("  pre-encoded", compare(stages.pre_encode, rows_of(read_npy((dir / "pre_encode.npy").u8string()), n, d)));
                    const Npy layers = read_npy((dir / "layers.npy").u8string());
                    Diff layer_worst;
                    for (size_t l = 0; l < stages.layers.size(); l++) keep_worst(layer_worst, compare(stages.layers[l], rows_of(layers, n, d, (int64_t) l)));
                    print_diff("  worst layer", layer_worst);
                    const Diff o = compare(out, rows_of(read_npy((dir / "output.npy").u8string()), n, d));
                    print_diff("  output", o);
                    keep_worst(worst_layer, layer_worst);
                    keep_worst(worst_output, o);
                }
            }
            // The bounds of float32 on the CPU, and of a GPU whose matrix kernels round their inputs to half precision.
            const double bound = cpu ? 80.0 : 30.0;
            print_diff("worst layer", worst_layer);
            print_diff("worst output", worst_output);
            ok = worst_output.snr_db >= bound && worst_layer.snr_db >= bound;
            std::printf("%s: output SNR %.1f dB against a bound of %.0f dB\n", ok ? "OK" : "FAIL", worst_output.snr_db, bound);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
