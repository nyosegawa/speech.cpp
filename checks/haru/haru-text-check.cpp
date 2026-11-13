// Checks Haru's language model, the dense Qwen3.5 text model Qwen35Decoder runs, against the official one on a dump of
// reference/haru/dump.py, a layer at a time from the layer's own input in the dump, so that the first layer that departs
// is the one reported and no more than one layer is loaded at once: for each layer, the prompt run at once and each
// step's row after it alone, as the official conversation runs them: the attention's or the Gated DeltaNet's output
// and the layer's, and for a Gated DeltaNet layer its recurrent state and its convolution's inputs after the prompt;
// after the last layer, the final norm.
//
// usage: haru-text-check <model.gguf> <dump dir> [gpu|cpu|device name]

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "haru/layout.h"
#include "npy.h"
#include "qwen35-decoder.h"
#include "weight-types.h"

using namespace haru;

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/** Rows [first, first + n) of a row-major array of rows of `width` values. */
std::vector<float> rows_of(const std::vector<float> & a, int64_t width, int64_t first, int64_t n) {
    return std::vector<float>(a.begin() + first * width, a.begin() + (first + n) * width);
}

/** The least SNR of a kind of stage over the layers, as keep_worst() keeps it, and where it was. */
struct Worst {
    Diff diff;
    int layer = -1;
    void keep(const Diff & d, int l) {
        if (keep_worst(diff, d)) layer = l;
    }
    void print(const std::string & what) const { print_diff(what + " (" + std::to_string(layer) + ")", diff); }
};

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <dump dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    const std::string path = args[1], dir = args[2];
    if (!takes_weights(path, model_layout, {"F32"}, "its thresholds are those measured with them")) return 2;
    ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
    std::printf("backend: %s\n", ggml_backend_name(backend));
    const Npy input = read_npy(dir + "/embeddings.npy");
    const Npy layers_out = read_npy(dir + "/lm_layers.npy");
    const Npy mixers = read_npy(dir + "/lm_mixer.npy");
    const Npy hidden = read_npy(dir + "/lm_hidden.npy");
    const Npy delta_states = read_npy(dir + "/delta_state.npy");
    const Npy conv_states = read_npy(dir + "/conv_state.npy");
    const Npy tokens = read_npy(dir + "/prompt_tokens.npy");
    const int64_t rows = tokens.shape[0], total = input.shape[0], h = input.shape[1];
    // Measured on an Apple M5 on 2026-11-14 for the ja-listen dump, the lowest SNR of a stage over the layers: 109.6 dB on
    // the CPU in F32 (layer 12's mixer), and 63.5 dB on Metal (layer 30's), whose matrix kernel rounds its inputs to half
    // precision; the bounds are Drama 3's text model's.
    const double bound_db = ggml_backend_is_cpu(backend) ? 100 : 50;
    bool ok = true;

    Worst mixer, layer, state, conv;
    int delta_index = 0;
    std::vector<float> last;
    const auto t0 = std::chrono::steady_clock::now();
    for (int l = 0;; l++) {
        const std::string prefix = "blk." + std::to_string(l) + ".";
        const ModelFile model(path, backend, model_layout, [&](const std::string & name) { return name.rfind(prefix, 0) == 0 || name == "output_norm"; });
        Qwen35Decoder text(model, "haru.text.", backend, l, l + 1);
        text.start(total);
        const std::vector<float> & x = l == 0 ? input.f32 : rows_of(layers_out.f32, total * h, l - 1, 1);
        Qwen35Stages stages, step;
        std::vector<float> out = text.run(rows_of(x, h, 0, rows), rows, &stages);
        if (!text.attention(l)) {
            // The official state is [value heads, key dim, value dim] with the value channel fastest.
            const std::vector<float> mine = text.delta_state(l);
            const int64_t heads = delta_states.shape[1], dk = delta_states.shape[2], dv = delta_states.shape[3];
            std::vector<float> theirs((size_t) (heads * dk * dv));
            const float * official = delta_states.f32.data() + delta_index * heads * dk * dv;
            for (int64_t hh = 0; hh < heads; hh++)
                for (int64_t i = 0; i < dk; i++)
                    for (int64_t j = 0; j < dv; j++) theirs[(size_t) (i + dk * (j + dv * hh))] = official[hh * dk * dv + i * dv + j];
            state.keep(compare(mine, theirs), l);
            // The official convolution state keeps the last `width` inputs of each channel, [channels, width], where
            // ggml_ssm_conv() takes the last width - 1, which are its last columns.
            const int64_t channels = conv_states.shape[1], width = conv_states.shape[2];
            std::vector<float> inputs((size_t) (channels * (width - 1)));
            const float * official_conv = conv_states.f32.data() + delta_index * channels * width;
            for (int64_t ch = 0; ch < channels; ch++)
                for (int64_t t = 1; t < width; t++) inputs[(size_t) (ch * (width - 1) + t - 1)] = official_conv[ch * width + t];
            conv.keep(compare(text.conv_state(l), inputs), l);
            delta_index++;
        }
        for (int64_t r = rows; r < total; r++) {
            const std::vector<float> y = text.run(rows_of(x, h, r, 1), 1, &step);
            out.insert(out.end(), y.begin(), y.end());
            for (auto [all, one] : {std::pair{&stages.mixer, &step.mixer}, {&stages.layers, &step.layers}}) {
                (*all)[0].insert((*all)[0].end(), (*one)[0].begin(), (*one)[0].end());
            }
        }
        mixer.keep(compare(stages.mixer[0], rows_of(mixers.f32, total * h, l, 1)), l);
        const Diff dl = compare(stages.layers[0], rows_of(layers_out.f32, total * h, l, 1));
        layer.keep(dl, l);
        std::printf("layer %2d (%s): %.1f dB, %.1f s so far\n", l, text.attention(l) ? "attention" : "Gated DeltaNet", dl.snr_db, seconds_since(t0));
        if (l + 1 == text.layers()) {
            const Diff dh = compare(out, hidden.f32);
            print_diff("final norm", dh);
            ok = ok && dh.snr_db > bound_db;
            break;
        }
    }
    mixer.print("mixers");
    layer.print("layers");
    state.print("Gated DeltaNet states");
    conv.print("convolutions' inputs");
    for (const Worst * w : {&mixer, &layer, &state, &conv}) ok = ok && w->diff.snr_db > bound_db;
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
