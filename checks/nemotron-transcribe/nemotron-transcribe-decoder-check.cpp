// Checks Nemotron 3.5 Transcribe's language model against Transformers' NemotronH on each dump of
// reference/nemotron-transcribe/dump.py, a layer at a time, each from the dump's own input of the layer: the prefill of
// the request's input, the audio's rows among the prompt's, and then the first decoded tokens one row at a time, each
// after the rows before it, so that the Mamba layers' states and the attention layers' keys and values come from the
// layer's own computation. It compares each layer's output over the prefill's rows and the decoded rows, the router's
// logits of each MoE layer, and the head's logits of the last layer's output of the prefill's last row, after the
// final norm. A row of a MoE layer whose last chosen expert and the next score within twice the router's error of each
// other may choose the other expert in either, so such rows are left out of the layer's comparison and counted. A layer is loaded
// alone, so that the F32 file, about 136 GB, need not fit. The dumps are those in <reference out dir>/<the file's
// general.name>/.
//
// usage: nemotron-transcribe-decoder-check <model.gguf> <reference out dir> [gpu|cpu|device name] [--layers N]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "json-reader.h"
#include "nemotron-transcribe/language-model.h"
#include "nemotron-transcribe/layout.h"
#include "npy.h"
#include "reference-dumps.h"
#include "weight-types.h"

using namespace nemotron_transcribe;

namespace {

/** Row r of [rows, width] row-major, or of plane `plane` of [planes, rows, width]. */
std::vector<float> rows_of(const Npy & a, int64_t first, int64_t n, int64_t width, int64_t plane = 0, int64_t rows = 0) {
    const float * from = &a.f32[(size_t) ((plane * rows + first) * width)];
    return std::vector<float>(from, from + n * width);
}

/**
 * The rows whose `used`-th and next largest selection scores of the official router, the sigmoid of its logits plus the
 * correction bias, lie within twice the largest difference of the port's scores from them in that row. The scores are
 * float32, as the official router computes them: a sigmoid that rounds to 1 makes two experts tie exactly, which
 * torch.topk() and ggml_top_k() may break either way.
 */
std::vector<bool> tied_rows(const std::vector<float> & mine, const std::vector<float> & official, const std::vector<float> & bias, int64_t rows,
                            int64_t experts, int used) {
    std::vector<bool> tied((size_t) rows);
    std::vector<float> row((size_t) experts);
    const auto score = [&](float logit, int64_t e) { return 1.0f / (1.0f + std::exp(-logit)) + bias[(size_t) e]; };
    for (int64_t r = 0; r < rows; r++) {
        float margin = 0;
        for (int64_t e = 0; e < experts; e++) {
            row[(size_t) e] = score(official[(size_t) (r * experts + e)], e);
            margin = std::max(margin, 2.0f * std::fabs(score(mine[(size_t) (r * experts + e)], e) - row[(size_t) e]));
        }
        std::nth_element(row.begin(), row.begin() + used, row.end(), std::greater<float>());
        const float next = row[(size_t) used];
        const float last = *std::min_element(row.begin(), row.begin() + used);
        tied[(size_t) r] = last - next <= margin;
    }
    return tied;
}

/** The rows of `a`, rows of `width` values, that `drop` does not mark. */
std::vector<float> without(const std::vector<float> & a, int64_t width, const std::vector<bool> & drop) {
    std::vector<float> out;
    for (size_t r = 0; r < drop.size(); r++) {
        if (!drop[r]) out.insert(out.end(), a.begin() + (std::ptrdiff_t) (r * width), a.begin() + (std::ptrdiff_t) ((r + 1) * width));
    }
    return out;
}

/** Whether `name` is a tensor of layer `l` of the language model. */
bool in_layer(const std::string & name, int l) {
    return name.rfind("blk." + std::to_string(l) + ".", 0) == 0;
}

}  // namespace

int main(int argc, char ** argv) {
    std::vector<std::string> args = utf8_args(argc, argv);
    int only_layers = -1;
    for (size_t i = 1; i + 1 < args.size(); i++) {
        if (args[i] == "--layers") {
            only_layers = std::stoi(args[i + 1]);
            args.erase(args.begin() + (std::ptrdiff_t) i, args.begin() + (std::ptrdiff_t) i + 2);
            break;
        }
    }
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name] [--layers N]\n", args[0].c_str());
        return 2;
    }
    try {
        if (!takes_weights(args[1], layout, {"F32"}, "the bounds are set from F32's arithmetic")) return 2;
        start_ggml();
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "auto");
        const bool cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        std::printf("device: %s\n", ggml_backend_name(backend));
        const ModelFile meta_file(args[1], layout);
        const std::vector<std::string> types = meta_file.str_array("nemotron-transcribe.llm.layer_types");
        const int64_t h = meta_file.size("nemotron-transcribe.llm.hidden_size"), experts = meta_file.size("nemotron-transcribe.llm.n_routed_experts");
        const int used = meta_file.size("nemotron-transcribe.llm.num_experts_per_tok");
        int64_t tied_count = 0, moe_rows = 0;
        const int layers = only_layers > 0 ? std::min<int>(only_layers, (int) types.size()) : (int) types.size();
        const std::vector<std::filesystem::path> dumps = reference_dumps(args[2], meta_file);
        Diff worst_prefill, worst_decode, worst_router;
        for (int l = 0; l < layers; l++) {
            const ModelFile model(args[1], backend, layout, [&](const std::string & name) {
                return in_layer(name, l) || (l == 0 && name == "token_embd.weight");
            });
            LanguageModel lm(model, backend, l, l + 1);
            Diff layer_prefill, layer_decode, layer_router;
            for (const auto & dump : dumps) {
                const std::filesystem::path d = dump / "llm";
                const Npy prefill = read_npy((d / "prefill.npy").u8string());
                const int64_t rows = prefill.shape[1];
                // A request whose first token ends its turn decodes no row, and its dump has none.
                const bool decoded = std::filesystem::exists(d / "decode.npy");
                const Npy decode = decoded ? read_npy((d / "decode.npy").u8string()) : Npy();
                const int64_t steps = decoded ? decode.shape[0] : 0;
                const Npy tokens = read_npy((d / "tokens.npy").u8string());
                const std::vector<float> input = l == 0 ? read_npy((d / "input_embeds.npy").u8string()).f32 : rows_of(prefill, 0, rows, h, l - 1, rows);
                lm.start(rows + steps);
                LanguageModelStages stages;
                const std::vector<float> out = lm.run(input, rows, &stages);
                std::vector<bool> tied((size_t) rows, false);
                if (types[(size_t) l] == "moe") {
                    const Npy router = read_npy((d / "router_logits.npy").u8string());
                    const int64_t moe_index = std::count(types.begin(), types.begin() + l, "moe");
                    const std::vector<float> official = rows_of(router, 0, rows, experts, moe_index, rows);
                    keep_worst(layer_router, compare(stages.router[0], official));
                    ggml_tensor * b = model.tensor("blk." + std::to_string(l) + ".exp_probs_b.bias");
                    std::vector<float> bias((size_t) experts);
                    ggml_backend_tensor_get(b, bias.data(), 0, ggml_nbytes(b));
                    tied = tied_rows(stages.router[0], official, bias, rows, experts, used);
                    tied_count += std::count(tied.begin(), tied.end(), true);
                    moe_rows += rows;
                }
                const std::vector<float> want = rows_of(prefill, 0, rows, h, l, rows);
                const Diff dp = compare(without(out, h, tied), without(want, h, tied));
                keep_worst(layer_prefill, dp);
                if (dp.snr_db < 80) {
                    // The rows that differ most, for a layer that departs.
                    std::vector<std::pair<double, int64_t>> worst_rows;
                    for (int64_t r = 0; r < rows; r++) {
                        double e = 0;
                        for (int64_t c = 0; c < h; c++) e = std::max(e, (double) std::fabs(out[(size_t) (r * h + c)] - want[(size_t) (r * h + c)]));
                        worst_rows.push_back({e, r});
                    }
                    std::sort(worst_rows.rbegin(), worst_rows.rend());
                    std::printf("  %s: %.1f dB, rows differing most:", dump.filename().u8string().c_str(), dp.snr_db);
                    for (size_t i = 0; i < 4 && i < worst_rows.size(); i++) std::printf(" %lld (%.2e)", (long long) worst_rows[i].second, worst_rows[i].first);
                    std::printf("\n");
                }
                for (int64_t s = 0; s < steps; s++) {
                    const std::vector<float> row = l == 0 ? lm.embed({tokens.i32[(size_t) s]}) : rows_of(decode, 0, 1, h, s * (int64_t) types.size() + l - 1, 1);
                    const std::vector<float> got = lm.run(row, 1);
                    keep_worst(layer_decode, compare(got, rows_of(decode, 0, 1, h, s * (int64_t) types.size() + l, 1)));
                }
            }
            std::printf("layer %d (%s)\n", l, types[(size_t) l].c_str());
            print_diff("  prefill", layer_prefill);
            print_diff("  decode", layer_decode);
            if (types[(size_t) l] == "moe") print_diff("  router", layer_router);
            keep_worst(worst_prefill, layer_prefill);
            keep_worst(worst_decode, layer_decode);
            if (types[(size_t) l] == "moe") keep_worst(worst_router, layer_router);
        }

        // The final norm and the head on the prefill's last row of the last layer.
        Diff worst_logits;
        if (layers == (int) types.size()) {
            const ModelFile model(args[1], backend, layout, [](const std::string & name) { return name == "output_norm.weight" || name == "output.weight"; });
            LanguageModel lm(model, backend, (int) types.size() - 1, (int) types.size());
            for (const auto & dump : dumps) {
                const std::filesystem::path d = dump / "llm";
                const Npy prefill = read_npy((d / "prefill.npy").u8string());
                const int64_t rows = prefill.shape[1];
                const Npy logits = read_npy((d / "logits.npy").u8string());
                const std::vector<float> got = lm.logits(rows_of(prefill, rows - 1, 1, h, (int64_t) types.size() - 1, rows));
                keep_worst(worst_logits, compare(got, rows_of(logits, 0, 1, logits.shape[1])));
            }
            print_diff("logits", worst_logits);
        }
        // The bounds of float32 on the CPU, and of a GPU whose matrix kernels round their inputs to half precision.
        const double bound = cpu ? 60.0 : 25.0;
        print_diff("worst prefill", worst_prefill);
        print_diff("worst decode", worst_decode);
        print_diff("worst router", worst_router);
        std::printf("%-34s %lld of %lld\n", "MoE rows whose experts tie, left out", (long long) tied_count, (long long) moe_rows);
        const bool ok = worst_prefill.snr_db >= bound && worst_decode.snr_db >= bound && worst_router.snr_db >= bound &&
                        (layers < (int) types.size() || worst_logits.snr_db >= bound);
        std::printf("%s: layers' outputs SNR %.1f dB over the prefill and %.1f dB decoding, against a bound of %.0f dB\n", ok ? "OK" : "FAIL",
                    worst_prefill.snr_db, worst_decode.snr_db, bound);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
