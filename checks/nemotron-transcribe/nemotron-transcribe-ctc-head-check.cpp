// Checks Nemotron 3.5 Transcribe's CTC timestamp head against NeMo on each dump of reference/nemotron-transcribe/align.py,
// from the dump's own input, the ASR encoder's kept frames rounded to bfloat16: each window's upsampled frames, the
// embedding norm's and each transformer layer's output, the final norm's and the trunk's, and the log probabilities of
// every class at every CTC frame, with the agreement of their largest. The official head runs on one frame more than the
// audio's where the preprocessor's extra feature frame starts one, which its attention masks out; the check runs the
// audio's frames alone and compares theirs. The dumps are the ctc/ folders under <reference out dir>/<the file's
// general.name>/.
//
// usage: nemotron-transcribe-ctc-head-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "json-reader.h"
#include "nemotron-transcribe/ctc-head.h"
#include "nemotron-transcribe/layout.h"
#include "npy.h"
#include "reference-dumps.h"
#include "weight-types.h"

using namespace nemotron_transcribe;

namespace {

/** The ctc/ folders of align.py's dumps under <root>/<name>/, sorted. */
std::vector<std::filesystem::path> ctc_dumps(const std::string & root, const ModelFile & model) {
    const std::filesystem::path dir = std::filesystem::u8path(root) / std::filesystem::u8path(model.str("general.name"));
    std::vector<std::filesystem::path> dumps;
    if (std::filesystem::is_directory(dir)) {
        for (const auto & e : std::filesystem::directory_iterator(dir)) {
            if (std::filesystem::is_regular_file(e.path() / "ctc" / "log_probs.npy")) dumps.push_back(e.path() / "ctc");
        }
    }
    std::sort(dumps.begin(), dumps.end());
    if (dumps.empty()) throw std::runtime_error("no dump of reference/nemotron-transcribe/align.py is under " + dir.u8string());
    return dumps;
}

/** The first `rows` rows of `width` values of `a`, or of its slice `layer` of [k, rows', width]. */
std::vector<float> rows_of(const Npy & a, int64_t rows, int64_t width, int64_t layer = -1) {
    const int64_t all = layer < 0 ? a.shape[0] : a.shape[1];
    const float * from = &a.f32[(size_t) ((layer < 0 ? 0 : layer * all) * width)];
    return std::vector<float>(from, from + rows * width);
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
            const ModelFile model(args[1], backend, layout, [](const std::string & name) { return name.rfind("ctc.", 0) == 0; });
            CtcHead head(model, backend);
            const int64_t d = model.size("nemotron-transcribe.asr.d_model"), up = head.upsampling(), classes = head.classes();
            std::vector<int32_t> all((size_t) classes);
            for (int32_t c = 0; c < (int32_t) classes; c++) all[(size_t) c] = c;
            Diff worst_stage, worst;
            double worst_agreement = 1;
            for (const auto & dump : ctc_dumps(args[2], model)) {
                const JsonValue meta = parse_json(dump_text(dump / "meta.json"));
                const int64_t valid = std::stoll(meta.member("valid_frames")->text);
                const Npy input = read_npy((dump / "input.npy").u8string());
                std::vector<uint16_t> states((size_t) (valid * d));
                for (size_t i = 0; i < states.size(); i++) {
                    uint32_t bits;
                    std::memcpy(&bits, &input.f32[i], sizeof bits);
                    states[i] = (uint16_t) (bits >> 16);
                }
                std::vector<CtcHeadStages> stages;
                const std::vector<float> out = head.log_probs(states, valid, all, &stages);
                const std::string name = dump.parent_path().filename().u8string();
                const JsonValue * windows = meta.member("windows");
                for (size_t k = 0; k < stages.size(); k++) {
                    const std::filesystem::path w = dump / "windows" / std::to_string(k);
                    const int64_t cs = std::stoll(windows->items[k].member("context_start")->text);
                    const int64_t rows = (std::min<int64_t>(std::stoll(windows->items[k].member("context_end")->text), valid) - cs) * up;
                    const CtcHeadStages & s = stages[k];
                    Diff window;
                    keep_worst(window, compare(s.upsampled, rows_of(read_npy((w / "upsampled.npy").u8string()), rows, d)));
                    const Npy layers = read_npy((w / "layers.npy").u8string());
                    for (size_t l = 0; l < s.layers.size(); l++) keep_worst(window, compare(s.layers[l], rows_of(layers, rows, d, (int64_t) l)));
                    keep_worst(window, compare(s.output, rows_of(read_npy((w / "output.npy").u8string()), rows, d)));
                    keep_worst(window, compare(s.trunk, rows_of(read_npy((w / "trunk.npy").u8string()), rows, d)));
                    print_diff(name + " window " + std::to_string(k) + " worst stage", window);
                    keep_worst(worst_stage, window);
                }
                const int64_t n = valid * up;
                const Npy reference = read_npy((dump / "log_probs.npy").u8string());
                const std::vector<float> expected = rows_of(reference, n, classes);
                const Diff diff = compare(out, expected);
                int64_t agree = 0;
                for (int64_t t = 0; t < n; t++) {
                    const float * a = &out[(size_t) (t * classes)], * b = &expected[(size_t) (t * classes)];
                    agree += std::max_element(a, a + classes) - a == std::max_element(b, b + classes) - b;
                }
                const double agreement = (double) agree / (double) n;
                print_diff(name + " log probabilities", diff);
                std::printf("  largest class agrees on %lld of %lld frames\n", (long long) agree, (long long) n);
                keep_worst(worst, diff);
                worst_agreement = std::min(worst_agreement, agreement);
            }
            // The bounds of float32 on the CPU, and of a GPU whose matrix kernels round their inputs to half precision: on
            // the dumps the CPU gave 123.5 dB and its worst stage 103.9 dB, an Apple M5's Metal 76.0 and 61.0 dB (2026-10-29).
            const double bound = cpu ? 90.0 : 45.0;
            print_diff("worst stage", worst_stage);
            print_diff("worst log probabilities", worst);
            ok = worst.snr_db >= bound && worst_stage.snr_db >= bound;
            std::printf("%s: log probability SNR %.1f dB against a bound of %.0f dB, largest class agreeing on %.4f of the frames\n",
                        ok ? "OK" : "FAIL", worst.snr_db, bound, worst_agreement);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
