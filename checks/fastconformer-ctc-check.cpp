// Checks the FastConformer CTC head, its greedy decoding and the detokenization against NeMo on each dump of
// reference/fastconformer/dump.py: first from the dump's own encoder output, then the whole path from the
// dump's audio to the text, which it also times.
//
// usage: fastconformer-ctc-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <chrono>
#include <cstdio>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "fastconformer-dumps.h"
#include "fastconformer/recognizer.h"
#include "npy.h"

using namespace fastconformer;

namespace {

/** The share of rows of two [rows, classes] arrays whose largest element is in the same column. */
double argmax_agreement(const std::vector<float> & a, const std::vector<float> & b, int classes) {
    const size_t rows = a.size() / (size_t) classes;
    size_t same = 0;
    for (size_t r = 0; r < rows; r++) {
        int ia = 0, ib = 0;
        for (int c = 1; c < classes; c++) {
            if (a[r * classes + c] > a[r * classes + ia]) ia = c;
            if (b[r * classes + c] > b[r * classes + ib]) ib = c;
        }
        same += ia == ib;
    }
    return rows == 0 ? 0 : (double) same / (double) rows;
}

std::string ids_text(const std::vector<int32_t> & ids) {
    std::string s;
    for (int32_t id : ids) s += " " + std::to_string(id);
    return s;
}

}  // namespace

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
            Recognizer recognizer(args[1], backend);
            const int classes = recognizer.ctc().classes();
            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            for (const auto & d : fastconformer_dumps(args[2])) {
                const Npy encoded = read_npy((d / "encoded.npy").u8string());
                const Npy log_probs = read_npy((d / "ctc_log_probs.npy").u8string());
                const Npy ctc_ids = read_npy((d / "ctc_ids.npy").u8string());
                const Npy audio = read_npy((d / "audio.npy").u8string());
                const std::string want_text = fastconformer_text(d);
                const std::vector<int32_t> & want_ids = ctc_ids.i32;
                std::printf("%s (%.2f s)\n", d.filename().u8string().c_str(), (double) audio.f32.size() / recognizer.sample_rate());

                Graph g;
                ggml_tensor * out = recognizer.ctc().build(g, g.input(encoded.f32, encoded.shape[1], encoded.shape[0]));
                g.output(out);
                g.compute(backend, allocr);
                const std::vector<float> logits = Graph::read(out);
                const Diff dl = compare(log_softmax(logits, classes), log_probs.f32);
                print_diff("  CTC log-probabilities", dl);
                std::printf("  argmax agreement %.4f\n", argmax_agreement(logits, log_probs.f32, classes));
                const std::vector<int32_t> ids = recognizer.ctc().greedy(logits);
                const std::string text = recognizer.detokenizer().text(ids);
                std::printf("  greedy ids %s, text %s\n", ids == want_ids ? "equal" : "DIFFER", text == want_text ? "equal" : "DIFFERS");
                if (ids != want_ids) std::printf("    got%s\n    want%s\n", ids_text(ids).c_str(), ids_text(want_ids).c_str());
                if (text != want_text) std::printf("    got  %s\n    want %s\n", text.c_str(), want_text.c_str());
                // Measured on an Apple M5 on 2026-10-05: 129 dB on the CPU with F32 weights and 75 to 76 dB with F16,
                // 88 dB on Metal with either.
                ok = ok && dl.snr_db > 60 && ids == want_ids && text == want_text;

                // The whole path, timed after a first run that builds Metal's pipelines.
                recognizer.recognize(audio.f32);
                const auto t0 = std::chrono::steady_clock::now();
                const std::vector<float> features = recognizer.frontend().features(audio.f32);
                const auto t1 = std::chrono::steady_clock::now();
                const std::vector<float> path_logits = recognizer.logits(features, recognizer.frontend().frames(audio.f32.size()));
                const auto t2 = std::chrono::steady_clock::now();
                const std::string path_text = recognizer.detokenizer().text(recognizer.ctc().greedy(path_logits));
                std::printf("  audio to text: argmax agreement %.4f, text %s; frontend %.1f ms, encoder and CTC head %.1f ms\n",
                            argmax_agreement(path_logits, log_probs.f32, classes), path_text == want_text ? "equal" : "DIFFERS",
                            std::chrono::duration<double, std::milli>(t1 - t0).count(),
                            std::chrono::duration<double, std::milli>(t2 - t1).count());
                if (path_text != want_text) std::printf("    got  %s\n    want %s\n", path_text.c_str(), want_text.c_str());
                ok = ok && path_text == want_text;
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
