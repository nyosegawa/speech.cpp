// Checks Nemotron 3.5 Transcribe's recognition against the official generation on each dump of
// reference/nemotron-transcribe/dump.py whose prompt a request makes, the model card's verbatim instruction or its
// contextual one with the request's terms: the prompt's ids; then the dump's audio with every stage ours, each step's
// logits at the dump's 16 largest against the dump's, and the ids and the text it writes. Where a greedy choice of ours
// differs from the dump's, the check takes it for the arithmetic's when the dump's margin between the two tokens is
// within the sum of our errors on their two logits, and for a defect otherwise, and compares no later step. The dumps
// of other prompts, the plain instruction's and the speaker tags', are the decoder check's alone. The F32 file takes
// about 136 GB of memory. The dumps are those in <reference out dir>/<the file's general.name>/.
//
// usage: nemotron-transcribe-recognizer-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "json-reader.h"
#include "nemotron-transcribe/layout.h"
#include "nemotron-transcribe/recognizer.h"
#include "npy.h"
#include "reference-dumps.h"
#include "weight-types.h"

using namespace nemotron_transcribe;

namespace {

/** The context of the request whose prompt is `text`, or none where no request's prompt is it. */
std::optional<std::string> context_of(const std::string & text, const ModelFile & m) {
    const std::string q = "nemotron-transcribe.prompt.";
    if (text == m.str(q + "instruction")) return "";
    const std::string before = m.str(q + "context_before"), after = m.str(q + "context_after");
    if (text.size() > before.size() + after.size() && text.compare(0, before.size(), before) == 0 &&
        text.compare(text.size() - after.size(), after.size(), after) == 0) {
        return text.substr(before.size(), text.size() - before.size() - after.size());
    }
    return std::nullopt;
}

/** How our steps compare with the dump's until the first whose choice differs. */
struct Steps {
    size_t steps = 0, differ = 0, unexplained = 0;
    double worst_snr = INFINITY, worst_abs = 0;
    bool following = true;

    /** Compares our logits at step `i` with the dump's 16 largest and its choice `want`. */
    void add(size_t i, const std::vector<float> & logits, const Npy & top_ids, const Npy & top, int32_t want) {
        if (!following || i >= (size_t) top_ids.shape[0]) return;
        const size_t k = (size_t) top_ids.shape[1];
        std::vector<float> ours(k), theirs(top.f32.begin() + (std::ptrdiff_t) (i * k), top.f32.begin() + (std::ptrdiff_t) ((i + 1) * k));
        for (size_t j = 0; j < k; j++) ours[j] = logits[(size_t) top_ids.i32[i * k + j]];
        const Diff d = compare(ours, theirs);
        worst_snr = std::min(worst_snr, d.snr_db);
        worst_abs = std::max(worst_abs, d.max_abs);
        steps++;
        // The first of equal logits, as torch.argmax() takes it.
        const int32_t got = (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (got == want) return;
        differ++;
        following = false;
        size_t at_want = k, at_got = k;
        for (size_t j = 0; j < k; j++) {
            if (top_ids.i32[i * k + j] == want) at_want = j;
            if (top_ids.i32[i * k + j] == got) at_got = j;
        }
        const bool explained = at_want < k && at_got < k &&
                               theirs[at_want] - theirs[at_got] <= std::fabs(ours[at_want] - theirs[at_want]) + std::fabs(ours[at_got] - theirs[at_got]);
        if (!explained) unexplained++;
        std::printf("    step %zu: ours %d where the dump's is %d, margin %.4f, %s\n", i, got, want,
                    at_got < k ? theirs[at_want] - theirs[at_got] : NAN, explained ? "within our error on the two logits" : "UNEXPLAINED");
    }
};

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
        double worst_snr = INFINITY;
        size_t checked = 0;
        {
            Recognizer recognizer(args[1], backend);
            const ModelFile & m = recognizer.model();
            const Guardrail guardrail(m);
            const int32_t audio_id = (int32_t) m.size("nemotron-transcribe.prompt.audio_id");
            for (const auto & dump : reference_dumps(args[2], m)) {
                const JsonValue meta = parse_json(dump_text(dump / "meta.json"));
                const std::optional<std::string> context = context_of(meta.member("prompt")->member("text")->text, m);
                std::printf("%s\n", dump.filename().u8string().c_str());
                if (!context) {
                    std::printf("  no request's prompt; the decoder check's alone\n");
                    continue;
                }
                checked++;

                const Npy prompt_ids = read_npy((dump / "prompt_ids.npy").u8string());
                const auto [before, after] = recognizer.prompt(*context);
                std::vector<int32_t> ids = before;
                ids.push_back(audio_id);
                ids.insert(ids.end(), after.begin(), after.end());
                const bool same_prompt = ids == prompt_ids.i32;
                std::printf("  %-32s %s\n", "prompt ids", same_prompt ? "the dump's" : "DIFFER");
                ok = ok && same_prompt;

                const Npy audio = read_npy((dump / "audio.npy").u8string());
                const Npy tokens = read_npy((dump / "llm" / "tokens.npy").u8string());
                const Npy top_ids = read_npy((dump / "llm" / "top_ids.npy").u8string());
                const Npy top = read_npy((dump / "llm" / "top.npy").u8string());
                Steps steps;
                RecognitionReport report;
                report.logits = [&](const std::vector<float> & logits) {
                    const size_t i = report.ids.size();
                    if (i < tokens.i32.size()) steps.add(i, logits, top_ids, top, tokens.i32[i]);
                };
                const Recognition got = *recognizer.recognize(audio.f32, RecognitionRequest{*context}, {}, &report);
                const bool same_ids = report.ids == tokens.i32;
                const std::string want = guardrail.apply(meta.member("text")->text);
                std::printf("  %-32s %zu of %zu steps ours, top logits' worst SNR %.1f dB, max |diff| %.2e\n", "greedy decoding",
                            steps.steps - steps.differ, tokens.i32.size(), steps.worst_snr, steps.worst_abs);
                std::printf("  %-32s %s\n", "ids", same_ids ? "the dump's" : "differ");
                std::printf("  %-32s %s\n", "text", got.text == want ? "the dump's" : "differs");
                if (got.text != want) std::printf("    ours:  %s\n    dump's: %s\n", got.text.c_str(), want.c_str());
                std::printf("  %-32s perception %.1f s, prefill %.1f s of %lld rows, decoding %.1f s\n", "time", report.perception, report.prefill,
                            (long long) report.prompt_rows, report.decode);
                worst_snr = std::min(worst_snr, steps.worst_snr);
                // A choice the arithmetic explains leaves the rest of the decoding to differ; the text is then not compared.
                ok = ok && steps.unexplained == 0 && (steps.differ > 0 || (same_ids && got.text == want));
            }
        }
        // The bounds of float32 on the CPU, and of a GPU whose matrix kernels round their inputs to half precision.
        const double bound = cpu ? 40.0 : 15.0;
        ok = ok && checked > 0 && worst_snr >= bound;
        std::printf("%s: %zu dumps, steps' top logits SNR %.1f dB against a bound of %.0f dB\n", ok ? "OK" : "FAIL", checked, worst_snr, bound);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
