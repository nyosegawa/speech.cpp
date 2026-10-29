// Checks Nemotron 3.5 Transcribe's recognition against the official generation on each dump of
// reference/nemotron-transcribe/dump.py whose prompt a request makes, the model card's verbatim instruction, its
// contextual one with the request's terms, or its prompt for speaker tags: the prompt's ids; then the dump's audio with
// every stage ours, each step's logits at the dump's 16 largest against the dump's, and the ids and the text it writes,
// without its speaker tags where it asks for speakers, with its tokens' and its segments' texts joining into it. Where
// a greedy choice of ours differs from the dump's, the check takes it for the arithmetic's when the dump's margin
// between the two tokens is within twice the largest of our errors on the step's 14 other logits, which bounds what
// rounding moves either of the two by, and for a defect otherwise, and compares no later step. A prompt of more than
// one block of rows reports its progress after each block, and a recognition that stops at the first such report ends
// before it decodes. The logits are compared but bound nothing: where two experts of a MoE layer score within float32's
// rounding of each other, the port may take the other, as the decoder check counts such rows, and the row's output then
// departs by the difference of the two experts, which every later layer and step carries. On the quiet input of
// reference/nemotron-transcribe, such a row of layer 13 departs by 3.0e-2 and the logits of its steps by up to 21, at
// an SNR of 2.8 dB, with every choice the dump's (2026-10-29). The dumps of the plain instruction are the decoder
// check's alone. The F32 file takes about 131 GB of memory, which Modal's CPU containers hold. The dumps are those in
// <reference out dir>/<the file's general.name>/.
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
#include "nemotron-transcribe/timing.h"
#include "npy.h"
#include "reference-dumps.h"
#include "weight-types.h"

using namespace nemotron_transcribe;

namespace {

/** The request whose prompt is `text`, with timestamps, or none where no request's prompt is it. */
std::optional<RecognitionRequest> request_of(const std::string & text, const ModelFile & m) {
    const std::string q = "nemotron-transcribe.prompt.";
    if (text == m.str(q + "instruction")) return RecognitionRequest{"", false, true};
    if (text == m.str(q + "speakers")) return RecognitionRequest{"", true, true};
    const std::string before = m.str(q + "context_before"), after = m.str(q + "context_after");
    if (text.size() > before.size() + after.size() && text.compare(0, before.size(), before) == 0 &&
        text.compare(text.size() - after.size(), after.size(), after) == 0) {
        return RecognitionRequest{text.substr(before.size(), text.size() - before.size() - after.size()), false, true};
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
        // The error of the two logits themselves always covers the margin a swap of them makes; the step's others do not.
        float others = 0;
        for (size_t j = 0; j < k; j++) {
            if (j != at_want && j != at_got) others = std::max(others, std::fabs(ours[j] - theirs[j]));
        }
        const bool explained = at_want < k && at_got < k && theirs[at_want] - theirs[at_got] <= 2 * others;
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
        if (!takes_weights(args[1], layout, {"F32"},
                           "a choice that differs is explained by the logits' error, which the weights of another type widen enough to explain a defect's")) {
            return 2;
        }
        start_ggml();
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "auto");
        std::printf("device: %s\n", ggml_backend_name(backend));
        bool ok = true;
        double worst_snr = INFINITY;
        size_t checked = 0, stops = 0;
        {
            Recognizer recognizer(args[1], backend);
            const ModelFile & m = recognizer.model();
            const Guardrail guardrail(m);
            Timing timing(m, backend);
            const int32_t audio_id = (int32_t) m.size("nemotron-transcribe.prompt.audio_id");
            for (const auto & dump : reference_dumps(args[2], m)) {
                const JsonValue meta = parse_json(dump_text(dump / "meta.json"));
                const std::optional<RecognitionRequest> request = request_of(meta.member("prompt")->member("text")->text, m);
                std::printf("%s\n", dump.filename().u8string().c_str());
                if (!request) {
                    std::printf("  no request's prompt; the decoder check's alone\n");
                    continue;
                }
                checked++;

                const Npy prompt_ids = read_npy((dump / "prompt_ids.npy").u8string());
                const auto [before, after] = recognizer.prompt(*request);
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
                std::vector<double> reports;
                const Recognition got = *recognizer.recognize(audio.f32, *request, [&](double done) {
                    reports.push_back(done);
                    return true;
                }, &report);
                const bool same_ids = report.ids == tokens.i32;
                const std::string decoded = guardrail.apply(meta.member("text")->text);
                const std::string want = request->speakers ? timing.untagged(decoded) : decoded;
                std::string tokens_text, segments_text;
                for (const TimedText & t : got.tokens) tokens_text += t.text;
                for (const TimedText & t : got.segments) segments_text += t.text;
                const bool joined = got.untimed.empty() ? tokens_text == got.text && segments_text == got.text : got.tokens.empty() && got.segments.empty();
                std::printf("  %-32s %zu of %zu steps ours, top logits' worst SNR %.1f dB, max |diff| %.2e\n", "greedy decoding",
                            steps.steps - steps.differ, tokens.i32.size(), steps.worst_snr, steps.worst_abs);
                std::printf("  %-32s %s\n", "ids", same_ids ? "the dump's" : "differ");
                std::printf("  %-32s %s\n", "text", got.text == want ? "the dump's" : "differs");
                if (got.text != want) std::printf("    ours:  %s\n    dump's: %s\n", got.text.c_str(), want.c_str());
                std::printf("  %-32s %zu tokens, %zu segments%s%s\n", "times", got.tokens.size(), got.segments.size(), joined ? ", joining into the text" : ", NOT JOINING INTO THE TEXT",
                            got.untimed.empty() ? "" : (", none: " + got.untimed).c_str());
                std::printf("  %-32s perception %.1f s, prefill %.1f s of %lld rows, decoding %.1f s, alignment %.1f s\n", "time", report.perception, report.prefill,
                            (long long) report.prompt_rows, report.decode, report.alignment);
                worst_snr = std::min(worst_snr, steps.worst_snr);
                const auto within_prefill = [](double done) { return done > 1.0 / 3 + 1e-9 && done < 2.0 / 3 - 1e-9; };
                const size_t between = (size_t) std::count_if(reports.begin(), reports.end(), within_prefill);
                if (between > 0) {
                    RecognitionReport stopped;
                    const bool none = !recognizer.recognize(audio.f32, *request, [&](double done) { return !within_prefill(done); }, &stopped);
                    const bool before_decoding = none && stopped.ids.empty();
                    std::printf("  %-32s %zu reports between the prefill's blocks; stopped at the first %s\n", "prefill", between,
                                before_decoding ? "before decoding" : "AFTER IT DECODED");
                    stops++;
                    ok = ok && before_decoding;
                }
                // A choice the arithmetic explains leaves the rest of the decoding to differ; the text is then not compared.
                ok = ok && joined && steps.unexplained == 0 && (steps.differ > 0 || (same_ids && got.text == want));
            }
        }
        // The meeting's prompt runs to more than one block of rows.
        ok = ok && checked > 0 && stops > 0;
        std::printf("%s: %zu dumps, every choice the dump's or within the arithmetic's error of it, steps' top logits SNR %.1f dB\n", ok ? "OK" : "FAIL",
                    checked, worst_snr);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
