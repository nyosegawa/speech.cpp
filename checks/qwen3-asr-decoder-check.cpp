// Checks Qwen3-ASR's prompt, decoder, decoding and parse against transformers' on each dump of
// reference/qwen3-asr/dump.py and each of its requests (auto, forced, auto-prompt, forced-prompt), after the parse of
// the cases of reference/qwen3-asr/parse_cases.py against qwen-asr's, in the order data flows: the prompt's ids; the decoder's input, the token embeddings with the dump's projector output spliced in; the
// logits of the prompt's last four rows from the dump's input; the decoder teacher-forced on the dump's ids; the
// tokenizer's decoding of the dump's ids and the parse of the dump's raw text; the greedy decoding from the dump's
// projector output with every later stage ours; and the text from the dump's audio with every stage ours, which it
// also times. The dumps are those of the model, in <reference out dir>/<its general.name>/.
//
// Where a greedy choice of ours differs from the dump's, the check takes it for the arithmetic's when the dump's margin
// between the two tokens is within the sum of our errors on their two logits, and for a defect otherwise.
//
// usage: qwen3-asr-decoder-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "json-reader.h"
#include "npy.h"
#include "qwen3-asr/recognizer.h"
#include "reference-dumps.h"

using namespace qwen3_asr;

namespace {

using Clock = std::chrono::steady_clock;

const char * const kVariants[] = {"auto", "forced", "auto-prompt", "forced-prompt"};

JsonValue read_meta(const std::filesystem::path & dir) {
    return parse_json(dump_text(dir / "meta.json"));
}

/** The request of a variant's meta.json: its context, and its forced language as an index of the model's languages. */
RecognitionRequest request_of(const JsonValue & meta, const ModelFile & m) {
    RecognitionRequest r;
    r.context = meta.member("prompt")->text;
    const JsonValue * language = meta.member("language");
    if (language && language->kind == JsonValue::Kind::String) {
        const std::vector<std::string> names = m.str_array("qwen3-asr.language_names");
        const auto it = std::find(names.begin(), names.end(), language->text);
        if (it == names.end()) throw std::runtime_error("the dump forces " + language->text + ", which the model does not name");
        r.language = (size_t) (it - names.begin());
    }
    return r;
}

/**
 * How our greedy choices compare with the dump's at the steps it compares: each step's top logits of the dump against
 * ours at the same ids, and every step where our choice differs, with whether the error explains it.
 */
struct Steps {
    size_t steps = 0, differ = 0, unexplained = 0;
    double worst_snr = INFINITY, worst_abs = 0;
    std::optional<size_t> first_differ;

    /**
     * Compares our logits at step `i` with the dump's top ids and logits; returns whether our choice is the dump's id
     * `want`.
     */
    bool add(size_t i, const std::vector<float> & logits, const Npy & top_ids, const Npy & top_logits, int32_t want) {
        const size_t k = (size_t) top_ids.shape[1];
        std::vector<float> ours(k), theirs(top_logits.f32.begin() + (std::ptrdiff_t) (i * k), top_logits.f32.begin() + (std::ptrdiff_t) ((i + 1) * k));
        for (size_t j = 0; j < k; j++) ours[j] = logits[(size_t) top_ids.i32[i * k + j]];
        const Diff d = compare(ours, theirs);
        worst_snr = std::min(worst_snr, d.snr_db);
        worst_abs = std::max(worst_abs, d.max_abs);
        steps++;
        const int32_t got = (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (got == want) return true;
        differ++;
        if (!first_differ) first_differ = i;
        // The dump's margin between its choice and ours, against our errors on those two logits.
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
        return false;
    }

    void print(const char * what) const {
        std::printf("  %-32s %zu of %zu steps ours, top logits' worst SNR %.1f dB, max |diff| %.2e\n", what, steps - differ, steps, worst_snr,
                    worst_abs);
    }
};

/**
 * Decodes greedily from the decoder's prefill, comparing each step with the dump's until the first that differs;
 * returns the ids it wrote.
 */
std::vector<int32_t> follow(Decoder & decoder, const Npy & ids, const Npy & top_ids, const Npy & top_logits, Steps & steps) {
    bool same = true;
    const auto compare_step = [&](size_t i) {
        if (same && i < (size_t) ids.shape[0]) same = steps.add(i, decoder.logits(), top_ids, top_logits, ids.i32[i]);
    };
    // The decoding calls back after each token but the last, before it feeds the token; the logits of the last stay.
    const Generation g = *decoder.generate([&](size_t n) {
        compare_step(n - 1);
        return true;
    });
    compare_step(g.ids.size() - 1);
    return g.ids;
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
            const ModelFile & m = recognizer.model();
            Decoder & decoder = recognizer.decoder();
            // The stages from the dump's input run with an F32 cache, which leaves the arithmetic of the weights alone;
            // the greedy decodings run as the recognizer does, with its F16 cache.
            Decoder exact(m, backend, GGML_TYPE_F32);
            // Measured on an Apple M5 on 2026-10-06, the prompt's last logits from the dumps' input against transformers'
            // float32, over the ten inputs and four requests of the 0.6B and the 1.7B model: with F32 weights 97.4 and
            // 96.1 dB on the CPU, and 43.3 and 46.6 dB on Metal, whose matrix kernel rounds its inputs to half precision,
            // as it does with F16 weights, which give the same there and 37.3 dB with the 0.6B model on the CPU, whose
            // dot product sums in half precision; with Q8_0 weights 19.3 and 18.2 dB on the CPU, which quantizes the
            // activations to 8 bits too, and 23.3 and 21.7 dB on Metal. The greedy choices where these errors exceed the
            // dump's margin between two tokens differ, which the steps below tell from a defect.
            const ggml_type type = m.tensor("dec.blk.0.attn_q")->type;
            const double threshold_db = type == GGML_TYPE_Q8_0 ? 15 : type == GGML_TYPE_F32 && ggml_backend_is_cpu(backend) ? 90 : 35;
            std::printf("prefill logits threshold: %.0f dB\n", threshold_db);
            // The parse of texts that hold what the dumps do not: repetitions, whitespace, and outputs without <asr_text>.
            {
                std::ifstream f(std::filesystem::u8path(args[2]) / "parse-cases.jsonl", std::ios::binary);
                if (!f) throw std::runtime_error("no parse-cases.jsonl of reference/qwen3-asr/parse_cases.py is in " + args[2]);
                int cases = 0, same = 0;
                for (std::string line; std::getline(f, line);) {
                    const JsonValue c = parse_json(line);
                    const std::string got = recognizer.transcript().text(c.member("raw")->text, c.member("forced")->boolean);
                    const bool equal = got == c.member("text")->text;
                    cases++;
                    same += equal;
                    if (!equal) std::printf("  parse of %s: %s where qwen-asr gives %s\n", to_json(*c.member("raw")).c_str(), got.c_str(), c.member("text")->text.c_str());
                }
                std::printf("parse: %d of %d cases qwen-asr's\n", same, cases);
                ok = ok && cases > 0 && same == cases;
            }
            // A second of silence first, so that the times below leave out the compilation of a GPU's kernels.
            recognizer.recognize(std::vector<float>((size_t) recognizer.sample_rate(), 0.0f), {}, [](double) { return true; });
            int texts = 0, same_texts = 0;
            for (const auto & d : reference_dumps(args[2], m)) {
                const Npy embeds_dump = read_npy((d / "audio_embeds.npy").u8string());
                const Npy audio = read_npy((d / "audio.npy").u8string());
                std::printf("%s (%.2f s, %lld audio tokens)\n", d.filename().u8string().c_str(), (double) audio.f32.size() / recognizer.sample_rate(),
                            (long long) embeds_dump.shape[0]);
                for (const char * variant : kVariants) {
                    const std::filesystem::path v = d / variant;
                    const RecognitionRequest request = request_of(read_meta(v), m);
                    const auto npy = [&](const char * name) { return read_npy((v / (std::string(name) + ".npy")).u8string()); };
                    const Npy prompt_ids = npy("prompt_ids"), embeds = npy("embeds"), prefill_logits = npy("prefill_logits");
                    const Npy ids = npy("ids"), top_ids = npy("step_top_ids"), top_logits = npy("step_top_logits");
                    const std::string raw = dump_text(v / "raw.txt"), text = dump_text(v / "text.txt");
                    std::printf(" %s (%lld ids written)\n", variant, (long long) ids.shape[0]);

                    const PromptIds prompt = recognizer.prompt().ids(request.context, request.language, embeds_dump.shape[0]);
                    const bool same_prompt = prompt.ids == prompt_ids.i32;
                    std::printf("  %-32s %s (%zu ids)\n", "prompt ids", same_prompt ? "the dump's" : "DIFFER", prompt.ids.size());
                    ok = ok && same_prompt;
                    if (!same_prompt) continue;
                    const std::vector<float> ours = decoder.embeddings(prompt, embeds_dump.f32);
                    print_diff("  decoder input", compare(ours, embeds.f32));

                    // The logits of the prompt's last rows from the dump's input, a prefill ending at each.
                    const int64_t n = (int64_t) prompt.ids.size(), rows = prefill_logits.shape[0], vocab = prefill_logits.shape[1];
                    for (int64_t r = 0; r < rows; r++) {
                        exact.prefill(embeds.f32, n - rows + 1 + r, [](int64_t) { return true; });
                        const Diff diff = compare(exact.logits().data(), &prefill_logits.f32[(size_t) (r * vocab)], (size_t) vocab);
                        print_diff("  prefill logits, row " + std::to_string(n - rows + r), diff);
                        ok = ok && diff.snr_db > threshold_db;
                    }

                    // Teacher forcing on the dump's ids from the dump's input.
                    Steps forced;
                    exact.prefill(embeds.f32, n, [](int64_t) { return true; });
                    for (int64_t i = 0; i < ids.shape[0]; i++) {
                        forced.add((size_t) i, exact.logits(), top_ids, top_logits, ids.i32[i]);
                        if (i + 1 < ids.shape[0]) exact.step(ids.i32[i]);
                    }
                    forced.print("teacher-forced");
                    ok = ok && forced.unexplained == 0;

                    const std::string decoded = recognizer.tokenizer().decode(ids.i32), parsed = recognizer.transcript().text(raw, request.language.has_value());
                    std::printf("  %-32s %s\n", "decoding of the dump's ids", decoded == raw ? "the dump's raw text" : "DIFFERS");
                    std::printf("  %-32s %s\n", "parse of the dump's raw text", parsed == text ? "the dump's text" : "DIFFERS");
                    ok = ok && decoded == raw && parsed == text;

                    // Greedy from the dump's projector output, the prompt and the embeddings ours.
                    Steps greedy;
                    decoder.prefill(ours, n, [](int64_t) { return true; });
                    const std::vector<int32_t> written = follow(decoder, ids, top_ids, top_logits, greedy);
                    const std::string greedy_text = recognizer.transcript().text(recognizer.tokenizer().decode(written), request.language.has_value());
                    greedy.print("greedy from the projector");
                    std::printf("  %-32s %s\n", "its ids and text", written == ids.i32 ? "the dump's" : greedy_text == text ? "other ids, the dump's text" : "DIFFER");
                    ok = ok && greedy.unexplained == 0 && (greedy.differ > 0 || written == ids.i32);

                    // The text from the audio, every stage ours.
                    std::vector<PartReport> parts;
                    const auto start = Clock::now();
                    const Recognition found = recognizer.recognize(audio.f32, request, [](double) { return true; }, &parts);
                    const PartReport & stats = parts.at(0);
                    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
                    const bool same_text = found.text == text;
                    texts++;
                    same_texts += same_text;
                    std::printf("  %-32s %s in %.3f s%s\n", "text from the audio", same_text ? "the dump's" : "DIFFERS", seconds,
                                found.limited ? ", at the model's limit" : "");
                    std::printf("    frontend %.3f s, encoder %.3f s, prefill of %lld rows %.3f s, %zu tokens in %.3f s (%.1f per second)\n",
                                stats.frontend, stats.encoder, (long long) stats.prompt_rows, stats.prefill, stats.ids.size(), stats.decode,
                                stats.ids.size() / stats.decode);
                    if (!same_text) {
                        std::printf("    got  %s\n    want %s\n", found.text.c_str(), text.c_str());
                        // The same stages one by one, to find the step where the decoding departs and whether the error
                        // explains it.
                        const std::vector<float> encoded = *recognizer.encoder().encode(recognizer.frontend().features(Frontend::normalize(audio.f32)));
                        decoder.prefill(decoder.embeddings(prompt, encoded), n, [](int64_t) { return true; });
                        Steps from_audio;
                        follow(decoder, ids, top_ids, top_logits, from_audio);
                        from_audio.print("greedy from the audio");
                        ok = ok && from_audio.unexplained == 0;
                    }
                }
            }
            std::printf("texts from the audio: %d of %d the dump's\n", same_texts, texts);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
