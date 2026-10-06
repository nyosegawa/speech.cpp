#include "recognizer.h"

#include <chrono>
#include <cstddef>
#include <utility>

#include "error.h"
#include "layout.h"

namespace qwen3_asr {

namespace {

/** Adds the seconds from its construction to its destruction to `total`, when `total` is set. */
struct Timer {
    double * total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~Timer() {
        if (total) *total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
};

}  // namespace

Recognizer::Recognizer(const std::string & path, ggml_backend_t backend)
    : model_(path, backend, layout),
      frontend_(model_),
      encoder_(model_, backend),
      decoder_(model_, backend, kCacheType),
      tokenizer_(model_),
      prompt_(model_, tokenizer_),
      transcript_(model_),
      splitter_(model_) {}

Recognition Recognizer::recognize(const std::vector<float> & samples, const RecognitionRequest & request,
                                  const std::function<bool(double)> & progress, std::vector<PartReport> * parts) {
    const std::vector<float> audio = Frontend::normalize(samples);
    const std::vector<int64_t> bounds = splitter_.bounds(audio);
    const size_t n_parts = bounds.size() - 1;
    // Every part's prompt is made, and checked against the decoder's positions, before any work.
    std::vector<PromptIds> prompts;
    for (size_t k = 0; k < n_parts; k++) {
        const int64_t audio_tokens = encoder_.tokens(frontend_.frames((size_t) (bounds[k + 1] - bounds[k])));
        prompts.push_back(prompt_.ids(request.context, request.language, audio_tokens));
        const int64_t n = (int64_t) prompts.back().ids.size();
        if (decoder_.positions(n) > decoder_.max_positions()) {
            throw Error(Fault::OutOfRange,
                        "the prompt takes " + std::to_string(n - audio_tokens) + " tokens, which with the audio's " + std::to_string(audio_tokens) +
                            " and the " + std::to_string(decoder_.max_new_tokens()) + " the model may write leave more than its " +
                            std::to_string(decoder_.max_positions()) + " positions; shorten it",
                        "prompt");
        }
    }
    Recognition out;
    for (size_t k = 0; k < n_parts; k++) {
        const std::vector<float> part(audio.begin() + (std::ptrdiff_t) bounds[k], audio.begin() + (std::ptrdiff_t) bounds[k + 1]);
        PartReport report;
        const std::optional<std::string> text = recognize_part(part, prompts[k], request.language.has_value(),
                                                                [&](double done) { return progress((k + done) / n_parts); }, report);
        if (!text) return {};
        // qwen-asr joins the texts of the parts without a separator.
        out.text += *text;
        out.limited = out.limited || report.limited;
        if (parts) parts->push_back(std::move(report));
    }
    return out;
}

std::optional<std::string> Recognizer::recognize_part(const std::vector<float> & part, const PromptIds & prompt, bool forced,
                                                      const std::function<bool(double)> & progress, PartReport & report) {
    const int64_t n = (int64_t) prompt.ids.size();
    std::vector<float> embeds;
    {
        std::vector<float> features;
        {
            Timer t{&report.frontend};
            features = frontend_.features(part);
        }
        const size_t windows = encoder_.windows((int64_t) features.size() / frontend_.mels()).size();
        std::optional<std::vector<float>> encoded;
        {
            Timer t{&report.encoder};
            encoded = encoder_.encode(features, [&](size_t done) { return progress((double) done / windows / 3); });
        }
        if (!encoded) return std::nullopt;
        Timer t{&report.prefill};
        embeds = decoder_.embeddings(prompt, *encoded);
    }
    report.prompt_rows = n;
    {
        Timer t{&report.prefill};
        if (!decoder_.prefill(embeds, n, [&](int64_t rows) { return progress((1 + (double) rows / n) / 3); })) return std::nullopt;
    }
    embeds.clear();
    embeds.shrink_to_fit();
    std::optional<Generation> generation;
    {
        Timer t{&report.decode};
        generation = decoder_.generate([&](size_t tokens) { return progress((2 + (double) tokens / decoder_.max_new_tokens()) / 3); });
    }
    if (!generation) return std::nullopt;
    report.ids = generation->ids;
    report.limited = generation->limited;
    return transcript_.text(tokenizer_.decode(generation->ids), forced);
}

}  // namespace qwen3_asr
