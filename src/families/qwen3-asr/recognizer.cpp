#include "recognizer.h"

#include <chrono>
#include <cstdio>
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
      max_samples_(model_.u32("qwen3-asr.audio.max_samples")) {}

Recognition Recognizer::recognize(const std::vector<float> & samples, const RecognitionRequest & request,
                                  const std::function<bool(double)> & progress, std::vector<PartReport> * parts) {
    PartReport st;
    const std::vector<float> audio = Frontend::normalize(samples);
    const double rate = sample_rate();
    if ((int64_t) audio.size() > max_samples_) {
        char message[160];
        std::snprintf(message, sizeof message, "the audio is %.2f s long, and Qwen3-ASR takes at most %.0f s at once; give a shorter part of it",
                      (double) audio.size() / rate, (double) max_samples_ / rate);
        throw Error(Fault::OutOfRange, message, "audio");
    }
    const int64_t frames = frontend_.frames(audio.size());
    const PromptIds prompt = prompt_.ids(request.context, request.language, encoder_.tokens(frames));
    const int64_t n = (int64_t) prompt.ids.size();
    if (decoder_.positions(n) > decoder_.max_positions()) {
        throw Error(Fault::OutOfRange,
                    "the prompt takes " + std::to_string(n - encoder_.tokens(frames)) + " tokens, which with the audio's " +
                        std::to_string(encoder_.tokens(frames)) + " and the " + std::to_string(decoder_.max_new_tokens()) +
                        " the model may write leave more than its " + std::to_string(decoder_.max_positions()) + " positions; shorten it",
                    "prompt");
    }

    std::vector<float> embeds;
    {
        std::vector<float> features;
        {
            Timer t{&st.frontend};
            features = frontend_.features(audio);
        }
        const size_t windows = encoder_.windows(frames).size();
        std::optional<std::vector<float>> encoded;
        {
            Timer t{&st.encoder};
            encoded = encoder_.encode(features, [&](size_t done) { return progress((double) done / windows / 3); });
        }
        if (!encoded) return {};
        Timer t{&st.prefill};
        embeds = decoder_.embeddings(prompt, *encoded);
    }
    st.prompt_rows = n;
    {
        Timer t{&st.prefill};
        if (!decoder_.prefill(embeds, n, [&](int64_t rows) { return progress((1 + (double) rows / n) / 3); })) return {};
    }
    embeds.clear();
    embeds.shrink_to_fit();
    Generation generation;
    {
        Timer t{&st.decode};
        generation = decoder_.generate([&](size_t tokens) { return progress((2 + (double) tokens / decoder_.max_new_tokens()) / 3); });
    }
    const Recognition out = {transcript_.text(tokenizer_.decode(generation.ids), request.language.has_value()), generation.limited};
    st.ids = generation.ids;
    st.limited = generation.limited;
    if (parts) parts->push_back(std::move(st));
    return out;
}

}  // namespace qwen3_asr
