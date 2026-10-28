#include "recognizer.h"

#include <algorithm>
#include <chrono>

#include "error.h"
#include "layout.h"

namespace nemotron_transcribe {

namespace {

const std::string kKeys = "nemotron-transcribe.";

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
      tokenizer_(model_),
      perception_(model_, backend),
      language_model_(model_, backend),
      guardrail_(model_),
      eos_(model_.i32_array(kKeys + "generation.eos_ids")),
      context_length_(model_.size(kKeys + "llm.context_length")),
      max_samples_(model_.size(kKeys + "audio.max_samples")) {}

std::pair<std::vector<int32_t>, std::vector<int32_t>> Recognizer::prompt(const std::string & context) const {
    const std::string q = kKeys + "prompt.";
    const std::string text = context.empty() ? model_.str(q + "instruction") : model_.str(q + "context_before") + context + model_.str(q + "context_after");
    std::vector<int32_t> before = naming("prompt", [&] { return tokenizer_.encode(model_.str(q + "before_text") + text + model_.str(q + "before_audio")); });
    return {std::move(before), tokenizer_.encode(model_.str(q + "after_audio"))};
}

std::optional<Recognition> Recognizer::recognize(const std::vector<float> & samples, const RecognitionRequest & request,
                                                 const std::function<bool(double)> & progress, RecognitionReport * report) {
    RecognitionReport local;
    RecognitionReport & r = report ? *report : local;
    if ((int64_t) samples.size() > max_samples_) {
        throw Error(Fault::OutOfRange,
                    "the audio is " + std::to_string((double) samples.size() / sample_rate()) + " s long; the model takes at most " +
                        std::to_string(max_samples_ / sample_rate()) + " s, so split it",
                    "audio");
    }
    const auto [before, after] = prompt(request.context);
    const int64_t frames = perception_.frames(samples.size()), rows = (int64_t) (before.size() + after.size()) + frames;
    if (rows >= context_length_) {
        throw Error(Fault::OutOfRange,
                    "the prompt takes " + std::to_string(before.size() + after.size()) + " tokens, which with the audio's " + std::to_string(frames) +
                        " rows leave none of the model's " + std::to_string(context_length_) + " positions to write in; shorten it",
                    "prompt");
    }
    r.prompt_rows = rows;

    std::vector<float> input;
    {
        Timer t{&r.perception};
        const std::vector<float> audio = perception_.encode(samples, [&](double done) { return !progress || progress(done / 3); });
        if (audio.empty()) return std::nullopt;
        input = language_model_.embed(before);
        input.insert(input.end(), audio.begin(), audio.end());
        const std::vector<float> tail = language_model_.embed(after);
        input.insert(input.end(), tail.begin(), tail.end());
    }
    const int h = language_model_.hidden();
    language_model_.start(context_length_);
    std::vector<float> last;
    {
        Timer t{&r.prefill};
        const std::vector<float> out = language_model_.run(input, rows);
        last.assign(out.end() - h, out.end());
    }
    if (progress && !progress(2.0 / 3)) return std::nullopt;

    Recognition result;
    {
        Timer t{&r.decode};
        while (true) {
            const std::vector<float> logits = language_model_.logits(last);
            // The most likely token, the first of equal logits, as torch.argmax() takes it.
            const int32_t token = (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
            r.ids.push_back(token);
            if (std::find(eos_.begin(), eos_.end(), token) != eos_.end()) break;
            if (language_model_.n_past() >= context_length_) {
                result.limited = true;
                break;
            }
            last = language_model_.run(language_model_.embed({token}), 1);
            if (progress && !progress((2 + (double) language_model_.n_past() / (double) context_length_) / 3)) return std::nullopt;
        }
    }
    std::vector<int32_t> written = r.ids;
    if (!result.limited) written.pop_back();
    result.text = guardrail_.apply(tokenizer_.decode(written));
    return result;
}

}  // namespace nemotron_transcribe
