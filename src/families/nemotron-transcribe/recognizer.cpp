#include "recognizer.h"

#include <algorithm>
#include <chrono>

#include "error.h"
#include "layout.h"

namespace nemotron_transcribe {

namespace {

const std::string kKeys = "nemotron-transcribe.";

/**
 * What the graphs and the backend allocate while a recognition computes, beyond the weights, the caches and the states:
 * the buffers of the perception module's graphs, a window at a time, of the language model's at their largest, a block
 * of 512 rows attending to the most positions, and of the CTC head's, a window of the alignment at a time. For an hour of
 * noise, the longest audio a request takes, with the IQ4_NL file on an Apple M5 the first two took 627 MiB of Metal's
 * memory, of which 72 MiB grew during the request (2026-10-29), and the head's 127 MiB more, its activations float32
 * whatever the weights' type (2026-10-30); Vulkan's backend takes about 0.3 GB more of its own.
 */
constexpr uint64_t kComputeBytes = (uint64_t) 1152 << 20;

/** Adds the seconds from its construction to its destruction to `total`, when `total` is set. */
struct Timer {
    double * total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~Timer() {
        if (total) *total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
};

}  // namespace

Recognizer::Recognizer(const std::string & path, ggml_backend_t backend, const ExpertPlacement * experts)
    : model_(path, layout,
             [&](const std::string & name) -> ggml_backend_t {
                 if (!reads(name)) return nullptr;
                 return experts ? experts->place(name, backend) : backend;
             }),
      tokenizer_(model_),
      perception_(model_, backend),
      language_model_(model_, backend, 0, -1, experts ? experts->cpu() : nullptr),
      guardrail_(model_),
      timing_(model_, backend),
      eos_(model_.i32_array(kKeys + "generation.eos_ids")),
      context_length_(model_.size(kKeys + "llm.context_length")),
      max_samples_(model_.size(kKeys + "audio.max_samples")) {
    // The keys and values of the whole context and the language model's largest graph are held from the load on, so
    // that a model loaded after this one cannot take the memory a long recording needs.
    language_model_.reserve(context_length_);
}

bool Recognizer::reads(const std::string & name) {
    return name.rfind("mtp.", 0) != 0;
}

uint64_t Recognizer::working_bytes(const ModelFile & m) {
    const std::string x = kKeys + "llm.";
    const std::vector<std::string> types = m.str_array(x + "layer_types");
    const uint64_t attention_layers = (uint64_t) std::count(types.begin(), types.end(), "attention");
    const uint64_t mamba_layers = (uint64_t) std::count(types.begin(), types.end(), "mamba");
    // A key and a value of float32 for each kv head of each position.
    const uint64_t keys_values = 2 * sizeof(float) * (uint64_t) m.size(x + "head_dim") * (uint64_t) m.size(x + "num_key_value_heads") *
                                 (uint64_t) m.size(x + "context_length") * attention_layers;
    const uint64_t channels = (uint64_t) m.size(x + "mamba_num_heads") * (uint64_t) m.size(x + "mamba_head_dim");
    const uint64_t conv = channels + 2 * (uint64_t) m.size(x + "n_groups") * (uint64_t) m.size(x + "ssm_state_size");
    const uint64_t states = sizeof(float) * mamba_layers * (channels * (uint64_t) m.size(x + "ssm_state_size") + conv * ((uint64_t) m.size(x + "conv_kernel") - 1));
    return keys_values + states + kComputeBytes;
}

std::pair<std::vector<int32_t>, std::vector<int32_t>> Recognizer::prompt(const RecognitionRequest & request) const {
    const std::string q = kKeys + "prompt.";
    if (request.speakers && !request.context.empty()) {
        throw Error(Fault::OutOfRange, "a request that asks for speakers takes no prompt: the model card's prompt for speaker tags has no context terms; leave prompt empty",
                    "prompt");
    }
    const std::string text = request.speakers ? model_.str(q + "speakers")
                             : request.context.empty() ? model_.str(q + "instruction")
                                                       : model_.str(q + "context_before") + request.context + model_.str(q + "context_after");
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
    const auto [before, after] = prompt(request);
    const int64_t frames = perception_.frames(samples.size()), rows = (int64_t) (before.size() + after.size()) + frames;
    if (rows >= context_length_) {
        throw Error(Fault::OutOfRange,
                    "the prompt takes " + std::to_string(before.size() + after.size()) + " tokens, which with the audio's " + std::to_string(frames) +
                        " rows leave none of the model's " + std::to_string(context_length_) + " positions to write in; shorten it",
                    "prompt");
    }
    r.prompt_rows = rows;

    const bool timed = request.speakers || request.timestamps;
    AlignmentFrames aligned;
    std::vector<float> input;
    {
        Timer t{&r.perception};
        const std::vector<float> audio =
            perception_.encode(samples, [&](double done) { return !progress || progress(done / 3); }, nullptr, timed ? &aligned : nullptr);
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
        const std::vector<float> out =
            language_model_.run(input, rows, nullptr, [&](int64_t done) { return !progress || progress((1 + (double) done / (double) rows) / 3); });
        if (out.empty()) return std::nullopt;
        last.assign(out.end() - h, out.end());
    }

    Recognition result;
    {
        Timer t{&r.decode};
        while (true) {
            const std::vector<float> logits = language_model_.logits(last);
            if (r.logits) r.logits(logits);
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
    if (timed) {
        Timer t{&r.alignment};
        TimedTranscript found = timing_.time(result.text, aligned, samples.size(), request.speakers);
        result.text = std::move(found.text);
        if (request.timestamps) result.tokens = std::move(found.tokens);
        result.segments = std::move(found.segments);
        result.untimed = std::move(found.failure);
    }
    return result;
}

}  // namespace nemotron_transcribe
