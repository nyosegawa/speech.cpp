#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "expert-placement.h"
#include "language-model.h"
#include "model-file.h"
#include "perception.h"
#include "tokenizer.h"
#include "transcript.h"

namespace nemotron_transcribe {

/** What a recognition asks of the model besides its audio: the terms of its context, or none. */
struct RecognitionRequest {
    std::string context;
};

/** What a recognition found: its text, and whether the decoding filled the model's context. */
struct Recognition {
    std::string text;
    bool limited = false;
};

/**
 * What a recognition did, for the checks and measurements: where its time went in seconds, the rows of its prompt with
 * the audio's, and the ids it wrote, the end of the turn included where it wrote one. `logits`, when set, hears each
 * step's logits before the step chooses its token.
 */
struct RecognitionReport {
    double perception = 0, prefill = 0, decode = 0;
    int64_t prompt_rows = 0;
    std::vector<int32_t> ids;
    std::function<void(const std::vector<float> & logits)> logits;
};

/**
 * Nemotron 3.5 Transcribe as NVIDIA's vLLM plugin serves a transcription at temperature 0 with reasoning off: the audio
 * through the perception module; the user's turn of the model card's instruction, its verbatim one or, for a request with
 * a context, its contextual one with the terms, one space and the audio locator, with no system turn, the locator's row
 * replaced by the audio's rows; the language model's prefill and greedy decoding until the end of the turn or the end of
 * the context; and the text decoded with its special tokens, cleaned by the model card's guardrail. It reads one GGUF
 * file and computes on one backend, but for the experts that a placement leaves to the CPU.
 */
class Recognizer {
public:
    /**
     * The model of the file `path` on `backend`, with the experts of the layers that `experts`, where it is given, does
     * not give the model's device in the CPU's memory; `experts` outlives the recognizer. The tensors a recognition
     * does not read (reads()) are not loaded.
     */
    Recognizer(const std::string & path, ggml_backend_t backend, const ExpertPlacement * experts = nullptr);

    /**
     * Whether a recognition reads the tensor `name`: every one but those of the multi-token prediction layers, which
     * greedy decoding does not run, and of the CTC head, which only timestamps read.
     */
    static bool reads(const std::string & name);

    /**
     * The memory a device needs beside the weights it holds to serve every request the model file `m` takes: the keys
     * and values of the attention layers for the whole context in float32, the Mamba layers' states, and what the
     * graphs and the backend allocate while they compute.
     */
    static uint64_t working_bytes(const ModelFile & m);

    /**
     * Recognizes mono samples at sample_rate(). `progress` hears how far the work has come, from 0 to 1: the perception
     * module's windows a third, the prefill's blocks a third and the tokens written, as a share of the context, the
     * rest; false stops the work at the next window, block or token, and what it returns is then none. Audio longer than the model takes, and a request whose prompt
     * with the audio leaves the model no room to write, are refused before any work, out of range naming the audio or
     * the option prompt.
     */
    std::optional<Recognition> recognize(const std::vector<float> & samples, const RecognitionRequest & request,
                                         const std::function<bool(double)> & progress, RecognitionReport * report = nullptr);

    /** The ids of the prompt before the audio's rows and after them, for a request with `context`. */
    std::pair<std::vector<int32_t>, std::vector<int32_t>> prompt(const std::string & context) const;

    int sample_rate() const { return perception_.sample_rate(); }
    const ModelFile & model() const { return model_; }

private:
    ModelFile model_;
    Tokenizer tokenizer_;
    Perception perception_;
    LanguageModel language_model_;
    Guardrail guardrail_;
    std::vector<int32_t> eos_;
    int64_t context_length_, max_samples_;
};

}  // namespace nemotron_transcribe
