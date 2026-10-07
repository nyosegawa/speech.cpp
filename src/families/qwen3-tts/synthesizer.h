#pragma once

#include <climits>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "codec.h"
#include "model-file.h"
#include "prompt.h"
#include "qwen2-tokenizer.h"
#include "sampler.h"
#include "talker.h"

struct SynthesisRequest {
    std::string text;
    std::string speaker;
    /** A BCP 47 tag of one of the model's languages (`ja`, `ja-JP`), or "auto". */
    std::string language = "auto";
    /** An instruction of how to speak, for a model that takes_instructions(), or empty for none. */
    std::string instructions;
    /** The frames to make at most; the model's own limit applies as well. */
    int max_frames = INT_MAX;
    uint64_t seed = 0;
    /** How the talker and the code predictor sample, where the request sets it rather than the model file. */
    std::optional<SamplingParams> talker, code_predictor;
};

/** How a synthesis ended: the frames it made, and whether the talker ended the speech or a limit stopped it. */
struct SynthesisOutcome {
    int frames = 0;
    bool ended = false;
};

/** Where the time of one synthesis went, in seconds. */
struct SynthesisStats {
    double prompt = 0, talker = 0, code_predictor = 0, codec = 0;
};

/** The prefix of the keys of a model file that hold the text's tokenizer. */
constexpr const char * kTextTokenizer = "qwen3-tts.tokenizer";

/**
 * The longest text in tokens that a model file speaks: what leaves the talker's positions room for a prompt and the
 * most frames.
 */
int text_token_limit(const ModelFile & m);

/** Called with each piece of 24 kHz audio as it is decoded; returning false stops the synthesis. */
using AudioSink = std::function<bool(const float * samples, size_t n)>;

/** Text in, audio out: the tokenizer, the talker, the code predictor and the codec decoder of one model file together. */
class Synthesizer {
public:
    Synthesizer(const std::string & path, ggml_backend_t backend);

    /**
     * Speaks `r.text`, decoding the first frame on its own so that audio starts as early as possible and later frames
     * `frames_per_piece` at a time, until the talker ends the speech, `r.max_frames` or the model's limit of frames is
     * reached, or the sink stops it. A text longer than max_text_tokens() throws, and so does an instruction whose
     * tokens leave the text fewer than it has.
     */
    SynthesisOutcome synthesize(const SynthesisRequest & r, const AudioSink & sink, int frames_per_piece = 4,
                                SynthesisStats * stats = nullptr);

    int sample_rate() const { return codec_.sample_rate(); }
    const ModelFile & model() const { return model_; }
    const PromptIds & ids() const { return ids_; }
    /** How the model file says the official generate() samples. */
    const Generation & generation() const { return generation_; }
    /** The frames a request makes at most. */
    int max_frames() const { return generation_.max_frames; }
    /** The samples of one frame at sample_rate(). */
    int samples_per_frame() const { return codec_.samples_per_frame(); }
    int max_text_tokens() const { return text_token_limit(model_); }

private:
    ModelFile model_;
    Talker talker_;
    CodecDecoder codec_;
    Qwen2Tokenizer tokenizer_;
    PromptIds ids_;
    Generation generation_;
};
