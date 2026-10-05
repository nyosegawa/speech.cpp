#pragma once

#include <climits>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "codec.h"
#include "model-file.h"
#include "prompt.h"
#include "sampler.h"
#include "talker.h"
#include "tokenizer.h"

struct SynthesisRequest {
    std::string text;
    std::string speaker;
    /** A BCP 47 tag of one of the model's languages (`ja`, `ja-JP`), or "auto". */
    std::string language = "auto";
    /** The frames to make at most; the model's own limit applies as well. */
    int max_frames = INT_MAX;
    uint64_t seed = 0;
};

/** Where the time of one synthesis went, in seconds. */
struct SynthesisStats {
    double prompt = 0, talker = 0, code_predictor = 0, codec = 0;
};

/** Called with each piece of 24 kHz audio as it is decoded; returning false stops the synthesis. */
using AudioSink = std::function<bool(const float * samples, size_t n)>;

/** Text in, audio out: the tokenizer, the talker, the code predictor and the codec decoder of one model file together. */
class Synthesizer {
public:
    Synthesizer(const std::string & path, ggml_backend_t backend);

    /**
     * Speaks `r.text`, decoding the first frame on its own so that audio starts as early as possible and
     * later frames `frames_per_piece` at a time, and stops at the model's limit of frames. Returns the number of frames
     * generated. A text longer than max_text_tokens() throws.
     */
    int synthesize(const SynthesisRequest & r, const AudioSink & sink, int frames_per_piece = 4,
                   SynthesisStats * stats = nullptr);

    int sample_rate() const { return codec_.sample_rate(); }
    const ModelFile & model() const { return model_; }
    const PromptIds & ids() const { return ids_; }
    /** The frames a request makes at most. */
    int max_frames() const { return generation_.max_frames; }
    /** The longest text in tokens: what leaves the talker's positions room for a prompt and the most frames. */
    int max_text_tokens() const { return talker_.max_positions() - generation_.max_frames - kPromptRows; }

private:
    ModelFile model_;
    Talker talker_;
    CodecDecoder codec_;
    Tokenizer tokenizer_;
    PromptIds ids_;
    Generation generation_;
};
