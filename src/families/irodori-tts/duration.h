#pragma once

#include <cstdint>

#include "graph.h"
#include "model-file.h"

namespace irodori {

/**
 * What a request asks of the length of its speech. The runtime takes `seconds` and `duration_scale`;
 * `speed` is Irodori-TTS-Server's, which divides both by it to serve OpenAI's speech API.
 */
struct LengthOptions {
    /** The length in seconds, or 0 for the length the duration predictor gives. */
    double seconds = 0;
    /** The factor of the predicted length, above 0. It cannot be given with `seconds`. */
    double duration_scale = 1;
    /** The speaking rate, which divides the length, fixed or predicted. */
    double speed = 1;

    /** Whether the length is fixed, so that the duration predictor does not run. */
    bool fixed() const { return seconds > 0; }
};

/** How long a synthesis is: the latent's frames, and the samples its audio is cut to before the cut where the latent goes flat. */
struct Length {
    int frames = 0;
    int64_t samples = 0;
};

/**
 * The duration predictor of v4.1: a stack of SwiGLU blocks over the text condition's tokens, each
 * modulated by the speaker's summary and, without a caption, by the learned null caption. Each token
 * contributes softplus(output) frames.
 */
class DurationPredictor {
public:
    /** The predictor of the model file `m`, which outlives it, with the bounds of the length and the speed it gives. */
    explicit DurationPredictor(const ModelFile & m);

    /** The predicted frames summed over the tokens, [1], for a text condition [text_dim, n] and a speaker summary [speaker_dim]. */
    ggml_tensor * build(Graph & g, ggml_tensor * text_state, ggml_tensor * speaker_summary) const;

    /**
     * Throws, naming the option at fault, unless the options are ones the runtime takes: a speed within the model's
     * bounds, a duration scale above 0, and seconds within the model's bounds of the length that, divided by the
     * speed, still lie within them. The runtime clamps such seconds into the bounds and ignores a duration scale given
     * with seconds; both are refused here instead.
     */
    void check(const LengthOptions & options) const;

    /**
     * The length the runtime synthesizes for checked options and, when they fix no length, the predicted sum.
     * Fixed seconds give int(seconds * sample rate) samples in the frames that hold them. A prediction goes
     * through the runtime's float32 log1p and expm1, the scale in double precision and its rounding to the
     * nearest frame with half to even, and gives whole frames of samples. At a scale and a speed of 1 the frames are
     * kept within the model's bounds of the length, as the runtime keeps them; a prediction that a scale or a speed
     * takes outside them throws, naming the scale, or the speed when the scale is 1.
     */
    Length length(const LengthOptions & options, float predicted_sum) const;

private:
    const ModelFile & m_;
    int layers_, sample_rate_, hop_, min_frames_, max_frames_;
    double min_seconds_, max_seconds_, min_speed_, max_speed_;
    float eps_;
};

}  // namespace irodori
