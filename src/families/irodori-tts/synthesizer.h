#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "codec.h"
#include "dit.h"
#include "duration.h"
#include "reference.h"
#include "sampler.h"
#include "speaker-encoder.h"
#include "text-encoder.h"
#include "tokenizer.h"

namespace irodori {

/** A voice: the latent of its reference and the speaker condition the model makes of it. */
struct Voice {
    std::vector<float> latent;
    int frames = 0;
    std::vector<float> speaker;
    int speaker_tokens = 0;
};

struct Request {
    std::string text;
    uint64_t seed = 0;
    /** The sampler's steps; 0 takes the model's default (4 for MeanFlow, 40 for RF). */
    int steps = 0;
    LengthOptions length;
    /** The sampler's starting point, row-major [frames, latent_dim], instead of noise from the seed. */
    std::vector<float> noise;
    /**
     * Told the fraction of the sampler's steps done before each step and once the last is done, while no audio
     * reaches the sink; once it answers false, synthesize() returns without audio.
     */
    std::function<bool(double done)> progress;
};

/** Where the time of one synthesis went, in seconds, and how long the speech is. */
struct Stats {
    double text = 0, sampling = 0, first_audio = 0, codec = 0;
    int tokens = 0, frames = 0;
    size_t samples = 0;
};

/**
 * The official runtime's cut where a sampled latent goes flat (find_flattening_point()), with its settings from the
 * model file: the first frame from which `window` frames, zeros past the end, have a standard deviation under
 * `std_threshold` and a mean within `mean_threshold` of zero, or the number of frames when none does.
 */
struct TailCut {
    int window = 0;
    double std_threshold = 0, mean_threshold = 0;

    explicit TailCut(const ModelFile & m);
    int flattening_point(const std::vector<float> & latent, int frames, int latent_dim) const;
};

/**
 * Text in, 48 kHz audio out, as the official runtime makes it: the normalized text's tokens, the text
 * condition and the predicted length, the sampler from seeded noise, the tail cut where the latent goes
 * flat, and the codec decoding window by window so that the first audio comes before the rest is decoded.
 */
class Synthesizer {
public:
    Synthesizer(const std::string & model_path, ggml_backend_t backend);
    ~Synthesizer();

    /**
     * A voice from a reference WAVE file or a voice file of this model's codec (voice-file.h). A voice file of another
     * codec throws.
     */
    Voice load_voice(const std::string & path);
    /** A voice from the latent of its reference, row-major [frames, latent_dim]. */
    Voice voice_from_latent(std::vector<float> latent);

    /** Speaks `r.text` in `voice`, passing the audio to `sink` window by window. Returns the samples. */
    size_t synthesize(const Request & r, const Voice & voice, const AudioSink & sink, Stats * stats = nullptr);

    int sample_rate() const { return codec_.sample_rate(); }
    const ModelFile & model() const { return *model_; }
    const Codec & codec() const { return codec_; }

    /** The decoder's first window, in frames; a short one brings the first audio early. */
    int first_window = 12;
    /** The decoder's later windows, in frames. */
    int window = 48;

private:
    ggml_backend_t backend_;
    std::unique_ptr<ModelFile> model_;
    Codec codec_;
    Tokenizer tokenizer_;
    TextEncoder text_;
    SpeakerEncoder speaker_;
    DurationPredictor duration_;
    Dit dit_;
    Sampler sampler_;
    ReferenceRules reference_;
    TailCut tail_;
    ggml_gallocr_t allocr_ = nullptr;
};

}  // namespace irodori
