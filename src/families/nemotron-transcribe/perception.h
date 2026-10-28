#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "encoder.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "model-file.h"
#include "nemo-frontend.h"
#include "speakers.h"

namespace nemotron_transcribe {

/** What the perception module computed, for the checks, each with its 80 ms frames as rows, row-major. */
struct PerceptionStages {
    /** The ASR encoder's kept frames of every window, [frames, d_model]. */
    std::vector<float> asr;
    /** The speaker branch's probabilities of the same frames, [frames, speakers]. */
    std::vector<float> speakers;
    /** The fusion's output, [frames, d_model]. */
    std::vector<float> fused;
};

/**
 * The perception module of Nemotron 3.5 Transcribe as SALMAutomodel.generate() runs it, with the parallel expert encoder
 * in its windowed inference (ParallelExpertEncoder._forward_online_impl()): NeMo's log-mel features; windows of
 * window_frames 80 ms frames with window_left and window_right frames of context on either side, each through the ASR
 * encoder on the features normalized per feature over the whole recording, of which the window's own frames are kept,
 * and through a streaming step of the speaker branch on the features as they are; the speakers whose probability passes
 * the activity threshold, normalized over the speakers and turned to the encoder's width by the sinusoidal kernel, added
 * to the normalized ASR frames; and the projection to the language model's width.
 *
 * The official preprocessor gives the audio one frame more than its length, zero, which every attention masks out and
 * the language model never takes; the windows here hold the audio's frames alone, which gives the same frames.
 */
class Perception {
public:
    /** The perception module of the model file `m`, which outlives it, computing on `backend`. */
    Perception(const ModelFile & m, ggml_backend_t backend);
    ~Perception();
    Perception(const Perception &) = delete;
    Perception & operator=(const Perception &) = delete;

    /**
     * The audio's rows of the language model's input of mono samples at sample_rate(), [frames(), hidden] row-major.
     * `progress` hears each window's end, from 0 to 1; false stops the work, and what it returns is then not the audio's
     * rows. `stages`, when given, receives what the module computed on the way.
     */
    std::vector<float> encode(const std::vector<float> & samples, const std::function<bool(double)> & progress = {},
                              PerceptionStages * stages = nullptr);

    /** The rows encode() gives that many samples. */
    int64_t frames(size_t samples) const;

    int sample_rate() const { return frontend_.sample_rate(); }
    int hidden() const { return hidden_; }

private:
    const ModelFile & m_;
    ggml_backend_t backend_;
    NemoFrontend frontend_;
    AsrEncoder asr_;
    SpeakerBranch speakers_;
    int window_, left_, right_, hidden_;
    float threshold_, scale_, eps_;
    ggml_gallocr_t allocr_;
};

}  // namespace nemotron_transcribe
