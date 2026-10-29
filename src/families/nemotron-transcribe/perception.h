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
 * What the CTC alignment of a recognition reads of the perception module, as the official encoder keeps it
 * (CTCTimestampInputs): the ASR encoder's kept frames rounded to bfloat16, [frames, d_model] row-major; the speaker
 * branch's probabilities of the same frames, [frames, speakers]; and its 10 ms probabilities of the audio's feature
 * frames, [feature frames, speakers].
 */
struct AlignmentFrames {
    std::vector<uint16_t> asr;
    std::vector<float> speakers, fine;
    int64_t frames = 0, fine_frames = 0;
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
     * rows. `stages`, when given, receives what the module computed on the way, and `alignment` what the CTC alignment
     * reads of it.
     */
    std::vector<float> encode(const std::vector<float> & samples, const std::function<bool(double)> & progress = {},
                              PerceptionStages * stages = nullptr, AlignmentFrames * alignment = nullptr);

    /** The rows encode() gives that many samples. */
    int64_t frames(size_t samples) const;

    int sample_rate() const { return frontend_.sample_rate(); }
    int hidden() const { return hidden_; }

private:
    /**
     * The fusion of a window's `frames` kept frames, the ASR encoder's `asr` and the speaker branch's `probs`, row-major:
     * LayerNorm(asr) + scale · LayerNorm(the speakers over the threshold) · kernel, then the projection, each frame on its
     * own, so that a window at a time holds the memory of one window's frames whatever the recording's length. It adds
     * the fusion's output to `stages` when given.
     */
    std::vector<float> fuse(const std::vector<float> & asr, const std::vector<float> & probs, int64_t frames, PerceptionStages * stages);

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
