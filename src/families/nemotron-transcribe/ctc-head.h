#pragma once

#include <cstdint>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "model-file.h"
#include "nemo-transformer.h"

namespace nemotron_transcribe {

/** What the CTC head computed on one window, for the checks, each [frames, d_model] row-major over the window's frames. */
struct CtcHeadStages {
    /** The transposed convolution's output. */
    std::vector<float> upsampled;
    /** The embedding norm's output and each transformer layer's. */
    std::vector<std::vector<float>> layers;
    /** The transformer's final norm. */
    std::vector<float> output;
    /** The upsampled frames plus the scaled transformer output, which the classifier reads. */
    std::vector<float> trunk;
};

/**
 * The CTC timestamp head of Nemotron 3.5 Transcribe (TransformerCTCDecoder of NeMo Speech's ctc_timestamp_utils.py), as
 * MultiSpeakerSOTWordTimestampAligner runs it over the ASR encoder's kept frames: windows of the encoder's window_frames
 * frames with window_left and window_right frames of context, each upsampled by the transposed convolution of kernel
 * and stride upsampling_factor, through NeMo's Transformer encoder, every frame of the window attending to every other,
 * added back to the upsampled frames scaled by residual_scale, and classified by a 1x1 convolution and a log-softmax over
 * the vocabulary and the blank, of which the window's own frames are kept.
 */
class CtcHead {
public:
    /** The head of the model file `m`, which outlives it, computing on `backend`. */
    CtcHead(const ModelFile & m, ggml_backend_t backend);
    ~CtcHead();
    CtcHead(const CtcHead &) = delete;
    CtcHead & operator=(const CtcHead &) = delete;

    /**
     * The log probabilities of the classes `classes` at each CTC frame of `frames` frames of the ASR encoder,
     * `states`, [frames, d_model] row-major, which the official encoder stores in bfloat16 for the head:
     * [frames * upsampling(), classes.size()] row-major. `stages`, when given, receives what each window computed.
     */
    std::vector<float> log_probs(const std::vector<uint16_t> & states, int64_t frames, const std::vector<int32_t> & classes,
                                 std::vector<CtcHeadStages> * stages = nullptr);

    /** The classes of the head: a piece of the CTC vocabulary for each but the last, the blank. */
    int classes() const { return classes_; }
    int upsampling() const { return upsampling_; }

private:
    const ModelFile & m_;
    ggml_backend_t backend_;
    NemoTransformer transformer_;
    int d_, classes_, upsampling_, window_, left_, right_;
    float residual_scale_;
    ggml_gallocr_t allocr_;
};

}  // namespace nemotron_transcribe
