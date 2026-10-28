#pragma once

#include <cstdint>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "model-file.h"
#include "nemo-transformer.h"

namespace nemotron_transcribe {

/** What one window of the ASR encoder computed, for the checks, each [frames, d_model] row-major. */
struct EncoderStages {
    /** The stacked frames' projection, before the embedding norm. */
    std::vector<float> pre_encode;
    /** The embedding norm's output and each layer's. */
    std::vector<std::vector<float>> layers;
};

/**
 * The ASR encoder of the parallel expert encoder (ParallelExpertEncoder.asr_encoder in NeMo Speech): FeatureStacking,
 * factor frames of features to each 80 ms frame, the last padded with zeros, projected to d_model without a bias, and
 * NeMo's Transformer encoder over them, each head's queries and keys normalized and the first rotary_dim channels of
 * each head rotated, every frame of the window attending to every frame of it.
 */
class AsrEncoder {
public:
    /** The encoder of the model file `m`, which outlives it, computing on `backend`. */
    AsrEncoder(const ModelFile & m, ggml_backend_t backend);
    ~AsrEncoder();
    AsrEncoder(const AsrEncoder &) = delete;
    AsrEncoder & operator=(const AsrEncoder &) = delete;

    /**
     * The output of `frames` frames of features normalized per feature, [frames, mels] row-major: [ceil(frames / factor),
     * d_model] row-major. `stages`, when given, receives what the encoder computed on the way.
     */
    std::vector<float> encode(const float * features, int64_t frames, EncoderStages * stages = nullptr);

    int d_model() const { return encoder_.shape().d_model; }
    int factor() const { return factor_; }

private:
    const ModelFile & m_;
    ggml_backend_t backend_;
    int factor_, mels_;
    NemoTransformer encoder_;
    ggml_gallocr_t allocr_;
};

}  // namespace nemotron_transcribe
