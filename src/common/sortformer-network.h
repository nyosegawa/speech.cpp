#pragma once

#include <cstdint>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"
#include "nemo-transformer.h"

namespace sortformer {

/** Intermediate results of one step's encoder, for the checks. */
struct StepStages {
    /** The embedding norm's output and each layer's, [d_model, frames] each. */
    std::vector<ggml_tensor *> layers;
    /** The final norm's output, [d_model, frames]. */
    ggml_tensor * hidden = nullptr;
    /** encoder_proj's output, [hidden, frames]. */
    ggml_tensor * proj = nullptr;
    /** The upsampling's output, [hidden, factor · frames]. */
    ggml_tensor * upsampled = nullptr;
};

/** What one step gives: the chunk's frames pre-encoded, [frames, d_model] row-major, and the probabilities. */
struct StepResult {
    std::vector<float> pre_encoded;
    /** The speakers' probabilities every 10 ms of the step's whole input, [factor · rows, speakers] row-major. */
    std::vector<float> probs;
};

/** The network's shape, which each family reads from its keys. */
struct NetworkShape {
    int d_model = 0, heads = 0, layers = 0, factor = 0, speakers = 0, mels = 0;
    float norm_eps = 0, rope_base = 0;
};

/**
 * Nemotron 3 Diarization's network for one streaming step, as NeMo runs it (SortformerEncLabelModel.forward_streaming_step()
 * in nemo/collections/asr/models/sortformer_diar_models.py, at the commit reference/sortformer pins).
 *
 * The chunk's features, factor (8) frames of 10 ms stacked into each of its 80 ms frames, the last padded with zeros,
 * are projected to d_model (FeatureStacking) and follow the speaker cache's and the FIFO queue's rows, which are such
 * frames from earlier chunks. The encoder normalizes them, runs its layers, each a pre-LayerNorm attention over every
 * frame, its queries and keys rotated by their positions in the step's input (RoPE, the halves of each head turned),
 * and a pre-LayerNorm GELU feed-forward, and normalizes their output. encoder_proj narrows it to the head's width, a
 * convolution over three frames widens each frame to factor frames of 10 ms, and two linear layers after ReLUs and a
 * sigmoid give each speaker's probability.
 */
class Network {
public:
    /**
     * The network of `shape` whose tensors are those under `prefix` of the model file `m`, which outlives it, computing
     * on `backend`: <prefix>pre_encode, the encoder's (NemoTransformer), <prefix>encoder_proj, <prefix>upsample and
     * <prefix>head.hidden and head.out.
     */
    Network(const ModelFile & m, ggml_backend_t backend, const NetworkShape & shape, std::string prefix);
    ~Network();
    Network(const Network &) = delete;
    Network & operator=(const Network &) = delete;

    /**
     * Builds the probabilities, [speakers, factor · (rows + frames)], of a step whose input is `rows` rows of the cache
     * and the queue, [rows, d_model] row-major, followed by the `frames` frames of `stacked`, [frames, factor · mels]
     * row-major. `pre` receives the frames pre-encoded, [d_model, frames], and `stages` the encoder's stages.
     */
    ggml_tensor * build(Graph & g, const std::vector<float> & state, int64_t rows, const std::vector<float> & stacked, int64_t frames,
                        ggml_tensor ** pre, StepStages * stages = nullptr) const;

    /** Computes a step as build() makes it. */
    StepResult step(const std::vector<float> & state, int64_t rows, const std::vector<float> & stacked, int64_t frames);

    int d_model() const { return shape_.d_model; }
    int factor() const { return shape_.factor; }
    int speakers() const { return shape_.speakers; }
    int mels() const { return shape_.mels; }

private:
    ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;

    const ModelFile & m_;
    ggml_backend_t backend_;
    NetworkShape shape_;
    std::string prefix_;
    int hidden_;
    NemoTransformer encoder_;
    ggml_gallocr_t allocr_;
};

}  // namespace sortformer
