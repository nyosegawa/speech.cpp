#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "graph.h"
#include "model-file.h"

namespace fastconformer {

/** Intermediate results of the encoder, for the checks. */
struct EncoderStages {
    /** The subsampling's output before the input scaling, [d_model, frames / factor]. */
    ggml_tensor * subsampled = nullptr;
    /** The output of each conformer layer, [d_model, frames / factor] each. */
    std::vector<ggml_tensor *> layers;
};

/**
 * NeMo's ConformerEncoder with dw_striding subsampling and relative positional self-attention over the whole
 * utterance (att_context_size [-1, -1]).
 *
 * ConformerEncoder.forward_internal() (nemo/collections/asr/modules/conformer_encoder.py) runs on a batch padded to
 * its longest utterance and masks the padding out of every convolution and attention; MaskedConvSequential
 * (parts/submodules/subsampling.py) zeroes the subsampling's padded frames between its layers. Every valid output
 * depends on the valid frames and on zeros alone, so this runs on the valid frames of one utterance.
 */
class Encoder {
public:
    explicit Encoder(const ModelFile & m);

    /** The encoder's output for `features` ([frames, mels] row-major), [d_model, subsampled_frames(frames)]. */
    ggml_tensor * build(Graph & g, const std::vector<float> & features, int64_t frames, EncoderStages * stages = nullptr) const;

    /** The frames after subsampling: calc_length() of the three stride-2 convolutions. */
    int64_t subsampled_frames(int64_t frames) const;

    int d_model() const { return d_model_; }

private:
    ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;
    ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;
    ggml_tensor * subsample(Graph & g, const std::vector<float> & features, int64_t frames) const;
    ggml_tensor * feed_forward(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;
    ggml_tensor * attention(Graph & g, ggml_tensor * x, ggml_tensor * pos, const std::string & name) const;
    ggml_tensor * convolution(Graph & g, ggml_tensor * x, const std::string & name) const;

    const ModelFile & m_;
    int mels_, d_model_, layers_, heads_, conv_kernel_, sub_layers_;
    float eps_, pos_base_, xscale_, ff_factor_;
};

}  // namespace fastconformer
