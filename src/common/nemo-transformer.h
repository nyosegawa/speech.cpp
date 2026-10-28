#pragma once

#include <string>
#include <vector>

#include "ggml.h"
#include "model-file.h"

/**
 * NeMo's TransformerEncoder after its pre-encoder (nemo/collections/asr/modules/transformer_encoder.py, as the
 * references of Sortformer and Nemotron 3.5 Transcribe pin it), in rope mode with full attention: the embedding norm,
 * the layers, each a pre-LayerNorm attention over every frame and a pre-LayerNorm GELU feed-forward with residuals, and
 * the final norm. A layer's queries and keys are, where the model normalizes them, LayerNorm-ed per head, and then
 * rotated by their positions (RotaryPositionalEncoding: the first rotary_dim channels of each head in two halves, the
 * rest as they are).
 *
 * Its tensors are <prefix>embed_norm, <prefix>blk.<l>.attn_norm, attn_q, attn_k, attn_v, attn_q_norm and attn_k_norm
 * where the queries and keys are normalized, attn_out, ffn_norm, ffn_up and ffn_down, and <prefix>final_norm, each
 * LayerNorm with a weight and a bias, each Linear with a weight and, but for the queries, keys and values, a bias.
 */
class NemoTransformer {
public:
    struct Shape {
        int d_model = 0, heads = 0, layers = 0, rotary_dim = 0;
        float norm_eps = 0, rope_base = 0;
        bool qk_norm = false;
    };

    /** The encoder of the tensors under `prefix` of the model file `m`, which outlives it. */
    NemoTransformer(const ModelFile & m, std::string prefix, const Shape & shape);

    /**
     * Builds the encoder on `x`, [d_model, frames], the pre-encoder's output, with `positions`, I32 [frames], each
     * frame's position for the rotation. `layers`, when given, receives the embedding norm's output and each layer's,
     * [d_model, frames] each. Returns the final norm's output, [d_model, frames].
     */
    ggml_tensor * build(ggml_context * ctx, ggml_tensor * x, ggml_tensor * positions, std::vector<ggml_tensor *> * layers = nullptr) const;

    const Shape & shape() const { return shape_; }

private:
    ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;
    ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;

    const ModelFile & m_;
    std::string prefix_;
    Shape shape_;
};
