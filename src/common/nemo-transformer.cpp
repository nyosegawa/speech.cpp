#include "nemo-transformer.h"

#include <cmath>
#include <utility>

#include "graph.h"

/*
 * Activations are [channels, frames] (ne0 = channels), the official [frames, channels] of a batch of one.
 */

NemoTransformer::NemoTransformer(const ModelFile & m, std::string prefix, const Shape & shape)
    : m_(m), prefix_(std::move(prefix)), shape_(shape) {}

ggml_tensor * NemoTransformer::layer_norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, shape_.norm_eps), m_.tensor(prefix_ + name + ".weight")),
                    m_.tensor(prefix_ + name + ".bias"));
}

ggml_tensor * NemoTransformer::linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, mul_mat(ctx, m_.tensor(prefix_ + name + ".weight"), x), m_.tensor(prefix_ + name + ".bias"));
}

ggml_tensor * NemoTransformer::build(ggml_context * ctx, ggml_tensor * x, ggml_tensor * positions, std::vector<ggml_tensor *> * layers) const {
    const int d = shape_.d_model, heads = shape_.heads;
    const int64_t n = x->ne[1], head_dim = d / heads;
    x = layer_norm(ctx, x, "embed_norm");
    if (layers) layers->push_back(x);
    for (int l = 0; l < shape_.layers; l++) {
        const std::string b = "blk." + std::to_string(l) + ".";
        ggml_tensor * h = layer_norm(ctx, x, b + "attn_norm");
        ggml_tensor * q = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(prefix_ + b + "attn_q.weight"), h), head_dim, heads, n);
        ggml_tensor * k = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(prefix_ + b + "attn_k.weight"), h), head_dim, heads, n);
        ggml_tensor * v = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(prefix_ + b + "attn_v.weight"), h), head_dim, heads, n);
        if (shape_.qk_norm) {
            q = layer_norm(ctx, q, b + "attn_q_norm");
            k = layer_norm(ctx, k, b + "attn_k_norm");
        }
        // NEOX rotation turns each pair (i, i + rotary_dim / 2) of the first rotary_dim channels, as _rotate_half() pairs
        // them, and leaves the rest of the head as it is.
        q = ggml_rope_ext(ctx, q, positions, nullptr, shape_.rotary_dim, GGML_ROPE_TYPE_NEOX, 0, shape_.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, positions, nullptr, shape_.rotary_dim, GGML_ROPE_TYPE_NEOX, 0, shape_.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor * weights = ggml_soft_max_ext(ctx, mul_mat(ctx, kh, qh), nullptr, 1.0f / std::sqrt((float) head_dim), 0.0f);
        ggml_tensor * o = mul_mat(ctx, vt, weights);
        o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)), d, n);
        x = ggml_add(ctx, x, linear(ctx, o, b + "attn_out"));
        h = ggml_gelu_erf(ctx, linear(ctx, layer_norm(ctx, x, b + "ffn_norm"), b + "ffn_up"));
        x = ggml_add(ctx, x, linear(ctx, h, b + "ffn_down"));
        if (layers) layers->push_back(x);
    }
    return layer_norm(ctx, x, "final_norm");
}
