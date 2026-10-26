#include "network.h"

#include <cmath>
#include <string>

namespace sortformer {

/*
 * Activations are [channels, frames] (ne0 = channels), the official [frames, channels] of a batch of one.
 */

namespace {

constexpr int kGraphSize = 4096;

}  // namespace

Network::Network(const ModelFile & m, ggml_backend_t backend)
    : m_(m),
      backend_(backend),
      d_((int) m.u32("sortformer.encoder.d_model")),
      heads_((int) m.u32("sortformer.encoder.num_heads")),
      layers_((int) m.u32("sortformer.encoder.num_layers")),
      factor_((int) m.u32("sortformer.encoder.subsampling_factor")),
      speakers_((int) m.u32("sortformer.speakers")),
      mels_((int) m.u32("sortformer.frontend.n_mels")),
      hidden_((int) m.tensor("encoder_proj.weight")->ne[1]),
      eps_(m.f32("sortformer.encoder.norm_eps")),
      base_(m.f32("sortformer.encoder.rope_base")),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {}

Network::~Network() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

ggml_tensor * Network::layer_norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps_), m_.tensor(name + ".weight")), m_.tensor(name + ".bias"));
}

ggml_tensor * Network::linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, mul_mat(ctx, m_.tensor(name + ".weight"), x), m_.tensor(name + ".bias"));
}

ggml_tensor * Network::build(Graph & g, const std::vector<float> & state, int64_t rows, const std::vector<float> & stacked, int64_t frames,
                             ggml_tensor ** pre, StepStages * stages) const {
    ggml_context * ctx = g.ctx();
    const int64_t n = rows + frames, head_dim = d_ / heads_;

    ggml_tensor * chunk = mul_mat(ctx, m_.tensor("pre_encode.weight"), g.input(stacked, (int64_t) factor_ * mels_, frames));
    *pre = chunk;
    ggml_tensor * x = rows > 0 ? ggml_concat(ctx, g.input(state, d_, rows), chunk, 1) : chunk;

    // RotaryPositionalEncoding rotates by the position of each frame in the step's input, the cache's first.
    std::vector<int32_t> positions((size_t) n);
    for (int64_t i = 0; i < n; i++) positions[(size_t) i] = (int32_t) i;
    ggml_tensor * pos = g.input(positions, n);

    x = layer_norm(ctx, x, "embed_norm");
    if (stages) stages->layers.push_back(x);
    for (int l = 0; l < layers_; l++) {
        const std::string b = "blk." + std::to_string(l) + ".";
        ggml_tensor * h = layer_norm(ctx, x, b + "attn_norm");
        ggml_tensor * q = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(b + "attn_q.weight"), h), head_dim, heads_, n);
        ggml_tensor * k = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(b + "attn_k.weight"), h), head_dim, heads_, n);
        ggml_tensor * v = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(b + "attn_v.weight"), h), head_dim, heads_, n);
        q = ggml_rope_ext(ctx, q, pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, 0, base_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, 0, base_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor * weights = ggml_soft_max_ext(ctx, mul_mat(ctx, kh, qh), nullptr, 1.0f / std::sqrt((float) head_dim), 0.0f);
        ggml_tensor * o = mul_mat(ctx, vt, weights);
        o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)), d_, n);
        x = ggml_add(ctx, x, linear(ctx, o, b + "attn_out"));
        h = ggml_gelu_erf(ctx, linear(ctx, layer_norm(ctx, x, b + "ffn_norm"), b + "ffn_up"));
        x = ggml_add(ctx, x, linear(ctx, h, b + "ffn_down"));
        if (stages) stages->layers.push_back(x);
    }
    x = layer_norm(ctx, x, "final_norm");
    if (stages) stages->hidden = x;
    x = linear(ctx, x, "encoder_proj");
    if (stages) stages->proj = x;

    // The upsampling Conv1d of width 3, padded by a zero frame on either side, as one product: each frame stacked
    // after the frame before it and before the frame after it, as the converter stacks the kernel's taps.
    ggml_tensor * zero = g.zeros(hidden_, 1);
    ggml_tensor * before = n > 1 ? ggml_concat(ctx, zero, ggml_view_2d(ctx, x, hidden_, n - 1, x->nb[1], 0), 1) : zero;
    ggml_tensor * after = n > 1 ? ggml_concat(ctx, ggml_view_2d(ctx, x, hidden_, n - 1, x->nb[1], x->nb[1]), zero, 1) : zero;
    ggml_tensor * up = linear(ctx, ggml_concat(ctx, ggml_concat(ctx, before, x, 0), after, 0), "upsample");
    // Each frame's channels are factor frames of the head's width, the earliest first.
    up = ggml_reshape_2d(ctx, up, hidden_, (int64_t) factor_ * n);
    if (stages) stages->upsampled = up;

    ggml_tensor * y = linear(ctx, ggml_relu(ctx, up), "head.hidden");
    y = linear(ctx, ggml_relu(ctx, y), "head.out");
    return ggml_sigmoid(ctx, y);
}

StepResult Network::step(const std::vector<float> & state, int64_t rows, const std::vector<float> & stacked, int64_t frames) {
    Graph g(kGraphSize);
    ggml_tensor * pre = nullptr;
    ggml_tensor * probs = build(g, state, rows, stacked, frames, &pre);
    g.output(pre);
    g.output(probs);
    g.compute(backend_, allocr_);
    return {Graph::read(pre), Graph::read(probs)};
}

}  // namespace sortformer
