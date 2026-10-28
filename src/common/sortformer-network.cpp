#include "sortformer-network.h"

#include <cmath>
#include <string>

namespace sortformer {

/*
 * Activations are [channels, frames] (ne0 = channels), the official [frames, channels] of a batch of one.
 */

namespace {

constexpr int kGraphSize = 4096;

/** The encoder's shape: every channel of a head rotated, and no normalization of its queries and keys. */
NemoTransformer::Shape encoder_shape(const NetworkShape & n) {
    NemoTransformer::Shape s;
    s.d_model = n.d_model;
    s.heads = n.heads;
    s.layers = n.layers;
    s.rotary_dim = n.d_model / n.heads;
    s.norm_eps = n.norm_eps;
    s.rope_base = n.rope_base;
    return s;
}

}  // namespace

Network::Network(const ModelFile & m, ggml_backend_t backend, const NetworkShape & shape, std::string prefix)
    : m_(m),
      backend_(backend),
      shape_(shape),
      prefix_(std::move(prefix)),
      hidden_((int) m.tensor(prefix_ + "encoder_proj.weight")->ne[1]),
      encoder_(m, prefix_, encoder_shape(shape)),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {}

Network::~Network() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

ggml_tensor * Network::linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, mul_mat(ctx, m_.tensor(prefix_ + name + ".weight"), x), m_.tensor(prefix_ + name + ".bias"));
}

ggml_tensor * Network::build(Graph & g, const std::vector<float> & state, int64_t rows, const std::vector<float> & stacked, int64_t frames,
                             ggml_tensor ** pre, StepStages * stages) const {
    ggml_context * ctx = g.ctx();
    const int64_t n = rows + frames;

    const int64_t factor = shape_.factor, d = shape_.d_model;
    ggml_tensor * chunk = mul_mat(ctx, m_.tensor(prefix_ + "pre_encode.weight"), g.input(stacked, factor * shape_.mels, frames));
    *pre = chunk;
    ggml_tensor * x = rows > 0 ? ggml_concat(ctx, g.input(state, d, rows), chunk, 1) : chunk;

    // RotaryPositionalEncoding rotates by the position of each frame in the step's input, the cache's first.
    std::vector<int32_t> positions((size_t) n);
    for (int64_t i = 0; i < n; i++) positions[(size_t) i] = (int32_t) i;
    ggml_tensor * pos = g.input(positions, n);

    x = encoder_.build(ctx, x, pos, stages ? &stages->layers : nullptr);
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
    up = ggml_reshape_2d(ctx, up, hidden_, factor * n);
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
