#include "encoder.h"

#include <cmath>
#include <stdexcept>

namespace fastconformer {

/*
 * Activations are [channels, frames] (ne0 = channels). The subsampling runs on [mels, frames, channels], the
 * layout of ggml's 2D convolutions, whose width is the mel axis and height the time axis.
 */

Encoder::Encoder(const ModelFile & m)
    : m_(m),
      mels_((int) m.u32("fastconformer.n_mels")),
      d_model_((int) m.u32("fastconformer.d_model")),
      layers_((int) m.u32("fastconformer.num_layers")),
      heads_((int) m.u32("fastconformer.num_heads")),
      conv_kernel_((int) m.u32("fastconformer.conv_kernel")),
      eps_(m.f32("fastconformer.norm_eps")),
      pos_base_(m.f32("fastconformer.pos_base")),
      xscale_(m.f32("fastconformer.xscale")),
      ff_factor_(m.f32("fastconformer.ff_factor")) {
    const uint32_t factor = m.u32("fastconformer.subsampling_factor");
    sub_layers_ = 0;
    for (uint32_t f = factor; f > 1; f /= 2) sub_layers_++;
    if (factor < 2 || (1u << sub_layers_) != factor) throw std::runtime_error("fastconformer.subsampling_factor is not a power of two");
    if (d_model_ % heads_ != 0) throw std::runtime_error("fastconformer.d_model is not a multiple of fastconformer.num_heads");
    if (conv_kernel_ % 2 != 1) throw std::runtime_error("fastconformer.conv_kernel is not odd");
}

int64_t Encoder::subsampled_frames(int64_t frames) const {
    for (int i = 0; i < sub_layers_; i++) frames = (frames - 1) / 2 + 1;
    return frames;
}

ggml_tensor * Encoder::linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, mul_mat(ctx, m_.tensor(name + ".weight"), x), m_.tensor(name + ".bias"));
}

ggml_tensor * Encoder::layer_norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps_), m_.tensor(name + ".weight")), m_.tensor(name + ".bias"));
}

ggml_tensor * Encoder::subsample(Graph & g, const std::vector<float> & features, int64_t frames) const {
    ggml_context * ctx = g.ctx();
    auto channel_bias = [&](ggml_tensor * x, const std::string & name) {
        ggml_tensor * b = m_.tensor(name);
        return ggml_add(ctx, x, ggml_reshape_3d(ctx, b, 1, 1, ggml_nelements(b)));
    };
    // ConvSubsampling "dw_striding" (nemo/collections/asr/parts/submodules/subsampling.py): a 3x3 convolution of
    // stride 2, then (depthwise 3x3 of stride 2, pointwise) for each further halving, each followed by a ReLU.
    ggml_tensor * x = g.input(features, mels_, frames);
    x = ggml_conv_2d_direct(ctx, m_.tensor("sub.conv.0.weight"), x, 2, 2, 1, 1, 1, 1);
    x = ggml_relu(ctx, channel_bias(x, "sub.conv.0.bias"));
    for (int i = 1; i < sub_layers_; i++) {
        const std::string p = "sub.conv." + std::to_string(i);
        x = channel_bias(ggml_conv_2d_dw_direct(ctx, m_.tensor(p + ".dw.weight"), x, 2, 2, 1, 1, 1, 1), p + ".dw.bias");
        const int64_t w = x->ne[0], h = x->ne[1], c = x->ne[2];
        // [mels, frames, channels] to [channels, mels * frames] for the pointwise convolution.
        x = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3)), c, w * h);
        x = ggml_relu(ctx, linear(ctx, x, p + ".pw"));
        x = ggml_reshape_3d(ctx, x, c, w, h);
        if (i + 1 < sub_layers_) x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
    }
    // ConvSubsampling.forward() flattens (channels, mels) with the mel axis fastest.
    const int64_t c = x->ne[0], w = x->ne[1], h = x->ne[2];
    x = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)), w * c, h);
    return linear(ctx, x, "sub.out");
}

ggml_tensor * Encoder::feed_forward(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    ggml_tensor * h = ggml_silu(ctx, linear(ctx, layer_norm(ctx, x, name + "_norm"), name + "_up"));
    return ggml_add(ctx, x, ggml_scale(ctx, linear(ctx, h, name + "_down"), ff_factor_));
}

/**
 * RelPositionMultiHeadAttention.forward() (nemo/collections/asr/parts/submodules/multi_head_attention.py):
 *
 *     scores = ((q + u) k^T + rel_shift((q + v) p^T)) / sqrt(d_k)
 *
 * where p is linear_pos of the encodings of the relative positions T - 1 down to -(T - 1). rel_shift() pads a
 * zero column on the left, reads the [T, 2T] result as [2T, T] and drops its first row; in the padded rows' flat
 * order, element (i, j) of the result is element T + i (2T - 1) + j, so it is a strided view. The zero column is
 * concatenated, since Metal's ggml_pad_ext() pads on the right only.
 */
ggml_tensor * Encoder::attention(Graph & g, ggml_tensor * x, ggml_tensor * pos, const std::string & name) const {
    ggml_context * ctx = g.ctx();
    const int64_t t = x->ne[1], dk = d_model_ / heads_;
    auto heads = [&](ggml_tensor * y, int64_t n) {
        return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, y, dk, heads_, n), 0, 2, 1, 3));
    };
    ggml_tensor * h = layer_norm(ctx, x, name + "_norm");
    ggml_tensor * q = ggml_reshape_3d(ctx, linear(ctx, h, name + "_q"), dk, heads_, t);
    ggml_tensor * k = heads(linear(ctx, h, name + "_k"), t);
    ggml_tensor * v = ggml_reshape_3d(ctx, linear(ctx, h, name + "_v"), dk, heads_, t);
    ggml_tensor * p = heads(mul_mat(ctx, m_.tensor(name + "_pos.weight"), pos), 2 * t - 1);
    ggml_tensor * qu = heads(ggml_add(ctx, q, m_.tensor(name + "_pos_bias_u")), t);
    ggml_tensor * qv = heads(ggml_add(ctx, q, m_.tensor(name + "_pos_bias_v")), t);

    ggml_tensor * ac = mul_mat(ctx, k, qu);
    ggml_tensor * bd = mul_mat(ctx, p, qv);
    bd = ggml_concat(ctx, g.input(std::vector<float>((size_t) (t * heads_), 0.0f), 1, t, heads_), bd, 0);
    bd = ggml_view_3d(ctx, bd, t, t, heads_, (size_t) (2 * t - 1) * sizeof(float), bd->nb[2], (size_t) t * sizeof(float));
    ggml_tensor * scores = ggml_add(ctx, ac, ggml_cont(ctx, bd));
    ggml_tensor * weights = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt((float) dk), 0.0f);

    // [d_k, heads, T] to [T, d_k, heads], so that the product contracts over the keys.
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    ggml_tensor * out = mul_mat(ctx, vt, weights);
    out = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3)), d_model_, t);
    return ggml_add(ctx, x, linear(ctx, out, name + "_out"));
}

/**
 * ConformerConvolution.forward() (nemo/collections/asr/parts/submodules/conformer_modules.py): pointwise to twice
 * the channels, GLU, a depthwise convolution over time with symmetric zero padding, batch norm (folded into the
 * depthwise convolution by the converter), Swish and pointwise back. The depthwise convolution runs as
 * ggml_ssm_conv(), which takes time as the fastest axis and slides the kernel over its input without padding, so
 * the zeros are concatenated on both sides.
 */
ggml_tensor * Encoder::convolution(Graph & g, ggml_tensor * x, const std::string & name) const {
    ggml_context * ctx = g.ctx();
    const int64_t t = x->ne[1], pad = (conv_kernel_ - 1) / 2;
    ggml_tensor * h = layer_norm(ctx, x, name + "_norm");
    h = ggml_mul(ctx, linear(ctx, h, name + "_pw1_a"), ggml_sigmoid(ctx, linear(ctx, h, name + "_pw1_gate")));
    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = ggml_concat(ctx, g.zeros(pad, d_model_), h, 0);
    h = ggml_concat(ctx, h, g.zeros(pad, d_model_), 0);
    h = ggml_ssm_conv(ctx, ggml_reshape_3d(ctx, h, t + 2 * pad, d_model_, 1), m_.tensor(name + "_dw.weight"));
    h = ggml_silu(ctx, ggml_add(ctx, ggml_reshape_2d(ctx, h, d_model_, t), m_.tensor(name + "_dw.bias")));
    return ggml_add(ctx, x, linear(ctx, h, name + "_pw2"));
}

ggml_tensor * Encoder::build(Graph & g, const std::vector<float> & features, int64_t frames, EncoderStages * stages) const {
    ggml_context * ctx = g.ctx();
    ggml_tensor * x = subsample(g, features, frames);
    if (stages) stages->subsampled = x;
    const int64_t t = x->ne[1];
    if (xscale_ != 1.0f) x = ggml_scale(ctx, x, xscale_);

    // RelPositionalEncoding.extend_pe() and create_pe() (multi_head_attention.py): the encodings of the positions
    // T - 1 down to -(T - 1), sines in the even channels and cosines in the odd. create_pe() computes them in
    // float32, so the angles round as they do here.
    std::vector<float> pe((size_t) ((2 * t - 1) * d_model_));
    const float step = (float) (-std::log((double) pos_base_) / d_model_);
    for (int64_t c = 0; c < 2 * t - 1; c++) {
        const float position = (float) (t - 1 - c);
        for (int i = 0; i < d_model_; i += 2) {
            const float angle = position * std::exp((float) i * step);
            pe[(size_t) (c * d_model_ + i)] = std::sin(angle);
            pe[(size_t) (c * d_model_ + i + 1)] = std::cos(angle);
        }
    }
    ggml_tensor * pos = g.input(pe, d_model_, 2 * t - 1);

    // ConformerLayer.forward() (conformer_modules.py): a half-step feed-forward, attention, convolution, another
    // half-step feed-forward, each on a pre-norm residual, and a final norm.
    for (int l = 0; l < layers_; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        x = feed_forward(ctx, x, p + "ff1");
        x = attention(g, x, pos, p + "attn");
        x = convolution(g, x, p + "conv");
        x = feed_forward(ctx, x, p + "ff2");
        x = layer_norm(ctx, x, p + "out_norm");
        if (stages) stages->layers.push_back(x);
    }
    return x;
}

}  // namespace fastconformer
