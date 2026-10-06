#include "encoder.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "layout.h"

namespace qwen3_asr {

/*
 * Activations are [channels, tokens] (ne0 = channels). The convolutions run on [time, mels, channels, chunks], the
 * layout of ggml's 2D convolutions with the official [chunks, channels, mels, time] read in reverse: time is their
 * width and the mel axis their height.
 */

namespace {

constexpr int kGraphSize = 8192;

}  // namespace

Encoder::Encoder(const ModelFile & m, ggml_backend_t backend)
    : m_(m),
      backend_(backend),
      mels_((int) m.u32("qwen3-asr.frontend.n_mels")),
      d_model_((int) m.u32("qwen3-asr.encoder.d_model")),
      heads_((int) m.u32("qwen3-asr.encoder.num_heads")),
      layers_((int) m.u32("qwen3-asr.encoder.num_layers")),
      chunk_frames_((int) m.u32("qwen3-asr.encoder.chunk_frames")),
      window_chunks_((int) (m.u32("qwen3-asr.encoder.window_frames") / m.u32("qwen3-asr.encoder.chunk_frames"))),
      output_dim_((int) m.u32("qwen3-asr.decoder.hidden_size")),
      channels_(m.tensor("enc.conv.1.weight")->ne[3]),
      chunk_tokens_(after_convolutions(chunk_frames_)),
      eps_(m.f32("qwen3-asr.encoder.norm_eps")),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {
    // SinusoidsPositionEmbedding computes the table in float32 from the increment of the log timescales, which numpy
    // computes in float64, so the angles round as they do here.
    const int half = d_model_ / 2;
    const float increment = (float) -(std::log((double) m.f32("qwen3-asr.encoder.max_timescale")) / (half - 1));
    positions_.resize((size_t) (chunk_tokens_ * d_model_));
    for (int64_t t = 0; t < chunk_tokens_; t++) {
        for (int i = 0; i < half; i++) {
            const float angle = (float) t * std::exp(increment * (float) i);
            positions_[(size_t) (t * d_model_ + i)] = std::sin(angle);
            positions_[(size_t) (t * d_model_ + half + i)] = std::cos(angle);
        }
    }
}

Encoder::~Encoder() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

int64_t Encoder::tokens(int64_t frames) const {
    return frames / chunk_frames_ * chunk_tokens_ + after_convolutions(frames % chunk_frames_);
}

std::vector<EncoderWindow> Encoder::windows(int64_t frames) const {
    std::vector<EncoderWindow> windows;
    const int64_t window_frames = (int64_t) window_chunks_ * chunk_frames_;
    for (int64_t first = 0; first < frames; first += window_frames) {
        EncoderWindow w;
        w.first_frame = first;
        w.frames = std::min(window_frames, frames - first);
        w.first_token = tokens(first);
        w.tokens = tokens(first + w.frames) - w.first_token;
        windows.push_back(w);
    }
    return windows;
}

ggml_tensor * Encoder::linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, mul_mat(ctx, m_.tensor(name + ".weight"), x), m_.tensor(name + ".bias"));
}

ggml_tensor * Encoder::layer_norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps_), m_.tensor(name + ".weight")), m_.tensor(name + ".bias"));
}

/**
 * A 3x3 convolution of stride 2 and padding 1 with its bias and GELU, from and to [time, mels, channels, chunks], as
 * im2col and one matrix product. On an Apple M5 it encoded 25.5 s with the 0.6B model in F16 in 0.13 s on Metal, where
 * ggml_conv_2d_direct() took 1.10 s; on the CPU both took 1.0 s (2026-10-06). ggml_conv_2d() does the same but rounds
 * the columns to half precision whatever the kernel's type, so this keeps them in the kernel's type: F32 for an F32
 * file, and F16 for the others, which the matrix product would round to half precision anyway.
 */
ggml_tensor * Encoder::convolution(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    ggml_tensor * kernel = m_.tensor(name + ".weight");
    // [taps * channels in, time, mels, chunks], the taps of each output position in a column.
    ggml_tensor * columns = ggml_im2col(ctx, kernel, x, 2, 2, 1, 1, 1, 1, true, kernel->type);
    const int64_t taps = columns->ne[0], time = columns->ne[1], mels = columns->ne[2], chunks = columns->ne[3];
    ggml_tensor * y = mul_mat(ctx, ggml_reshape_2d(ctx, kernel, taps, channels_), ggml_reshape_2d(ctx, columns, taps, time * mels * chunks));
    y = ggml_gelu_erf(ctx, ggml_add(ctx, y, m_.tensor(name + ".bias")));
    // [channels, time, mels, chunks] to [time, mels, channels, chunks].
    return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, y, channels_, time, mels, chunks), 2, 0, 1, 3));
}

/**
 * Qwen3ASRAudioEncoderLayer: multi-head attention over the window's tokens, without a mask, after a LayerNorm, and a
 * GELU feed-forward after another, each added to its input.
 */
ggml_tensor * Encoder::layer(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    const int64_t n = x->ne[1], dk = d_model_ / heads_;
    ggml_tensor * h = layer_norm(ctx, x, name + "attn_norm");
    // [d_k, heads, tokens] to [d_k, tokens, heads].
    auto heads = [&](ggml_tensor * y) { return ggml_permute(ctx, ggml_reshape_3d(ctx, y, dk, heads_, n), 0, 2, 1, 3); };
    ggml_tensor * q = heads(linear(ctx, h, name + "attn_q"));
    ggml_tensor * k = heads(linear(ctx, h, name + "attn_k"));
    // [d_k, heads, tokens] to [tokens, d_k, heads], so that the product with the weights contracts over the keys.
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, linear(ctx, h, name + "attn_v"), dk, heads_, n), 1, 2, 0, 3));
    ggml_tensor * weights = ggml_soft_max_ext(ctx, mul_mat(ctx, k, q), nullptr, 1.0f / std::sqrt((float) dk), 0.0f);
    ggml_tensor * out = mul_mat(ctx, vt, weights);
    out = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3)), d_model_, n);
    x = ggml_add(ctx, x, linear(ctx, out, name + "attn_out"));

    h = ggml_gelu_erf(ctx, linear(ctx, layer_norm(ctx, x, name + "ffn_norm"), name + "ffn_up"));
    return ggml_add(ctx, x, linear(ctx, h, name + "ffn_down"));
}

ggml_tensor * Encoder::build(Graph & g, const std::vector<float> & features, const EncoderWindow & w, EncoderStages * stages) const {
    ggml_context * ctx = g.ctx();
    const int64_t total = (int64_t) features.size() / mels_, chunks = (w.frames + chunk_frames_ - 1) / chunk_frames_;
    // The window's chunks as [time, mels, 1, chunks], the frames past the utterance zero.
    std::vector<float> input((size_t) (chunk_frames_ * mels_ * chunks), 0.0f);
    for (int64_t c = 0; c < chunks; c++) {
        for (int64_t t = 0; t < chunk_frames_; t++) {
            const int64_t frame = w.first_frame + c * chunk_frames_ + t;
            if (frame >= total) break;
            for (int m = 0; m < mels_; m++) input[(size_t) (((c * mels_) + m) * chunk_frames_ + t)] = features[(size_t) (frame * mels_ + m)];
        }
    }
    ggml_tensor * x = g.input(input, chunk_frames_, mels_, 1, chunks);
    for (int i = 1; i <= kConvLayers; i++) {
        x = convolution(ctx, x, "enc.conv." + std::to_string(i));
        if (stages) stages->convolutions.push_back(x);
    }
    // conv_out reads each token's [channels, mels] with the mel axis fastest: [time, mels, channels, chunks] to
    // [mels, channels, time, chunks].
    const int64_t mels = x->ne[1];
    x = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3)), mels * channels_, chunk_tokens_ * chunks);
    x = mul_mat(ctx, m_.tensor("enc.conv_out.weight"), x);
    x = ggml_add(ctx, ggml_reshape_3d(ctx, x, d_model_, chunk_tokens_, chunks), g.input(positions_, d_model_, chunk_tokens_));
    // The tokens of the padding of the utterance's last chunk end the window, and are dropped.
    x = ggml_view_2d(ctx, x, d_model_, w.tokens, x->nb[1], 0);
    // A copy: ggml's allocator frees the tensor a view reads once the view's last reader has run, even when the view
    // is a graph's output.
    if (stages) stages->input = ggml_cont(ctx, x);

    for (int l = 0; l < layers_; l++) {
        x = layer(ctx, x, "enc.blk." + std::to_string(l) + ".");
        if (stages) stages->layers.push_back(x);
    }
    x = layer_norm(ctx, x, "enc.norm");
    if (stages) stages->output = x;
    return linear(ctx, ggml_gelu_erf(ctx, linear(ctx, x, "proj.1")), "proj.2");
}

std::optional<std::vector<float>> Encoder::encode(const std::vector<float> & features, const std::function<bool(size_t windows)> & keep_going) {
    std::vector<float> out;
    const std::vector<EncoderWindow> all = windows((int64_t) features.size() / mels_);
    for (size_t i = 0; i < all.size(); i++) {
        Graph g(kGraphSize);
        ggml_tensor * embeds = build(g, features, all[i], nullptr);
        g.output(embeds);
        g.compute(backend_, allocr_);
        const std::vector<float> window = Graph::read(embeds);
        out.insert(out.end(), window.begin(), window.end());
        if (keep_going && !keep_going(i + 1)) return std::nullopt;
    }
    return out;
}

}  // namespace qwen3_asr
