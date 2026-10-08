#include "network.h"

#include <string>
#include <utility>

namespace silero_vad {

namespace {

/**
 * Columns [rows, 1] joined into [rows, n] in order, in pairs and pairs of pairs, so that each value is copied as often as
 * the pairs are deep, not once for each column after it.
 */
ggml_tensor * stacked(ggml_context * ctx, std::vector<ggml_tensor *> columns) {
    while (columns.size() > 1) {
        std::vector<ggml_tensor *> next;
        for (size_t i = 0; i + 1 < columns.size(); i += 2) next.push_back(ggml_concat(ctx, columns[i], columns[i + 1], 1));
        if (columns.size() % 2) next.push_back(columns.back());
        columns = std::move(next);
    }
    return columns[0];
}

}  // namespace

Network::Network(const ModelFile & m)
    : m_(m),
      chunk_((int) m.u32("silero-vad.chunk_size")),
      context_((int) m.u32("silero-vad.context_size")),
      n_fft_((int) m.u32("silero-vad.stft.n_fft")),
      hop_((int) m.u32("silero-vad.stft.hop_length")),
      reflect_((int) m.u32("silero-vad.stft.reflect")),
      frames_((context_ + chunk_ + reflect_ - n_fft_) / hop_ + 1),
      hidden_((int) m.tensor("lstm.hh.weight")->ne[0]),
      strides_(m.i32_array("silero-vad.encoder.strides")),
      padding_(m.i32_array("silero-vad.encoder.padding")) {}

ggml_tensor * Network::stft(ggml_context * ctx, ggml_tensor * input) const {
    const int64_t n = input->ne[1];
    ggml_tensor * padded = ggml_pad_reflect_1d(ctx, input, 0, reflect_);
    // The STFT is the official's convolution of one channel with the basis at a stride of a hop: ggml_im2col() cuts the
    // overlapping frames, which a view cannot, ggml refusing a view larger than its source.
    ggml_tensor * basis = m_.tensor("stft.basis");
    ggml_tensor * frames = ggml_im2col(ctx, ggml_reshape_3d(ctx, basis, n_fft_, 1, basis->ne[1]), ggml_reshape_3d(ctx, padded, padded->ne[0], 1, n), hop_, 0,
                                       0, 0, 1, 0, false, GGML_TYPE_F32);
    frames = ggml_reshape_2d(ctx, frames, n_fft_, frames_ * n);
    // The basis holds the real parts' filters in its first n_fft / 2 + 1 rows and the imaginary parts' in the rest.
    ggml_tensor * real = mul_mat(ctx, ggml_view_2d(ctx, basis, n_fft_, bins(), basis->nb[1], 0), frames);
    ggml_tensor * imag = mul_mat(ctx, ggml_view_2d(ctx, basis, n_fft_, bins(), basis->nb[1], (size_t) bins() * basis->nb[1]), frames);
    ggml_tensor * magnitude = ggml_sqrt(ctx, ggml_add(ctx, ggml_sqr(ctx, real), ggml_sqr(ctx, imag)));
    return ggml_reshape_3d(ctx, magnitude, bins(), frames_, n);
}

ggml_tensor * Network::block(ggml_context * ctx, int i, ggml_tensor * x) const {
    const std::string name = "encoder." + std::to_string(i);
    ggml_tensor * w = m_.tensor(name + ".weight");
    const int64_t n = x->ne[2];
    // ggml_im2col() takes the frames on the first axis and gives, for each output frame, the window of every channel,
    // [channels * width, frames', n], in the order of the kernel's [width, channels, out].
    ggml_tensor * by_frame = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    ggml_tensor * columns = ggml_im2col(ctx, w, by_frame, strides_[i], 0, padding_[i], 0, 1, 0, false, GGML_TYPE_F32);
    const int64_t out_frames = columns->ne[1];
    ggml_tensor * y = mul_mat(ctx, ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]),
                              ggml_reshape_2d(ctx, columns, columns->ne[0], out_frames * n));
    y = ggml_relu(ctx, ggml_add(ctx, y, m_.tensor(name + ".bias")));
    return ggml_reshape_3d(ctx, y, w->ne[2], out_frames, n);
}

ggml_tensor * Network::encode(ggml_context * ctx, ggml_tensor * magnitude, std::vector<ggml_tensor *> * blocks) const {
    ggml_tensor * x = magnitude;
    for (int i = 0; i < this->blocks(); i++) {
        x = block(ctx, i, x);
        if (blocks) blocks->push_back(x);
    }
    return ggml_reshape_2d(ctx, x, x->ne[0], x->ne[2]);
}

/**
 * torch.nn.LSTMCell's step: gates = W_ih x + W_hh h + b, stacked as input, forget, cell and output;
 * c' = sigmoid(f) c + sigmoid(i) tanh(g) and h' = sigmoid(o) tanh(c'). The converter sums b_ih and b_hh, and the input's
 * part of every chunk's gates is one product before the steps.
 */
Network::Steps Network::lstm(ggml_context * ctx, ggml_tensor * encoded, ggml_tensor * h0, ggml_tensor * c0, bool keep_c) const {
    const int64_t n = encoded->ne[1];
    const size_t width = (size_t) hidden_ * sizeof(float);
    ggml_tensor * inputs = ggml_add(ctx, mul_mat(ctx, m_.tensor("lstm.ih.weight"), encoded), m_.tensor("lstm.bias"));
    ggml_tensor * hh = m_.tensor("lstm.hh.weight");
    ggml_tensor * h = h0;
    ggml_tensor * c = c0;
    std::vector<ggml_tensor *> hs, cs;
    for (int64_t t = 0; t < n; t++) {
        ggml_tensor * gates = ggml_add(ctx, ggml_view_1d(ctx, inputs, 4 * hidden_, (size_t) t * inputs->nb[1]), mul_mat(ctx, hh, h));
        // The sigmoid of the cell's gate goes unused, which costs less than a node of its own for each of the other three.
        ggml_tensor * s = ggml_sigmoid(ctx, gates);
        auto gate = [&](ggml_tensor * x, int i) { return ggml_view_1d(ctx, x, hidden_, (size_t) i * width); };
        c = ggml_add(ctx, ggml_mul(ctx, gate(s, 1), c), ggml_mul(ctx, gate(s, 0), ggml_tanh(ctx, gate(gates, 2))));
        h = ggml_mul(ctx, gate(s, 3), ggml_tanh(ctx, c));
        hs.push_back(h);
        if (keep_c) cs.push_back(c);
    }
    return {stacked(ctx, hs), keep_c ? stacked(ctx, cs) : nullptr, h, c};
}

ggml_tensor * Network::decode(ggml_context * ctx, ggml_tensor * h) const {
    ggml_tensor * logits = ggml_add(ctx, mul_mat(ctx, m_.tensor("decoder.weight"), ggml_relu(ctx, h)), m_.tensor("decoder.bias"));
    return ggml_reshape_1d(ctx, ggml_sigmoid(ctx, logits), h->ne[1]);
}

}  // namespace silero_vad
