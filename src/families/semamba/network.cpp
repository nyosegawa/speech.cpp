#include "network.h"

#include <cstdint>
#include <string>
#include <vector>

namespace semamba {

namespace {

/**
 * A matrix product whose operands stay in float32 on every device: RE-USE's activations reach 7.07e4 before its output
 * projection, past the 65504 of half precision, in which Metal and Vulkan otherwise stage the operands of an F32 matrix.
 */
ggml_tensor * mul_mat_f32(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * t = mul_mat(ctx, a, b);
    ggml_prec_set_src(t, GGML_PREC_F32, 1);
    return t;
}

}  // namespace

Network::Network(const ModelFile & m)
    : m_(m),
      hidden_((int) m.width("enc.in.conv.weight", 3)),
      depth_(m.count("semamba.dense_depth")),
      frame_stride_(m.size("semamba.frame_stride")),
      bin_stride_(m.size("semamba.bin_stride")),
      blocks_(m.count("semamba.block_count")),
      padding_((int) m.u32("semamba.input_padding")),
      instance_eps_(m.f32("semamba.instance_norm_eps")),
      layer_eps_(m.f32("semamba.layer_norm_eps")) {}

ggml_tensor * Network::conv(ggml_context * ctx, const std::string & name, ggml_tensor * x, int s0, int s1, int p0, int p1, int d1) const {
    ggml_tensor * k = w(name + ".weight");
    ggml_tensor * y = ggml_conv_2d_direct(ctx, k, x, s0, s1, p0, p1, 1, d1);
    return ggml_add(ctx, y, ggml_reshape_4d(ctx, w(name + ".bias"), 1, 1, k->ne[3], 1));
}

ggml_tensor * Network::norm_prelu(ggml_context * ctx, const std::string & name, ggml_tensor * x) const {
    // InstanceNorm2d normalizes each channel over its frames and bins.
    const int64_t c = x->ne[2];
    ggml_tensor * y = ggml_norm(ctx, ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], c), instance_eps_);
    y = ggml_reshape_4d(ctx, y, x->ne[0], x->ne[1], c, 1);
    y = ggml_add(ctx, ggml_mul(ctx, y, ggml_reshape_4d(ctx, w(name + ".norm.weight"), 1, 1, c, 1)), ggml_reshape_4d(ctx, w(name + ".norm.bias"), 1, 1, c, 1));
    // PReLU of a slope for each channel: max(0, y) + a min(0, y).
    ggml_tensor * a = ggml_reshape_4d(ctx, w(name + ".prelu"), 1, 1, c, 1);
    return ggml_sub(ctx, ggml_relu(ctx, y), ggml_mul(ctx, ggml_relu(ctx, ggml_neg(ctx, y)), a));
}

ggml_tensor * Network::dense(ggml_context * ctx, const std::string & name, ggml_tensor * x) const {
    // Layer i sees every output before it, newest first, through a 3x3 kernel dilated 2^i frames and padded to keep the size.
    ggml_tensor * skip = x;
    ggml_tensor * y = x;
    for (int i = 0; i < depth_; i++) {
        const std::string layer = name + "." + std::to_string(i);
        const int dilation = 1 << i;
        y = norm_prelu(ctx, layer, conv(ctx, layer + ".conv", skip, 1, 1, 1, dilation, dilation));
        skip = ggml_concat(ctx, y, skip, 2);
    }
    return y;
}

ggml_tensor * Network::encode(ggml_context * ctx, ggml_tensor * input) const {
    // SEMamba.forward() pads the input with zeros after its last bin and its last frame.
    ggml_tensor * x = ggml_pad(ctx, ggml_reshape_4d(ctx, input, input->ne[0], input->ne[1], input->ne[2], 1), padding_, padding_, 0, 0);
    x = norm_prelu(ctx, "enc.in", conv(ctx, "enc.in.conv", x, 1, 1, 0, 0, 1));
    x = dense(ctx, "enc.dense", x);
    return norm_prelu(ctx, "enc.out", conv(ctx, "enc.out.conv", x, bin_stride_, frame_stride_, 0, 0, 1));
}

ggml_tensor * Network::mamba(ggml_context * ctx, const std::string & prefix, ggml_tensor * x, bool reverse) const {
    ggml_tensor * xz = mul_mat_f32(ctx, w(prefix + ".in_proj.weight"), x);
    ggml_tensor * a = ggml_neg(ctx, ggml_exp(ctx, w(prefix + ".A_log")));
    ggml_tensor * y = ggml_mamba1(ctx, xz, w(prefix + ".conv1d.weight"), w(prefix + ".conv1d.bias"), w(prefix + ".x_proj.weight"),
                                  w(prefix + ".dt_proj.weight"), w(prefix + ".dt_proj.bias"), a, w(prefix + ".D"), reverse);
    return mul_mat_f32(ctx, w(prefix + ".out_proj.weight"), y);
}

ggml_tensor * Network::bidirectional(ggml_context * ctx, const std::string & prefix, ggml_tensor * x) const {
    ggml_tensor * forward = ggml_add(ctx, mamba(ctx, prefix + ".fw", x, false), x);
    ggml_tensor * backward = ggml_add(ctx, mamba(ctx, prefix + ".bw", x, true), x);
    ggml_tensor * y = mul_mat_f32(ctx, w(prefix + ".proj.weight"), ggml_concat(ctx, forward, backward, 0));
    y = ggml_add(ctx, y, w(prefix + ".proj.bias"));
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, y, layer_eps_), w(prefix + ".norm.weight")), w(prefix + ".norm.bias"));
}

ggml_tensor * Network::block(ggml_context * ctx, int b, ggml_tensor * x) const {
    const std::string name = "blk." + std::to_string(b);
    // The products and additions read the transposed views as they are, which saves copying the activations twice a block.
    x = ggml_add(ctx, bidirectional(ctx, name + ".time", x), x);
    x = ggml_permute(ctx, x, 0, 2, 1, 3);
    x = ggml_add(ctx, bidirectional(ctx, name + ".freq", x), x);
    return ggml_permute(ctx, x, 0, 2, 1, 3);
}

ggml_tensor * Network::up(ggml_context * ctx, const std::string & name, ggml_tensor * x, int r) const {
    // SPConvTranspose2d pads one bin at each side, convolves to r times the channels, the r groups of channels one
    // after another, and interleaves the groups along the bins: bin j of group g goes to bin j r + g.
    ggml_tensor * y = conv(ctx, name + ".conv", x, 1, 1, 1, 0, 1);
    const int64_t bins = y->ne[0], frames = y->ne[1], c = y->ne[2] / r;
    y = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, y, bins, frames, c, r), 1, 2, 3, 0));
    return norm_prelu(ctx, name, ggml_reshape_4d(ctx, y, r * bins, frames, c, 1));
}

ggml_tensor * Network::decode(ggml_context * ctx, const std::string & decoder, ggml_tensor * x) const {
    ggml_tensor * y = dense(ctx, decoder + ".dense", x);
    y = up(ctx, decoder + ".up1", y, bin_stride_);
    // The second up-sampling runs on the transpose, along the frames.
    y = ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));
    y = up(ctx, decoder + ".up2", y, frame_stride_);
    return ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));
}

ggml_tensor * Network::project(ggml_context * ctx, const std::string & name, ggml_tensor * x) const {
    return conv(ctx, name, x, 1, 1, 0, 0, 1);
}

Network::Outputs Network::forward(Graph & g, const std::vector<float> & magnitude, const std::vector<float> & phase, int64_t frames, int64_t bins,
                                  bool keep_blocks) const {
    ggml_context * ctx = g.ctx();
    std::vector<float> input(magnitude);
    input.insert(input.end(), phase.begin(), phase.end());
    ggml_tensor * x = encode(ctx, g.input(input, bins, frames, 2));
    Outputs out{};
    out.encoded = x;
    // The blocks run on [hidden, frames', bins'].
    ggml_tensor * h = ggml_cont(ctx, ggml_permute(ctx, x, 2, 1, 0, 3));
    for (int b = 0; b < blocks_; b++) {
        h = block(ctx, b, h);
        if (keep_blocks) out.blocks.push_back(ggml_cont(ctx, ggml_permute(ctx, h, 2, 1, 0, 3)));
    }
    x = ggml_cont(ctx, ggml_permute(ctx, h, 2, 1, 0, 3));
    out.magnitude = project(ctx, "mag.out", decode(ctx, "mag", x));
    ggml_tensor * phase_x = decode(ctx, "pha", x);
    out.real = project(ctx, "pha.r", phase_x);
    out.imag = project(ctx, "pha.i", phase_x);
    return out;
}

}  // namespace semamba
