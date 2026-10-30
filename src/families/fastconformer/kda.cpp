#include "kda.h"

namespace fastconformer {

namespace {

ggml_tensor * layer_norm(ggml_context * ctx, const ModelFile & m, ggml_tensor * x, const std::string & name, float eps) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), m.tensor(name + ".weight")), m.tensor(name + ".bias"));
}

/**
 * fused_recurrent_kda()'s normalization of q and k: each head [rows of ne0] divided by the square root of its squares'
 * sum plus `eps`, which ggml_l2_norm() would instead bound below by eps.
 */
ggml_tensor * qk_norm(ggml_context * ctx, ggml_tensor * x, float eps) {
    return ggml_div(ctx, x, ggml_sqrt(ctx, ggml_scale_bias(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, x)), 1.0f, eps)));
}

}  // namespace

KdaLayers::KdaLayers(const ModelFile & m)
    : m_(m),
      layers_((int) m.u32("fastconformer.decoder.prediction_layers")),
      heads_(m.u32("fastconformer.decoder.kda.num_heads")),
      head_dim_(m.width("pred.kda.0.o_norm.weight", 0)),
      conv_size_(m.width("pred.kda.0.q_conv.weight", 0)),
      norm_eps_(m.f32("fastconformer.decoder.kda.norm_eps")),
      gate_norm_eps_(m.f32("fastconformer.decoder.kda.gate_norm_eps")),
      qk_norm_eps_(m.f32("fastconformer.decoder.kda.qk_norm_eps")),
      final_norm_(m.boolean("fastconformer.decoder.kda.final_norm")) {}

ggml_tensor * KdaLayers::build(ggml_context * ctx, ggml_tensor * x, ggml_tensor * state, ggml_tensor ** next) const {
    const int64_t d = head_dim_, h = heads_, width = h * d, past = conv_size_ - 1;
    const size_t f = sizeof(float);
    ggml_tensor * out = nullptr;
    auto keep = [&](ggml_tensor * t) {
        t = ggml_reshape_1d(ctx, ggml_cont(ctx, t), ggml_nelements(t));
        out = out ? ggml_concat(ctx, out, t, 0) : t;
    };
    for (int l = 0; l < layers_; l++) {
        const std::string p = "pred.kda." + std::to_string(l) + ".";
        const size_t base = (size_t) (l * layer_state()) * f;
        ggml_tensor * normed = layer_norm(ctx, m_, x, p + "norm", norm_eps_);
        auto weight = [&](const std::string & name) { return m_.tensor(p + name); };

        // Each convolution takes its projection after the inputs before it, [conv_size, width], whose last
        // conv_size - 1 are the next state's.
        ggml_tensor * windows[3];
        ggml_tensor * convolved[3];
        const char * names[3] = {"q", "k", "v"};
        for (int i = 0; i < 3; i++) {
            ggml_tensor * before = ggml_view_2d(ctx, state, past, width, (size_t) past * f, base + (size_t) (width * d + i * past * width) * f);
            ggml_tensor * projected = mul_mat(ctx, weight(std::string(names[i]) + ".weight"), normed);
            windows[i] = ggml_concat(ctx, before, ggml_reshape_2d(ctx, projected, 1, width), 0);
            convolved[i] = ggml_silu(ctx, ggml_ssm_conv(ctx, ggml_reshape_3d(ctx, windows[i], conv_size_, width, 1), weight(std::string(names[i]) + "_conv.weight")));
        }
        ggml_tensor * q = qk_norm(ctx, ggml_reshape_2d(ctx, convolved[0], d, h), qk_norm_eps_);
        ggml_tensor * k = qk_norm(ctx, ggml_reshape_2d(ctx, convolved[1], d, h), qk_norm_eps_);
        ggml_tensor * v = ggml_reshape_2d(ctx, convolved[2], d, h);

        ggml_tensor * gate = mul_mat(ctx, weight("f_b.weight"), mul_mat(ctx, weight("f_a.weight"), normed));
        gate = ggml_softplus(ctx, ggml_add(ctx, ggml_reshape_2d(ctx, gate, d, h), ggml_reshape_2d(ctx, weight("dt_bias"), d, h)));
        gate = ggml_mul(ctx, gate, ggml_neg(ctx, ggml_exp(ctx, ggml_reshape_2d(ctx, weight("a_log"), 1, h))));
        ggml_tensor * beta = ggml_sigmoid(ctx, ggml_reshape_2d(ctx, mul_mat(ctx, weight("b.weight"), normed), 1, h));
        ggml_tensor * recurrent = ggml_view_4d(ctx, state, d, d, h, 1, d * f, d * d * f, width * d * f, base);
        ggml_tensor * attended =
            ggml_gated_delta_net(ctx, ggml_reshape_4d(ctx, q, d, h, 1, 1), ggml_reshape_4d(ctx, k, d, h, 1, 1), ggml_reshape_4d(ctx, v, d, h, 1, 1),
                                 ggml_reshape_4d(ctx, gate, d, h, 1, 1), ggml_reshape_4d(ctx, beta, 1, h, 1, 1), recurrent, 1);
        // The result holds the output of the heads, [head_dim, heads], then the next recurrent state.
        keep(ggml_view_1d(ctx, attended, width * d, (size_t) width * f));
        for (ggml_tensor * w : windows) keep(ggml_view_2d(ctx, w, past, width, w->nb[1], f));

        ggml_tensor * o = ggml_rms_norm(ctx, ggml_view_2d(ctx, attended, d, h, d * f, 0), gate_norm_eps_);
        ggml_tensor * output_gate = ggml_add(ctx, mul_mat(ctx, weight("g_b.weight"), mul_mat(ctx, weight("g_a.weight"), normed)), weight("g_b.bias"));
        o = ggml_mul(ctx, ggml_mul(ctx, o, weight("o_norm.weight")), ggml_sigmoid(ctx, ggml_reshape_2d(ctx, output_gate, d, h)));
        x = ggml_add(ctx, x, mul_mat(ctx, weight("o.weight"), ggml_reshape_1d(ctx, o, width)));
    }
    *next = out;
    return final_norm_ ? layer_norm(ctx, m_, x, "pred.norm", norm_eps_) : x;
}

std::vector<std::string> KdaLayers::matrices() const {
    std::vector<std::string> names;
    for (int l = 0; l < layers_; l++) {
        for (const char * part : {"q", "k", "v", "f_a", "f_b", "b", "g_a", "g_b", "o"}) {
            names.push_back("pred.kda." + std::to_string(l) + "." + part + ".weight");
        }
    }
    return names;
}

}  // namespace fastconformer
