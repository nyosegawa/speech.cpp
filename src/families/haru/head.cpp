#include "head.h"

#include <cmath>
#include <string>

#include "error.h"
#include "json.h"

namespace haru {

namespace {

/**
 * The sinusoidal features of a scalar, as the official ScalarEmbed computes them in float32: the cosines and then the
 * sines of value x base^(-i / half) for i < half.
 */
std::vector<float> features(float value, int64_t dim, float base) {
    const int64_t half = dim / 2;
    const float log_base = (float) -std::log((double) base);
    std::vector<float> f((size_t) dim);
    for (int64_t i = 0; i < half; i++) {
        const float arg = value * std::exp(log_base * (float) i / (float) half);
        f[(size_t) i] = std::cos(arg);
        f[(size_t) (half + i)] = std::sin(arg);
    }
    return f;
}

ggml_tensor * linear(ggml_context * ctx, const ModelFile & m, const std::string & name, ggml_tensor * x) {
    return mul_mat(ctx, m.tensor(name), x);
}

/** Rows [part x width, (part + 1) x width) of `y`, [parts x width, n], as a tensor of their own. */
ggml_tensor * part(ggml_context * ctx, ggml_tensor * y, int64_t width, int part) {
    return ggml_cont(ctx, ggml_view_2d(ctx, y, width, y->ne[1], y->nb[1], (size_t) part * width * ggml_element_size(y)));
}

/** x (1 + scale) + shift, (1 + scale) formed first as the official code forms it. */
ggml_tensor * modulate(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * shift) {
    return ggml_add(ctx, ggml_mul(ctx, x, ggml_scale_bias(ctx, scale, 1.0f, 1.0f)), shift);
}

}  // namespace

Head::Head(const ModelFile & m, ggml_backend_t backend) : m_(m), backend_(backend) {
    hidden_ = m.size("haru.text.hidden_size");
    frequencies_ = m.size("haru.head.frequency_dim");
    eps_ = m.f32("haru.head.norm_eps");
    interval_ = m.f32("haru.head.interval_embed");
    frequency_base_ = m.f32("haru.head.frequency_base");
    latent_scale_ = m.f32("haru.latent_scale");
    latent_bias_ = m.f32("haru.latent_bias");
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!allocr_) throw Error(Fault::OutOfMemory, "cannot create a graph allocator for the head");
    set_guidance(m.f32("haru.head.guidance_scale"));
}

Head::~Head() { ggml_gallocr_free(allocr_); }

void Head::set_guidance(float guidance) {
    const float lo = m_.f32("haru.head.guidance_min"), hi = m_.f32("haru.head.guidance_max");
    if (!(guidance >= lo && guidance <= hi)) {
        throw Error(Fault::OutOfRange, "the guidance must be from " + json_number(lo) + " to " + json_number(hi), "guidance_scale");
    }
    Graph g;
    ggml_context * ctx = g.ctx();
    const float values[4] = {0.0f, interval_, 1.0f, 1.0f - 1.0f / guidance};
    const char * names[4] = {"t_embed", "dt_embed", "h_embed", "omega_embed"};
    ggml_tensor * embeds[4];
    for (int i = 0; i < 4; i++) {
        const std::string e = std::string("head.") + names[i];
        ggml_tensor * f = g.input(features(values[i], frequencies_, frequency_base_), frequencies_);
        embeds[i] = linear(ctx, m_, e + ".fc2", ggml_silu(ctx, linear(ctx, m_, e + ".fc1", f)));
        g.output(embeds[i]);
    }
    g.compute(backend_, allocr_);
    constants_.clear();
    for (ggml_tensor * e : embeds) constants_.push_back(Graph::read(e));
    guidance_ = guidance;
}

ggml_tensor * Head::build(Graph & g, ggml_tensor * hidden, ggml_tensor * noise, std::vector<ggml_tensor *> * layers, ggml_tensor ** cond) const {
    ggml_context * ctx = g.ctx();
    const int64_t h = hidden_;
    // The projected hidden state plus each embedding in turn, the order in which the official code sums them.
    ggml_tensor * c = linear(ctx, m_, "head.cond_proj", hidden);
    for (const std::vector<float> & e : constants_) c = ggml_add(ctx, c, g.input(e, h));
    c = ggml_silu(ctx, c);
    if (cond) *cond = c;
    ggml_tensor * x = linear(ctx, m_, "head.latent_proj", noise);
    const int count = m_.count("haru.head.num_layers");
    for (int i = 0; i < count; i++) {
        const std::string p = "head.blk." + std::to_string(i) + ".";
        ggml_tensor * ada = linear(ctx, m_, p + "ada", c);
        ggml_tensor * n = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps_), m_.tensor(p + "norm"));
        ggml_tensor * in = modulate(ctx, n, part(ctx, ada, h, 1), part(ctx, ada, h, 0));
        ggml_tensor * f = linear(ctx, m_, p + "ffn_down", ggml_mul(ctx, ggml_silu(ctx, linear(ctx, m_, p + "ffn_gate", in)), linear(ctx, m_, p + "ffn_up", in)));
        x = ggml_add(ctx, x, ggml_mul(ctx, part(ctx, ada, h, 2), f));
        if (layers) layers->push_back(x);
    }
    ggml_tensor * ada = linear(ctx, m_, "head.final.ada", c);
    ggml_tensor * velocity = linear(ctx, m_, "head.final.proj", modulate(ctx, ggml_rms_norm(ctx, x, eps_), part(ctx, ada, h, 1), part(ctx, ada, h, 0)));
    return ggml_add(ctx, noise, velocity);
}

ggml_tensor * build_logits(Graph & g, const ModelFile & m, ggml_tensor * hidden) { return mul_mat(g.ctx(), m.tensor("token_embd"), hidden); }

ggml_tensor * build_silent_logits(Graph & g, const ModelFile & m, ggml_tensor * hidden) {
    return mul_mat(g.ctx(), m.tensor("output_silent"), hidden);
}

}  // namespace haru
