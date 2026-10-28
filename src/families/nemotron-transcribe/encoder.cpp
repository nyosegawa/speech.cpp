#include "encoder.h"

#include <algorithm>

#include "graph.h"

namespace nemotron_transcribe {

namespace {

NemoTransformer::Shape shape_of(const ModelFile & m) {
    const std::string e = "nemotron-transcribe.asr.";
    NemoTransformer::Shape s;
    s.d_model = m.size(e + "d_model");
    s.heads = m.size(e + "num_heads");
    s.layers = m.count(e + "num_layers");
    s.rotary_dim = m.size(e + "rotary_dim");
    s.norm_eps = m.f32(e + "norm_eps");
    s.rope_base = m.f32(e + "rope_base");
    s.qk_norm = true;
    return s;
}

}  // namespace

AsrEncoder::AsrEncoder(const ModelFile & m, ggml_backend_t backend)
    : m_(m),
      backend_(backend),
      factor_(m.size("nemotron-transcribe.asr.subsampling_factor")),
      mels_(m.size("nemotron-transcribe.frontend.n_mels")),
      encoder_(m, "asr.", shape_of(m)),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {}

AsrEncoder::~AsrEncoder() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

std::vector<float> AsrEncoder::encode(const float * features, int64_t frames, EncoderStages * stages) {
    const int64_t n = (frames + factor_ - 1) / factor_;
    // FeatureStacking: each 80 ms frame holds its factor frames of features one after another, the last zero-padded.
    std::vector<float> stacked((size_t) (n * factor_ * mels_), 0.0f);
    std::copy(features, features + frames * mels_, stacked.begin());
    std::vector<int32_t> positions((size_t) n);
    for (int64_t i = 0; i < n; i++) positions[(size_t) i] = (int32_t) i;

    Graph g;
    ggml_context * ctx = g.ctx();
    ggml_tensor * pre = mul_mat(ctx, m_.tensor("asr.pre_encode.weight"), g.input(stacked, (int64_t) factor_ * mels_, n));
    std::vector<ggml_tensor *> layers;
    ggml_tensor * out = encoder_.build(ctx, pre, g.input(positions, n), stages ? &layers : nullptr);
    g.output(out);
    if (stages) {
        g.output(pre);
        for (ggml_tensor * t : layers) g.output(t);
    }
    g.compute(backend_, allocr_);
    if (stages) {
        stages->pre_encode = Graph::read(pre);
        stages->layers.clear();
        for (ggml_tensor * t : layers) stages->layers.push_back(Graph::read(t));
    }
    return Graph::read(out);
}

}  // namespace nemotron_transcribe
