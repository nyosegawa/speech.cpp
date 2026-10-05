#include "recognizer.h"

#include <stdexcept>

#include "layout.h"

namespace fastconformer {

Recognizer::Recognizer(const std::string & path, ggml_backend_t backend)
    : backend_(backend),
      model_(path, backend, layout),
      frontend_(model_),
      encoder_(model_),
      prediction_(model_),
      joint_(model_),
      decoder_(make_decoder(model_, prediction_, joint_)),
      detokenizer_(model_),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {
    if (!allocr_) throw std::runtime_error("cannot create a graph allocator");
}

Recognizer::~Recognizer() {
    ggml_gallocr_free(allocr_);
}

std::vector<float> Recognizer::encode(const std::vector<float> & features, int64_t frames) {
    Graph g;
    ggml_tensor * out = joint_.project_encoder(g.ctx(), encoder_.build(g, features, frames));
    g.output(out);
    g.compute(backend_, allocr_);
    return Graph::read(out);
}

Transcript Recognizer::recognize(const std::vector<float> & samples) {
    const std::vector<float> features = frontend_.features(samples);
    Transcript t;
    t.decoding = decoding(encode(features, frontend_.frames(samples.size())));
    t.text = detokenizer_.text(t.decoding.ids);
    return t;
}

double Recognizer::seconds(int64_t frame) const {
    const double window_stride = (double) frontend_.hop_length() / (double) frontend_.sample_rate();
    return (double) frame * window_stride * (double) encoder_.subsampling_factor();
}

}  // namespace fastconformer
