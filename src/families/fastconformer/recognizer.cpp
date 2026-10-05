#include "recognizer.h"

#include <stdexcept>

namespace fastconformer {

Recognizer::Recognizer(const std::string & path, ggml_backend_t backend)
    : backend_(backend),
      model_(path, backend),
      frontend_(model_),
      encoder_(model_),
      ctc_(model_),
      detokenizer_(model_),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {
    if (!allocr_) throw std::runtime_error("cannot create a graph allocator");
}

Recognizer::~Recognizer() {
    ggml_gallocr_free(allocr_);
}

std::vector<float> Recognizer::logits(const std::vector<float> & features, int64_t frames) {
    Graph g;
    ggml_tensor * out = ctc_.build(g, encoder_.build(g, features, frames));
    g.output(out);
    g.compute(backend_, allocr_);
    return Graph::read(out);
}

Transcript Recognizer::recognize(const std::vector<float> & samples) {
    const std::vector<float> features = frontend_.features(samples);
    Transcript t;
    t.ids = ctc_.greedy(logits(features, frontend_.frames(samples.size())));
    t.text = detokenizer_.text(t.ids);
    return t;
}

}  // namespace fastconformer
