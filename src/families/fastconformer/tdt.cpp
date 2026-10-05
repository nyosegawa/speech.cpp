#include "tdt.h"

#include <memory>
#include <stdexcept>

namespace fastconformer {

namespace {

using Allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;

Allocator new_allocator(ggml_backend_t backend) {
    Allocator a(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), &ggml_gallocr_free);
    if (!a) throw std::runtime_error("cannot create a graph allocator");
    return a;
}

/** The index of the largest of `n` values; torch.max() and argmax() keep the first of equal values. */
int argmax(const float * v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

}  // namespace

TdtDecoder::TdtDecoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint)
    : prediction_(prediction),
      joint_(joint),
      blank_((int) m.u32("fastconformer.decoder.blank_id")),
      durations_(m.i32_array("fastconformer.decoder.tdt.durations")),
      max_symbols_(m.u32("fastconformer.decoder.tdt.max_symbols")) {
    if (joint_.outputs() != blank_ + 1 + (int) durations_.size()) {
        throw std::runtime_error("joint.out.weight does not have an output for each token, the blank and each duration");
    }
    for (int32_t d : durations_) {
        if (d < 0) throw std::runtime_error("fastconformer.decoder.tdt.durations holds a negative duration");
    }
    if (max_symbols_ == 0) throw std::runtime_error("fastconformer.decoder.tdt.max_symbols is 0");
}

std::vector<int32_t> TdtDecoder::decode(const std::vector<float> & projected, ggml_backend_t backend) const {
    const int hidden = joint_.hidden();
    const int64_t frames = (int64_t) (projected.size() / (size_t) hidden);
    auto frame = [&](int64_t t) {
        return std::vector<float>(projected.begin() + t * hidden, projected.begin() + (t + 1) * hidden);
    };
    // The step's graph and the joint's alone keep an allocator each, so that neither is planned again when the
    // other runs.
    const Allocator step_allocator = new_allocator(backend), joint_allocator = new_allocator(backend);

    std::vector<int32_t> ids;
    PredictionState state = prediction_.initial_state();
    int32_t label = blank_;
    int64_t t = 0, last_token_frame = -1;
    uint32_t tokens_on_frame = 0;
    while (t < frames) {
        // The prediction network takes the last label, and the joint is evaluated at the current frame with its
        // output, in one graph.
        Graph g(512);
        const PredictionNetwork::Step step = prediction_.build(g, label, state);
        ggml_tensor * predicted = joint_.project_prediction(g.ctx(), step.output);
        ggml_tensor * logits = joint_.build(g.ctx(), g.input(frame(t), hidden), predicted);
        g.output(logits);
        g.output(predicted);
        g.output(step.h);
        g.output(step.c);
        g.compute(backend, step_allocator.get());
        state = PredictionNetwork::read_state(step);
        const std::vector<float> prediction = Graph::read(predicted);
        std::vector<float> out = Graph::read(logits);

        int64_t label_frame;
        for (;;) {
            label = argmax(out.data(), blank_ + 1);
            int32_t duration = durations_[(size_t) argmax(out.data() + blank_ + 1, (int) durations_.size())];
            if (label == blank_ && duration == 0) duration = 1;
            label_frame = t;
            t += duration;
            if (label != blank_ || t >= frames) break;
            Graph j(64);
            ggml_tensor * next = joint_.build(j.ctx(), j.input(frame(t), hidden), j.input(prediction, hidden));
            j.output(next);
            j.compute(backend, joint_allocator.get());
            out = Graph::read(next);
        }
        if (label == blank_) break;
        ids.push_back(label);
        tokens_on_frame = label_frame == last_token_frame ? tokens_on_frame + 1 : 1;
        last_token_frame = label_frame;
        if (t < frames && t == label_frame && tokens_on_frame >= max_symbols_) t++;
    }
    return ids;
}

}  // namespace fastconformer
