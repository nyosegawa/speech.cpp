#include "transducer.h"

#include <stdexcept>
#include <string>

namespace fastconformer {

namespace {

ggml_tensor * linear(ggml_context * ctx, const ModelFile & m, ggml_tensor * x, const std::string & name) {
    return ggml_add(ctx, mul_mat(ctx, m.tensor(name + ".weight"), x), m.tensor(name + ".bias"));
}

}  // namespace

PredictionNetwork::PredictionNetwork(const ModelFile & m)
    : m_(m), layers_((int) m.u32("fastconformer.prediction.num_layers")), hidden_((int) m.tensor("pred.embed.weight")->ne[0]) {
    if (m.tensor("pred.embed.weight")->ne[1] != (int64_t) m.u32("fastconformer.blank_id") + 1) {
        throw std::runtime_error("pred.embed.weight does not have a row for each token and the blank");
    }
    for (int l = 0; l < layers_; l++) {
        const std::string p = "pred.lstm." + std::to_string(l);
        if (m.tensor(p + ".ih.weight")->ne[1] != 4 * hidden_ || m.tensor(p + ".hh.weight")->ne[0] != hidden_) {
            throw std::runtime_error(p + " does not have four gates of the embedding's width");
        }
    }
}

PredictionState PredictionNetwork::initial_state() const {
    const size_t n = (size_t) hidden_ * (size_t) layers_;
    return {std::vector<float>(n, 0.0f), std::vector<float>(n, 0.0f)};
}

/**
 * torch.nn.LSTM's step: gates = W_ih x + W_hh h + b, stacked as input, forget, cell and output;
 * c' = sigmoid(f) c + sigmoid(i) tanh(g) and h' = sigmoid(o) tanh(c'). The converter sums b_ih and b_hh.
 */
PredictionNetwork::Step PredictionNetwork::build(Graph & g, int32_t label, const PredictionState & state) const {
    ggml_context * ctx = g.ctx();
    const size_t width = (size_t) hidden_ * sizeof(float);
    ggml_tensor * h_in = g.input(state.h, hidden_, layers_);
    ggml_tensor * c_in = g.input(state.c, hidden_, layers_);
    ggml_tensor * x = ggml_get_rows(ctx, m_.tensor("pred.embed.weight"), g.input(std::vector<int32_t>{label}, 1));
    ggml_tensor * h_out = nullptr;
    ggml_tensor * c_out = nullptr;
    for (int l = 0; l < layers_; l++) {
        const std::string p = "pred.lstm." + std::to_string(l);
        ggml_tensor * h = ggml_view_1d(ctx, h_in, hidden_, (size_t) l * width);
        ggml_tensor * c = ggml_view_1d(ctx, c_in, hidden_, (size_t) l * width);
        ggml_tensor * gates = ggml_add(ctx, ggml_add(ctx, mul_mat(ctx, m_.tensor(p + ".ih.weight"), x), mul_mat(ctx, m_.tensor(p + ".hh.weight"), h)),
                                       m_.tensor(p + ".bias"));
        auto gate = [&](int i) { return ggml_view_1d(ctx, gates, hidden_, (size_t) i * width); };
        ggml_tensor * next_c = ggml_add(ctx, ggml_mul(ctx, ggml_sigmoid(ctx, gate(1)), c), ggml_mul(ctx, ggml_sigmoid(ctx, gate(0)), ggml_tanh(ctx, gate(2))));
        ggml_tensor * next_h = ggml_mul(ctx, ggml_sigmoid(ctx, gate(3)), ggml_tanh(ctx, next_c));
        h_out = h_out ? ggml_concat(ctx, h_out, next_h, 0) : next_h;
        c_out = c_out ? ggml_concat(ctx, c_out, next_c, 0) : next_c;
        x = next_h;
    }
    return {x, h_out, c_out};
}

PredictionState PredictionNetwork::read_state(const Step & step) {
    return {Graph::read(step.h), Graph::read(step.c)};
}

Joint::Joint(const ModelFile & m)
    : m_(m), hidden_((int) m.tensor("joint.out.weight")->ne[0]), outputs_((int) m.tensor("joint.out.weight")->ne[1]) {}

ggml_tensor * Joint::project_encoder(ggml_context * ctx, ggml_tensor * encoded) const {
    return linear(ctx, m_, encoded, "joint.enc");
}

ggml_tensor * Joint::project_prediction(ggml_context * ctx, ggml_tensor * prediction) const {
    return linear(ctx, m_, prediction, "joint.pred");
}

ggml_tensor * Joint::build(ggml_context * ctx, ggml_tensor * f, ggml_tensor * g) const {
    return linear(ctx, m_, ggml_relu(ctx, ggml_add(ctx, f, g)), "joint.out");
}

}  // namespace fastconformer
