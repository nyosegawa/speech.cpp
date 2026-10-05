#pragma once

#include <cstdint>
#include <vector>

#include "graph.h"
#include "model-file.h"

namespace fastconformer {

/** The LSTM state of the prediction network: h and c, [hidden, layers] each in ggml order. */
struct PredictionState {
    std::vector<float> h, c;
};

/**
 * NeMo's RNNTDecoder (nemo/collections/asr/modules/rnnt.py) with the blank as the embedding's padding: an
 * embedding of the tokens and the blank, whose blank row is zero, then stacked LSTM layers. One step takes the
 * last label and the state and gives the last layer's h. The blank fed first is the start of the sequence.
 */
class PredictionNetwork {
public:
    explicit PredictionNetwork(const ModelFile & m);

    /** The tensors of one step: the output [hidden] and the next state, [hidden, layers] each. */
    struct Step {
        ggml_tensor * output;
        ggml_tensor * h;
        ggml_tensor * c;
    };

    /** The state before the first label: zeros, as NeMo's LSTM starts when it is given none. */
    PredictionState initial_state() const;

    Step build(Graph & g, int32_t label, const PredictionState & state) const;

    /** The next state of a step that has been computed. */
    static PredictionState read_state(const Step & step);

    int hidden() const { return hidden_; }

private:
    const ModelFile & m_;
    int layers_, hidden_;
};

/**
 * NeMo's RNNTJoint: the encoder's frames and the prediction network's output each projected to the joint's width,
 * summed, a ReLU and a linear layer to the outputs, the tokens and the blank then, for TDT, one per duration.
 * NeMo applies a log-softmax over all outputs on the CPU only, which changes no argmax and is left out.
 */
class Joint {
public:
    explicit Joint(const ModelFile & m);

    /** The encoder's output [d_model, T] projected, [hidden, T]; computed once per utterance. */
    ggml_tensor * project_encoder(ggml_context * ctx, ggml_tensor * encoded) const;
    /** The prediction network's output [prediction hidden, n] projected, [hidden, n]. */
    ggml_tensor * project_prediction(ggml_context * ctx, ggml_tensor * prediction) const;

    /**
     * The logits of projected frames `f` ([hidden, n]) with projected predictions `g`, one for all frames ([hidden])
     * or one each ([hidden, n]), [outputs, n].
     */
    ggml_tensor * build(ggml_context * ctx, ggml_tensor * f, ggml_tensor * g) const;

    int hidden() const { return hidden_; }
    int outputs() const { return outputs_; }

private:
    const ModelFile & m_;
    int hidden_, outputs_;
};

}  // namespace fastconformer
