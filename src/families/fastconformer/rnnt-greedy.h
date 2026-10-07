#pragma once

#include <cstdint>
#include <vector>

#include "decoder.h"

namespace fastconformer {

/**
 * NeMo's greedy decoding of an RNN-T for one utterance as change_decoding_strategy() sets it up for the strategy
 * greedy_batch, GreedyBatchedRNNTLabelLoopingComputer.torch_impl()
 * (nemo/collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py), which emits what the strategy
 * greedy, GreedyRNNTInfer, emits frame by frame.
 *
 * From frame 0 with the blank fed to the prediction network, the joint is evaluated with the prediction network's
 * output at the frame and the label is the argmax of the tokens and the blank, the first of equal values. A blank
 * moves to the next frame with the same prediction, until a token is found or the frames run out; a token is
 * emitted on the frame it was found on and fed to the prediction network, and the next label is sought on the same
 * frame. After max_symbols tokens on one frame the decoding moves to the next. NeMo argmaxes the joint's log-softmax on
 * the CPU, which keeps the order of the logits, and the logits are argmaxed here.
 *
 * The prediction network's output changes only when a token is emitted, so one graph computes the prediction for the
 * last token and the joint at a run of frames from the current one, and the host takes the first frame whose label
 * is a token: one graph per token where the beam search takes one per step of its search. A run of only blanks is
 * followed by a run twice as long, so that silence takes few graphs.
 */
class RnntGreedyDecoder : public Decoder {
public:
    RnntGreedyDecoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint);

    Decoding decode(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress) const override;

    int blank() const override { return blank_; }

private:
    const PredictionNetwork & prediction_;
    const Joint & joint_;
    int blank_;
    uint32_t max_symbols_;
};

}  // namespace fastconformer
