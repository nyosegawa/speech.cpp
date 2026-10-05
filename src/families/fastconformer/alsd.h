#pragma once

#include <cstdint>
#include <vector>

#include "decoder.h"

namespace fastconformer {

/**
 * NeMo's alignment-length synchronous beam search over an RNN-T as transcribe() runs it for one utterance,
 * BeamRNNTInfer.align_length_sync_decoding() (nemo/collections/asr/parts/submodules/rnnt_beam_decoding.py), the
 * search of Saon et al. (ICASSP 2020).
 *
 * Every hypothesis starts from the blank. Step i of T + U_max (U_max the frames times max_target_ratio) takes each
 * hypothesis of u labels whose frame t = i - u is still in the utterance, and evaluates the joint at frame t with the
 * prediction network's output for its labels; a log-softmax over the tokens and the blank gives the scores. Each
 * gives one hypothesis that adds the blank's score and keeps its labels, which is finished when t is the last frame,
 * and one for each of the beam's best tokens, which adds the token. The beam keeps the best hypotheses of the step
 * (a stable sort by score), then merges those with the same labels into the first of them, its score the log of
 * the sum of their probabilities; a finished hypothesis that is merged into takes the new score too. The search ends
 * when no hypothesis has a frame left, and the result is the finished hypothesis with the best score, divided by
 * its length with the starting blank when score_norm is set, or the beam's first when none finished.
 */
class AlsdDecoder : public Decoder {
public:
    AlsdDecoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint);

    std::vector<int32_t> decode(const std::vector<float> & projected, ggml_backend_t backend) const override;

    int blank() const override { return blank_; }

private:
    const PredictionNetwork & prediction_;
    const Joint & joint_;
    int blank_;
    int beam_;
    bool score_norm_;
    float max_target_ratio_;
};

}  // namespace fastconformer
