#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "graph.h"
#include "model-file.h"

namespace fastconformer {

/** NeMo's ConvASRDecoder: a pointwise projection of the encoder's output to the tokens and a blank after them. */
class CtcHead {
public:
    explicit CtcHead(const ModelFile & m);

    /** The logits of `encoded` ([d_model, T]), [classes, T]; the official head returns their log-softmax. */
    ggml_tensor * build(Graph & g, ggml_tensor * encoded) const;

    /**
     * GreedyCTCInfer and AbstractCTCDecoding.decode_hypothesis(): the best class of each frame of `logits`
     * ([T, classes] row-major), with repeats merged and blanks dropped.
     */
    std::vector<int32_t> greedy(const std::vector<float> & logits) const;

    int classes() const { return blank_ + 1; }

private:
    const ModelFile & m_;
    int blank_;
};

/**
 * The text of SentencePiece ids as NeMo's CTC decoding writes it: SentencePieceProcessor.DecodeIds(), then
 * one whitespace removed before each punctuation mark of the vocabulary
 * (AbstractCTCDecoding.decode_tokens_to_str_with_strip_punctuation()).
 */
class Detokenizer {
public:
    explicit Detokenizer(const ModelFile & m);

    std::string text(const std::vector<int32_t> & ids) const;

private:
    std::vector<std::string> pieces_;
    std::vector<std::string> punctuation_;
    int unknown_id_;
    std::string unknown_surface_;
    bool strip_leading_space_;
};

/** The log-softmax of each row of `logits` ([rows, classes] row-major), in double precision. */
std::vector<float> log_softmax(const std::vector<float> & logits, int classes);

}  // namespace fastconformer
