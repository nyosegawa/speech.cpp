#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "model-file.h"

namespace fastconformer {

/**
 * The text of SentencePiece ids as NeMo's decoding writes it: SentencePieceProcessor.DecodeIds(), then one
 * whitespace removed before each punctuation mark of the vocabulary
 * (AbstractRNNTDecoding.decode_tokens_to_str_with_strip_punctuation()).
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

}  // namespace fastconformer
