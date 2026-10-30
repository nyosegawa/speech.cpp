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
 *
 * A run of byte pieces, which a vocabulary with byte fallback writes a character it lacks with, is decoded as
 * DecodeIds() decodes it: each character of its bytes is the text of the piece of its last byte, the pieces before
 * having none, and each byte that begins no well-formed UTF-8 character is U+FFFD.
 */
class Detokenizer {
public:
    explicit Detokenizer(const ModelFile & m);

    std::string text(const std::vector<int32_t> & ids) const;

    /**
     * The text of each token as it stands in text(ids), which they make when joined: a whitespace the decoding
     * removes before a punctuation mark is taken from the token that holds it.
     */
    std::vector<std::string> token_texts(const std::vector<int32_t> & ids) const;

    /**
     * Whether the token, decoded alone, is one of the vocabulary's punctuation marks, as compute_rnnt_timestamps()
     * tests it: decode_ids_to_str([id]) in supported_punctuation.
     */
    bool punctuation(int32_t id) const;

    /** The text of the token decoded alone, as decode_ids_to_str([id]) writes it: U+FFFD for a byte piece past ASCII. */
    std::string decoded_alone(int32_t id) const;

    /** Whether the token is a byte piece. */
    bool byte_piece(int32_t id) const { return byte_of(id) >= 0; }

    /**
     * Whether each token begins a word as get_words_offsets() tells it for a SentencePiece vocabulary: its piece
     * differs from its text decoded alone, as a piece with a leading "▁" and the unknown piece do, and it is not a
     * punctuation mark, which NeMo adds to the word before it.
     */
    std::vector<bool> word_starts(const std::vector<int32_t> & ids) const;

private:
    /** The piece of `id`, not a byte piece, as DecodeIds() writes it when the text before it is `empty` or not. */
    std::string piece_text(int32_t id, bool empty) const;

    /** The text of each id as DecodeIds() writes it: piece_text(), or a byte piece's share of the run it is in. */
    std::vector<std::string> surfaces(const std::vector<int32_t> & ids) const;

    /** The byte of a byte piece, -1 for any other piece. */
    int byte_of(int32_t id) const;

    std::vector<std::string> pieces_;
    /** The byte each id stands for, -1 for a piece that is no byte piece; empty for a vocabulary without them. */
    std::vector<int> bytes_;
    std::vector<std::string> punctuation_;
    int unknown_id_;
    std::string unknown_surface_;
    bool strip_leading_space_;
};

}  // namespace fastconformer
