#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "model-file.h"

namespace nemotron_transcribe {

/**
 * The SentencePiece model of the CTC timestamp head, the ModelProto that ctc.tokenizer.model_proto holds as the checkpoint
 * does, encoding as sentencepiece 0.2.2's SentencePieceProcessor.encode() does, which NeMo's SentencePieceTokenizer calls
 * outside its legacy mode. The normalizer replaces the longest prefix its precompiled character map holds, nmt_nfkc's,
 * or keeps one character, but for the user-defined pieces, which it keeps whole; it writes a dummy "▁" before the text
 * and "▁" for each space, and the model's settings say whether it drops the spaces at the ends and runs of them. BPE then
 * splits the normalized text into characters and user-defined pieces and, until no adjacent pair joins into a piece of
 * the vocabulary, joins the pair whose join has the highest score, the leftmost of equal scores; a user-defined piece
 * joins nothing. Each run of symbols outside the vocabulary becomes one id, the unknown's. A ModelProto of another model
 * than BPE, with byte fallback, unused pieces, a whitespace suffix or rules of normalization in text, which this reader
 * does not run, throws.
 */
class CtcTokenizer {
public:
    explicit CtcTokenizer(const ModelFile & m);

    std::vector<int32_t> encode(const std::string & text) const;

    /** The id of the piece `piece`, the unknown's for one outside the vocabulary, as PieceToId() gives it. */
    int32_t piece_to_id(const std::string & piece) const;

    /**
     * The text of `ids`, none of them unknown or a control piece, as decode() gives it: their pieces joined, each "▁" a
     * space, but the dummy one that starts the first.
     */
    std::string decode(const std::vector<int32_t> & ids) const;

    const std::string & piece(int32_t id) const { return pieces_[(size_t) id].text; }
    int32_t unknown() const { return unknown_; }
    int32_t size() const { return (int32_t) pieces_.size(); }

private:
    struct Piece {
        std::string text;
        float score;
        int type;
    };

    /** The normalized text, as Normalizer::Normalize() gives it. */
    std::string normalize(const std::string & text) const;
    /** The length of the longest user-defined piece at `pos` of `text`, 0 where none starts. */
    size_t user_defined_at(const std::string & text, size_t pos) const;

    std::vector<Piece> pieces_;
    /** The ids of the pieces of the vocabulary, PieceToId()'s pieces_, and of the reserved ones, its reserved_id_map_. */
    std::unordered_map<std::string, int32_t> vocabulary_, reserved_;
    std::vector<std::string> user_defined_;
    int32_t unknown_ = 0;
    bool add_dummy_prefix_ = true, remove_extra_whitespaces_ = true, escape_whitespaces_ = true;
    /** The double-array trie of the character map, its units, and the normalized strings its values point into. */
    std::vector<uint32_t> trie_;
    std::string normalized_;
};

}  // namespace nemotron_transcribe
