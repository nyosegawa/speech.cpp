#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fastconformer {

/** The type of a SentencePiece piece, with the values of sentencepiece_model.proto's ModelProto.SentencePiece.Type. */
enum class PieceType : int32_t {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
};

/**
 * What SentencePiece reads of a BPE model (sentencepiece_model.proto's ModelProto, with trainer_spec.model_type BPE
 * and treat_whitespace_as_suffix false) to encode a text: its pieces and the settings of its normalizer.
 */
struct PieceModel {
    /** Every piece, by id. */
    std::vector<std::string> pieces;
    /** The score of each piece; of the pairs a merge can join, the one whose piece scores highest joins first. */
    std::vector<float> scores;
    /**
     * The type of each piece: normal pieces, the one unknown piece, and either none or all 256 byte pieces, `<0x00>`
     * to `<0xFF>`, which a model with byte fallback has.
     */
    std::vector<PieceType> types;
    /** normalizer_spec.precompiled_charsmap: the rules of the normalizer, or none for a normalizer that rewrites nothing. */
    std::string charsmap;
    /** Whether the normalizer puts a space before the text, so that its first word begins with one as the others do. */
    bool add_dummy_prefix;
    /** Whether the normalizer drops the spaces at the start and the end and keeps one of each run of spaces. */
    bool remove_extra_whitespaces;
    /** Whether the normalizer writes each space as "▁" (U+2581). */
    bool escape_whitespaces;
};

/**
 * The ids sentencepiece 0.2's SentencePieceProcessor.encode() gives a text, with a BPE model: its normalizer
 * (normalizer.cc), the merges of BPE (bpe_model.cc) and, for a piece the model lacks, its UTF-8 bytes as byte pieces or,
 * without byte fallback, the unknown piece once for each run of pieces it lacks (EncodeOptimized() of
 * sentencepiece_processor.cc).
 *
 * A model with control, user-defined or unused pieces is refused: SentencePiece matches a user-defined piece before
 * it normalizes, keeps a control piece out of the merges and splits an unused piece back into the pieces it joins, which
 * this does not do.
 */
class PieceEncoder {
public:
    explicit PieceEncoder(PieceModel model);

    /** The ids of any bytes: each byte that begins no well-formed UTF-8 character is U+FFFD to the normalizer. */
    std::vector<int32_t> encode(const std::string & text) const;

    /** The text as the normalizer gives it to the model (Normalizer::Normalize()), its spaces as "▁" by default. */
    std::string normalize(const std::string & text) const;

    /** The ids of a text the normalizer gave: the merges of BPE and the byte pieces or unknown pieces of what it lacks. */
    std::vector<int32_t> encode_normalized(const std::string & normalized) const;

private:
    /**
     * What the normalizer makes of the start of `input`, and the bytes of `input` it takes (Normalizer::NormalizePrefix()):
     * the replacement of the longest key of the charsmap the input begins with, or else the input's first character as
     * it is, or U+FFFD for a byte that begins no well-formed character.
     */
    std::pair<std::string_view, size_t> normalize_prefix(std::string_view input) const;

    PieceModel model_;
    /** The id of each normal piece, which is what the merges may form. */
    std::unordered_map<std::string, int32_t> mergeable_;
    int32_t unknown_id_ = -1;
    /** The id of each byte's piece, or none for a model without byte fallback. */
    std::vector<int32_t> byte_pieces_;
    /**
     * The double array of darts-clone (darts.h) that the charsmap begins with, one unit each: a node holds its label in
     * the low byte, whether a key ends at it in bit 8, and the offset of its children in bits 10 to 31, shifted left by 8
     * when bit 9 is set; a leaf has bit 31 set and holds its value, the offset of a replacement in `replacements_`, in
     * the other 31 bits. The keys are UTF-8, a node's child of the byte b is at its index ^ offset ^ b, and its leaf at
     * its index ^ offset.
     */
    std::vector<uint32_t> trie_;
    /** The replacements of the charsmap, each ended by a NUL. */
    std::string replacements_;
};

}  // namespace fastconformer
