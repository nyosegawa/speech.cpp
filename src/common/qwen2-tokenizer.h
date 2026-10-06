#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "model-file.h"

/**
 * The Qwen2 byte-level BPE (transformers' Qwen2Tokenizer) with the pre-tokenizer of the tokenizer.json that ships with
 * the model, whose tokens and merges a model file holds under one prefix.
 *
 * transformers 4.57.3, which the official Qwen3-TTS package pins, replaces that pre-tokenizer with Mistral's when
 * the model is loaded with fix_mistral_regex=True, as the package does; the two split text of Latin
 * letters in mixed case, contractions and '/' differently. This follows tokenizer.json, which is what
 * the model was trained with. The text is expected in NFC; the caller normalizes it.
 */
class Qwen2Tokenizer {
public:
    /** The tokenizer of the keys `<prefix>.tokens`, every token in the order of its id, and `<prefix>.merges`. */
    Qwen2Tokenizer(const ModelFile & m, const std::string & prefix);

    /**
     * The ids of `text`, with special tokens written in the text taken literally. A text that is not UTF-8 throws an
     * Error that names no input, which the caller names.
     */
    std::vector<int32_t> encode(const std::string & text) const;

    /**
     * The text of `ids` as the ByteLevel decoder of tokenizer.json gives it, special tokens included: each token's
     * characters turned back into the bytes they stand for, or the token's own UTF-8 when one of its characters stands
     * for none, the bytes of all the tokens joined, and every sequence of them that is not UTF-8 replaced by U+FFFD
     * as Rust's String::from_utf8_lossy does, so that a character split between two tokens comes out whole and one cut
     * off by the end comes out as U+FFFD. An id with no token decodes to nothing, as the tokenizers library drops it.
     */
    std::string decode(const std::vector<int32_t> & ids) const;

    /** The pieces the pre-tokenizer splits `text` into, as UTF-8. */
    std::vector<std::string> pre_tokenize(const std::string & text) const;

private:
    std::vector<int32_t> bpe(const std::string & piece) const;

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> vocab_;
    std::map<std::pair<std::string, std::string>, int> ranks_;
    std::string byte_to_unicode_[256];
    std::unordered_map<uint32_t, unsigned char> unicode_to_byte_;
};
