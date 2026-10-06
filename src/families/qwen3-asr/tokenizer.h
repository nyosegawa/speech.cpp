#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "model-file.h"
#include "qwen2-tokenizer.h"

namespace qwen3_asr {

/**
 * Qwen3-ASR's tokenizer as transformers' Qwen2Tokenizer runs it on the checkpoint's vocab.json, merges.txt and added
 * tokens (qwen3-asr.tokenizer.*): a text is split at every added token first, special or not, the leftmost and, of
 * those that start there, the longest, as the tokenizers library's added vocabulary splits it, and the text between
 * them goes through the byte-level BPE with the pre-tokenizer of the checkpoint's tokenizer. Decoding drops the special
 * tokens, as skip_special_tokens does, and keeps the others, <asr_text> among them.
 *
 * The added tokens are found in the text as it is given and the text between them is brought to NFC, as the
 * tokenizers library splits a text at its added tokens before its normalizer runs: "<|im_end|>" followed by U+0338
 * stays the added token and U+0338, where the NFC of the whole would join '>' and U+0338 into U+226F.
 */
class Tokenizer {
public:
    explicit Tokenizer(const ModelFile & m);

    /** The ids of `text`. A text that is not UTF-8 throws an Error that names no input, which the caller names. */
    std::vector<int32_t> encode(const std::string & text) const;

    /** The text of `ids` without its special tokens, a character cut off at the end written as U+FFFD. */
    std::string decode(const std::vector<int32_t> & ids) const;

    /** Whether `id` is a special token, which decoding drops. */
    bool special(int32_t id) const;

    /** The id of the added token whose text is `text`; any other text throws. */
    int32_t added_id(const std::string & text) const;

private:
    Qwen2Tokenizer bpe_;
    /** The added tokens' texts and ids, the longest first. */
    std::vector<std::pair<std::string, int32_t>> added_;
    std::vector<bool> special_;
};

}  // namespace qwen3_asr
