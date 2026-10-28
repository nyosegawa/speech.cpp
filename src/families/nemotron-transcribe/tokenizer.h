#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "added-vocabulary.h"
#include "model-file.h"
#include "qwen2-tokenizer.h"

namespace nemotron_transcribe {

/**
 * Nemotron 3.5's tokenizer as the tokenizers library runs the checkpoint's tokenizer.json (nemotron-transcribe.tokenizer.*):
 * a text is split at every added token first, and the text between them goes through the byte-level BPE with
 * Nemotron's pre-tokenizer, which splits words by case, no normalizer, and ignore_merges. Decoding keeps every token,
 * the special ones included, as the model card's request asks (skip_special_tokens=False), so that the speaker and
 * event tags the model writes stay in its text.
 */
class Tokenizer {
public:
    explicit Tokenizer(const ModelFile & m);

    /** The ids of `text`. A text that is not UTF-8 throws an Error that names no input, which the caller names. */
    std::vector<int32_t> encode(const std::string & text) const;

    /** The text of `ids`, a character cut off at the end written as U+FFFD. */
    std::string decode(const std::vector<int32_t> & ids) const;

private:
    Qwen2Tokenizer bpe_;
    AddedVocabulary added_;
};

}  // namespace nemotron_transcribe
