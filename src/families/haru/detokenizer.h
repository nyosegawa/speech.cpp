#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "qwen2-tokenizer.h"

namespace haru {

/**
 * The text of the model's text stream as it comes, as haru.sampling's StreamDetokenizer gives it: a run of the silent
 * tokens, PAD and EOP, is one space, and a token whose bytes end inside a character is held until the character's
 * last token, the text of the tokens held so far decoded together.
 */
class StreamDetokenizer {
public:
    StreamDetokenizer(const Qwen2Tokenizer & tokenizer, std::vector<int32_t> silent);

    /** The new text of `token`, empty where it adds none. */
    std::string push(int32_t token);

private:
    const Qwen2Tokenizer & tokenizer_;
    std::vector<int32_t> silent_, pending_;
    bool in_gap_ = false;
};

}  // namespace haru
