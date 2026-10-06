#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "model-file.h"
#include "tokenizer.h"

namespace qwen3_asr {

/** The ids of a request's prompt, and where the audio's tokens begin among them. */
struct PromptIds {
    std::vector<int32_t> ids;
    /** The position of the first of the audio's tokens, which follow one another. */
    int64_t audio_start = 0;
};

/**
 * The prompt of a request as qwen-asr writes it (_build_text_prompt() in qwen_asr/inference/qwen3_asr.py): the
 * checkpoint's chat template with a system turn that holds the context, empty without one, a user turn that holds the
 * audio's tokens between <|audio_start|> and <|audio_end|>, and the start of the assistant's turn, followed for a
 * forced language by the prefill "language <Name><asr_text>". The file holds the template's text before the context
 * (qwen3-asr.prompt.before_context), between the context and the audio's tokens (before_audio) and after them
 * (after_audio). The text on either side of the audio's tokens is tokenized whole, which gives the official ids of the
 * whole prompt since the audio's tokens border on added tokens, at which the tokenizer splits.
 */
class Prompt {
public:
    /** The prompt of the model file `m` with its tokenizer, both of which outlive it. */
    Prompt(const ModelFile & m, const Tokenizer & tokenizer);

    /**
     * The prompt of a request with `audio_tokens` tokens of audio, `context` in its system turn, and `language`, an index
     * of general.languages, forced, or the language left to the model. A context that is not UTF-8 is refused naming
     * the option prompt.
     */
    PromptIds ids(const std::string & context, std::optional<size_t> language, int64_t audio_tokens) const;

    /** The model's languages, general.languages. */
    const std::vector<std::string> & languages() const { return languages_; }

private:
    const Tokenizer & tokenizer_;
    std::string before_context_, before_audio_, after_audio_, language_prefix_, asr_text_;
    int32_t audio_token_;
    std::vector<std::string> languages_, names_;
};

}  // namespace qwen3_asr
