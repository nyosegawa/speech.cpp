#pragma once

#include <string>
#include <vector>

#include "talker.h"

/**
 * The rows a prompt adds to its text's tokens when it names a language, one fewer for auto: the role, the think
 * tags, the language, the speaker, and the rows that close the text and open the speech.
 */
constexpr int kPromptRows = 11;

/** The ids a CustomVoice prompt is built from, and the voices and languages they belong to, read from the model file. */
struct PromptIds {
    int32_t tts_bos = 0, tts_eos = 0, tts_pad = 0;
    int32_t im_start = 0, im_end = 0, assistant = 0, newline = 0;
    int32_t codec_bos = 0, codec_eos = 0, codec_pad = 0;
    int32_t think = 0, nothink = 0, think_bos = 0, think_eos = 0;
    /** The voices, sorted, with the codec id of each and of the dialect it speaks, or -1. */
    std::vector<std::string> voices;
    std::vector<int32_t> speaker_ids, dialect_ids;
    /** The languages as BCP 47 tags, sorted, with the codec id of each. */
    std::vector<std::string> languages;
    std::vector<int32_t> language_ids;
    /** The language in which a voice with a dialect speaks its dialect, as it does with auto. */
    std::string dialect_language;

    explicit PromptIds(const ModelFile & m);

    /**
     * The index of a voice, its name compared with case: the official code lowers the name it is given, and the C API
     * takes the names as speech.voices writes them. An unknown name throws.
     */
    size_t voice(const std::string & name) const;
    /**
     * The codec id of the language that a prompt for voice `voice` names when a request asks for the BCP 47 tag `tag`
     * (`ja`, `ja-JP`), "auto" or "": the voice's dialect, for a voice that has one, in the dialect's language and with
     * auto, else the language, or -1 for auto. A language the model does not speak throws.
     */
    int32_t language_id(const std::string & tag, size_t voice) const;
};

/** What the talker needs to speak one text: the prefill embeddings and the vector added to every frame. */
struct Prompt {
    std::vector<float> embeds;
    int n = 0;
    std::vector<float> frame_extra;
};

/**
 * Builds the CustomVoice prompt of the official generate() in its non-streaming mode, which puts the
 * whole text in the prefill. `text_ids` are the tokens of `<|im_start|>assistant\n{text}<|im_end|>\n
 * <|im_start|>assistant\n`, and `language` is a BCP 47 tag of one of the model's languages or "auto".
 */
Prompt build_prompt(Talker & talker, const PromptIds & ids, const std::vector<int32_t> & text_ids,
                    const std::string & voice, const std::string & language);
