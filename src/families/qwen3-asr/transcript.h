#pragma once

#include <string>

#include "model-file.h"

namespace qwen3_asr {

/**
 * The text of a recognition from the model's decoded output, as qwen-asr's parse_asr_output() gives it
 * (qwen_asr/inference/utils.py): the output stripped of whitespace as Python's str.strip() strips it, its repetitions
 * fixed, and then, for a request that forced its language, whose output follows the prefill, the whole; otherwise
 * what follows the first <asr_text> (qwen3-asr.prompt.asr_text), stripped, or the whole without one. The language
 * the model writes before <asr_text>, None for audio without speech, is not part of the text.
 */
class Transcript {
public:
    explicit Transcript(const ModelFile & m);

    /** The text of the output `raw`; `forced` for a request that forced its language. */
    std::string text(const std::string & raw, bool forced) const;

    /**
     * detect_and_fix_repetitions(): a run of more than qwen3-asr.output.repetition_threshold of one character kept
     * once, then, from the first place where a pattern of up to repetition_max_period characters is repeated at least
     * repetition_threshold times, the pattern kept once and the rest fixed the same way. It counts characters as Python
     * does, in code points.
     */
    std::string fix_repetitions(const std::string & text) const;

private:
    std::string asr_text_;
    size_t threshold_, max_period_;
};

}  // namespace qwen3_asr
