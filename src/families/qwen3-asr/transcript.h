#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "model-file.h"

namespace qwen3_asr {

/** What qwen-asr's parse_asr_output() makes of the model's decoded output. */
struct Parsed {
    std::string text;
    /**
     * The language of the output as an index of general.languages: none for an output that is empty once stripped;
     * otherwise, for a request that forced its language, the forced one; otherwise the one the model wrote before
     * <asr_text>, or none where it wrote none, wrote the language None of audio without speech, or wrote a name that
     * qwen3-asr.language_names does not hold.
     */
    std::optional<size_t> language;
    /** The name the model wrote that qwen3-asr.language_names does not hold, as it wrote it, or "". */
    std::string other_language;
};

/**
 * The text and the language of a recognition from the model's decoded output, as qwen-asr's parse_asr_output() gives
 * them (qwen_asr/inference/utils.py): the output stripped of whitespace as Python's str.strip() strips it and its
 * repetitions fixed; then, for a request that forced its language, whose output follows the prefill, the whole as the
 * text and the forced language; otherwise the text after the first <asr_text> (qwen3-asr.prompt.asr_text), stripped,
 * or the whole without one, and the language from what precedes it.
 */
class Transcript {
public:
    explicit Transcript(const ModelFile & m);

    /** The parse of the output `raw`; `forced` is the language a request forced, an index of general.languages. */
    Parsed parse(const std::string & raw, std::optional<size_t> forced) const;

    /**
     * detect_and_fix_repetitions(): a run of more than qwen3-asr.output.repetition_threshold of one character kept
     * once, then, from the first place where a pattern of up to repetition_max_period characters is repeated at least
     * repetition_threshold times, the pattern kept once and the rest fixed the same way. It counts characters as Python
     * does, in code points.
     */
    std::string fix_repetitions(const std::string & text) const;

private:
    /**
     * The language that the part of an output before <asr_text> names: none where it holds the language None, in any
     * case and anywhere; otherwise the name on its first line that begins with qwen3-asr.prompt.language_prefix, in any
     * case, normalized as normalize_language_name() normalizes it.
     */
    void read_language(const std::vector<uint32_t> & meta, Parsed & out) const;

    std::string asr_text_;
    std::vector<uint32_t> prefix_;
    size_t threshold_, max_period_;
    std::vector<std::vector<uint32_t>> names_;
};

}  // namespace qwen3_asr
