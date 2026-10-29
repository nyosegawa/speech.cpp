#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "alignment.h"
#include "ctc-head.h"
#include "ctc-text.h"
#include "ggml-backend.h"
#include "model-file.h"
#include "perception.h"

namespace nemotron_transcribe {

/** A token or a segment of a timed transcript: its start and end in seconds, its text, and its speaker tag, or -1. */
struct TimedText {
    double start, end;
    std::string text;
    int speaker;
};

/**
 * A transcript with its times: the text, without its speaker tags where speakers were asked; each unit of the alignment
 * as a token whose text stands in the text, the texts of the tokens joined being the text; and the segments of tokens,
 * whose texts joined are the text too. Where the alignment cannot time the transcript, it has neither tokens nor
 * segments, and `failure` says why.
 */
struct TimedTranscript {
    std::string text;
    std::vector<TimedText> tokens, segments;
    std::string failure;
};

/**
 * The CTC timestamps of a recognition: its transcript aligned with the perception module's frames (AlignmentFrames) by
 * the CTC head and NeMo's alignment, each unit timed, rounded to the millisecond. A token takes the whitespace and the
 * tags before its unit, the marks after it within its word, and the last token the rest of the text. Its segments end
 * where nemotron-transcribe.segment.separators and .breaks end a sentence and where a speaker tag starts a turn, in the
 * order of the text, each from the start of its first token to the end of its last and, where speakers were asked, of its
 * turn's speaker. With speakers, the text drops each <spk:N> tag and the whitespace after it.
 */
class Timing {
public:
    /** The timing of the model file `m`, which outlives it, its CTC head computing on `backend`. */
    Timing(const ModelFile & m, ggml_backend_t backend);

    /**
     * The times of `transcript`, the text of `samples` samples of audio whose frames are `frames`. `speakers` takes the
     * speaker tags out of the text and gives each segment its speaker. `alignment`, when given, receives the alignment.
     * `proceed`, asked before each window of the CTC head and each speaker's search, stops the work where it says false,
     * and the times are then none.
     */
    std::optional<TimedTranscript> time(const std::string & transcript, const AlignmentFrames & frames, size_t samples, bool speakers,
                                        Alignment * alignment = nullptr, const std::function<bool()> & proceed = {});

    /** `transcript` without its speaker tags and the whitespace after each, as time() gives it with speakers. */
    std::string untagged(const std::string & transcript) const;

private:
    /** Which bytes of `transcript` its text keeps. */
    std::vector<bool> kept(const std::string & transcript, bool speakers) const;

    CtcText text_;
    CtcHead head_;
    Aligner aligner_;
    std::vector<std::string> separators_, breaks_;
    int sample_rate_;
};

}  // namespace nemotron_transcribe
