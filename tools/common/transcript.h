#pragma once

#include <optional>
#include <string>
#include <vector>

#include "speech.h"

// A recognition as the subcommands write it: read from one request's result, or joined from the results of the parts
// of one audio, each recognized alone, as transcription by regions joins them.

/**
 * A token or a segment with its start and end in seconds from the start of the audio, and a segment's speaker where the
 * recognition set speakers, -1 where it has none.
 */
struct Timed {
    double start = 0;
    double end = 0;
    std::string text;
    int speaker = -1;
};

/**
 * A recognition's text, why it stopped, the tags of the languages heard in the order of the audio, with timestamps its
 * segments and tokens and with speakers its segments, whose texts, joined in order, are the text; or why it has none
 * (speech_result_untimed()).
 */
struct Transcript {
    std::string text;
    speech_stop stop = SPEECH_STOP_COMPLETE;
    std::vector<std::string> languages;
    std::vector<Timed> segments;
    std::vector<Timed> tokens;
    std::optional<std::string> untimed;
};

/**
 * The recognition of a result: its segments read when `timestamps` or `speakers` is set, its tokens with `timestamps`,
 * and its segments' speakers with `speakers`.
 */
Transcript transcript_of(const speech_result * result, bool timestamps, bool speakers = false);

/**
 * The words of `tokens`: a word begins with the first token, with a token that begins with whitespace, and at each
 * token of a script written without spaces and the token after it, so that each such token is a word of its own; it
 * spans from the start of its first token to the end of its last, and its text is theirs without the whitespace before
 * it. A token without text and a word of whitespace alone are left out.
 */
std::vector<Timed> words_of(const std::vector<Timed> & tokens);

/**
 * Appends a part of the audio that begins `offset` seconds into it: its text after the text so far, with a space
 * between them unless either side is written without spaces or already has one; its segments and tokens moved by the
 * offset, the space, where one is added, beginning the first of each; its languages after the others, a run of the same
 * language counted once, as qwen-asr merges the languages of its parts; and model_limit as the stop when the part
 * stopped there. A part without times makes the whole without them, for the first such part's reason.
 */
void append_part(Transcript & whole, const Transcript & part, double offset);

/**
 * The beginning that two readings of the beginning of one utterance agree on, `earlier` read from less audio than
 * `later`: the code points both begin with, back to the end of the last word both have whole where the script puts
 * spaces between words, so that a word that the end of the shorter audio cut is not taken as agreed; and where that is
 * all of `later`, only up to the last mark or space before the end, since the end of the audio closes whatever was said
 * last, with a mark the recognizer writes there.
 */
std::string agreed_beginning(const std::string & earlier, const std::string & later);

/**
 * The members of a recognition in the form of the worker's messages, each after a comma: "text", "stop" ("complete" or
 * "model_limit"), "languages" where it has any, with `timestamps` or `speakers` "segments" and with `timestamps`
 * "tokens", each a list of {"start", "end", "text"} with the times in seconds, a segment with `speakers` also with its
 * "speaker", null where it has none; and "untimed", why there are no segments or tokens, where the recognition says.
 */
std::string recognition_members(const Transcript & transcript, bool timestamps, bool speakers = false);
