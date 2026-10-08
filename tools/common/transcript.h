#pragma once

#include <string>
#include <vector>

#include "speech.h"

// A recognition as the subcommands write it: read from one request's result, or joined from the results of the parts
// of one audio, each recognized alone, as transcription by regions joins them.

/** A token or a segment with its start and end in seconds from the start of the audio. */
struct Timed {
    double start = 0;
    double end = 0;
    std::string text;
};

/**
 * A recognition's text, why it stopped, the tags of the languages heard in the order of the audio, and with timestamps
 * its segments and tokens, whose texts, joined in order, are the text.
 */
struct Transcript {
    std::string text;
    speech_stop stop = SPEECH_STOP_COMPLETE;
    std::vector<std::string> languages;
    std::vector<Timed> segments;
    std::vector<Timed> tokens;
};

/** The recognition of a result, its segments and tokens read when `timestamps` is set. */
Transcript transcript_of(const speech_result * result, bool timestamps);

/**
 * Appends a part of the audio that begins `offset` seconds into it: its text after the text so far, with a space
 * between them unless either side is written without spaces or already has one; its segments and tokens moved by the
 * offset, the space, where one is added, beginning the first of each; its languages after the others, a run of the same
 * language counted once, as qwen-asr merges the languages of its parts; and model_limit as the stop when the part
 * stopped there.
 */
void append_part(Transcript & whole, const Transcript & part, double offset);

/**
 * The members of a recognition in the form of the worker's messages, each after a comma: "text", "stop" ("complete" or
 * "model_limit"), "languages" where it has any, and with `timestamps` "segments" and "tokens", each a list of
 * {"start", "end", "text"} with the times in seconds.
 */
std::string recognition_members(const Transcript & transcript, bool timestamps);
