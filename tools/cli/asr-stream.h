#pragma once

#include <cstdio>
#include <vector>

#include "command-line.h"

// speech asr - and speech asr --live: audio that arrives while someone speaks, cut into utterances by the regions where a
// detection model finds speech, each written as it ends.

/**
 * Transcribes the 16-bit PCM on stdin at --rate, or the microphone with --live, by the regions where `detector` finds
 * speech with `detection` options, each recognized on `recognizer` with `options`, and writes each utterance's text, or
 * with `json` OpenAI's Realtime server events, to `out`. It returns the exit code: 3 when a recognition stopped at the
 * most the model writes, 0 otherwise; a failure throws.
 */
int transcribe_stream(const CommandLine & line, FILE * out, speech_model * recognizer, const std::vector<RequestOption> & options,
                      speech_model * detector, const std::vector<RequestOption> & detection, bool json);
