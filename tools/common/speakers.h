#pragma once

#include <optional>
#include <vector>

#include "cancellation.h"
#include "library.h"
#include "request-options.h"

// Diarization in the tools: who speaks where in a whole recording, as speech_diarize() finds it.

/** A segment of one speaker, in seconds from the start of the audio, and the model's number for the speaker. */
struct SpeakerSegment {
    double start = 0;
    double end = 0;
    int speaker = 0;
};

/** A diarization request on `diarizer` with the audio and the options set, which the library checks as they are set. */
Request diarization_request(speech_model * diarizer, const std::vector<float> & samples, int rate, const std::vector<RequestOption> & options);

/** The segments that a diarization request set up by diarization_request() finds, or nothing once it is cancelled. */
std::optional<std::vector<SpeakerSegment>> diarize_segments(speech_request * diarization, Cancellation & cancellation);
