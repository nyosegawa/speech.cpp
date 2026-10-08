#pragma once

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "cancellation.h"
#include "library.h"
#include "request-options.h"
#include "transcript.h"

// Transcription by regions: a detection model finds where someone speaks in the audio, a recognition model recognizes
// each region alone, and the texts are joined. A FastConformer model given 15 to 20 s that hold several sentences drops
// whole sentences, as NeMo's own implementation does, and writes words for audio in which no one speaks; given one
// region at a time it hears one or a few sentences and nothing where no one speaks.

/** A region where someone speaks, in seconds from the start of the audio. */
struct Region {
    double start = 0;
    double end = 0;
};

/** The samples [first, last) of a region of audio that has `n` samples at `rate`. */
std::pair<size_t, size_t> region_samples(const Region & region, int rate, size_t n);

/**
 * The detection options of transcription by regions, `given` in place of the defaults: OpenAI's server_vad defaults
 * (threshold 0.5, speech_pad_ms 300 for its prefix_padding_ms, min_silence_duration_ms 500 for its
 * silence_duration_ms) and a longest region, max_speech_duration_s, which its form has no member for.
 */
std::vector<RequestOption> region_options(const std::vector<RequestOption> & given);

/** A detection request on `detector` with the audio and the options set, which the library checks as they are set. */
Request detection_request(speech_model * detector, const std::vector<float> & samples, int rate, const std::vector<RequestOption> & options);

/** The regions that a detection request set up by detection_request() finds, or nothing once it is cancelled. */
std::optional<std::vector<Region>> detect_regions(speech_request * detection, Cancellation & cancellation);

/**
 * Recognizes each region of `samples` alone on `recognizer` with `options`, and joins the results with append_part(),
 * the times of the segments and tokens (with `timestamps`) moved to those of the whole audio. The options are checked
 * against the model before any region, so audio without regions gives "" and still refuses an option the model does not
 * take. It returns nothing once cancelled.
 */
std::optional<Transcript> transcribe_regions(speech_model * recognizer, const std::vector<float> & samples, int rate,
                                             const std::vector<Region> & regions, const std::vector<RequestOption> & options,
                                             bool timestamps, Cancellation & cancellation);
