#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "cancellation.h"
#include "library.h"
#include "request-options.h"

// Enhancement in the tools: a recording's audio enhanced by speech_enhance(), passed to a callback as it comes, which
// speech enhance writes to WAVE files, the worker to its chunks and the server to its responses; or gathered whole, for
// a recognition of the enhanced audio, as speech asr --enhance and the server's noise_reduction ask for it.

/** An enhancement request on `enhancer` with the audio and the options set, which the library checks as they are set. */
Request enhancement_request(speech_model * enhancer, const std::vector<float> & samples, int rate, const std::vector<RequestOption> & options);

/**
 * The rate an enhancement with `options` passes audio given at `rate` at: the model's own where its information gives
 * one, else the option bwe where the options set it, else the audio's, as speech_enhance() enhances it.
 */
int enhanced_rate(const speech_model_info * info, int rate, const std::vector<RequestOption> & options);

/**
 * Runs an enhancement request set up by enhancement_request(), passing its audio to `on_audio` with `user_data`, and
 * returns the samples passed, or nothing once it is cancelled.
 */
std::optional<uint64_t> enhance_audio(speech_request * enhancement, speech_audio_callback on_audio, void * user_data, Cancellation & cancellation);

/** Audio and its rate. */
struct Enhanced {
    std::vector<float> samples;
    int rate = 0;
};

/**
 * Runs an enhancement request set up by enhancement_request(), whose audio comes at `rate` (enhanced_rate()), and gathers
 * the enhanced audio whole, or nothing once it is cancelled.
 */
std::optional<Enhanced> gather_enhanced(speech_request * enhancement, int rate, Cancellation & cancellation);

/** Enhances `samples` at `rate` on `enhancer` with `options` and gathers the audio whole, or nothing once it is cancelled. */
std::optional<Enhanced> enhance_whole(speech_model * enhancer, const std::vector<float> & samples, int rate, const std::vector<RequestOption> & options,
                                      Cancellation & cancellation);
