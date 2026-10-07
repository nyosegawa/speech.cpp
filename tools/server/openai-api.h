#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "failure.h"
#include "request-options.h"

// OpenAI's audio API as speech.cpp reads and writes it: the create speech request, the create transcription request,
// the error object with the library's categories mapped onto it, and the events of an SSE stream. The shapes follow
// OpenAI's API reference, components CreateSpeechRequest, CreateTranscriptionRequest, CreateTranscriptionResponseJson,
// CreateTranscriptionResponseVerboseJson, TranscriptionSegment, Error, SpeechAudioDeltaEvent and SpeechAudioDoneEvent
// and the path /audio/transcriptions of github.com/openai/openai-openapi at commit 31af4fc (2026-10-05).

namespace openai {

/** An answer that is not audio: an HTTP status and an error in OpenAI's shape, its type following from the status. */
struct ApiError {
    int status;
    std::string message;
    std::string param;
    std::string code;
};

/**
 * The error of a failure of the library, by its category alone: invalid_argument, unsupported and out_of_range are
 * the request's (400, invalid_value, unsupported_parameter and unsupported_value), every other category the server's
 * (500, the category's name as the code). The param is the input at fault, "text" written as "input" and "audio" as
 * "file", the names OpenAI's requests give them.
 */
ApiError library_error(const Failure & failure);

/** The members of OpenAI's error object: {"message", "type", "param", "code"}. */
std::string error_object(const ApiError & e);

/** An error response's body: {"error": {...}}. */
std::string error_json(const ApiError & e);

/** A create speech request as read from its JSON body. */
struct SpeechRequest {
    std::string input;
    std::string format = "wav";
    bool sse = false;
    /** The options of the vocabulary that the body sets, by their names. */
    std::vector<RequestOption> options;
};

/**
 * Reads a create speech request: "model", "input", "response_format", "stream_format" and every option of the
 * vocabulary by its name. A member it does not have is refused rather than ignored, OpenAI's "instructions" among
 * them, and a member set to null counts as left out. The model's name, which "model" may give, is the one /v1/models
 * lists. Anything it cannot read throws an ApiError; the values of the options are checked by the library's setters.
 */
SpeechRequest read_speech_request(const std::string & body, const std::string & model_name);

/** One part of a multipart/form-data body: a field, or a file when it has a filename. */
struct FormPart {
    std::string name, content, filename;
    bool file = false;
};

/** A create transcription request as read from its form, with the file's audio decoded. */
struct TranscriptionRequest {
    std::vector<float> samples;
    int sample_rate = 0;
    /** "json", "text" or "verbose_json". */
    std::string format = "json";
    /** The language, the prompt, and timestamps for verbose_json. */
    std::vector<RequestOption> options;
    /** Whether the options set timestamps, so that the answer carries the segments. */
    bool timestamps = false;
};

/**
 * Reads a create transcription request from the parts of its form: "file", a WAV file, its channels averaged;
 * "model"; "language"; "prompt", the option prompt; "response_format"; and "timestamp_granularities[]", "segment"
 * alone, with verbose_json. verbose_json sets the option timestamps for a model that `gives_times`, and for any model
 * when the request names a granularity, which a model without times then refuses. A member it does not have is
 * refused, and so is one given twice but timestamp_granularities[]. Anything it cannot read throws an ApiError.
 */
TranscriptionRequest read_transcription_request(bool multipart, const std::vector<FormPart> & parts, const std::string & model_name,
                                                bool gives_times);

/** A speech.audio.delta event with a chunk of 16-bit PCM. */
std::string sse_delta(const std::string & pcm);

/**
 * The speech.audio.done event. OpenAI's carries the usage in tokens, which speech.cpp does not count, so it carries
 * the seed, the number of samples and the stop reason instead.
 */
std::string sse_done(int64_t seed, uint64_t samples, const char * stop);

/** An error after an SSE stream has begun, {"type": "error", "error": {...}}, which OpenAI's speech stream does not define. */
std::string sse_error(const ApiError & e);

/** A transcription in OpenAI's json form; OpenAI's also carries the usage, which speech.cpp does not count. */
std::string transcription_json(const std::string & text);

/**
 * A transcription in OpenAI's verbose_json form: the task; the language, the tag of the language the model heard, or
 * the tags joined with commas where it heard several in the parts of long audio, as qwen-asr joins their names; the
 * audio's duration in seconds; the text; and with `timestamps` the segments. A value the recognizer does not give is
 * left out rather than made up: the language where the result has none, and the members of OpenAI's segment the
 * recognizers have no value for (seek, tokens, temperature, avg_logprob, compression_ratio, no_speech_prob).
 */
std::string transcription_verbose_json(const speech_result * result, double duration, bool timestamps);

}  // namespace openai
