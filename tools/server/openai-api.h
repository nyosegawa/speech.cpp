#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "failure.h"
#include "openai-error.h"
#include "regions.h"
#include "request-options.h"
#include "speakers.h"
#include "transcript.h"

namespace httplib {
struct Response;
}

// OpenAI's audio API as speech.cpp reads and writes it: the create speech request, the create transcription request,
// its errors as HTTP answers, and the events of an SSE stream. The shapes follow
// OpenAI's API reference, components CreateSpeechRequest, CreateTranscriptionRequest, CreateTranscriptionResponseJson,
// CreateTranscriptionResponseVerboseJson, TranscriptionSegment, Error, SpeechAudioDeltaEvent and SpeechAudioDoneEvent
// and the path /audio/transcriptions of github.com/openai/openai-openapi at commit 31af4fc (2026-10-05), and its
// chunking_strategy as TranscriptionChunkingStrategy and VadConfig give it at commit 234829e (2026-10-07), whose object
// openai-python 3.26.0 sends in a form as one field per member, chunking_strategy[type] and so on, with
// CreateTranscriptionResponseDiarizedJson and TranscriptionDiarizedSegment at the same commit.

namespace openai {

/** Answers with the error's status and body. */
void send_error(httplib::Response & res, const ApiError & e);

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
 * vocabulary by its name, OpenAI's "instructions" among them. A member it does not have is refused rather than ignored,
 * and a member set to null counts as left out. The model's name, which "model" may give, is the one /v1/models
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
    /** "json", "text", "verbose_json" or "diarized_json". */
    std::string format = "json";
    /** The language, the prompt, the decoding, the keywords, and timestamps for verbose_json. */
    std::vector<RequestOption> options;
    /** Whether the options set timestamps, so that the answer carries the segments, and which granularities it gives. */
    bool timestamps = false, segments = true, words = false;
    /** Whether the options set speakers, for diarized_json of a model that tells who speaks. */
    bool speakers = false;
    /**
     * With chunking_strategy, the members of server_vad it gives as detection options, which region_options() completes
     * with the detection model's defaults; without it, nothing.
     */
    std::optional<std::vector<RequestOption>> regions;
};

/**
 * Reads a create transcription request from the parts of its form: "file", a WAV file, its channels averaged;
 * "model"; "language"; "prompt", the option prompt; "response_format"; "decoding", the option decoding, speech.cpp's
 * own; "keywords[]", each a line of the option keywords; "keywords_alpha" and "keywords_guard", speech.cpp's own;
 * "chunking_strategy", "auto" or the members of server_vad, which diarized_json needs unless the model
 * `tells_speakers` itself, as one that takes the option speakers does, and which such a model's diarized_json refuses;
 * and "timestamp_granularities[]", "segment" or "word", with verbose_json. verbose_json sets the option timestamps for a
 * model that `gives_times` unless keywords are in effect, from the request or the model's `defaults`, and for any
 * model when the request names a granularity, which a model without times, or with keywords, then refuses;
 * diarized_json of a model that tells speakers sets the option speakers. A member it does not have is refused,
 * known_speaker_names[] and known_speaker_references[] among them, and so is one given twice but keywords[] and
 * timestamp_granularities[]. Anything it cannot read throws an ApiError.
 */
TranscriptionRequest read_transcription_request(bool multipart, const std::vector<FormPart> & parts, const std::string & model_name,
                                                bool gives_times, bool tells_speakers, const std::vector<RequestOption> & defaults);

/** A library failure of a create transcription request as OpenAI's error (library_error()), the option keywords named by its field, keywords[]. */
ApiError transcription_error(const Failure & failure);

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
 * the tags joined with commas where it heard several in the parts of long audio or in its regions, as qwen-asr joins
 * their names; the
 * audio's duration in seconds; the text; and with `timestamps` the segments where `segments` asks for them and the words
 * (words_of() the tokens) where `words` does, or "untimed", speech.cpp's own, why the model gives neither. A value the
 * recognizer does not give is left out rather than made up: the language where the result has none, and the members of
 * OpenAI's segment the recognizers have no value for (seek, tokens, temperature, avg_logprob, compression_ratio,
 * no_speech_prob).
 */
std::string transcription_verbose_json(const Transcript & transcript, double duration, bool timestamps, bool segments = true, bool words = false);

/**
 * A transcription in OpenAI's diarized_json form: the task, the audio's duration, the text, a segment for each of the
 * `regions`, recognized as `parts`, and the usage as the duration. A segment's speaker is the one who speaks the most
 * in it as `speakers` give them, labeled A, B and on in the order the diarization numbers them, or "" where no one does;
 * each segment also carries "speakers", every speaker's parts of it, which OpenAI's segment does not define.
 */
std::string transcription_diarized_json(const Transcript & transcript, double duration, const std::vector<Region> & regions,
                                        const std::vector<Transcript> & parts, const std::vector<SpeakerSegment> & speakers);

/**
 * A transcription in OpenAI's diarized_json form from a model that tells who speaks: a segment for each of the
 * recognition's segments, its speaker labeled A, B and on in the order they first speak, or "" for words before the
 * model named one, and "speakers" with the segment's one speaker; or no segments and "untimed", why the model gives none.
 */
std::string transcription_diarized_json(const Transcript & transcript, double duration);

}  // namespace openai
