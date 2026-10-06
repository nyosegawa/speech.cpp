#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "base64.h"
#include "flat-json.h"
#include "speech.h"
#include "wav.h"

// OpenAI's audio API as speech.cpp reads and writes it: the create speech request, the create transcription request,
// the error object, and the events of an SSE stream. The shapes follow OpenAI's API reference, components
// CreateSpeechRequest, CreateTranscriptionRequest, CreateTranscriptionResponseJson, AudioResponseFormat, Error,
// SpeechAudioDeltaEvent and SpeechAudioDoneEvent and the path /audio/transcriptions of
// github.com/openai/openai-openapi at commit 31af4fc (2026-10-05).

/** An answer that is not audio: an HTTP status and an error in OpenAI's shape. */
struct ApiError {
    int status;
    std::string message;
    std::string param;
    std::string code;
};

/** The members of OpenAI's error object. */
inline std::string error_object(const ApiError & e) {
    const char * type = e.status >= 500 ? "server_error" : "invalid_request_error";
    return "{\"message\":" + json_string(e.message) + ",\"type\":\"" + type + "\",\"param\":" +
           (e.param.empty() ? "null" : json_string(e.param)) + ",\"code\":" +
           (e.code.empty() ? "null" : json_string(e.code)) + "}";
}

inline std::string error_json(const ApiError & e) {
    return "{\"error\":" + error_object(e) + "}";
}

/** A create speech request as read from its JSON body; a number left out is one the request does not set. */
struct SpeechRequest {
    std::string input, voice, language;
    std::string format = "wav";
    bool sse = false;
    uint64_t seed = 0;
    std::optional<double> speed, seconds, duration_scale;
};

/**
 * Reads a create speech request. A member OpenAI's API has and speech.cpp does not follow ("instructions",
 * another format) is refused rather than ignored, and so is a member neither has; a null member is one left out.
 * The model's name, which "model" may give, is the one `/v1/models` lists. Anything it cannot read throws an
 * ApiError.
 */
inline SpeechRequest read_request(const std::string & body, const std::string & model_name) {
    FlatJson json;
    try {
        json = parse_flat_json(body);
    } catch (const std::exception & e) {
        throw ApiError{400, std::string("The body cannot be read as a JSON object: ") + e.what() + ". Send one JSON object.", "", ""};
    }
    for (auto it = json.begin(); it != json.end();) {
        it = it->second.kind == FlatValue::OTHER && it->second.text == "null" ? json.erase(it) : std::next(it);
    }
    static const char * known[] = {"model", "input", "voice", "response_format", "speed", "stream_format",
                                   "language", "seed", "seconds", "duration_scale"};
    for (const auto & member : json) {
        const std::string & key = member.first;
        if (std::find_if(std::begin(known), std::end(known), [&](const char * k) { return key == k; }) == std::end(known)) {
            throw ApiError{400, "Unrecognized request argument supplied: " + key + ". speech.cpp takes model, input, voice, "
                           "response_format, speed, stream_format, language, seed, seconds and duration_scale.", key, "unknown_parameter"};
        }
    }
    auto text = [&](const char * key, bool required) -> std::string {
        const auto it = json.find(key);
        if (it == json.end()) {
            if (required) throw ApiError{400, std::string("The request has no \"") + key + "\"; it is required.", key, "missing_required_parameter"};
            return "";
        }
        if (it->second.kind != FlatValue::STRING) {
            throw ApiError{400, std::string("\"") + key + "\" is " + it->second.text + "; give it as a string.", key, "invalid_type"};
        }
        return it->second.text;
    };
    auto number = [&](const char * key) -> std::optional<double> {
        const auto it = json.find(key);
        if (it == json.end()) return std::nullopt;
        char * end = nullptr;
        const double v = std::strtod(it->second.text.c_str(), &end);
        if (it->second.kind != FlatValue::NUMBER || *end != '\0' || !std::isfinite(v)) {
            throw ApiError{400, std::string("\"") + key + "\" is " + json_string(it->second.text) + "; give it as a number.", key, "invalid_type"};
        }
        return v;
    };

    if (json.count("model") && text("model", false) != model_name) {
        throw ApiError{404, "The model " + json_string(text("model", false)) + " is not served here; this server serves " +
                       json_string(model_name) + ". Name it in \"model\" or leave \"model\" out.", "model", "model_not_found"};
    }
    SpeechRequest r;
    r.input = text("input", true);
    r.voice = text("voice", true);
    r.language = text("language", false);
    if (json.count("response_format")) r.format = text("response_format", false);
    if (r.format != "wav" && r.format != "pcm") {
        throw ApiError{400, "The response_format " + json_string(r.format) + " is not supported; speech.cpp answers with \"wav\" or \"pcm\".",
                       "response_format", "unsupported_value"};
    }
    const std::string stream_format = json.count("stream_format") ? text("stream_format", false) : "audio";
    if (stream_format != "audio" && stream_format != "sse") {
        throw ApiError{400, "The stream_format " + json_string(stream_format) + " is not supported; give \"audio\" or \"sse\".",
                       "stream_format", "unsupported_value"};
    }
    r.sse = stream_format == "sse";
    if (r.sse && r.format != "pcm") {
        throw ApiError{400, "stream_format \"sse\" carries \"pcm\" audio; give \"response_format\": \"pcm\" with it.",
                       "response_format", "unsupported_value"};
    }
    const auto seed = json.find("seed");
    if (seed == json.end()) {
        // The library takes a seed from 0 to 2^53 - 1, the integers a JSON reader in JavaScript holds exactly.
        r.seed = (std::random_device{}() * 0x100000000ull + std::random_device{}()) & ((1ull << 53) - 1);
    } else {
        const std::string & t = seed->second.text;
        const bool digits = seed->second.kind == FlatValue::NUMBER && !t.empty() &&
                            std::all_of(t.begin(), t.end(), [](char c) { return c >= '0' && c <= '9'; });
        errno = 0;
        r.seed = digits ? std::strtoull(t.c_str(), nullptr, 10) : 0;
        if (!digits || errno == ERANGE) {
            throw ApiError{400, "\"seed\" is " + json_string(t) + "; give an integer from 0 to 9007199254740991.", "seed", "invalid_type"};
        }
    }
    r.speed = number("speed");
    r.seconds = number("seconds");
    r.duration_scale = number("duration_scale");
    return r;
}

/** One SSE event; OpenAI's stream sends each event as a data line alone. */
inline std::string sse_event(const std::string & json) {
    return "data: " + json + "\n\n";
}

/** A speech.audio.delta event with a chunk of 16-bit PCM. */
inline std::string sse_delta(const std::string & pcm) {
    return sse_event("{\"type\":\"speech.audio.delta\",\"audio\":\"" + base64((const uint8_t *) pcm.data(), pcm.size()) + "\"}");
}

/**
 * The speech.audio.done event. OpenAI's carries the usage in tokens; speech.cpp counts none, so it carries no
 * usage, which would be made up, and gives the number of samples instead.
 */
inline std::string sse_done(size_t samples) {
    return sse_event("{\"type\":\"speech.audio.done\",\"samples\":" + std::to_string(samples) + "}");
}

/**
 * An error after an SSE stream has begun. OpenAI's speech stream defines no error event; this one carries the
 * error object of an error response: {"type": "error", "error": {...}}.
 */
inline std::string sse_error(const ApiError & e) {
    return sse_event("{\"type\":\"error\",\"error\":" + error_object(e) + "}");
}

/** One part of a multipart/form-data body: a field, or a file when it has a filename. */
struct FormPart {
    std::string name, content, filename;
    bool file = false;
};

/** A create transcription request as read from its form, with the file's audio decoded. */
struct TranscriptionRequest {
    std::vector<float> samples;
    int sample_rate = 0;
    std::string language;
    std::string format = "json";
};

/**
 * Reads a create transcription request from the parts of its form. The file is decoded as WAV, the one format
 * speech.cpp reads, and its channels are averaged; any other file is refused, as is a member speech.cpp does not
 * follow (prompt, temperature, timestamps and the rest) and a member given twice. The model's name, which "model"
 * may give, is the one `/v1/models` lists. Anything it cannot read throws an ApiError.
 */
inline TranscriptionRequest read_transcription_request(bool multipart, const std::vector<FormPart> & parts,
                                                       const std::string & model_name) {
    if (!multipart) {
        throw ApiError{400, "The body is not multipart/form-data. Send the audio as the form's \"file\", as OpenAI's create "
                       "transcription takes it.", "", ""};
    }
    static const char * known[] = {"file", "model", "language", "response_format"};
    const FormPart * given[4] = {};
    for (const FormPart & part : parts) {
        const auto k = std::find_if(std::begin(known), std::end(known), [&](const char * name) { return part.name == name; });
        if (k == std::end(known)) {
            throw ApiError{400, "Unrecognized request argument supplied: " + part.name + ". speech.cpp takes file, model, language "
                           "and response_format.", part.name, "unknown_parameter"};
        }
        const FormPart *& slot = given[k - std::begin(known)];
        if (slot) throw ApiError{400, "\"" + part.name + "\" is given twice; give it once.", part.name, "invalid_value"};
        slot = &part;
    }
    const FormPart * file = given[0], * model = given[1], * language = given[2], * format = given[3];
    if (model && model->content != model_name) {
        throw ApiError{404, "The model " + json_string(model->content) + " is not served here; this server serves " +
                       json_string(model_name) + ". Name it in \"model\" or leave \"model\" out.", "model", "model_not_found"};
    }
    if (!file) throw ApiError{400, "The request has no \"file\"; it is required.", "file", "missing_required_parameter"};
    if (!file->file) throw ApiError{400, "\"file\" is a field; send it as a file, with a filename.", "file", "invalid_type"};
    TranscriptionRequest r;
    if (format) r.format = format->content;
    if (r.format != "json" && r.format != "text") {
        throw ApiError{400, "The response_format " + json_string(r.format) + " is not supported; speech.cpp answers with \"json\" "
                       "or \"text\", since it gives neither timestamps nor speakers.", "response_format", "unsupported_value"};
    }
    if (language) r.language = language->content;
    const std::string & bytes = file->content;
    if (bytes.size() < 12 || bytes.compare(0, 4, "RIFF") != 0 || bytes.compare(8, 4, "WAVE") != 0) {
        throw ApiError{400, "The file " + json_string(file->filename) + " is not a WAV file. speech.cpp reads WAV alone (16-, 24- "
                       "or 32-bit PCM or 32-bit float); convert the audio first, for example with ffmpeg -i in.mp3 out.wav.",
                       "file", "unsupported_value"};
    }
    try {
        const Wav wav = parse_wav(bytes, file->filename);
        r.samples = wav.mono();
        r.sample_rate = wav.sample_rate;
    } catch (const std::exception & e) {
        throw ApiError{400, std::string("The file cannot be read: ") + e.what() + ".", "file", "invalid_value"};
    }
    return r;
}

/**
 * A transcription in OpenAI's json format. OpenAI's also carries the usage, in tokens or in seconds billed, which
 * speech.cpp does not count, so it carries the text alone.
 */
inline std::string transcription_json(const std::string & text) {
    return "{\"text\":" + json_string(text) + "}";
}
