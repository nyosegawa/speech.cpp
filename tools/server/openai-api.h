#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <random>
#include <string>

#include "flat-json.h"
#include "speech.h"

// OpenAI's speech API as speech.cpp reads and writes it: the create speech request, the error object, and the
// events of an SSE stream. The shapes follow OpenAI's API reference, components CreateSpeechRequest, Error,
// SpeechAudioDeltaEvent and SpeechAudioDoneEvent of github.com/openai/openai-openapi at commit 31af4fc
// (2026-10-05).

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

inline std::string base64(const uint8_t * data, size_t n) {
    static const char * table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t) data[i] << 16 | (i + 1 < n ? (uint32_t) data[i + 1] << 8 : 0) | (i + 2 < n ? data[i + 2] : 0);
        out += table[(v >> 18) & 63];
        out += table[(v >> 12) & 63];
        out += i + 1 < n ? table[(v >> 6) & 63] : '=';
        out += i + 2 < n ? table[v & 63] : '=';
    }
    return out;
}

/** A create speech request as read from its JSON body. */
struct SpeechRequest {
    std::string input, voice, language;
    std::string format = "wav";
    bool sse = false;
    uint64_t seed = 0;
    double speed, seconds, duration_scale;
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
    auto number = [&](const char * key, double absent) {
        const auto it = json.find(key);
        if (it == json.end()) return absent;
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
        r.seed = std::random_device{}() * 0x100000000ull + std::random_device{}();
    } else {
        const std::string & t = seed->second.text;
        const bool digits = seed->second.kind == FlatValue::NUMBER && !t.empty() &&
                            std::all_of(t.begin(), t.end(), [](char c) { return c >= '0' && c <= '9'; });
        errno = 0;
        r.seed = digits ? std::strtoull(t.c_str(), nullptr, 10) : 0;
        if (!digits || errno == ERANGE) {
            throw ApiError{400, "\"seed\" is " + json_string(t) + "; give an integer from 0 to 18446744073709551615.", "seed", "invalid_type"};
        }
    }
    const speech_request defaults = speech_request_default();
    r.speed = number("speed", defaults.speed);
    r.seconds = number("seconds", defaults.seconds);
    r.duration_scale = number("duration_scale", defaults.duration_scale);
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
