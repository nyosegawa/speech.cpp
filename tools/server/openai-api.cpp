#include "openai-api.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>

#include "base64.h"
#include "error.h"
#include "json-reader.h"
#include "json.h"
#include "wav.h"

namespace openai {

namespace {

ApiError unknown_member(const std::string & name, const std::string & takes) {
    return {400, "Unrecognized request argument supplied: " + name + ". speech.cpp takes " + takes + ".", name, "unknown_parameter"};
}

ApiError not_served(const std::string & asked, const std::string & model_name) {
    return {404, "The model " + json_string(asked) + " is not served here; this server serves " + json_string(model_name) +
                     ". Name it in \"model\" or leave \"model\" out.",
            "model", "model_not_found"};
}

}  // namespace

ApiError library_error(const Failure & failure) {
    const std::string & option = failure.option();
    const std::string param = option == "text" ? "input" : option == "audio" ? "file" : option;
    const std::string & code = failure.code();
    if (code == speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT)) return {400, failure.what(), param, "invalid_value"};
    if (code == speech_status_name(SPEECH_ERROR_UNSUPPORTED)) return {400, failure.what(), param, "unsupported_parameter"};
    if (code == speech_status_name(SPEECH_ERROR_OUT_OF_RANGE)) return {400, failure.what(), param, "unsupported_value"};
    return {500, failure.what(), param, code};
}

std::string error_object(const ApiError & e) {
    const char * type = e.status >= 500 ? "server_error" : "invalid_request_error";
    return "{\"message\":" + json_string(e.message) + ",\"type\":\"" + type + "\",\"param\":" + (e.param.empty() ? "null" : json_string(e.param)) +
           ",\"code\":" + (e.code.empty() ? "null" : json_string(e.code)) + "}";
}

std::string error_json(const ApiError & e) {
    return "{\"error\":" + error_object(e) + "}";
}

SpeechRequest read_speech_request(const std::string & body, const std::string & model_name) {
    JsonValue json;
    try {
        json = parse_json(body);
    } catch (const std::invalid_argument & e) {
        throw ApiError{400, std::string("The body cannot be read as JSON: ") + e.what() + ". Send one JSON object.", "", ""};
    }
    if (json.kind != JsonValue::Kind::Object) throw ApiError{400, "The body is not a JSON object. Send one JSON object.", "", ""};
    const std::vector<std::string> own = {"model", "input", "response_format", "stream_format"};
    SpeechRequest r;
    std::vector<std::pair<std::string, const JsonValue *>> text;
    for (const auto & [name, value] : json.members) {
        if (value.kind == JsonValue::Kind::Null) continue;
        speech_option option;
        if (std::find(own.begin(), own.end(), name) != own.end()) {
            if (value.kind != JsonValue::Kind::String) {
                throw ApiError{400, "\"" + name + "\" is " + json_excerpt(value) + "; give it as a string.", name, "invalid_type"};
            }
            text.push_back({name, &value});
        } else if (option_named(name, option)) {
            try {
                r.options.push_back({option, option_from_json(option, value)});
            } catch (const std::invalid_argument & e) {
                throw ApiError{400, std::string(e.what()) + ".", name, "invalid_type"};
            }
        } else {
            throw unknown_member(name, "model, input, response_format, stream_format and the options " + option_names());
        }
    }
    const auto member = [&](const std::string & name) -> const std::string * {
        for (const auto & [key, value] : text) {
            if (key == name) return &value->text;
        }
        return nullptr;
    };
    if (const std::string * model = member("model"); model && *model != model_name) throw not_served(*model, model_name);
    const std::string * input = member("input");
    if (!input) throw ApiError{400, "The request has no \"input\"; it is required.", "input", "missing_required_parameter"};
    r.input = *input;
    if (const std::string * format = member("response_format")) r.format = *format;
    if (r.format != "wav" && r.format != "pcm") {
        throw ApiError{400, "The response_format " + json_string(r.format) + " is not supported; speech.cpp answers with \"wav\" or \"pcm\".",
                       "response_format", "unsupported_value"};
    }
    const std::string stream_format = member("stream_format") ? *member("stream_format") : "audio";
    if (stream_format != "audio" && stream_format != "sse") {
        throw ApiError{400, "The stream_format " + json_string(stream_format) + " is not supported; give \"audio\" or \"sse\".", "stream_format",
                       "unsupported_value"};
    }
    r.sse = stream_format == "sse";
    if (r.sse && r.format != "pcm") {
        throw ApiError{400, "stream_format \"sse\" carries \"pcm\" audio; give \"response_format\": \"pcm\" with it.", "response_format",
                       "unsupported_value"};
    }
    return r;
}

TranscriptionRequest read_transcription_request(bool multipart, const std::vector<FormPart> & parts, const std::string & model_name,
                                                bool gives_times) {
    if (!multipart) {
        throw ApiError{400, "The body is not multipart/form-data. Send the audio as the form's \"file\", as OpenAI's create transcription takes it.",
                       "", ""};
    }
    static const char * known[] = {"file", "model", "language", "prompt", "response_format", "decoding", "timestamp_granularities[]"};
    constexpr std::ptrdiff_t kGranularities = 6;
    const FormPart * given[kGranularities] = {};
    std::vector<std::string> granularities;
    for (const FormPart & part : parts) {
        const auto k = std::find_if(std::begin(known), std::end(known), [&](const char * name) { return part.name == name; });
        if (k == std::end(known)) throw unknown_member(part.name, "file, model, language, prompt, response_format, decoding and timestamp_granularities[]");
        if (k - std::begin(known) == kGranularities) {
            granularities.push_back(part.content);
            continue;
        }
        const FormPart *& slot = given[k - std::begin(known)];
        if (slot) throw ApiError{400, "\"" + part.name + "\" is given twice; give it once.", part.name, "invalid_value"};
        slot = &part;
    }
    const FormPart * file = given[0], * model = given[1], * language = given[2], * prompt = given[3], * format = given[4], * decoding = given[5];
    if (model && model->content != model_name) throw not_served(model->content, model_name);
    if (!file) throw ApiError{400, "The request has no \"file\"; it is required.", "file", "missing_required_parameter"};
    if (!file->file) throw ApiError{400, "\"file\" is a field; send it as a file, with a filename.", "file", "invalid_type"};
    TranscriptionRequest r;
    if (format) r.format = format->content;
    if (r.format != "json" && r.format != "text" && r.format != "verbose_json") {
        throw ApiError{400, "The response_format " + json_string(r.format) + " is not supported; speech.cpp answers with \"json\", \"text\" or "
                       "\"verbose_json\", since it gives neither subtitles nor speakers.",
                       "response_format", "unsupported_value"};
    }
    for (const std::string & g : granularities) {
        if (g != "segment") {
            throw ApiError{400, "The timestamp granularity " + json_string(g) + " is not supported; speech.cpp gives \"segment\".",
                           "timestamp_granularities[]", "unsupported_value"};
        }
        if (r.format != "verbose_json") {
            throw ApiError{400, "timestamp_granularities[] needs \"response_format\": \"verbose_json\", which carries the segments.",
                           "timestamp_granularities[]", "invalid_value"};
        }
    }
    if (language) r.options.push_back({SPEECH_OPT_LANGUAGE, language->content});
    if (prompt) r.options.push_back({SPEECH_OPT_PROMPT, prompt->content});
    if (decoding) r.options.push_back({SPEECH_OPT_DECODING, decoding->content});
    // OpenAI's verbose_json requires the language, the duration and the text alone, and its timestamp granularity is
    // segment unless the request names one: a model that gives no times answers it without segments, and refuses a
    // granularity the request names.
    r.timestamps = r.format == "verbose_json" && (gives_times || !granularities.empty());
    if (r.timestamps) r.options.push_back({SPEECH_OPT_TIMESTAMPS, true});
    const std::string & bytes = file->content;
    if (bytes.size() < 12 || bytes.compare(0, 4, "RIFF") != 0 || bytes.compare(8, 4, "WAVE") != 0) {
        throw ApiError{400, "The file " + json_string(file->filename) + " is not a WAV file. speech.cpp reads WAV alone (16-, 24- or 32-bit PCM or "
                       "32-bit float); convert the audio first, for example with ffmpeg -i in.mp3 out.wav.",
                       "file", "unsupported_value"};
    }
    try {
        const Wav wav = parse_wav(bytes, file->filename);
        r.samples = wav.mono();
        r.sample_rate = wav.sample_rate;
    } catch (const Error & e) {
        throw ApiError{400, std::string("The file cannot be read: ") + e.what() + ".", "file", "invalid_value"};
    }
    return r;
}

std::string sse_delta(const std::string & pcm) {
    return "data: {\"type\":\"speech.audio.delta\",\"audio\":\"" + base64((const uint8_t *) pcm.data(), pcm.size()) + "\"}\n\n";
}

std::string sse_done(int64_t seed, uint64_t samples, const char * stop) {
    return "data: {\"type\":\"speech.audio.done\",\"seed\":" + std::to_string(seed) + ",\"samples\":" + std::to_string(samples) +
           ",\"stop\":\"" + stop + "\"}\n\n";
}

std::string sse_error(const ApiError & e) {
    return "data: {\"type\":\"error\",\"error\":" + error_object(e) + "}\n\n";
}

std::string transcription_json(const std::string & text) {
    return "{\"text\":" + json_string(text) + "}";
}

std::string transcription_verbose_json(const speech_result * result, double duration, bool timestamps) {
    std::string language;
    for (size_t i = 0; i < speech_result_language_count(result); i++) language += (i ? "," : "") + std::string(speech_result_language(result, i));
    std::string out = "{\"task\":\"transcribe\"" + (language.empty() ? "" : ",\"language\":" + json_string(language)) +
                      ",\"duration\":" + json_number(duration) + ",\"text\":" + json_string(speech_result_text(result));
    if (!timestamps) return out + "}";
    out += ",\"segments\":[";
    for (size_t i = 0; i < speech_result_segment_count(result); i++) {
        double start = 0, end = 0;
        const char * text = nullptr;
        check(speech_result_segment(result, i, &start, &end, &text));
        out += std::string(i ? "," : "") + "{\"id\":" + std::to_string(i) + ",\"start\":" + json_number(start) + ",\"end\":" + json_number(end) +
               ",\"text\":" + json_string(text) + "}";
    }
    return out + "]}";
}

}  // namespace openai
