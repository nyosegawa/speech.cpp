#include "realtime-session.h"

#include <algorithm>
#include <cstdio>
#include <initializer_list>
#include <random>
#include <stdexcept>

#include "base64.h"
#include "json.h"

namespace realtime {

using openai::ApiError;

namespace {

/** The client events a session takes. */
const char * const kTaken = "session.update, input_audio_buffer.append, input_audio_buffer.commit and input_audio_buffer.clear";

ApiError refusal(const std::string & code, const std::string & param, const std::string & message) {
    return {400, message, param, code};
}

/** The value at `param` as an object, or an error that says it is not one. */
const JsonValue & object_at(const JsonValue & value, const std::string & param) {
    if (value.kind != JsonValue::Kind::Object) throw refusal("invalid_type", param, param + " is " + json_excerpt(value) + "; give an object.");
    return value;
}

/** The value at `param` as a string, or an error that says it is not one. */
const std::string & string_at(const JsonValue & value, const std::string & param) {
    if (value.kind != JsonValue::Kind::String) throw refusal("invalid_type", param, param + " is " + json_excerpt(value) + "; give a string.");
    return value.text;
}

/** Refuses a member of `object`, at `param` ("" for the event itself), that is not one of `taken`. */
void only(const JsonValue & object, const std::string & param, std::initializer_list<const char *> taken) {
    for (const auto & member : object.members) {
        const std::string & name = member.first;
        if (std::find_if(taken.begin(), taken.end(), [&](const char * t) { return name == t; }) != taken.end()) continue;
        std::string listed;
        for (const char * t : taken) listed += std::string(listed.empty() ? "" : ", ") + t;
        const std::string at = param.empty() ? name : param + "." + name;
        throw refusal("unknown_parameter", at, "Unknown parameter: '" + at + "'. speech.cpp's transcription session takes " + listed + " there.");
    }
}

/** A member that is present and not null. */
const JsonValue * given(const JsonValue & object, const char * name) {
    const JsonValue * value = object.member(name);
    return value && value->kind != JsonValue::Kind::Null ? value : nullptr;
}

/** 16-bit little-endian PCM as the library takes samples, each scaled as a 16-bit WAVE file is read. */
std::vector<float> samples_of(const std::string & pcm) {
    std::vector<float> out(pcm.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = (float) (int16_t) ((uint16_t) (unsigned char) pcm[2 * i] | (uint16_t) (unsigned char) pcm[2 * i + 1] << 8) / 32768.0f;
    }
    return out;
}

const char * const kTranscription = "session.audio.input.transcription";

/** The error of the library's failure of a transcription's option, its param the member of the session that set it. */
ApiError option_error(const Failure & failure) {
    ApiError e = openai::library_error(failure);
    if (failure.option() == speech_option_name(SPEECH_OPT_LANGUAGE) || failure.option() == speech_option_name(SPEECH_OPT_PROMPT)) {
        e.param = std::string(kTranscription) + "." + failure.option();
    }
    return e;
}

}  // namespace

std::vector<RequestOption> Session::Config::options() const {
    std::vector<RequestOption> out;
    if (language) out.push_back({SPEECH_OPT_LANGUAGE, *language});
    if (prompt) out.push_back({SPEECH_OPT_PROMPT, *prompt});
    return out;
}

Session::Session(Recognizer & recognizer, std::function<void(const std::string &)> send) : recognizer_(recognizer), send_(std::move(send)) {
    std::random_device device;
    char unique[17];
    std::snprintf(unique, sizeof unique, "%08x%08x", (unsigned) device(), (unsigned) device());
    unique_ = unique;
    id_ = "sess_" + unique_;
    worker_ = std::thread([this] { work(); });
}

Session::~Session() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closing_ = true;
        waiting_.clear();
        if (running_) running_->cancel();
    }
    changed_.notify_all();
    worker_.join();
}

void Session::open() {
    emit("session.created", ",\"session\":" + session_json());
}

std::string Session::new_id(const char * prefix) {
    char n[24];
    std::snprintf(n, sizeof n, "%06llx", (unsigned long long) counter_++);
    return prefix + unique_ + n;
}

void Session::emit(const std::string & type, const std::string & members) {
    std::lock_guard<std::mutex> lock(send_mutex_);
    send_("{\"event_id\":\"" + new_id("event_") + "\",\"type\":\"" + type + "\"" + members + "}");
}

void Session::error(const ApiError & e, const std::string & event_id) {
    std::string object = openai::error_object(e);
    object.insert(object.size() - 1, ",\"event_id\":" + (event_id.empty() ? std::string("null") : json_string(event_id)));
    emit("error", ",\"error\":" + object);
}

std::string Session::session_json() {
    std::string transcription = "null";
    if (config_.transcribe) {
        const std::string model = config_.model.value_or(recognizer_.held());
        transcription = "{" + (model.empty() ? "" : "\"model\":" + json_string(model) + ",") +
                        "\"language\":" + (config_.language ? json_string(*config_.language) : "null") +
                        (config_.prompt ? ",\"prompt\":" + json_string(*config_.prompt) : "") + "}";
    }
    return "{\"type\":\"transcription\",\"id\":\"" + id_ + "\",\"object\":\"realtime.transcription_session\",\"audio\":{\"input\":{\"format\":"
           "{\"type\":\"audio/pcm\",\"rate\":" + std::to_string(kRate) + "},\"transcription\":" + transcription +
           ",\"noise_reduction\":null,\"turn_detection\":null}},\"include\":null}";
}

void Session::receive(const std::string & message, bool text) {
    std::string event_id;
    try {
        if (!text) {
            throw refusal("invalid_value", "", "The message is binary. speech.cpp takes each client event as JSON in a text message, the audio "
                                               "in input_audio_buffer.append as base64.");
        }
        JsonValue event;
        try {
            event = parse_json(message);
        } catch (const std::invalid_argument & e) {
            throw refusal("invalid_json", "", std::string("The message cannot be read as JSON: ") + e.what() + ". Send one JSON object per event.");
        }
        if (event.kind != JsonValue::Kind::Object) throw refusal("invalid_json", "", "The message is not a JSON object. Send one JSON object per event.");
        if (const JsonValue * id = given(event, "event_id")) event_id = string_at(*id, "event_id");
        const JsonValue * type = event.member("type");
        if (!type) throw refusal("missing_required_parameter", "type", "The event has no \"type\"; speech.cpp takes " + std::string(kTaken) + ".");
        const std::string & name = string_at(*type, "type");
        if (name == "session.update") {
            update(event);
        } else if (name == "input_audio_buffer.append") {
            append(event);
        } else if (name == "input_audio_buffer.commit") {
            only(event, "", {"type", "event_id"});
            commit();
        } else if (name == "input_audio_buffer.clear") {
            only(event, "", {"type", "event_id"});
            clear();
        } else {
            throw refusal("unsupported_value", "type", "The event " + json_string(name) + " is not supported; speech.cpp's transcription session takes " +
                                                           kTaken + ".");
        }
    } catch (const ApiError & e) {
        error(e, event_id);
    } catch (const Failure & e) {
        error(option_error(e), event_id);
    }
}

void Session::update(const JsonValue & event) {
    only(event, "", {"type", "event_id", "session"});
    const JsonValue * member = given(event, "session");
    if (!member) throw refusal("missing_required_parameter", "session", "session.update needs \"session\", the configuration to change.");
    const JsonValue & session = object_at(*member, "session");
    only(session, "session", {"type", "audio", "include"});
    const JsonValue * type = session.member("type");
    if (!type) throw refusal("missing_required_parameter", "session.type", "session.type is required; give \"transcription\".");
    if (string_at(*type, "session.type") == "realtime") {
        throw refusal("unsupported_value", "session.type", "A conversation session (session.type \"realtime\") is not supported; speech.cpp's "
                                                           "Realtime sessions transcribe. Give session.type \"transcription\".");
    }
    if (type->text != "transcription") {
        throw refusal("invalid_value", "session.type", "session.type is " + json_string(type->text) + "; give \"transcription\".");
    }
    if (const JsonValue * include = given(session, "include")) {
        if (include->kind != JsonValue::Kind::Array) throw refusal("invalid_type", "session.include", "session.include is not an array.");
        for (const JsonValue & item : include->items) {
            const std::string & what = string_at(item, "session.include");
            throw refusal(what == "item.input_audio_transcription.logprobs" ? "unsupported_value" : "invalid_value", "session.include",
                          "session.include holds " + json_string(what) + ", which speech.cpp does not give; leave it empty.");
        }
    }
    Config next = config_;
    const JsonValue * audio = given(session, "audio");
    const JsonValue * input = audio ? given(object_at(*audio, "session.audio"), "input") : nullptr;
    if (audio) only(*audio, "session.audio", {"input"});
    if (input) {
        object_at(*input, "session.audio.input");
        only(*input, "session.audio.input", {"format", "transcription", "noise_reduction", "turn_detection"});
        if (const JsonValue * format = given(*input, "format")) {
            object_at(*format, "session.audio.input.format");
            only(*format, "session.audio.input.format", {"type", "rate"});
            const JsonValue * kind = format->member("type");
            if (!kind) throw refusal("missing_required_parameter", "session.audio.input.format.type", "The format needs its type; give \"audio/pcm\".");
            if (string_at(*kind, "session.audio.input.format.type") != "audio/pcm") {
                throw refusal(kind->text == "audio/pcmu" || kind->text == "audio/pcma" ? "unsupported_value" : "invalid_value",
                              "session.audio.input.format.type",
                              "The audio format " + json_string(kind->text) + " is not supported; speech.cpp takes \"audio/pcm\", 16-bit PCM at 24000 Hz.");
            }
            if (const JsonValue * rate = given(*format, "rate"); rate && !(rate->is_integer() && rate->text == std::to_string(kRate))) {
                throw refusal("invalid_value", "session.audio.input.format.rate",
                              "audio/pcm is " + std::to_string(kRate) + " Hz, not " + json_excerpt(*rate) + "; give rate 24000 or leave it out.");
            }
        }
        if (const JsonValue * transcription = input->member("transcription")) {
            next.transcribe = transcription->kind != JsonValue::Kind::Null;
            next.model.reset();
            next.language.reset();
            next.prompt.reset();
            if (next.transcribe) {
                object_at(*transcription, kTranscription);
                const std::string at = kTranscription;
                for (const char * own : {"keywords", "delay"}) {
                    if (given(*transcription, own)) {
                        throw refusal("unsupported_parameter", at + "." + own,
                                      at + "." + own + " is not supported; speech.cpp takes the model, the language or languages, and the prompt.");
                    }
                }
                only(*transcription, at, {"model", "language", "languages", "prompt", "keywords", "delay"});
                if (const JsonValue * model = given(*transcription, "model")) next.model = string_at(*model, at + ".model");
                if (const JsonValue * language = given(*transcription, "language")) next.language = string_at(*language, at + ".language");
                if (const JsonValue * languages = given(*transcription, "languages")) {
                    if (next.language) throw refusal("invalid_value", at + ".languages", "Give the language or the languages, not both.");
                    if (languages->kind != JsonValue::Kind::Array || languages->items.empty()) {
                        throw refusal("invalid_type", at + ".languages", at + ".languages is " + json_excerpt(*languages) + "; give an array of one tag.");
                    }
                    if (languages->items.size() > 1) {
                        throw refusal("unsupported_value", at + ".languages",
                                      "A recognition takes one language or leaves it to the model; give one language, or none for the model to tell.");
                    }
                    next.language = string_at(languages->items[0], at + ".languages");
                }
                if (const JsonValue * prompt = given(*transcription, "prompt")) next.prompt = string_at(*prompt, at + ".prompt");
            }
        }
        if (given(*input, "noise_reduction")) {
            throw refusal("unsupported_parameter", "session.audio.input.noise_reduction",
                          "Noise reduction is not supported; give session.audio.input.noise_reduction null or leave it out.");
        }
        if (const JsonValue * turns = given(*input, "turn_detection")) {
            object_at(*turns, "session.audio.input.turn_detection");
            const JsonValue * kind = turns->member("type");
            const std::string what = kind && kind->kind == JsonValue::Kind::String ? kind->text : "";
            throw refusal(what == "server_vad" || what == "semantic_vad" ? "unsupported_value" : "invalid_value", "session.audio.input.turn_detection",
                          "turn_detection " + (what.empty() ? json_excerpt(*turns) : json_string(what)) +
                              " is not supported; speech.cpp transcribes what the client commits. Give session.audio.input.turn_detection null and "
                              "send input_audio_buffer.commit at the end of each utterance.");
        }
    }
    if (next.transcribe) {
        try {
            recognizer_.check(next.model.value_or(""), next.options());
        } catch (ApiError & e) {
            if (e.param.empty()) e.param = std::string(kTranscription) + ".model";
            throw;
        }
    }
    config_ = next;
    emit("session.updated", ",\"session\":" + session_json());
}

void Session::append(const JsonValue & event) {
    only(event, "", {"type", "event_id", "audio"});
    const JsonValue * audio = event.member("audio");
    if (!audio) throw refusal("missing_required_parameter", "audio", "input_audio_buffer.append needs \"audio\", base64 of 16-bit PCM at 24000 Hz.");
    std::string pcm;
    try {
        pcm = base64_decode(string_at(*audio, "audio"));
    } catch (const std::invalid_argument & e) {
        throw refusal("invalid_value", "audio", std::string("The audio is not base64: ") + e.what() + ".");
    }
    if (pcm.size() % 2) {
        throw refusal("invalid_value", "audio", "The audio has " + std::to_string(pcm.size()) + " bytes, and 16-bit PCM has two for each sample.");
    }
    buffer_ += pcm;
}

void Session::commit() {
    if (buffer_.empty()) {
        throw refusal("input_audio_buffer_commit_empty", "", "The input audio buffer is empty; append audio before committing it.");
    }
    Commit c{new_id("item_"), samples_of(buffer_), config_};
    buffer_.clear();
    emit("input_audio_buffer.committed",
         ",\"previous_item_id\":" + (previous_item_.empty() ? std::string("null") : json_string(previous_item_)) + ",\"item_id\":" + json_string(c.item));
    previous_item_ = c.item;
    if (!c.config.transcribe) return;
    std::lock_guard<std::mutex> lock(mutex_);
    waiting_.push_back(std::move(c));
    changed_.notify_all();
}

void Session::clear() {
    buffer_.clear();
    emit("input_audio_buffer.cleared", "");
}

void Session::work() {
    for (;;) {
        Commit c;
        std::shared_ptr<Cancellation> cancellation;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait(lock, [&] { return closing_ || !waiting_.empty(); });
            if (closing_) return;
            c = std::move(waiting_.front());
            waiting_.pop_front();
            running_ = cancellation = std::make_shared<Cancellation>();
        }
        const std::string item = ",\"item_id\":" + json_string(c.item) + ",\"content_index\":0";
        try {
            const std::optional<Transcript> t = recognizer_.transcribe(c.config.model.value_or(""), c.samples, c.config.options(), *cancellation, c.item);
            if (t) {
                // The library gives the text once the recognition has ended, so the one delta carries all of it, for a
                // client that builds the text from the deltas.
                if (!t->text.empty()) emit("conversation.item.input_audio_transcription.delta", item + ",\"delta\":" + json_string(t->text));
                std::string languages;
                for (const std::string & tag : t->languages) languages += (languages.empty() ? "" : ",") + std::string("{\"code\":") + json_string(tag) + "}";
                emit("conversation.item.input_audio_transcription.completed",
                     item + ",\"transcript\":" + json_string(t->text) + ",\"usage\":{\"type\":\"duration\",\"seconds\":" +
                         json_number((double) c.samples.size() / kRate) + "}" + (languages.empty() ? "" : ",\"languages\":[" + languages + "]") +
                         ",\"stop\":\"" + speech_stop_name(t->stop) + "\"");
            }
        } catch (const ApiError & e) {
            emit("conversation.item.input_audio_transcription.failed", item + ",\"error\":" + openai::error_object(e));
        } catch (const Failure & e) {
            emit("conversation.item.input_audio_transcription.failed", item + ",\"error\":" + openai::error_object(option_error(e)));
        } catch (const std::exception & e) {
            emit("conversation.item.input_audio_transcription.failed",
                 item + ",\"error\":" + openai::error_object({500, e.what(), "", speech_status_name(SPEECH_ERROR_INTERNAL)}));
        }
        std::lock_guard<std::mutex> lock(mutex_);
        running_.reset();
    }
}

}  // namespace realtime
