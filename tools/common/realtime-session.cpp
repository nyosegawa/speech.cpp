#include "realtime-session.h"

#include <algorithm>
#include <cstdio>
#include <initializer_list>
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

}  // namespace

std::vector<RequestOption> Session::Config::options() const {
    std::vector<RequestOption> out;
    if (language) out.push_back({SPEECH_OPT_LANGUAGE, *language});
    if (prompt) out.push_back({SPEECH_OPT_PROMPT, *prompt});
    return out;
}

Session::Session(Recognizer & recognizer, std::function<void(const std::string &)> send, size_t limit, const std::string & model)
    : recognizer_(recognizer), events_(std::move(send)), limit_(limit) {
    if (!model.empty()) config_.model = model;
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
    events_.emit("session.created", ",\"session\":" + session_json());
}

std::string Session::session_json() {
    std::string transcription = "null";
    if (config_.transcribe) {
        const std::string model = config_.model.value_or(recognizer_.held());
        transcription = "{" + (model.empty() ? "" : "\"model\":" + json_string(model) + ",") +
                        "\"language\":" + (config_.language ? json_string(*config_.language) : "null") +
                        (config_.prompt ? ",\"prompt\":" + json_string(*config_.prompt) : "") + "}";
    }
    return "{\"type\":\"transcription\",\"id\":\"" + events_.session_id() + "\",\"object\":\"realtime.transcription_session\",\"audio\":{\"input\":{\"format\":"
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
        events_.error(e, event_id);
    } catch (const Failure &) {
        events_.error(error_of(std::current_exception()), event_id);
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
                // A member given sets its value, null taking it away, and one left out keeps it, as OpenAI's
                // session.update changes only the members it holds.
                const auto text = [&](const char * name, std::optional<std::string> & into) {
                    const JsonValue * value = transcription->member(name);
                    if (value) into = value->kind == JsonValue::Kind::Null ? std::nullopt : std::optional<std::string>(string_at(*value, at + "." + name));
                };
                text("model", next.model);
                text("language", next.language);
                text("prompt", next.prompt);
                if (const JsonValue * languages = transcription->member("languages")) {
                    if (transcription->member("language")) throw refusal("invalid_value", at + ".languages", "Give the language or the languages, not both.");
                    next.language.reset();
                    if (languages->kind != JsonValue::Kind::Null) {
                        if (languages->kind != JsonValue::Kind::Array || languages->items.empty()) {
                            throw refusal("invalid_type", at + ".languages", at + ".languages is " + json_excerpt(*languages) + "; give an array of one tag.");
                        }
                        if (languages->items.size() > 1) {
                            throw refusal("unsupported_value", at + ".languages",
                                          "A recognition takes one language or leaves it to the model; give one language, or none for the model to tell.");
                        }
                        next.language = string_at(languages->items[0], at + ".languages");
                    }
                }
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
    events_.emit("session.updated", ",\"session\":" + session_json());
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
    size_t held = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        held = buffer_.size() + committed_;
    }
    if (held + pcm.size() > limit_) {
        char message[320];
        std::snprintf(message, sizeof message,
                      "The session holds %.1f s of audio not yet transcribed, and this append would take it past the %.1f s this server "
                      "holds; commit shorter utterances, wait for their transcriptions, or clear the buffer.",
                      (double) held / 2 / kRate, (double) limit_ / 2 / kRate);
        throw refusal("input_audio_buffer_full", "audio", message);
    }
    buffer_ += pcm;
}

void Session::commit() {
    if (buffer_.empty()) {
        throw refusal("input_audio_buffer_commit_empty", "", "The input audio buffer is empty; append audio before committing it.");
    }
    Commit c{commits_++, samples_of(buffer_), config_, nullptr, std::nullopt};
    const size_t bytes = buffer_.size();
    buffer_.clear();
    // The commit takes its place on the model as it is accepted, so that it runs before any request that arrives later.
    if (c.config.transcribe) {
        try {
            c.reservation = recognizer_.reserve(c.config.model.value_or(""));
        } catch (const ApiError & e) {
            c.refused = e;
            c.samples.clear();
        }
    }
    utterances::Event committed;
    committed.kind = utterances::Event::Kind::Committed;
    committed.utterance = c.number;
    committed.recognized = c.config.transcribe;
    events_.utterance(committed);
    if (!c.config.transcribe) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (c.reservation) committed_ += bytes;
    waiting_.push_back(std::move(c));
    changed_.notify_all();
}

void Session::clear() {
    buffer_.clear();
    events_.emit("input_audio_buffer.cleared", "");
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
        utterances::Event e;
        e.utterance = c.number;
        try {
            if (c.refused) throw *c.refused;
            std::optional<Transcript> t =
                recognizer_.transcribe(*c.reservation, c.samples, c.config.options(), *cancellation, events_.item(c.number));
            // The model's turn passes as soon as the recognition has ended.
            c.reservation.reset();
            if (t) {
                // The library gives the text once the recognition has ended, so the one delta carries all of it, for a
                // client that builds the text from the deltas.
                if (!t->text.empty()) {
                    e.kind = utterances::Event::Kind::Delta;
                    e.delta = t->text;
                    events_.utterance(e);
                }
                e.kind = utterances::Event::Kind::Completed;
                e.transcript = std::move(*t);
                e.duration = (double) c.samples.size() / kRate;
                events_.utterance(e);
            }
        } catch (...) {
            e.kind = utterances::Event::Kind::Failed;
            e.failure = std::current_exception();
            events_.utterance(e);
        }
        c.reservation.reset();
        std::lock_guard<std::mutex> lock(mutex_);
        committed_ -= c.samples.size() * 2;
        running_.reset();
    }
}

}  // namespace realtime
