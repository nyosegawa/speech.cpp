#include "dialogue.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "jobs.h"
#include "json-reader.h"
#include "json.h"
#include "log.h"
#include "request-options.h"

namespace server {

namespace {

using openai::ApiError;

std::string error_event(const ApiError & e) {
    return "{\"type\":\"error\",\"error\":" + openai::error_object(e) + "}";
}

/** What the frame callback needs: the socket, and the frames held back until the client is told the conversation started. */
struct Talking {
    httplib::ws::WebSocket & ws;
    bool started = false;
    /** The frames passed before "started" went out, the one the prompt gives, with their text. */
    std::vector<std::pair<std::vector<int16_t>, std::string>> early;
    bool gone = false;
    uint64_t frames = 0, samples = 0;
    std::vector<int16_t> pcm;
};

void send_frame(Talking & t, const std::vector<int16_t> & pcm, const std::string & text) {
    if (!text.empty() && !t.ws.send("{\"type\":\"text\",\"text\":" + json_string(text) + "}")) t.gone = true;
    if (!t.gone && !t.ws.send(reinterpret_cast<const char *>(pcm.data()), pcm.size() * sizeof(int16_t))) t.gone = true;
}

int on_frame(const float * s, size_t n, const char * text, void * user_data) {
    Talking & t = *static_cast<Talking *>(user_data);
    t.pcm.resize(n);
    for (size_t i = 0; i < n; i++) t.pcm[i] = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s[i])) * 32767.0f);
    if (t.started) {
        send_frame(t, t.pcm, text);
    } else {
        t.early.emplace_back(t.pcm, text);
    }
    t.frames++;
    t.samples += n;
    // A client that went away stops the conversation at its next frame.
    return t.gone ? 1 : 0;
}

/** The start message's request options and the rate of the audio the client sends, `model_rate` unless it gives one. */
std::vector<RequestOption> read_start(const JsonValue & message, int model_rate, int & rate) {
    rate = model_rate;
    std::vector<RequestOption> options;
    for (const auto & [name, value] : message.members) {
        if (name == "type") continue;
        if (name == "sample_rate") {
            if (value.kind != JsonValue::Kind::Number || !value.is_integer() || value.text.size() > 7 || value.text[0] == '-' ||
                std::stoi(value.text) < 1) {
                throw ApiError{400, "sample_rate is the rate of the audio the client sends, a whole number of samples a second.", name,
                               "invalid_value"};
            }
            rate = std::stoi(value.text);
            continue;
        }
        speech_option option;
        if (!option_named(name, option)) {
            throw ApiError{400, "Unknown parameter: " + json_string(name) + ". start takes sample_rate and the request options of the dialogue "
                                "model, which speech info lists.",
                           name, "unknown_parameter"};
        }
        try {
            options.push_back({option, option_from_json(option, value)});
        } catch (const std::invalid_argument & e) {
            throw ApiError{400, std::string(e.what()) + ".", name, "invalid_type"};
        }
    }
    return options;
}

/** 16-bit little-endian samples as floats. */
std::vector<float> samples_of(const std::string & bytes) {
    std::vector<float> out(bytes.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = (float) (int16_t) ((uint16_t) (unsigned char) bytes[2 * i] | (uint16_t) (unsigned char) bytes[2 * i + 1] << 8) / 32768.0f;
    }
    return out;
}

/**
 * Holds one conversation on the socket until the client hangs up or goes away, or it fails. The server's one place for a
 * conversation, `busy`, is freed with the conversation's memory before the client hears that it has ended, so that a
 * client may start the next as soon as it does.
 */
class Conversation {
public:
    Conversation(httplib::ws::WebSocket & ws, std::shared_ptr<Served> model, std::atomic<bool> & busy)
        : ws_(ws), model_(std::move(model)), busy_(busy), talking_{ws} {}
    ~Conversation() { release(); }
    Conversation(const Conversation &) = delete;
    Conversation & operator=(const Conversation &) = delete;

    /** Runs the conversation, whose place it frees however it ends; a failure is sent to the client as an error event. */
    void run() {
        try {
            if (start()) listen();
        } catch (const Failure & e) {
            fail(openai::library_error(e));
        } catch (const ApiError & e) {
            fail(e);
        }
        release();
    }

    const Talking & talking() const { return talking_; }
    /** How the conversation ended: "hung up", "gone" or the failure's message. */
    const std::string & outcome() const { return outcome_; }

private:
    using Dialogue = std::unique_ptr<speech_dialogue, decltype(&speech_dialogue_free)>;

    /** Reads the start message and starts the conversation, which passes the model's first frame; false if the client went away. */
    bool start() {
        std::string message;
        httplib::ws::ReadResult got;
        while ((got = ws_.read(message)) == httplib::ws::Timeout) {
        }
        if (got == httplib::ws::Fail) {
            outcome_ = "gone";
            return false;
        }
        const JsonValue start = parse(message, got);
        const JsonValue * type = start.member("type");
        if (!type || type->text != "start") {
            throw ApiError{400, "A conversation begins with {\"type\": \"start\"} and the request options it takes.", "type", "invalid_value"};
        }
        const auto info = model_->info();
        int rate = 0;
        const std::vector<RequestOption> options = read_start(start, speech_model_info_sample_rate(info.get()), rate);
        const Request request = new_request(model_->get());
        apply_options(request.get(), model_->with_defaults(options));
        speech_dialogue * raw = nullptr;
        check(speech_dialogue_start(request.get(), rate, on_frame, &talking_, &raw));
        dialogue_.reset(raw);
        if (!ws_.send("{\"type\":\"started\",\"sample_rate\":" + std::to_string(speech_model_info_sample_rate(info.get())) +
                      ",\"frame_samples\":" + std::to_string(speech_model_info_frame_samples(info.get())) + "}")) {
            outcome_ = "gone";
            return false;
        }
        talking_.started = true;
        for (const auto & [pcm, text] : talking_.early) send_frame(talking_, pcm, text);
        talking_.early.clear();
        return !gone();
    }

    /** Passes the audio heard to the conversation until the client hangs up or goes away. */
    void listen() {
        std::string message;
        for (;;) {
            const httplib::ws::ReadResult got = ws_.read(message);
            if (got == httplib::ws::Fail) {
                outcome_ = "gone";
                return;
            }
            if (got == httplib::ws::Timeout) continue;
            if (got == httplib::ws::Binary) {
                if (message.size() % 2 != 0) {
                    throw ApiError{400, "The audio is 16-bit PCM, two bytes a sample, and a message held " + std::to_string(message.size()) + " bytes.",
                                   "", "invalid_value"};
                }
                const std::vector<float> samples = samples_of(message);
                check(speech_dialogue_push(dialogue_.get(), samples.data(), samples.size()));
                if (gone()) return;
                continue;
            }
            const JsonValue event = parse(message, got);
            const JsonValue * type = event.member("type");
            if (!type || type->text != "hang_up") {
                throw ApiError{400, "During a conversation the client sends its audio in binary messages and ends it with {\"type\": \"hang_up\"}.",
                               "type", "invalid_value"};
            }
            check(speech_dialogue_end(dialogue_.get()));
            if (gone()) return;
            release();
            ws_.send("{\"type\":\"end\",\"frames\":" + std::to_string(talking_.frames) + ",\"samples\":" + std::to_string(talking_.samples) + "}");
            ws_.close();
            outcome_ = "hung up";
            return;
        }
    }

    JsonValue parse(const std::string & message, httplib::ws::ReadResult got) const {
        if (got != httplib::ws::Text) throw ApiError{400, "A conversation begins with a text message, {\"type\": \"start\"}.", "", "invalid_value"};
        try {
            JsonValue value = parse_json(message);
            if (value.kind != JsonValue::Kind::Object) throw std::invalid_argument("");
            return value;
        } catch (const std::invalid_argument &) {
            throw ApiError{400, "A text message is a JSON object with its \"type\".", "", "invalid_json"};
        }
    }

    /** Whether the client went away, which is when the frame callback stops the conversation. */
    bool gone() {
        if (talking_.gone) outcome_ = "gone";
        return talking_.gone;
    }

    void fail(const ApiError & e) {
        outcome_ = e.message;
        release();
        ws_.send(error_event(e));
        ws_.close();
    }

    void release() {
        dialogue_.reset();
        busy_ = false;
    }

    httplib::ws::WebSocket & ws_;
    std::shared_ptr<Served> model_;
    std::atomic<bool> & busy_;
    Talking talking_;
    Dialogue dialogue_{nullptr, speech_dialogue_free};
    std::string outcome_;
};

}  // namespace

std::optional<ApiError> Dialogues::refusal(const httplib::Request & req) const {
    if (access_.loopback()) {
        if (auto refused = access_.host_refusal(req)) return refused;
    }
    if (!req.params.empty()) {
        return ApiError{400, "/v1/dialogue takes no query; the start message carries the conversation's options.", req.params.begin()->first,
                        "unknown_parameter"};
    }
    if (!models_.of(SPEECH_TASK_DIALOGUE)) {
        const std::string coming = models_.replacing(SPEECH_TASK_DIALOGUE);
        if (!coming.empty()) {
            return ApiError{503, "The speech dialogue model is being loaded (" + coming + "); connect again once it is in place.", "", "model_loading"};
        }
        return ApiError{404, "This server holds no speech dialogue model, which /v1/dialogue talks with. Give speech serve one.", "", ""};
    }
    if (talking_) {
        return ApiError{409, "A conversation is going on with the dialogue model, which holds one at a time; connect again once it has ended.", "",
                        "conversation_in_progress"};
    }
    return std::nullopt;
}

void Dialogues::route(httplib::Server & http) {
    http.WebSocket("/v1/dialogue", [this](const httplib::Request & req, httplib::ws::WebSocket & ws) {
        bool idle = false;
        if (!talking_.compare_exchange_strong(idle, true)) {
            ws.send(error_event({409, "A conversation is going on with the dialogue model, which holds one at a time; connect again once it has "
                                      "ended.", "", "conversation_in_progress"}));
            ws.close();
            return;
        }
        const auto model = models_.of(SPEECH_TASK_DIALOGUE);
        if (!model) {
            talking_ = false;
            ws.send(error_event({503, "The speech dialogue model is being replaced; connect again once it is in place.", "", "model_loading"}));
            ws.close();
            return;
        }
        log("dialogue from " + req.remote_addr + " opened");
        const Clock::time_point opened = Clock::now();
        Conversation conversation(ws, model, talking_);
        conversation.run();
        char line[160];
        std::snprintf(line, sizeof line, ": %llu frames, %.1f s of the model's audio in %.1f s, ",
                      (unsigned long long) conversation.talking().frames,
                      (double) conversation.talking().samples / speech_model_info_sample_rate(model->info().get()), seconds_between(opened, Clock::now()));
        log("dialogue from " + req.remote_addr + line + conversation.outcome());
    });
}

}  // namespace server
