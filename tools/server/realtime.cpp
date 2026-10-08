#include "realtime.h"

#include <chrono>
#include <cstdio>

#include "jobs.h"
#include "json.h"
#include "realtime-session.h"

namespace server {

namespace {

using openai::ApiError;

std::string name_of(const Served & served) {
    return speech_model_info_name(served.info().get());
}

/** The recognition model a session transcribes with: the one the server holds when a commit takes its turn. */
class HeldRecognizer : public realtime::Recognizer {
public:
    explicit HeldRecognizer(ServedModels & models) : models_(models) {}

    std::string held() override {
        const auto served = models_.of(SPEECH_TASK_RECOGNITION);
        return served ? name_of(*served) : "";
    }

    void check(const std::string & model, const std::vector<RequestOption> & options) override {
        apply_options(new_request(served(model)->get()).get(), options);
    }

    std::optional<Transcript> transcribe(const std::string & model, const std::vector<float> & samples, const std::vector<RequestOption> & options,
                                         Cancellation & cancellation, const std::string & item) override {
        const std::shared_ptr<Served> on = served(model);
        const Request request = new_request(on->get());
        ::check(speech_request_set_audio(request.get(), samples.data(), samples.size(), realtime::kRate));
        apply_options(request.get(), options);
        const Clock::time_point arrived = Clock::now();
        Turns & turns = on->turns();
        turns.wait(turns.take());
        struct Pass {
            Turns & turns;
            ~Pass() { turns.pass(); }
        } pass{turns};
        const Clock::time_point started = Clock::now();
        speech_status status = SPEECH_CANCELLED;
        try {
            status = ::check(cancellation.run(request.get(), [&] { return speech_transcribe(request.get()); }));
        } catch (const Failure & e) {
            log(item, samples.size(), arrived, started, ", failed: " + e.code() + ": " + e.what());
            throw;
        }
        if (status == SPEECH_CANCELLED) {
            log(item, samples.size(), arrived, started, ", stopped: the session ended");
            return std::nullopt;
        }
        Transcript t = transcript_of(speech_request_result(request.get()), false);
        log(item, samples.size(), arrived, started, ", " + std::to_string(t.text.size()) + " bytes of text");
        return t;
    }

private:
    static void log(const std::string & item, size_t samples, Clock::time_point arrived, Clock::time_point started, const std::string & outcome) {
        std::fprintf(stderr, "realtime %s: %.2f s of audio: waited %.3f s%s in %.3f s\n", item.c_str(), (double) samples / realtime::kRate,
                     seconds_between(arrived, started), outcome.c_str(), seconds_between(started, Clock::now()));
    }

    /** The recognition model held, named `model` unless that is "", which a commit keeps until its recognition ends. */
    std::shared_ptr<Served> served(const std::string & model) const {
        const auto on = models_.of(SPEECH_TASK_RECOGNITION);
        if (!on) {
            const std::string coming = models_.replacing(SPEECH_TASK_RECOGNITION);
            if (!coming.empty()) {
                throw ApiError{503, "The speech recognition model is being loaded (" + coming + "); commit the audio again once it is in place.", "",
                               "model_loading"};
            }
            throw ApiError{404, "This server holds no speech recognition model to transcribe with; give speech serve one.", "", "model_not_found"};
        }
        if (!model.empty() && model != name_of(*on)) {
            throw ApiError{404, "The model " + json_string(model) + " is not served here; this server transcribes with " + json_string(name_of(*on)) +
                                    ". Name it, or leave the model out.",
                           "", "model_not_found"};
        }
        return on;
    }

    ServedModels & models_;
};

}  // namespace

std::optional<ApiError> Realtime::refusal(const httplib::Request & req) const {
    if (access_.loopback()) {
        if (auto refused = access_.host_refusal(req)) return refused;
    }
    std::string model;
    for (const auto & [name, value] : req.params) {
        if (name != "model") {
            return ApiError{400, "Unknown parameter in the address: " + name + ". speech.cpp's /v1/realtime takes model alone, in a query.", name,
                            "unknown_parameter"};
        }
        model = value;
    }
    const auto on = models_.of(SPEECH_TASK_RECOGNITION);
    if (!on) {
        const std::string coming = models_.replacing(SPEECH_TASK_RECOGNITION);
        if (!coming.empty()) {
            return ApiError{503, "The speech recognition model is being loaded (" + coming + "); connect again once it is in place.", "",
                            "model_loading"};
        }
        return ApiError{404, "This server holds no speech recognition model, which /v1/realtime transcribes with. Give speech serve one.", "", ""};
    }
    if (!model.empty() && model != name_of(*on)) {
        return ApiError{404, "The model " + json_string(model) + " is not served here; this server transcribes with " + json_string(name_of(*on)) +
                                 ". Name it in \"model\" or leave \"model\" out.",
                        "model", "model_not_found"};
    }
    return std::nullopt;
}

void Realtime::route(httplib::Server & http) {
    http.WebSocket("/v1/realtime", [this](const httplib::Request & req, httplib::ws::WebSocket & ws) {
        HeldRecognizer recognizer(models_);
        realtime::Session session(recognizer, [&ws](const std::string & event) { ws.send(event); });
        std::fprintf(stderr, "%s realtime session opened\n", req.remote_addr.c_str());
        session.open();
        std::string message;
        for (;;) {
            const httplib::ws::ReadResult got = ws.read(message);
            if (got == httplib::ws::Fail) break;
            if (got == httplib::ws::Timeout) continue;
            session.receive(message, got == httplib::ws::Text);
        }
        std::fprintf(stderr, "%s realtime session closed\n", req.remote_addr.c_str());
    });
}

}  // namespace server
