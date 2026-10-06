#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "httplib.h"

#include "commands.h"
#include "jobs.h"
#include "json.h"
#include "openai-api.h"

// speech serve: serves one model over HTTP with OpenAI's audio API, so that a program that speaks HTTP (a web app,
// Python with requests, curl) can use speech.cpp without starting the worker. A synthesis model answers
// POST /v1/audio/speech and a recognition model POST /v1/audio/transcriptions; GET /v1/models gives the model with its
// information, and GET /health says the server is up, which it is once the model is loaded. A failure of the library
// becomes OpenAI's error object by its category alone (openai::library_error()). The model serves one request at a
// time in the order they arrive, and a client that goes away while it waits or while its request runs cancels it.

using namespace server;
using openai::ApiError;

namespace {

void send_error(httplib::Response & res, const ApiError & e) {
    res.status = e.status;
    res.set_content(openai::error_json(e), "application/json");
}

std::string wav_header(size_t data_bytes, int sample_rate) {
    std::string h;
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; i++) h += (char) ((v >> (8 * i)) & 0xFF); };
    auto u16 = [&](uint16_t v) { h += (char) (v & 0xFF); h += (char) (v >> 8); };
    h += "RIFF";
    u32((uint32_t) (36 + data_bytes));
    h += "WAVEfmt ";
    u32(16);
    u16(1);
    u16(1);
    u32((uint32_t) sample_rate);
    u32((uint32_t) sample_rate * 2);
    u16(2);
    u16(16);
    h += "data";
    u32((uint32_t) data_bytes);
    return h;
}

/** A seed from 0 to 2^53 - 1, the range the library draws from, for a request that sets none. */
int64_t draw_seed() {
    std::random_device device;
    return (int64_t) (((uint64_t) device() << 32 | device()) & ((1ull << 53) - 1));
}

/** The interval at which a waiting handler looks whether its client is still there. */
constexpr auto POLL = std::chrono::milliseconds(50);

class Server {
public:
    Server(speech_model * model, std::vector<std::string> cors_origins)
        : model_(model), info_(model_info(model)), cors_origins_(std::move(cors_origins)), created_((long long) std::time(nullptr)) {}

    void route(httplib::Server & http) {
        http.set_pre_routing_handler([this](const httplib::Request & req, httplib::Response & res) { return cors(req, res); });
        http.set_error_handler([](const httplib::Request & req, httplib::Response & res) {
            if (res.body.empty()) {
                send_error(res, {res.status, res.status == 404 ? "Invalid URL (" + req.method + " " + req.path + ")."
                                                               : std::string(httplib::status_message(res.status)) + ".", "", ""});
            }
        });
        http.set_logger([](const httplib::Request & req, const httplib::Response & res) {
            std::fprintf(stderr, "%s %s %s %d\n", req.remote_addr.c_str(), req.method.c_str(), req.path.c_str(), res.status);
        });
        http.Get("/health", [](const httplib::Request &, httplib::Response & res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
        http.Get("/v1/models", [this](const httplib::Request &, httplib::Response & res) {
            res.set_content("{\"object\":\"list\",\"data\":[" + model_json() + "]}", "application/json");
        });
        http.Get("/v1/models/(.+)", [this](const httplib::Request & req, httplib::Response & res) {
            if (req.matches[1] != name()) {
                send_error(res, {404, "The model " + json_string(req.matches[1]) + " does not exist.", "model", "model_not_found"});
                return;
            }
            res.set_content(model_json(), "application/json");
        });
        http.Post("/v1/audio/speech", [this](const httplib::Request & req, httplib::Response & res) {
            if (serves(SPEECH_TASK_SYNTHESIS, req, res)) speech(req, res);
        });
        http.Post("/v1/audio/transcriptions", [this](const httplib::Request & req, httplib::Response & res) {
            if (serves(SPEECH_TASK_RECOGNITION, req, res)) transcription(req, res);
        });
    }

private:
    speech_model * model_;
    /** The information of the model, whose voices are all added before the server listens. */
    ModelInfo info_;
    std::vector<std::string> cors_origins_;
    long long created_;
    Turns turns_;

    std::string name() const { return speech_model_info_name(info_.get()); }
    speech_task task() const { return speech_model_info_task(info_.get()); }
    int sample_rate() const { return speech_model_info_sample_rate(info_.get()); }

    /** OpenAI's model object, with the release of speech.cpp and the model's information. */
    std::string model_json() const {
        return "{\"id\":" + json_string(name()) + ",\"object\":\"model\",\"created\":" + std::to_string(created_) +
               ",\"owned_by\":\"speech.cpp\",\"version\":" + json_string(speech_version()) + ",\"speech\":" + speech_model_info_json(info_.get()) + "}";
    }

    /**
     * Whether the model does what the endpoint asks; otherwise answers a 404, as for a path the server does not have,
     * with a message that names the endpoint of the model's task.
     */
    bool serves(speech_task wanted, const httplib::Request & req, httplib::Response & res) const {
        if (task() == wanted) return true;
        const bool synthesis = task() == SPEECH_TASK_SYNTHESIS;
        send_error(res, {404, "This server serves " + name() + ", a speech " + task_name(task()) + " model, which " + req.path + " is not for; " +
                                  (synthesis ? "POST its text to /v1/audio/speech." : "POST its audio to /v1/audio/transcriptions."),
                         "", ""});
        return false;
    }

    /** Adds the CORS headers for an allowed origin, and answers a preflight. */
    httplib::Server::HandlerResponse cors(const httplib::Request & req, httplib::Response & res) {
        const std::string origin = req.get_header_value("Origin");
        const bool any = std::find(cors_origins_.begin(), cors_origins_.end(), "*") != cors_origins_.end();
        const bool allowed = !origin.empty() && (any || std::find(cors_origins_.begin(), cors_origins_.end(), origin) != cors_origins_.end());
        if (allowed) {
            res.set_header("Access-Control-Allow-Origin", any ? "*" : origin);
            res.set_header("Access-Control-Expose-Headers", "X-Sample-Rate, X-Speech-Seed, X-Speech-Stop");
            if (!any) res.set_header("Vary", "Origin");
        }
        if (req.method != "OPTIONS") return httplib::Server::HandlerResponse::Unhandled;
        if (allowed) {
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            const std::string headers = req.get_header_value("Access-Control-Request-Headers");
            res.set_header("Access-Control-Allow-Headers", headers.empty() ? "Content-Type, Authorization" : headers);
            res.set_header("Access-Control-Max-Age", "86400");
        }
        res.status = 204;
        return httplib::Server::HandlerResponse::Handled;
    }

    /**
     * Waits until the job has finished, or with `until` until it says so; returns false for a client that went away
     * meanwhile, whose request it cancels.
     */
    template <typename Until>
    static bool wait(Job & job, const httplib::Request & req, std::unique_lock<std::mutex> & lock, Until until) {
        while (!job.finished && !until()) {
            if (job.changed.wait_for(lock, POLL) == std::cv_status::timeout && req.is_connection_closed()) {
                lock.unlock();
                job.abandon();
                return false;
            }
        }
        return true;
    }

    void speech(const httplib::Request & req, httplib::Response & res) {
        openai::SpeechRequest asked;
        try {
            asked = openai::read_speech_request(req.body, name());
        } catch (const ApiError & e) {
            send_error(res, e);
            return;
        }
        auto job = std::make_shared<SpeechJob>();
        job->sample_rate = sample_rate();
        std::optional<int64_t> seed;
        std::string voice;
        for (const RequestOption & o : asked.options) {
            if (o.option == SPEECH_OPT_SEED) seed = std::get<int64_t>(o.value);
            if (o.option == SPEECH_OPT_VOICE) voice = std::get<std::string>(o.value);
        }
        // A pcm stream's headers leave before its result, so the server draws the seed the library would draw.
        if (!seed && speech_model_info_takes(info_.get(), SPEECH_OPT_SEED)) {
            seed = draw_seed();
            asked.options.push_back({SPEECH_OPT_SEED, *seed});
        }
        try {
            job->request = new_request(model_);
            check(speech_request_set_text(job->request.get(), asked.input.c_str()));
            apply_options(job->request.get(), asked.options);
            check(speech_request_set_progress(job->request.get(), on_progress, job.get()));
        } catch (const Failure & e) {
            send_error(res, openai::library_error(e));
            return;
        }
        char log[256];
        std::snprintf(log, sizeof log, "speech: voice %s, seed %lld", voice.c_str(), (long long) seed.value_or(-1));
        const uint64_t ticket = turns_.take();
        std::thread(synthesize, job, std::ref(turns_), ticket, std::string(log)).detach();

        // A wav is sent whole, so it waits for the end; a stream waits until the library has begun the request's work,
        // so that a request it refuses is answered with an error status rather than a stream that breaks off.
        const bool whole = asked.format == "wav";
        {
            std::unique_lock<std::mutex> lock(job->mutex);
            if (!wait(*job, req, lock, [&] { return !whole && job->accepted; })) return;
            if (job->finished && job->status == SPEECH_CANCELLED) return;
            if (job->finished && job->failure && (whole || !job->accepted)) {
                send_error(res, openai::library_error(*job->failure));
                return;
            }
        }
        res.set_header("X-Sample-Rate", std::to_string(sample_rate()));
        if (seed) res.set_header("X-Speech-Seed", std::to_string(*seed));
        if (whole) {
            res.set_header("X-Speech-Stop", speech_stop_name(job->stop));
            res.set_content(wav_header(job->pending.size(), sample_rate()) + job->pending, "audio/wav");
            return;
        }
        const bool sse = asked.sse;
        if (sse) res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider(
            sse ? "text/event-stream" : "audio/pcm", [job, sse](size_t, httplib::DataSink & sink) { return stream(*job, sse, sink); },
            [job](bool success) {
                if (!success) job->abandon();
            });
    }

    /** Recognizes the form's WAV file and answers with its text once it is done, as json, text or verbose_json. */
    void transcription(const httplib::Request & req, httplib::Response & res) {
        openai::TranscriptionRequest asked;
        try {
            std::vector<openai::FormPart> parts;
            for (const auto & [field_name, field] : req.form.fields) parts.push_back({field_name, field.content, "", false});
            for (const auto & [file_name, file] : req.form.files) parts.push_back({file_name, file.content, file.filename, true});
            asked = openai::read_transcription_request(req.is_multipart_form_data(), parts, name());
        } catch (const ApiError & e) {
            send_error(res, e);
            return;
        }
        auto job = std::make_shared<TranscriptionJob>();
        job->sample_rate = asked.sample_rate;
        job->duration = (double) asked.samples.size() / asked.sample_rate;
        try {
            job->request = new_request(model_);
            check(speech_request_set_audio(job->request.get(), asked.samples.data(), asked.samples.size(), asked.sample_rate));
            apply_options(job->request.get(), asked.options);
        } catch (const Failure & e) {
            send_error(res, openai::library_error(e));
            return;
        }
        const uint64_t ticket = turns_.take();
        std::thread(transcribe, job, std::ref(turns_), ticket).detach();
        std::unique_lock<std::mutex> lock(job->mutex);
        if (!wait(*job, req, lock, [] { return false; })) return;
        if (job->failure) {
            send_error(res, openai::library_error(*job->failure));
            return;
        }
        if (job->status != SPEECH_OK) return;
        const speech_result * result = speech_request_result(job->request.get());
        if (asked.format == "text") {
            res.set_content(speech_result_text(result), "text/plain; charset=utf-8");
        } else if (asked.format == "verbose_json") {
            res.set_content(openai::transcription_verbose_json(result, job->duration), "application/json");
        } else {
            res.set_content(openai::transcription_json(speech_result_text(result)), "application/json");
        }
    }

    /**
     * Sends the audio as it is made. A failure after the stream has begun ends a pcm stream without its last chunk,
     * which the client reads as a broken transfer, and an SSE stream with an error event.
     */
    static bool stream(SpeechJob & job, bool sse, httplib::DataSink & sink) {
        std::unique_lock<std::mutex> lock(job.mutex);
        for (;;) {
            if (!job.pending.empty()) {
                std::string pcm;
                pcm.swap(job.pending);
                lock.unlock();
                const std::string out = sse ? openai::sse_delta(pcm) : pcm;
                // A write to a socket the client has closed succeeds once more before it fails, so a client that went
                // away is noticed a chunk earlier by looking first.
                if (!sink.is_writable() || !sink.write(out.data(), out.size())) {
                    job.abandon();
                    return false;
                }
                lock.lock();
                continue;
            }
            if (job.finished) break;
            if (job.changed.wait_for(lock, POLL) == std::cv_status::timeout && !job.finished && job.pending.empty()) {
                lock.unlock();
                if (!sink.is_writable()) {
                    job.abandon();
                    return false;
                }
                lock.lock();
            }
        }
        const speech_status status = job.status;
        const std::optional<Failure> failure = job.failure;
        const int64_t seed = job.seed;
        const uint64_t samples = job.samples;
        const speech_stop stop = job.stop;
        lock.unlock();
        if (status != SPEECH_OK && !sse) return false;
        if (sse) {
            std::string out;
            if (status == SPEECH_OK) out = openai::sse_done(seed, samples, speech_stop_name(stop));
            else if (failure) out = openai::sse_error(openai::library_error(*failure));
            else return false;
            if (!sink.write(out.data(), out.size())) return false;
        }
        sink.done();
        return true;
    }
};

int run_serve(const CommandLine & line, FILE *) {
    const std::string host = line.value("--host").value_or("127.0.0.1");
    const int port = line.integer("--port").value_or(8080);
    if (port < 0 || port > 65535) throw UsageError("--port takes a port from 0 to 65535, not " + std::to_string(port));
    const Model model = load_model(line.args[0], line.loading(true));
    const ModelInfo info = model_info(model.get());
    std::fprintf(stderr, "speech serve: %s (%s) on %s, %d Hz, speech.cpp %s\n", speech_model_info_name(info.get()),
                 speech_model_info_architecture(info.get()), speech_model_info_device(info.get()), speech_model_info_sample_rate(info.get()),
                 speech_version());

    httplib::Server http;
    // Nagle's algorithm would hold a small chunk of a stream until the client acknowledges the previous one.
    http.set_tcp_nodelay(true);
    // A text to speak fits in 1 MB; a file to recognize may take OpenAI's limit for an upload, 25 MB.
    http.set_payload_max_length(speech_model_info_task(info.get()) == SPEECH_TASK_SYNTHESIS ? 1 << 20 : 25 << 20);
    Server server(model.get(), line.values("--cors-origin"));
    server.route(http);
    if (!http.bind_to_port(host, port)) {
        throw Failure(speech_status_name(SPEECH_ERROR_IO), "", "cannot listen on " + host + ":" + std::to_string(port) + "; choose another --port or --host");
    }
    std::fprintf(stderr, "speech serve: listening on http://%s:%d\n", host.c_str(), port);
    if (!http.listen_after_bind()) throw Failure(speech_status_name(SPEECH_ERROR_IO), "", "the server stopped listening");
    return 0;
}

}  // namespace

Command serve_command() {
    Command c;
    c.name = "serve";
    c.usage = "serve MODEL [options]";
    c.summary = "serve a model over HTTP with OpenAI's audio API";
    c.description =
        "Loads MODEL, warmed up unless --no-warmup, adds the voices of --add-voice, and serves it over HTTP: POST\n"
        "/v1/audio/speech for a synthesis model, POST /v1/audio/transcriptions for a recognition model, GET /v1/models and\n"
        "GET /health. It has no authentication and no TLS.";
    c.flags = {
        {"--host", "ADDRESS", false, "the address to listen on, 127.0.0.1 unless given"},
        {"--port", "N", false, "the port, 8080 unless given"},
        {"--cors-origin", "ORIGIN|*", true, "an origin a web page may call the server from, or * for any"},
        add_voice_flag(),
        device_flag("auto (the first GPU, or the CPU without one), gpu, cpu or a name `speech devices` lists"),
        threads_flag(),
        no_warmup_flag(),
    };
    c.min_args = 1;
    c.max_args = 1;
    c.run = run_serve;
    return c;
}
