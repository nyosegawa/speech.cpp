#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "httplib.h"

#include "commands.h"
#include "fetch.h"
#include "jobs.h"
#include "json.h"
#include "openai-api.h"

#include "access.h"
#include "page.h"
#include "served-models.h"

// speech serve: serves a model of each task over HTTP with OpenAI's audio API, so that a program that speaks HTTP (a web
// app, Python with requests, curl) can use speech.cpp without starting the worker: the synthesis model answers
// POST /v1/audio/speech and the recognition model POST /v1/audio/transcriptions; GET /v1/models gives the models with
// their information, and GET /health says the server is up, which it is once the models given are loaded. A failure of
// the library becomes OpenAI's error object by its category alone (openai::library_error()). A model serves one request
// at a time in the order they arrive, and a client that goes away while it waits or while its request runs cancels it.
// On a loopback address the server also serves its page (page.h), which replaces the model of a task by one of the
// catalog.

using namespace server;
using openai::ApiError;
using openai::send_error;

namespace {

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

/** OpenAI's model object, with the release of speech.cpp and the model's information. */
std::string model_json(const Served & served) {
    const auto info = served.info();
    return "{\"id\":" + json_string(speech_model_info_name(info.get())) + ",\"object\":\"model\",\"created\":" +
           std::to_string(served.created()) + ",\"owned_by\":\"speech.cpp\",\"version\":" + json_string(speech_version()) +
           ",\"speech\":" + speech_model_info_json(info.get()) + "}";
}

class Server {
public:
    Server(ServedModels & models, const Access & access) : models_(models), access_(access) {}

    void route(httplib::Server & http) {
        http.set_pre_routing_handler([this](const httplib::Request & req, httplib::Response & res) { return admit(req, res); });
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
            std::string data;
            for (const auto & served : models_.all()) data += (data.empty() ? "" : ",") + model_json(*served);
            res.set_content("{\"object\":\"list\",\"data\":[" + data + "]}", "application/json");
        });
        http.Get("/v1/models/(.+)", [this](const httplib::Request & req, httplib::Response & res) {
            for (const auto & served : models_.all()) {
                if (req.matches[1] == speech_model_info_name(served->info().get())) {
                    res.set_content(model_json(*served), "application/json");
                    return;
                }
            }
            send_error(res, {404, "The model " + json_string(req.matches[1]) + " does not exist.", "model", "model_not_found"});
        });
        http.Post("/v1/audio/speech", [this](const httplib::Request & req, httplib::Response & res) {
            if (auto served = held(SPEECH_TASK_SYNTHESIS, req, res)) speech(std::move(served), req, res);
        });
        http.Post("/v1/audio/transcriptions", [this](const httplib::Request & req, httplib::Response & res) {
            if (auto served = held(SPEECH_TASK_RECOGNITION, req, res)) transcription(std::move(served), req, res);
        });
    }

private:
    ServedModels & models_;
    const Access & access_;

    /**
     * The model of the task the endpoint asks for; otherwise answers a 404, as for a path the server does not have,
     * or a 503 while the page loads one in its place.
     */
    std::shared_ptr<Served> held(speech_task wanted, const httplib::Request & req, httplib::Response & res) const {
        if (auto served = models_.of(wanted)) return served;
        const std::string task = std::string("speech ") + task_name(wanted);
        if (const std::string coming = models_.replacing(wanted); !coming.empty()) {
            res.set_header("Retry-After", "5");
            send_error(res, {503, "The " + task + " model is being loaded (" + coming + "); send the request again once it is in place.", "",
                             "model_loading"});
            return nullptr;
        }
        std::string message = "This server holds no " + task + " model, which " + req.path + " is for";
        const speech_task other = wanted == SPEECH_TASK_SYNTHESIS ? SPEECH_TASK_RECOGNITION : SPEECH_TASK_SYNTHESIS;
        if (const auto served = models_.of(other)) {
            message += "; it holds " + std::string(speech_model_info_name(served->info().get())) + ", a speech " + task_name(other) +
                       " model, " + (other == SPEECH_TASK_SYNTHESIS ? "which /v1/audio/speech is for" : "which /v1/audio/transcriptions is for");
        }
        send_error(res, {404, message + ". Give speech serve a " + task + " model" + (access_.loopback() ? ", or pick one on its page." : "."), "", ""});
        return nullptr;
    }

    /**
     * Refuses a request from an origin that may not call the server, adds the CORS headers for an origin --cors-origin
     * allows, and answers a preflight.
     */
    httplib::Server::HandlerResponse admit(const httplib::Request & req, httplib::Response & res) const {
        if (const auto refusal = access_.origin_refusal(req)) {
            send_error(res, *refusal);
            return httplib::Server::HandlerResponse::Handled;
        }
        const std::string origin = req.get_header_value("Origin");
        const bool allowed = access_.cross_origin_allowed(origin);
        if (allowed) {
            res.set_header("Access-Control-Allow-Origin", access_.any_origin() ? "*" : origin);
            res.set_header("Access-Control-Expose-Headers", "X-Sample-Rate, X-Speech-Seed, X-Speech-Stop");
            if (!access_.any_origin()) res.set_header("Vary", "Origin");
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

    void speech(std::shared_ptr<Served> served, const httplib::Request & req, httplib::Response & res) {
        const auto info = served->info();
        const int sample_rate = speech_model_info_sample_rate(info.get());
        openai::SpeechRequest asked;
        try {
            asked = openai::read_speech_request(req.body, speech_model_info_name(info.get()));
        } catch (const ApiError & e) {
            send_error(res, e);
            return;
        }
        auto job = std::make_shared<SpeechJob>();
        job->sample_rate = sample_rate;
        std::optional<int64_t> seed;
        std::string voice;
        for (const RequestOption & o : asked.options) {
            if (o.option == SPEECH_OPT_SEED) seed = std::get<int64_t>(o.value);
            if (o.option == SPEECH_OPT_VOICE) voice = std::get<std::string>(o.value);
        }
        // A pcm stream's headers leave before its result, so the server draws the seed the library would draw.
        if (!seed && speech_model_info_takes(info.get(), SPEECH_OPT_SEED)) {
            seed = draw_seed();
            asked.options.push_back({SPEECH_OPT_SEED, *seed});
        }
        job->served = served;
        try {
            job->request = new_request(served->get());
            check(speech_request_set_text(job->request.get(), asked.input.c_str()));
            apply_options(job->request.get(), asked.options);
            check(speech_request_set_progress(job->request.get(), on_progress, job.get()));
        } catch (const Failure & e) {
            send_error(res, openai::library_error(e));
            return;
        }
        char log[256];
        std::snprintf(log, sizeof log, "speech: voice %s, seed %lld", voice.c_str(), (long long) seed.value_or(-1));
        const uint64_t ticket = served->turns().take();
        std::thread(synthesize, job, std::ref(served->turns()), ticket, std::string(log)).detach();

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
        res.set_header("X-Sample-Rate", std::to_string(sample_rate));
        if (seed) res.set_header("X-Speech-Seed", std::to_string(*seed));
        if (whole) {
            res.set_header("X-Speech-Stop", speech_stop_name(job->stop));
            res.set_content(wav_header(job->pending.size(), sample_rate) + job->pending, "audio/wav");
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

    /**
     * Recognizes the form's WAV file and answers with its text once it is done, as json, text or verbose_json, which
     * also carries the language the model heard, and with why the recognition ended in X-Speech-Stop.
     */
    void transcription(std::shared_ptr<Served> served, const httplib::Request & req, httplib::Response & res) {
        const auto info = served->info();
        openai::TranscriptionRequest asked;
        try {
            std::vector<openai::FormPart> parts;
            for (const auto & [field_name, field] : req.form.fields) parts.push_back({field_name, field.content, "", false});
            for (const auto & [file_name, file] : req.form.files) parts.push_back({file_name, file.content, file.filename, true});
            asked = openai::read_transcription_request(req.is_multipart_form_data(), parts, speech_model_info_name(info.get()),
                                                       speech_model_info_takes(info.get(), SPEECH_OPT_TIMESTAMPS));
        } catch (const ApiError & e) {
            send_error(res, e);
            return;
        }
        auto job = std::make_shared<TranscriptionJob>();
        job->sample_rate = asked.sample_rate;
        job->duration = (double) asked.samples.size() / asked.sample_rate;
        job->served = served;
        try {
            job->request = new_request(served->get());
            check(speech_request_set_audio(job->request.get(), asked.samples.data(), asked.samples.size(), asked.sample_rate));
            apply_options(job->request.get(), asked.options);
        } catch (const Failure & e) {
            send_error(res, openai::library_error(e));
            return;
        }
        const uint64_t ticket = served->turns().take();
        std::thread(transcribe, job, std::ref(served->turns()), ticket).detach();
        std::unique_lock<std::mutex> lock(job->mutex);
        if (!wait(*job, req, lock, [] { return false; })) return;
        if (job->failure) {
            send_error(res, openai::library_error(*job->failure));
            return;
        }
        if (job->status != SPEECH_OK) return;
        const speech_result * result = speech_request_result(job->request.get());
        res.set_header("X-Speech-Stop", speech_stop_name(speech_result_stop(result)));
        if (asked.format == "text") {
            res.set_content(speech_result_text(result), "text/plain; charset=utf-8");
        } else if (asked.format == "verbose_json") {
            res.set_content(openai::transcription_verbose_json(result, job->duration, asked.timestamps), "application/json");
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

/** Says on stderr which model a task has, as the server starts or the page loads one. */
void report(const Served & served) {
    const auto info = served.info();
    std::fprintf(stderr, "speech serve: %s (%s) on %s, %d Hz, for %s\n", speech_model_info_name(info.get()), speech_model_info_architecture(info.get()),
                 speech_model_info_device(info.get()), speech_model_info_sample_rate(info.get()),
                 speech_model_info_task(info.get()) == SPEECH_TASK_SYNTHESIS ? "/v1/audio/speech" : "/v1/audio/transcriptions");
}

int run_serve(const CommandLine & line, FILE *) {
    const std::string host = line.value("--host").value_or("127.0.0.1");
    const int port = line.integer("--port").value_or(8080);
    if (port < 0 || port > 65535) throw UsageError("--port takes a port from 0 to 65535, not " + std::to_string(port));
    const bool open = line.has("--open");
    if (open && host != "127.0.0.1" && host != "::1" && host != "localhost") {
        throw UsageError("--open opens the page, which the server has only on 127.0.0.1, ::1 or localhost, not on " + host +
                         "; leave --host out");
    }
    if (line.args.empty() && !open) {
        throw UsageError(no_model_message("serve MODEL [options]", ModelKind::Any) + "\nOr give --open, and pick the models on the page.");
    }
    const Loading loading = line.loading(true);
    ServedModels models(loading);
    for (const std::string & argument : line.args) {
        models.load_given(model_file(argument), is_model_path(argument) ? "" : choice_name(find_model(argument)));
    }
    if (!loading.voices.empty()) {
        const auto synthesis = models.of(SPEECH_TASK_SYNTHESIS);
        if (!synthesis) throw UsageError("--add-voice adds voices to the synthesis model, and none is given");
        for (const auto & [name, file] : loading.voices) synthesis->add_voice(name, file);
    }
    for (const auto & served : models.all()) report(*served);

    httplib::Server http;
    // Nagle's algorithm would hold a small chunk of a stream until the client acknowledges the previous one.
    http.set_tcp_nodelay(true);
    // A file to recognize, or a recording to make a voice of, may take OpenAI's limit for an upload, 25 MB.
    http.set_payload_max_length(25 << 20);
    const int bound = port == 0 ? http.bind_to_any_port(host) : (http.bind_to_port(host, port) ? port : -1);
    if (bound < 0) {
        throw Failure(speech_status_name(SPEECH_ERROR_IO), "", "cannot listen on " + host + ":" + std::to_string(port) + "; choose another --port or --host");
    }
    const Access access(host, bound, line.values("--cors-origin"));
    Server server(models, access);
    server.route(http);
    Page page(models, access, report);
    page.route(http);
    const std::string address = host.find(':') == std::string::npos ? host : "[" + host + "]";
    std::fprintf(stderr, "speech serve: speech.cpp %s, listening on http://%s:%d\n", speech_version(), address.c_str(), bound);
    if (access.loopback()) {
        std::fprintf(stderr, "speech serve: the page, which fetches and loads models, is at %s\n", access.page_url().c_str());
        if (open) open_in_browser(access.page_url());
    }
    if (!http.listen_after_bind()) throw Failure(speech_status_name(SPEECH_ERROR_IO), "", "the server stopped listening");
    return 0;
}

}  // namespace

Command serve_command() {
    Command c;
    c.name = "serve";
    c.usage = "serve [MODEL [MODEL]] [options]";
    c.summary = "serve models over HTTP with OpenAI's audio API, and a page to try them";
    c.description =
        "Loads each MODEL, a synthesis model and a recognition model at most, warmed up unless --no-warmup, adds the voices\n"
        "of --add-voice to the synthesis model, and serves them over HTTP: POST /v1/audio/speech, POST\n"
        "/v1/audio/transcriptions, GET /v1/models and GET /health. It has no authentication and no TLS, and refuses a\n"
        "request that a web page at another origin than --cors-origin's sends. On 127.0.0.1, ::1 or localhost it also\n"
        "serves a page on which to pick models of the catalog, fetch them and try them, at the address with a token\n"
        "that it prints; --open opens it, and starts the server without a model if none is given.";
    c.flags = {
        {"--host", "ADDRESS", false, "the address to listen on, 127.0.0.1 unless given"},
        {"--port", "N", false, "the port, 8080 unless given; 0 for any free one"},
        {"--open", "", false, "open the page in the browser once the server listens"},
        {"--cors-origin", "ORIGIN|*", true, "an origin a web page may call the server from, or * for any"},
        add_voice_flag(),
        device_flag("auto (the first GPU, or the CPU without one), gpu, cpu or a name `speech devices` lists"),
        threads_flag(),
        no_warmup_flag(),
    };
    c.model = ModelKind::Any;
    c.max_args = 2;
    c.run = run_serve;
    return c;
}
