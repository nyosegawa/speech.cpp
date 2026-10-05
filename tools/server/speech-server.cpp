// An HTTP server that serves one model through OpenAI's speech API, so that any program that speaks HTTP (a web
// app, Python with requests, curl) can use speech.cpp without starting the worker. It reaches the model only
// through the C API.
//
//   POST /v1/audio/speech   OpenAI's create speech: {"model", "input", "voice", "response_format", "speed",
//                           "stream_format"}, and speech.cpp's own "language", "seed", "seconds" and
//                           "duration_scale". "wav" answers with the whole file; "pcm" streams 16-bit
//                           little-endian mono at the model's rate as it is made; "stream_format": "sse" streams
//                           the same PCM as speech.audio.delta events and ends with speech.audio.done.
//   GET  /v1/models         the loaded model, with its voices, languages, sample rate and streaming kind.
//   GET  /health            {"status": "ok"} once the model is loaded, which is before the server listens.
//
// A request names the loaded model in "model" or leaves it out, and any other model is a 404. A request
// without "seed" gets a random one; the response's X-Speech-Seed header gives the seed that was used, so the
// same request with that seed gives the same audio. Errors have OpenAI's shape, {"error": {"message", "type",
// "param", "code"}}; a request the model cannot follow is a 400 with the library's message.
//
// The model speaks one request at a time, in the order they arrive, and the others wait. A client that goes
// away, while it waits or while its audio streams, cancels its request.
//
// usage: speech-server <model.gguf> <codec.gguf> [--host 127.0.0.1] [--port 8080] [--cors-origin ORIGIN|*]...
//                      [--device NAME|gpu|cpu]
//                      [--ctx n]                               (Qwen3-TTS)
//                      [--voice NAME=FILE]... [--steps n]      (Irodori-TTS; FILE is a WAVE or voice file)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "httplib.h"

#include "args.h"
#include "flat-json.h"
#include "openai-api.h"
#include "speech.h"

namespace {

using Clock = std::chrono::steady_clock;


void send_error(httplib::Response & res, const ApiError & e) {
    res.status = e.status;
    res.set_content(error_json(e), "application/json");
}

template <typename Get>
std::string json_array(size_t count, Get get) {
    std::string out = "[";
    for (size_t i = 0; i < count; i++) out += (i ? "," : "") + json_string(get(i));
    return out + "]";
}

/** The audio of a request as the worker sends it: 16-bit little-endian samples, clamped to [-1, 1] and rounded. */
void append_pcm(std::string & out, const float * s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const int16_t v = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s[i])) * 32767.0f);
        out += (char) (v & 0xFF);
        out += (char) ((uint16_t) v >> 8);
    }
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

/** What the server was started with; anything else on the command line throws. */
struct Options {
    std::string model, codec, device;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::vector<std::string> cors_origins;
    int context = speech_model_default_params().context;
    int steps = 0;
    std::vector<std::pair<std::string, std::string>> voices;
};

int integer(const std::string & key, const std::string & value) {
    char * end = nullptr;
    const long v = std::strtol(value.c_str(), &end, 10);
    if (value.empty() || *end != '\0' || v < 0 || v > 1 << 30) throw std::runtime_error(key + " takes a whole number, not " + value);
    return (int) v;
}

Options parse_options(const std::vector<std::string> & a) {
    if (a.size() < 3) throw std::runtime_error("expected the model and codec GGUF paths: speech-server <model.gguf> <codec.gguf> [options], with the options the README lists");
    Options o;
    o.model = a[1];
    o.codec = a[2];
    for (size_t i = 3; i < a.size(); i++) {
        const std::string & key = a[i];
        if (i + 1 >= a.size()) throw std::runtime_error(key + " needs a value");
        const std::string & value = a[++i];
        if (key == "--host") o.host = value;
        else if (key == "--port") o.port = integer(key, value);
        else if (key == "--cors-origin") o.cors_origins.push_back(value);
        else if (key == "--device") o.device = value;
        else if (key == "--ctx") o.context = integer(key, value);
        else if (key == "--steps") o.steps = integer(key, value);
        else if (key == "--voice") {
            const size_t eq = value.find('=');
            if (eq == std::string::npos || eq == 0) throw std::runtime_error("--voice takes NAME=FILE");
            o.voices.push_back({value.substr(0, eq), value.substr(eq + 1)});
        } else {
            throw std::runtime_error("unknown option " + key);
        }
    }
    return o;
}

speech_model * load(const Options & o) {
    std::vector<speech_voice_source> voices;
    for (const auto & [name, path] : o.voices) voices.push_back({name.c_str(), path.c_str()});
    speech_model_params params = speech_model_default_params();
    params.model_path = o.model.c_str();
    params.codec_path = o.codec.c_str();
    params.device = o.device.c_str();
    params.context = o.context;
    params.voices = voices.data();
    params.n_voices = voices.size();
    params.steps = o.steps;
    speech_model * model = nullptr;
    if (speech_model_load(&params, &model) != SPEECH_OK) throw std::runtime_error(speech_last_error());
    return model;
}

/** The model as OpenAI's model object, with what speech.cpp adds to it. */
std::string model_json(const speech_model * m, long long created) {
    std::string out = "{\"id\":" + json_string(speech_model_name(m)) + ",\"object\":\"model\",\"created\":" +
                      std::to_string(created) + ",\"owned_by\":\"speech.cpp\",\"architecture\":" +
                      json_string(speech_model_architecture(m)) +
                      ",\"sample_rate\":" + std::to_string(speech_model_sample_rate(m)) + ",\"streaming\":\"" +
                      (speech_model_streaming(m) == SPEECH_STREAMING_FRAME ? "frame" : "sentence") + "\",\"voices\":" +
                      json_array(speech_model_voice_count(m), [&](size_t i) { return speech_model_voice(m, i); }) +
                      ",\"languages\":" +
                      json_array(speech_model_language_count(m), [&](size_t i) { return speech_model_language(m, i); }) +
                      ",\"language_selectable\":" + (speech_model_language_selectable(m) ? "true" : "false");
    if (speech_model_steps(m) > 0) out += ",\"steps\":" + std::to_string(speech_model_steps(m));
    return out + ",\"backend\":" + json_string(speech_model_backend(m)) +
           ",\"version\":" + json_string(speech_version()) + "}";
}

/**
 * Lets requests take the model in the order they arrive. speech_synthesize() serializes concurrent calls by
 * itself, but in no particular order, and a request must know when it is its turn to tell a client that went
 * away while waiting from one whose synthesis is under way.
 */
class Turns {
public:
    uint64_t take() {
        std::lock_guard<std::mutex> lock(mutex_);
        return next_++;
    }
    void wait(uint64_t ticket) {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return serving_ == ticket; });
    }
    void pass() {
        std::lock_guard<std::mutex> lock(mutex_);
        serving_++;
        changed_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    uint64_t next_ = 0, serving_ = 0;
};

/** One request's synthesis, which runs on a thread of its own and hands its audio to the HTTP handler. */
struct Job {
    speech_model * model;
    SpeechRequest request;
    std::mutex mutex;
    std::condition_variable changed;
    /** The PCM made and not yet sent. */
    std::string pending;
    /** The callback has been called, so the library has accepted the request. */
    bool accepted = false;
    bool running = false;
    bool finished = false;
    bool abandoned = false;
    speech_status status = SPEECH_OK;
    std::string error;
    size_t samples = 0;
    Clock::time_point arrived = Clock::now(), first_audio, gone;

    /** Stops the request for a client that went away, whether it waits or speaks. */
    void abandon() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!abandoned) gone = Clock::now();
        abandoned = true;
        // Only while this request runs: once it has returned, speech_cancel() would stop the next one.
        if (running) speech_cancel(model);
    }
};

int on_audio(const float * s, size_t n, void * user_data) {
    Job & job = *static_cast<Job *>(user_data);
    std::lock_guard<std::mutex> lock(job.mutex);
    if (job.abandoned) return 1;
    job.accepted = true;
    if (n > 0 && job.samples == 0) job.first_audio = Clock::now();
    append_pcm(job.pending, s, n);
    job.samples += n;
    job.changed.notify_all();
    return 0;
}

double seconds_since(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

void synthesize(const std::shared_ptr<Job> & job, Turns & turns, uint64_t ticket) {
    turns.wait(ticket);
    const Clock::time_point started = Clock::now();
    {
        std::lock_guard<std::mutex> lock(job->mutex);
        job->running = !job->abandoned;
    }
    const bool ran = job->running;
    speech_status status = SPEECH_STOPPED;
    std::string error;
    if (ran) {
        speech_request r = speech_request_default();
        r.text = job->request.input.c_str();
        r.voice = job->request.voice.c_str();
        r.language = job->request.language.c_str();
        r.seed = job->request.seed;
        r.speed = job->request.speed;
        r.seconds = job->request.seconds;
        r.duration_scale = job->request.duration_scale;
        status = speech_synthesize(job->model, &r, on_audio, job.get());
        if (status == SPEECH_ERROR) error = speech_last_error();
    }
    {
        std::lock_guard<std::mutex> lock(job->mutex);
        job->running = false;
        job->finished = true;
        job->status = status;
        job->error = error;
        job->changed.notify_all();
        const Clock::time_point now = Clock::now();
        char line[256];
        std::snprintf(line, sizeof line, "speech: voice %s, seed %llu: waited %.3f s", job->request.voice.c_str(),
                      (unsigned long long) job->request.seed, seconds_since(job->arrived, started));
        std::string log = line;
        if (!ran) {
            log += ", not started: the client went away while it waited";
        } else {
            if (job->samples) {
                std::snprintf(line, sizeof line, ", first audio after %.3f s, %.2f s of audio", seconds_since(started, job->first_audio),
                              (double) job->samples / speech_model_sample_rate(job->model));
                log += line;
            }
            std::snprintf(line, sizeof line, " in %.3f s", seconds_since(started, now));
            log += line;
            if (status == SPEECH_ERROR) log += ", failed: " + error;
            if (job->abandoned) {
                std::snprintf(line, sizeof line, ", stopped %.3f s after the client went away", seconds_since(job->gone, now));
                log += line;
            }
        }
        std::fprintf(stderr, "%s\n", log.c_str());
    }
    turns.pass();
}

/** The interval at which a waiting handler looks whether its client is still there. */
constexpr auto POLL = std::chrono::milliseconds(50);

class Server {
public:
    Server(speech_model * model, std::vector<std::string> cors_origins)
        : model_(model), cors_origins_(std::move(cors_origins)), created_((long long) std::time(nullptr)) {}

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
        http.Get("/health", [](const httplib::Request &, httplib::Response & res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });
        http.Get("/v1/models", [this](const httplib::Request &, httplib::Response & res) {
            res.set_content("{\"object\":\"list\",\"data\":[" + model_json(model_, created_) + "]}", "application/json");
        });
        http.Get("/v1/models/(.+)", [this](const httplib::Request & req, httplib::Response & res) {
            if (req.matches[1] != speech_model_name(model_)) {
                send_error(res, {404, "The model " + json_string(req.matches[1]) + " does not exist.", "model", "model_not_found"});
                return;
            }
            res.set_content(model_json(model_, created_), "application/json");
        });
        http.Post("/v1/audio/speech", [this](const httplib::Request & req, httplib::Response & res) { speech(req, res); });
    }

private:
    speech_model * model_;
    std::vector<std::string> cors_origins_;
    long long created_;
    Turns turns_;

    /** Adds the CORS headers for an allowed origin, and answers a preflight. */
    httplib::Server::HandlerResponse cors(const httplib::Request & req, httplib::Response & res) {
        const std::string origin = req.get_header_value("Origin");
        const bool any = std::find(cors_origins_.begin(), cors_origins_.end(), "*") != cors_origins_.end();
        const bool allowed = !origin.empty() &&
                             (any || std::find(cors_origins_.begin(), cors_origins_.end(), origin) != cors_origins_.end());
        if (allowed) {
            res.set_header("Access-Control-Allow-Origin", any ? "*" : origin);
            res.set_header("Access-Control-Expose-Headers", "X-Sample-Rate, X-Speech-Seed");
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

    void speech(const httplib::Request & req, httplib::Response & res) {
        auto job = std::make_shared<Job>();
        job->model = model_;
        try {
            job->request = read_request(req.body, speech_model_name(model_));
        } catch (const ApiError & e) {
            send_error(res, e);
            return;
        }
        const uint64_t ticket = turns_.take();
        std::thread(synthesize, job, std::ref(turns_), ticket).detach();

        // A wav is sent whole, so it waits for the end; a stream waits until the library has accepted the
        // request, so that a request it refuses is answered with a 400 instead of a stream that breaks off.
        const bool whole = job->request.format == "wav";
        {
            std::unique_lock<std::mutex> lock(job->mutex);
            while (!job->finished && (whole || !job->accepted)) {
                if (job->changed.wait_for(lock, POLL) == std::cv_status::timeout && req.is_connection_closed()) {
                    lock.unlock();
                    job->abandon();
                    return;
                }
            }
            // The library checks a request before it calls back, so an error before then is the request's, and one
            // after it is the synthesis's.
            if (job->finished && job->status != SPEECH_OK && (whole || !job->accepted)) {
                if (job->status == SPEECH_ERROR) send_error(res, {job->accepted ? 500 : 400, job->error, "", ""});
                return;
            }
        }
        const int rate = speech_model_sample_rate(model_);
        res.set_header("X-Sample-Rate", std::to_string(rate));
        res.set_header("X-Speech-Seed", std::to_string(job->request.seed));
        if (whole) {
            res.set_content(wav_header(job->pending.size(), rate) + job->pending, "audio/wav");
            return;
        }
        const bool sse = job->request.sse;
        if (sse) res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider(
            sse ? "text/event-stream" : "audio/pcm",
            [job, sse](size_t, httplib::DataSink & sink) { return stream(*job, sse, sink); },
            [job](bool success) {
                if (!success) job->abandon();
            });
    }

    /**
     * Sends the audio as it is made. A failure after the stream has begun ends a pcm stream without its last
     * chunk, which the client reads as a broken transfer, and an SSE stream with an error event.
     */
    static bool stream(Job & job, bool sse, httplib::DataSink & sink) {
        std::unique_lock<std::mutex> lock(job.mutex);
        for (;;) {
            if (!job.pending.empty()) {
                std::string pcm;
                pcm.swap(job.pending);
                lock.unlock();
                const std::string out = sse ? sse_delta(pcm) : pcm;
                // A write to a socket the client has closed succeeds once more before it fails, so a client that
                // went away is noticed a chunk earlier by looking first.
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
        const std::string error = job.error;
        const size_t samples = job.samples;
        lock.unlock();
        if (status != SPEECH_OK && !sse) return false;
        if (sse) {
            const std::string out = status == SPEECH_OK
                ? sse_done(samples)
                : sse_error({500, status == SPEECH_ERROR ? error : "The synthesis was cancelled.", "", ""});
            if (!sink.write(out.data(), out.size())) return false;
        }
        sink.done();
        return true;
    }
};

}  // namespace

int main(int argc, char ** argv) {
    Options options;
    speech_model * model = nullptr;
    try {
        options = parse_options(utf8_args(argc, argv));
        model = load(options);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "speech-server: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "speech-server: %s (%s) on %s, %d Hz, speech.cpp %s\n", speech_model_name(model),
                 speech_model_architecture(model), speech_model_backend(model), speech_model_sample_rate(model), speech_version());

    httplib::Server http;
    // Nagle's algorithm would hold a small chunk of a stream until the client acknowledges the previous one.
    http.set_tcp_nodelay(true);
    http.set_payload_max_length(1 << 20);
    Server server(model, options.cors_origins);
    server.route(http);
    if (!http.bind_to_port(options.host, options.port)) {
        std::fprintf(stderr, "speech-server: cannot listen on %s:%d; choose another --port or --host\n", options.host.c_str(), options.port);
        speech_model_free(model);
        return 1;
    }
    std::fprintf(stderr, "speech-server: listening on http://%s:%d\n", options.host.c_str(), options.port);
    const bool ok = http.listen_after_bind();
    speech_model_free(model);
    return ok ? 0 : 1;
}
