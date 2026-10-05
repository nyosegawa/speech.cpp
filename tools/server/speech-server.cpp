// An HTTP server that serves one model through OpenAI's audio API, so that any program that speaks HTTP (a web
// app, Python with requests, curl) can use speech.cpp without starting the worker. It reaches the model only
// through the C API. A synthesis model answers /v1/audio/speech and a recognition model
// /v1/audio/transcriptions; the endpoint of the other task is a 404 that names the right one.
//
//   POST /v1/audio/speech   OpenAI's create speech: {"model", "input", "voice", "response_format", "speed",
//                           "stream_format"}, and speech.cpp's own "language", "seed", "seconds" and
//                           "duration_scale". "wav" answers with the whole file; "pcm" streams 16-bit
//                           little-endian mono at the model's rate as it is made; "stream_format": "sse" streams
//                           the same PCM as speech.audio.delta events and ends with speech.audio.done.
//   POST /v1/audio/transcriptions
//                           OpenAI's create transcription: a multipart/form-data form with "file", a WAV file at the
//                           model's rate, and "model", "language" and "response_format" ("json" or "text").
//   GET  /v1/models         the loaded model, with its task, languages and sample rate, and a synthesis model's
//                           voices and streaming kind.
//   GET  /health            {"status": "ok"} once the model is loaded, which is before the server listens.
//
// A request names the loaded model in "model" or leaves it out, and any other model is a 404. A request
// without "seed" gets a random one; the response's X-Speech-Seed header gives the seed that was used, so the
// same request with that seed gives the same audio. Errors have OpenAI's shape, {"error": {"message", "type",
// "param", "code"}}; a request the model cannot follow is a 400 with the library's message.
//
// The model serves one request at a time, in the order they arrive, and the others wait. A client that goes
// away, while it waits, while its audio streams or while its audio is recognized, cancels its request.
//
// usage: speech-server <model.gguf> [<codec.gguf>] [--host 127.0.0.1] [--port 8080] [--cors-origin ORIGIN|*]...
//                      [--device NAME|gpu|cpu]                 (the codec for a synthesis model)
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
#include "jobs.h"
#include "openai-api.h"
#include "speech.h"

namespace {

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
    if (a.size() < 2) {
        throw std::runtime_error("expected the model's GGUF path, and the codec's for a synthesis model: speech-server "
                                 "<model.gguf> [<codec.gguf>] [options], with the options the README lists");
    }
    Options o;
    o.model = a[1];
    // A recognition model has no codec, so the second path is there only when it is not an option.
    size_t first = 2;
    if (a.size() > 2 && a[2].compare(0, 2, "--") != 0) o.codec = a[first++];
    for (size_t i = first; i < a.size(); i++) {
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
    const bool synthesis = speech_model_task(m) == SPEECH_TASK_SYNTHESIS;
    std::string out = "{\"id\":" + json_string(speech_model_name(m)) + ",\"object\":\"model\",\"created\":" +
                      std::to_string(created) + ",\"owned_by\":\"speech.cpp\",\"task\":\"" +
                      (synthesis ? "synthesis" : "recognition") + "\",\"architecture\":" + json_string(speech_model_architecture(m)) +
                      ",\"sample_rate\":" + std::to_string(speech_model_sample_rate(m));
    if (synthesis) {
        out += std::string(",\"streaming\":\"") + (speech_model_streaming(m) == SPEECH_STREAMING_FRAME ? "frame" : "sentence") +
               "\",\"voices\":" + json_array(speech_model_voice_count(m), [&](size_t i) { return speech_model_voice(m, i); });
    }
    out += ",\"languages\":" + json_array(speech_model_language_count(m), [&](size_t i) { return speech_model_language(m, i); }) +
           ",\"language_selectable\":" + (speech_model_language_selectable(m) ? "true" : "false");
    if (speech_model_steps(m) > 0) out += ",\"steps\":" + std::to_string(speech_model_steps(m));
    return out + ",\"backend\":" + json_string(speech_model_backend(m)) +
           ",\"version\":" + json_string(speech_version()) + "}";
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
        http.Post("/v1/audio/speech", [this](const httplib::Request & req, httplib::Response & res) {
            if (serves(SPEECH_TASK_SYNTHESIS, req, res)) speech(req, res);
        });
        http.Post("/v1/audio/transcriptions", [this](const httplib::Request & req, httplib::Response & res) {
            if (serves(SPEECH_TASK_RECOGNITION, req, res)) transcription(req, res);
        });
    }

private:
    speech_model * model_;
    std::vector<std::string> cors_origins_;
    long long created_;
    Turns turns_;

    /**
     * Whether the model does what the endpoint asks; otherwise answers a 404, as for a path the server does not have,
     * with a message that names the endpoint of the model's task.
     */
    bool serves(speech_task task, const httplib::Request & req, httplib::Response & res) const {
        if (speech_model_task(model_) == task) return true;
        const bool synthesis = speech_model_task(model_) == SPEECH_TASK_SYNTHESIS;
        send_error(res, {404, "This server serves " + std::string(speech_model_name(model_)) + ", a speech " +
                                  (synthesis ? "synthesis" : "recognition") + " model, which " + req.path + " is not for; " +
                                  (synthesis ? "POST its text to /v1/audio/speech." : "POST its audio to /v1/audio/transcriptions."),
                         "", ""});
        return false;
    }

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
        auto job = std::make_shared<SpeechJob>();
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

    /** Recognizes the form's WAV file and answers with its text once it is done, as json or text. */
    void transcription(const httplib::Request & req, httplib::Response & res) {
        auto job = std::make_shared<TranscriptionJob>();
        job->model = model_;
        try {
            std::vector<FormPart> parts;
            for (const auto & [name, field] : req.form.fields) parts.push_back({name, field.content, "", false});
            for (const auto & [name, file] : req.form.files) parts.push_back({name, file.content, file.filename, true});
            job->request = read_transcription_request(req.is_multipart_form_data(), parts, speech_model_name(model_));
        } catch (const ApiError & e) {
            send_error(res, e);
            return;
        }
        const uint64_t ticket = turns_.take();
        std::thread(transcribe, job, std::ref(turns_), ticket).detach();
        std::unique_lock<std::mutex> lock(job->mutex);
        while (!job->finished) {
            if (job->changed.wait_for(lock, POLL) == std::cv_status::timeout && req.is_connection_closed()) {
                lock.unlock();
                job->abandon();
                return;
            }
        }
        // The library refuses the audio and the language before it recognizes anything. The recognition itself fails
        // only when the device does, which the status cannot tell apart, so every error is answered as the request's.
        if (job->status == SPEECH_ERROR) {
            send_error(res, {400, job->error, "", ""});
            return;
        }
        if (job->status != SPEECH_OK) return;
        if (job->request.format == "text") res.set_content(job->text, "text/plain; charset=utf-8");
        else res.set_content(transcription_json(job->text), "application/json");
    }

    /**
     * Sends the audio as it is made. A failure after the stream has begun ends a pcm stream without its last
     * chunk, which the client reads as a broken transfer, and an SSE stream with an error event.
     */
    static bool stream(SpeechJob & job, bool sse, httplib::DataSink & sink) {
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
    // A text to speak fits in 1 MB; a file to recognize may take OpenAI's limit for an upload, 25 MB.
    http.set_payload_max_length(speech_model_task(model) == SPEECH_TASK_SYNTHESIS ? 1 << 20 : 25 << 20);
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
