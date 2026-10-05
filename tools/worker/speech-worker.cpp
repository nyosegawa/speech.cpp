// A worker process that speaks texts for another program over JSON Lines. It runs Qwen3-TTS or Irodori-TTS,
// chosen by general.architecture of the model's GGUF.
//
// Reads one JSON object per line from stdin and answers with one JSON object per line on stdout. Nothing else
// reaches stdout: every log goes to stderr, so a line on stdout that is not a JSON object is a defect.
//   in : {"id": "...", "text": "...", "voice": "...", "language": "...", "speed": 1.0, "seconds": 2.5,
//         "durationScale": 1.2}
//        {"type": "cancel", "id": "..."}
//   out: {"type": "ready", "model": "...", "architecture": "...", "sampleRate": 24000, "streaming": "frame",
//         "voices": [...], "languages": [...], "languageSelectable": true, "backend": "MTL0", "version": "0.4.0"}
//        {"type": "chunk", "id": "...", "seq": 0, "pcm": "<base64 int16le mono>"}
//        {"type": "end", "id": "...", "samples": 123456}
//        {"type": "error", "id": "...", "error": "..."}
//        {"type": "fatal", "error": "..."}
//
// Every family lists its languages as BCP 47 tags, and a request's "language", when given, must be one of
// them or a region or script of one ("ja", "ja-JP"); "auto" or no language leaves the choice to the model.
// Qwen3-TTS streams frame by frame ("streaming": "frame"); its voices are the model's speakers, and the
// language goes into its prompt. Irodori-TTS makes a sentence at once and streams it as the codec decodes it
// ("streaming": "sentence"), so a request is one sentence; its voices are the ones given with --voice, the
// model is not told the language ("languageSelectable": false), and its "steps" is the sampler's.
//
// "speed", "seconds" and "durationScale" are the JSON numbers of speech_request's speed, seconds and
// duration_scale, each optional. Irodori-TTS takes them; Qwen3-TTS answers an error to any but the defaults.
//
// Requests are served one at a time in arrival order. A cancel takes effect between two chunks, and for
// Irodori-TTS also between two of the sampler's steps, before the first chunk; a cancelled request sends no
// end, and one cancelled before it starts is dropped.
//
// Every line on stdin gets an answer. A line that is not a request or a cancel the worker can read (not one
// JSON object, a member that is not a string or a number, no "id", an unknown "type") is answered at once
// with an error naming the problem, with the "id" when one could be read and without one otherwise. Such an
// error is the caller's defect, not the model's.
//
// `--devices` instead prints the devices ggml can run on and exits, so that the caller can tell whether
// the machine has a GPU and how much memory it has before starting a worker:
//   {"type": "devices", "devices": [{"name": "Vulkan0", "description": "NVIDIA GeForce RTX 2080",
//                                    "kind": "gpu", "memoryTotal": 8589934592, "memoryFree": 7516192768}]}
//
// usage: speech-worker <model.gguf> <codec.gguf> [--device NAME|gpu|cpu] [--seed n]
//                      [--ctx n]                               (Qwen3-TTS)
//                      [--voice NAME=FILE]... [--steps n]      (Irodori-TTS; FILE is a WAVE or voice file)
//        speech-worker --devices

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#include "args.h"
#include "flat-json.h"
#include "speech.h"

namespace {

std::mutex out_mutex;
FILE * protocol = stdout;

/**
 * Keeps the caller's stdout for the protocol alone. The stream moves to a descriptor of its own and
 * descriptor 1 then writes to stderr, so whatever ggml, a GPU driver or a system framework prints to stdout
 * lands among the logs instead of between two messages. On Windows, _dup2() onto descriptor 1 also sets the
 * process's standard output handle, which code writing with WriteFile() reads.
 */
FILE * take_stdout() {
    std::fflush(stdout);
#ifdef _WIN32
    const int fd = _dup(_fileno(stdout));
    if (fd < 0 || _dup2(_fileno(stderr), _fileno(stdout)) != 0) {
        throw std::runtime_error("cannot send stdout to stderr");
    }
    _setmode(fd, _O_BINARY);
    FILE * stream = _fdopen(fd, "wb");
#else
    const int fd = dup(STDOUT_FILENO);
    if (fd < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) throw std::runtime_error("cannot send stdout to stderr");
    FILE * stream = fdopen(fd, "w");
#endif
    if (!stream) throw std::runtime_error("cannot open a stream on the protocol's descriptor");
    return stream;
}

void emit(const std::string & json) {
    std::lock_guard<std::mutex> lock(out_mutex);
    std::fwrite(json.data(), 1, json.size(), protocol);
    std::fputc('\n', protocol);
    std::fflush(protocol);
}

std::string base64(const uint8_t * data, size_t n) {
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

/** A JSON array of the strings a getter gives for 0 to count - 1. */
template <typename Get>
std::string json_array(size_t count, Get get) {
    std::string out = "[";
    for (size_t i = 0; i < count; i++) out += (i ? "," : "") + json_string(get(i));
    return out + "]";
}

std::string value(const FlatJson & request, const char * key) {
    const auto it = request.find(key);
    return it == request.end() ? "" : it->second.text;
}

/** A number member of a request, or `absent` when the request leaves it out; a string or a non-finite number throws. */
double number(const FlatJson & request, const char * key, double absent) {
    const auto it = request.find(key);
    if (it == request.end()) return absent;
    const std::string & text = it->second.text;
    char * end = nullptr;
    const double v = std::strtod(text.c_str(), &end);
    if (it->second.kind != FlatValue::NUMBER || end != text.c_str() + text.size() || !std::isfinite(v)) {
        throw std::invalid_argument(std::string("\"") + key + "\" is " + json_string(text) + "; give it as a JSON number");
    }
    return v;
}

struct Inbox {
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<FlatJson> requests;
    std::set<std::string> cancelled;
    bool closed = false;

    bool is_cancelled(const std::string & id) {
        std::lock_guard<std::mutex> lock(mutex);
        return cancelled.count(id) > 0;
    }
    void forget(const std::string & id) {
        std::lock_guard<std::mutex> lock(mutex);
        cancelled.erase(id);
    }
};

/** Why a message is not a request or a cancel the worker can serve, or nothing when it is one. */
std::string unreadable(const FlatJson & message) {
    for (const auto & [key, v] : message) {
        if (v.kind == FlatValue::OTHER) {
            return "the member " + json_string(key) + " is " + v.text + "; the protocol's members are strings and numbers";
        }
    }
    if (!message.count("id")) return "the message has no \"id\"; every request and cancel needs one";
    const auto type = message.find("type");
    if (type != message.end() && type->second.text != "cancel") {
        return "the message's \"type\" is " + json_string(type->second.text) + "; a request has none and a cancel has \"cancel\"";
    }
    return "";
}

void read_requests(Inbox & inbox) {
    std::string line;
    while (std::getline(std::cin, line)) {
        FlatJson message;
        std::string problem;
        try {
            message = parse_flat_json(line);
            problem = unreadable(message);
        } catch (const std::exception & e) {
            problem = std::string("a line on stdin cannot be read: ") + e.what() + "; send one JSON object per line";
        }
        if (!problem.empty()) {
            const auto id = message.find("id");
            const bool has_id = id != message.end() && id->second.kind != FlatValue::OTHER;
            emit("{\"type\":\"error\"," + (has_id ? "\"id\":" + json_string(id->second.text) + "," : std::string()) +
                 "\"error\":" + json_string(problem) + "}");
            continue;
        }
        std::lock_guard<std::mutex> lock(inbox.mutex);
        if (value(message, "type") == "cancel") {
            inbox.cancelled.insert(value(message, "id"));
        } else {
            inbox.requests.push_back(message);
            inbox.ready.notify_one();
        }
    }
    std::lock_guard<std::mutex> lock(inbox.mutex);
    inbox.closed = true;
    inbox.ready.notify_one();
}

void list_devices() {
    std::string list;
    for (size_t i = 0; i < speech_device_count(); i++) {
        speech_device d;
        if (speech_device_get(i, &d) != SPEECH_OK) throw std::runtime_error(speech_last_error());
        // ggml's accelerators (BLAS) run with the CPU, and ASIST chooses only between a GPU and the CPU.
        const char * kind = d.kind == SPEECH_DEVICE_GPU ? "gpu" : d.kind == SPEECH_DEVICE_IGPU ? "igpu" : "cpu";
        list += (i ? "," : "") + std::string("{\"name\":") + json_string(d.name) + ",\"description\":" + json_string(d.description) +
                ",\"kind\":\"" + kind + "\",\"memoryTotal\":" + std::to_string(d.memory_total) +
                ",\"memoryFree\":" + std::to_string(d.memory_free) + "}";
    }
    emit("{\"type\":\"devices\",\"devices\":[" + list + "]}");
}

/** The command line after the two model files, as the library's parameters; anything else on it throws. */
struct Options {
    std::string model, codec, device;
    int context = speech_model_default_params().context;
    int steps = 0;
    std::vector<std::pair<std::string, std::string>> voices;
    uint64_t seed = std::random_device{}();
};

Options parse_options(const std::vector<std::string> & a) {
    Options o;
    o.model = a[1];
    o.codec = a[2];
    for (size_t i = 3; i < a.size(); i++) {
        const std::string & key = a[i];
        if (i + 1 >= a.size()) throw std::runtime_error(key + " needs a value");
        const std::string & value = a[++i];
        if (key == "--device" || key == "--backend") o.device = value;
        else if (key == "--seed") o.seed = std::stoull(value);
        else if (key == "--ctx") o.context = std::stoi(value);
        else if (key == "--steps") o.steps = std::stoi(value);
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

/** The ready message: the model, which the members after "type" describe, and the release of speech.cpp. */
std::string ready_message(const speech_model * m) {
    std::string out = "{\"type\":\"ready\",\"model\":" + json_string(speech_model_name(m)) +
                      ",\"architecture\":" + json_string(speech_model_architecture(m)) +
                      ",\"sampleRate\":" + std::to_string(speech_model_sample_rate(m)) + ",\"streaming\":\"" +
                      (speech_model_streaming(m) == SPEECH_STREAMING_FRAME ? "frame" : "sentence") + "\",\"voices\":" +
                      json_array(speech_model_voice_count(m), [&](size_t i) { return speech_model_voice(m, i); }) +
                      ",\"languages\":" +
                      json_array(speech_model_language_count(m), [&](size_t i) { return speech_model_language(m, i); }) +
                      ",\"languageSelectable\":" + (speech_model_language_selectable(m) ? "true" : "false");
    if (speech_model_steps(m) > 0) out += ",\"steps\":" + std::to_string(speech_model_steps(m));
    return out + ",\"backend\":" + json_string(speech_model_backend(m)) +
           ",\"version\":" + json_string(speech_version()) + "}";
}

/** What the audio callback needs of one request. */
struct Speaking {
    Inbox & inbox;
    const std::string & id;
    int seq = 0;
    size_t samples = 0;
    std::vector<int16_t> pcm;
};

int on_audio(const float * s, size_t n, void * user_data) {
    Speaking & r = *static_cast<Speaking *>(user_data);
    if (r.inbox.is_cancelled(r.id)) return 1;
    if (n == 0) return 0;
    r.pcm.resize(n);
    for (size_t i = 0; i < n; i++) r.pcm[i] = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s[i])) * 32767.0f);
    emit("{\"type\":\"chunk\",\"id\":" + json_string(r.id) + ",\"seq\":" + std::to_string(r.seq++) + ",\"pcm\":\"" +
         base64((const uint8_t *) r.pcm.data(), n * sizeof(int16_t)) + "\"}");
    r.samples += n;
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    try {
        protocol = take_stdout();
    } catch (const std::exception & e) {
        emit("{\"type\":\"fatal\",\"error\":" + json_string(e.what()) + "}");
        return 1;
    }
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() == 2 && args[1] == "--devices") {
        try {
            list_devices();
        } catch (const std::exception & e) {
            emit("{\"type\":\"fatal\",\"error\":" + json_string(e.what()) + "}");
            return 1;
        }
        return 0;
    }
    if (args.size() < 3) {
        emit("{\"type\":\"fatal\",\"error\":\"expected the model and codec GGUF paths\"}");
        return 2;
    }

    speech_model * model = nullptr;
    uint64_t seed = 0;
    try {
        const Options options = parse_options(args);
        seed = options.seed;
        model = load(options);
    } catch (const std::exception & e) {
        emit("{\"type\":\"fatal\",\"error\":" + json_string(e.what()) + "}");
        return 1;
    }
    emit(ready_message(model));

    Inbox inbox;
    std::thread reader(read_requests, std::ref(inbox));
    reader.detach();

    for (;;) {
        FlatJson request;
        {
            std::unique_lock<std::mutex> lock(inbox.mutex);
            inbox.ready.wait(lock, [&] { return !inbox.requests.empty() || inbox.closed; });
            if (inbox.requests.empty()) break;
            request = inbox.requests.front();
            inbox.requests.pop_front();
        }
        const std::string id = value(request, "id");
        if (inbox.is_cancelled(id)) {
            inbox.forget(id);
            continue;
        }
        const std::string text = value(request, "text"), voice = value(request, "voice"), language = value(request, "language");
        speech_request r = speech_request_default();
        r.text = text.c_str();
        r.voice = voice.c_str();
        r.language = language.c_str();
        r.seed = seed++;
        std::string error;
        try {
            r.speed = number(request, "speed", r.speed);
            r.seconds = number(request, "seconds", r.seconds);
            r.duration_scale = number(request, "durationScale", r.duration_scale);
        } catch (const std::exception & e) {
            error = e.what();
        }
        Speaking speaking{inbox, id, 0, 0, {}};
        if (error.empty() && speech_synthesize(model, &r, on_audio, &speaking) == SPEECH_ERROR) error = speech_last_error();
        if (!error.empty()) {
            emit("{\"type\":\"error\",\"id\":" + json_string(id) + ",\"error\":" + json_string(error) + "}");
        } else if (!inbox.is_cancelled(id)) {
            emit("{\"type\":\"end\",\"id\":" + json_string(id) + ",\"samples\":" + std::to_string(speaking.samples) + "}");
        }
        inbox.forget(id);
    }
    speech_model_free(model);
    return 0;
}
