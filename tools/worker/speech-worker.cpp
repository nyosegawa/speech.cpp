// A worker process that speaks texts or recognizes speech for another program over JSON Lines. It runs the family
// that general.architecture of the model's GGUF names: Qwen3-TTS or Irodori-TTS, which speak, or FastConformer,
// which recognizes speech. The ready message's "task" says which, and the requests follow from it.
//
// Reads one JSON object per line from stdin and answers with one JSON object per line on stdout. Nothing else
// reaches stdout: every log goes to stderr, so a line on stdout that is not a JSON object is a defect.
//
// A synthesis worker ("task": "synthesis"):
//   in : {"id": "...", "text": "...", "voice": "...", "language": "...", "speed": 1.0, "seconds": 2.5,
//         "durationScale": 1.2}
//        {"type": "cancel", "id": "..."}
//   out: {"type": "ready", "task": "synthesis", "model": "...", "architecture": "...", "sampleRate": 24000,
//         "streaming": "frame", "voices": [...], "languages": [...], "languageSelectable": true, "backend": "MTL0",
//         "version": "0.4.0"}
//        {"type": "chunk", "id": "...", "seq": 0, "pcm": "<base64 int16le mono>"}
//        {"type": "end", "id": "...", "samples": 123456}
//        {"type": "error", "id": "...", "error": "..."}
//        {"type": "fatal", "error": "..."}
//
// A recognition worker ("task": "recognition") takes the audio of a request in the chunks a synthesis worker sends,
// and the request itself in its end:
//   in : {"type": "chunk", "id": "...", "seq": 0, "pcm": "<base64 int16le mono>"}, seq counting from 0
//        {"type": "end", "id": "...", "sampleRate": 16000, "language": "..."}
//        {"type": "cancel", "id": "..."}
//   out: {"type": "ready", "task": "recognition", "model": "...", "architecture": "fastconformer",
//         "sampleRate": 16000, "languages": [...], "languageSelectable": false, "backend": "MTL0", "version": "..."}
//        {"type": "text", "id": "...", "text": "..."}
//        {"type": "error", "id": "...", "error": "..."}
// A request is answered once: with its text, or with one error, which comes as soon as one of its lines is refused
// (a chunk out of order or not base64, an end without a whole-number sampleRate), after which its other lines are
// dropped. sampleRate is the rate of the chunks' audio, any rate; the library resamples it to the model's.
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
// end, and one cancelled before it starts is dropped. A cancelled recognition request sends no text, whether it
// was still taking chunks, waiting or being recognized.
//
// Every line on stdin gets an answer, a recognition request's chunks through the answer to the request. A line
// that is not a message the worker can read (not one JSON object, a member that is not a string or a number, no
// "id", a "type" the worker's task does not take) is answered at once with an error naming the problem, with the
// "id" when one could be read and without one otherwise. Such an error is the caller's defect, not the model's.
//
// `--devices` instead prints the devices ggml can run on and exits, so that the caller can tell whether
// the machine has a GPU and how much memory it has before starting a worker:
//   {"type": "devices", "devices": [{"name": "Vulkan0", "description": "NVIDIA GeForce RTX 2080",
//                                    "kind": "gpu", "memoryTotal": 8589934592, "memoryFree": 7516192768}]}
//
// usage: speech-worker <model.gguf> <codec.gguf> [--device NAME|gpu|cpu] [--seed n]    (synthesis)
//                      [--ctx n]                               (Qwen3-TTS)
//                      [--voice NAME=FILE]... [--steps n]      (Irodori-TTS; FILE is a WAVE or voice file)
//        speech-worker <model.gguf> [--device NAME|gpu|cpu]                      (recognition)
//        speech-worker --devices

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "args.h"
#include "base64.h"
#include "flat-json.h"
#include "inbox.h"
#include "speech.h"
#include "take-stdout.h"

namespace {

std::mutex out_mutex;
FILE * protocol = stdout;

void emit(const std::string & json) {
    std::lock_guard<std::mutex> lock(out_mutex);
    std::fwrite(json.data(), 1, json.size(), protocol);
    std::fputc('\n', protocol);
    std::fflush(protocol);
}

/** A JSON array of the strings a getter gives for 0 to count - 1. */
template <typename Get>
std::string json_array(size_t count, Get get) {
    std::string out = "[";
    for (size_t i = 0; i < count; i++) out += (i ? "," : "") + json_string(get(i));
    return out + "]";
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
    bool has_seed = false;
    uint64_t seed = std::random_device{}();
};

Options parse_options(const std::vector<std::string> & a) {
    Options o;
    o.model = a[1];
    // A recognition model has no codec, so the second path is there only when it is not an option.
    size_t first = 2;
    if (a.size() > 2 && a[2].compare(0, 2, "--") != 0) o.codec = a[first++];
    for (size_t i = first; i < a.size(); i++) {
        const std::string & key = a[i];
        if (i + 1 >= a.size()) throw std::runtime_error(key + " needs a value");
        const std::string & value = a[++i];
        if (key == "--device" || key == "--backend") o.device = value;
        else if (key == "--seed") {
            o.seed = std::stoull(value);
            o.has_seed = true;
        }
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

/**
 * The ready message: the task, the model, which the members after "type" describe, and the release of speech.cpp. A
 * recognition model has no streaming, voices or steps, so its message leaves them out.
 */
std::string ready_message(const speech_model * m) {
    const bool synthesis = speech_model_task(m) == SPEECH_TASK_SYNTHESIS;
    std::string out = std::string("{\"type\":\"ready\",\"task\":\"") + (synthesis ? "synthesis" : "recognition") +
                      "\",\"model\":" + json_string(speech_model_name(m)) +
                      ",\"architecture\":" + json_string(speech_model_architecture(m)) +
                      ",\"sampleRate\":" + std::to_string(speech_model_sample_rate(m));
    if (synthesis) {
        out += std::string(",\"streaming\":\"") + (speech_model_streaming(m) == SPEECH_STREAMING_FRAME ? "frame" : "sentence") +
               "\",\"voices\":" + json_array(speech_model_voice_count(m), [&](size_t i) { return speech_model_voice(m, i); });
    }
    out += ",\"languages\":" + json_array(speech_model_language_count(m), [&](size_t i) { return speech_model_language(m, i); }) +
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

int collect_text(const char * text, void * user_data) {
    *static_cast<std::string *>(user_data) += text;
    return 0;
}

/** Recognizes one request and answers it, unless it is cancelled before its text is sent. */
void recognize(Inbox & inbox, speech_model * model, const std::string & id, const Request & request) {
    // 16-bit samples are scaled as a 16-bit WAVE file is read.
    std::vector<float> samples(request.pcm.size());
    for (size_t i = 0; i < samples.size(); i++) samples[i] = (float) request.pcm[i] / 32768.0f;
    const std::string language = value(request.message, "language");
    uint64_t rate = 0;
    whole_number(request.message, "sampleRate", rate);
    speech_transcription_request r = speech_transcription_request_default();
    r.samples = samples.data();
    r.n_samples = samples.size();
    r.sample_rate = (int) rate;
    r.language = language.c_str();
    {
        std::lock_guard<std::mutex> lock(inbox.mutex);
        inbox.running = id;
    }
    std::string text;
    const speech_status status = speech_transcribe(model, &r, collect_text, &text);
    const std::string error = status == SPEECH_ERROR ? speech_last_error() : "";
    {
        std::lock_guard<std::mutex> lock(inbox.mutex);
        inbox.running.clear();
    }
    if (status == SPEECH_ERROR) {
        emit("{\"type\":\"error\",\"id\":" + json_string(id) + ",\"error\":" + json_string(error) + "}");
    } else if (status == SPEECH_OK && !inbox.is_cancelled(id)) {
        emit("{\"type\":\"text\",\"id\":" + json_string(id) + ",\"text\":" + json_string(text) + "}");
    }
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
    if (args.size() < 2) {
        emit("{\"type\":\"fatal\",\"error\":\"expected the model's GGUF path, and the codec's for a synthesis model\"}");
        return 2;
    }

    speech_model * model = nullptr;
    uint64_t seed = 0;
    try {
        const Options options = parse_options(args);
        seed = options.seed;
        model = load(options);
        if (speech_model_task(model) == SPEECH_TASK_RECOGNITION && options.has_seed) {
            speech_model_free(model);
            throw std::runtime_error("a speech recognition model samples nothing and takes no seed; leave --seed out");
        }
    } catch (const std::exception & e) {
        emit("{\"type\":\"fatal\",\"error\":" + json_string(e.what()) + "}");
        return 1;
    }
    emit(ready_message(model));

    Inbox inbox;
    inbox.task = speech_model_task(model);
    inbox.model = model;
    std::thread reader(read_requests, std::ref(inbox), emit);
    reader.detach();

    for (;;) {
        Request taken;
        {
            std::unique_lock<std::mutex> lock(inbox.mutex);
            inbox.ready.wait(lock, [&] { return !inbox.requests.empty() || inbox.closed; });
            if (inbox.requests.empty()) break;
            taken = std::move(inbox.requests.front());
            inbox.requests.pop_front();
        }
        const FlatJson & request = taken.message;
        const std::string id = value(request, "id");
        if (inbox.is_cancelled(id)) {
            inbox.forget(id);
            continue;
        }
        if (inbox.task == SPEECH_TASK_RECOGNITION) {
            recognize(inbox, model, id, taken);
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
