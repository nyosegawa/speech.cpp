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
// "speed", "seconds" and "durationScale" are JSON numbers, each optional, for the request options speed, seconds and
// duration_scale; a "seconds" of 0 leaves the length to the model. Irodori-TTS takes them; Qwen3-TTS answers an error
// to any but a speed or a scale of 1. --seed sets the seed of the first request, each later one taking the next, and
// --steps the sampler's steps of every request.
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
// `--devices` instead prints the devices the library can run a model on and exits, so that the caller can tell
// whether the machine has a GPU and how much memory it has before starting a worker:
//   {"type": "devices", "devices": [{"name": "Vulkan0", "description": "NVIDIA GeForce RTX 2080",
//                                    "kind": "gpu", "memoryTotal": 8589934592, "memoryFree": 7516192768}]}
//
// usage: speech-worker <model.gguf> [--device NAME|gpu|cpu] [--seed n]                (synthesis)
//                      [--voice NAME=FILE]... [--steps n]      (Irodori-TTS; FILE is a WAVE or voice file)
//        speech-worker <model.gguf> [--device NAME|gpu|cpu]                           (recognition)
//        speech-worker --devices

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
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
#include "list-devices.h"
#include "load-model.h"
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
        uint64_t total = 0, free = 0;
        if (speech_device_memory(i, &total, &free) != SPEECH_OK) throw_last_error();
        list += (i ? "," : "") + std::string("{\"name\":") + json_string(speech_device_name(i)) +
                ",\"description\":" + json_string(speech_device_description(i)) + ",\"kind\":\"" + device_kind_name(i) +
                "\",\"memoryTotal\":" + std::to_string(total) + ",\"memoryFree\":" + std::to_string(free) + "}";
    }
    emit("{\"type\":\"devices\",\"devices\":[" + list + "]}");
}

/** The command line after the model file; anything else on it throws. */
struct Options {
    std::string model, device;
    int steps = 0;
    std::vector<std::pair<std::string, std::string>> voices;
    bool has_seed = false;
    uint64_t seed = std::random_device{}();
};

Options parse_options(const std::vector<std::string> & a) {
    Options o;
    o.model = a[1];
    for (size_t i = 2; i < a.size(); i++) {
        const std::string & key = a[i];
        if (key.compare(0, 2, "--") != 0) {
            throw std::runtime_error("unexpected " + key + "; give the model's one GGUF file, which holds its codec, and then the options");
        }
        if (i + 1 >= a.size()) throw std::runtime_error(key + " needs a value");
        const std::string & value = a[++i];
        if (key == "--device" || key == "--backend") o.device = value;
        else if (key == "--seed") {
            o.seed = std::stoull(value);
            o.has_seed = true;
        }
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

using RequestHandle = std::unique_ptr<speech_request, decltype(&speech_request_free)>;

RequestHandle new_request(speech_model * model) {
    speech_request * r = nullptr;
    if (speech_request_new(model, &r) != SPEECH_OK) throw_last_error();
    return RequestHandle(r, speech_request_free);
}

/**
 * Refuses what the command line asks of a model that cannot do it, before the first request: steps for a model without
 * them or out of their range, a seed for a model that samples nothing, and a synthesis model without a voice.
 */
void check_options(speech_model * model, const speech_model_info * info, const Options & o) {
    const std::string name = speech_model_info_name(info);
    if (o.steps != 0) {
        if (!speech_model_info_takes(info, SPEECH_OPT_STEPS)) throw std::runtime_error(name + " has no sampler steps; leave --steps out");
        const RequestHandle r = new_request(model);
        if (speech_request_set_int(r.get(), SPEECH_OPT_STEPS, o.steps) != SPEECH_OK) throw_last_error();
    }
    if (o.has_seed && !speech_model_info_takes(info, SPEECH_OPT_SEED)) {
        throw std::runtime_error("a speech recognition model samples nothing and takes no seed; leave --seed out");
    }
    if (speech_model_info_option_required(info, SPEECH_OPT_VOICE) && speech_model_info_voice_count(info) == 0) {
        throw std::runtime_error(name + " has no voices of its own; give it at least one with --voice NAME=FILE, a WAVE or voice file");
    }
}

/**
 * The ready message: the task, the model, which the members after "type" describe, and the release of speech.cpp. A
 * recognition model has no streaming, voices or steps, so its message leaves them out.
 */
std::string ready_message(const speech_model_info * m, int steps) {
    const bool synthesis = speech_model_info_task(m) == SPEECH_TASK_SYNTHESIS;
    std::string out = std::string("{\"type\":\"ready\",\"task\":\"") + (synthesis ? "synthesis" : "recognition") +
                      "\",\"model\":" + json_string(speech_model_info_name(m)) +
                      ",\"architecture\":" + json_string(speech_model_info_architecture(m)) +
                      ",\"sampleRate\":" + std::to_string(speech_model_info_sample_rate(m));
    if (synthesis) {
        out += std::string(",\"streaming\":\"") + (speech_model_info_incremental(m) ? "frame" : "sentence") +
               "\",\"voices\":" + json_array(speech_model_info_voice_count(m), [&](size_t i) { return speech_model_info_voice_name(m, i); });
    }
    out += ",\"languages\":" + json_array(speech_model_info_language_count(m), [&](size_t i) { return speech_model_info_language(m, i); }) +
           ",\"languageSelectable\":" + (speech_model_info_option_steers(m, SPEECH_OPT_LANGUAGE) ? "true" : "false");
    if (steps > 0) out += ",\"steps\":" + std::to_string(steps);
    return out + ",\"backend\":" + json_string(speech_model_info_device(m)) +
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
    r.pcm.resize(n);
    for (size_t i = 0; i < n; i++) r.pcm[i] = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s[i])) * 32767.0f);
    emit("{\"type\":\"chunk\",\"id\":" + json_string(r.id) + ",\"seq\":" + std::to_string(r.seq++) + ",\"pcm\":\"" +
         base64((const uint8_t *) r.pcm.data(), n * sizeof(int16_t)) + "\"}");
    r.samples += n;
    return 0;
}

/**
 * Runs a request, which a cancel of its id stops while it runs, unless the id was cancelled before; returns
 * SPEECH_CANCELLED for one that did not run.
 */
template <typename Body>
speech_status run_request(Inbox & inbox, const std::string & id, speech_request * request, Body && body) {
    {
        std::lock_guard<std::mutex> lock(inbox.mutex);
        if (inbox.cancelled.count(id)) return SPEECH_CANCELLED;
        inbox.running = id;
        inbox.request = request;
    }
    const speech_status status = body();
    std::lock_guard<std::mutex> lock(inbox.mutex);
    inbox.running.clear();
    inbox.request = nullptr;
    return status;
}

/** The error of a request: the library's last one. */
void emit_error(const std::string & id, const std::string & error) {
    emit("{\"type\":\"error\",\"id\":" + json_string(id) + ",\"error\":" + json_string(error) + "}");
}

/** Speaks one request and answers it, unless it is cancelled. */
void speak(Inbox & inbox, speech_model * model, const std::string & id, const FlatJson & message, const Options & options, uint64_t seed) {
    const RequestHandle request = new_request(model);
    speech_request * r = request.get();
    std::string error;
    try {
        if (speech_request_set_text(r, value(message, "text").c_str()) != SPEECH_OK) throw_last_error();
        if (message.count("voice") && speech_request_set_string(r, SPEECH_OPT_VOICE, value(message, "voice").c_str()) != SPEECH_OK) throw_last_error();
        const std::string language = value(message, "language");
        if (!language.empty() && speech_request_set_string(r, SPEECH_OPT_LANGUAGE, language.c_str()) != SPEECH_OK) throw_last_error();
        if (speech_request_set_int(r, SPEECH_OPT_SEED, (int64_t) seed) != SPEECH_OK) throw_last_error();
        if (options.steps > 0 && speech_request_set_int(r, SPEECH_OPT_STEPS, options.steps) != SPEECH_OK) throw_last_error();
        if (message.count("speed") && speech_request_set_float(r, SPEECH_OPT_SPEED, number(message, "speed", 1)) != SPEECH_OK) throw_last_error();
        const double seconds = number(message, "seconds", 0);
        if (seconds != 0 && speech_request_set_float(r, SPEECH_OPT_SECONDS, seconds) != SPEECH_OK) throw_last_error();
        if (message.count("durationScale") &&
            speech_request_set_float(r, SPEECH_OPT_DURATION_SCALE, number(message, "durationScale", 1)) != SPEECH_OK) {
            throw_last_error();
        }
    } catch (const std::exception & e) {
        error = e.what();
    }
    Speaking speaking{inbox, id, 0, 0, {}};
    speech_status status = SPEECH_CANCELLED;
    if (error.empty()) {
        status = run_request(inbox, id, r, [&] { return speech_synthesize(r, on_audio, &speaking); });
        if (status < 0) error = speech_last_error();
    }
    if (!error.empty()) {
        emit_error(id, error);
    } else if (status == SPEECH_OK && !inbox.is_cancelled(id)) {
        emit("{\"type\":\"end\",\"id\":" + json_string(id) + ",\"samples\":" + std::to_string(speaking.samples) + "}");
    }
}

/** Recognizes one request and answers it, unless it is cancelled before its text is sent. */
void recognize(Inbox & inbox, speech_model * model, const std::string & id, const Request & taken) {
    // 16-bit samples are scaled as a 16-bit WAVE file is read.
    std::vector<float> samples(taken.pcm.size());
    for (size_t i = 0; i < samples.size(); i++) samples[i] = (float) taken.pcm[i] / 32768.0f;
    const std::string language = value(taken.message, "language");
    uint64_t rate = 0;
    whole_number(taken.message, "sampleRate", rate);
    const RequestHandle request = new_request(model);
    speech_request * r = request.get();
    speech_status status = speech_request_set_audio(r, samples.data(), samples.size(), (int) rate);
    if (status == SPEECH_OK && !language.empty()) status = speech_request_set_string(r, SPEECH_OPT_LANGUAGE, language.c_str());
    if (status == SPEECH_OK) status = run_request(inbox, id, r, [&] { return speech_transcribe(r); });
    if (status < 0) {
        emit_error(id, speech_last_error());
    } else if (status == SPEECH_OK && !inbox.is_cancelled(id)) {
        emit("{\"type\":\"text\",\"id\":" + json_string(id) + ",\"text\":" + json_string(speech_result_text(speech_request_result(r))) + "}");
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
        emit("{\"type\":\"fatal\",\"error\":\"expected the model's GGUF path\"}");
        return 2;
    }

    Options options;
    Model model(nullptr, speech_model_free);
    std::string ready;
    try {
        options = parse_options(args);
        model = load_model(options.model, options.device, options.voices);
        const ModelInfo info = model_info(model.get());
        check_options(model.get(), info.get(), options);
        ready = ready_message(info.get(), (int) steps_in_effect(info.get(), options.steps));
    } catch (const std::exception & e) {
        emit("{\"type\":\"fatal\",\"error\":" + json_string(e.what()) + "}");
        return 1;
    }
    emit(ready);

    Inbox inbox;
    {
        const ModelInfo info = model_info(model.get());
        inbox.task = speech_model_info_task(info.get());
    }
    std::thread reader(read_requests, std::ref(inbox), emit);
    reader.detach();

    uint64_t seed = options.seed;
    for (;;) {
        Request taken;
        {
            std::unique_lock<std::mutex> lock(inbox.mutex);
            inbox.ready.wait(lock, [&] { return !inbox.requests.empty() || inbox.closed; });
            if (inbox.requests.empty()) break;
            taken = std::move(inbox.requests.front());
            inbox.requests.pop_front();
        }
        const std::string id = value(taken.message, "id");
        if (!inbox.is_cancelled(id)) {
            if (inbox.task == SPEECH_TASK_RECOGNITION) recognize(inbox, model.get(), id, taken);
            else speak(inbox, model.get(), id, taken.message, options, seed++);
        }
        inbox.forget(id);
    }
    return 0;
}
