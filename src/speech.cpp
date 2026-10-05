#include "speech.h"

#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.h"
#include "engine.h"
#include "irodori-tts/synthesizer.h"
#include "language.h"
#include "model-file.h"

namespace {

/**
 * A family of models: the general.architecture its GGUF files carry, what it does, which fields of
 * speech_model_params it takes, and its engine. Adding a family is adding its line to `families`.
 */
struct Family {
    const char * architecture;
    /** The name a message gives it, such as Qwen3-TTS. */
    const char * name;
    speech_task task;
    /** Whether it needs a codec, and takes voices, a context and sampler steps; any it does not take is refused. */
    bool codec, voices, context, steps;
    std::unique_ptr<Engine> (*make)(const EngineOptions & options, ggml_backend_t backend);
};

const Family families[] = {
    {"qwen3tts-talker", "Qwen3-TTS", SPEECH_TASK_SYNTHESIS, true, false, true, false, make_qwen3_tts},
    {"irodori-tts", "Irodori-TTS", SPEECH_TASK_SYNTHESIS, true, true, false, true, make_irodori_tts},
    {"fastconformer", "FastConformer", SPEECH_TASK_RECOGNITION, false, false, false, false, make_fastconformer},
};

}  // namespace

std::optional<std::string> Engine::transcribe(const std::vector<float> &, const StopCheck &) {
    throw std::logic_error("the family's engine does not recognize speech, though the table of families says it does");
}

void Engine::speak(const EngineRequest &, const AudioCallback &) {
    throw std::logic_error("the family's engine does not speak, though the table of families says it does");
}

struct speech_model {
    const Family * family = nullptr;
    ggml_backend_t backend = nullptr;
    std::unique_ptr<Engine> engine;
    /** Held for the whole of a request, so that requests on one model run one at a time. */
    std::mutex speaking;
    std::atomic<bool> cancelled{false};

    ~speech_model() {
        engine.reset();
        if (backend) ggml_backend_free(backend);
    }
};

namespace {

thread_local std::string last_error;

/** Runs `body`, turning what it throws into SPEECH_ERROR and the message speech_last_error() returns. */
template <typename Body>
speech_status guarded(Body && body) {
    try {
        return body();
    } catch (const std::exception & e) {
        last_error = e.what();
    } catch (...) {
        last_error = "an unknown C++ exception";
    }
    return SPEECH_ERROR;
}

template <typename Pointer>
void require(Pointer pointer, const char * what) {
    if (!pointer) throw std::invalid_argument(std::string(what) + " is NULL");
}

std::string text_or_empty(const char * s) {
    return s ? s : "";
}

EngineOptions engine_options(const speech_model_params & params) {
    require(params.model_path, "speech_model_params.model_path");
    if (params.n_voices > 0) require(params.voices, "speech_model_params.voices");
    EngineOptions o;
    o.model = params.model_path;
    o.codec = text_or_empty(params.codec_path);
    o.context = params.context;
    o.steps = params.steps;
    for (size_t i = 0; i < params.n_voices; i++) {
        require(params.voices[i].name, "a voice's name");
        require(params.voices[i].path, "a voice's path");
        o.voices.push_back({params.voices[i].name, params.voices[i].path});
    }
    return o;
}

const char * task_name(speech_task task) {
    return task == SPEECH_TASK_SYNTHESIS ? "speech synthesis" : "speech recognition";
}

/** The family of the model's GGUF file; an architecture no family has throws. */
const Family & family_of(const std::string & model) {
    const std::string architecture = gguf_architecture(model);
    for (const Family & f : families) {
        if (architecture == f.architecture) return f;
    }
    std::string known;
    for (const Family & f : families) known += (known.empty() ? "" : ", ") + std::string(f.architecture);
    throw std::runtime_error(model + " is a model of " + architecture + ", which speech.cpp does not run; it runs " + known);
}

/** Refuses a codec the family needs and lacks, and any field it does not take that is not at its default. */
void check_inputs(const Family & family, const EngineOptions & o) {
    const std::string model = o.model + " is a model of " + family.name + " for " + task_name(family.task);
    const speech_model_params defaults = speech_model_default_params();
    if (family.codec && o.codec.empty()) {
        throw std::invalid_argument(model + ", which needs its codec; give the codec's GGUF file in speech_model_params.codec_path");
    }
    if (!family.codec && !o.codec.empty()) {
        throw std::invalid_argument(model + ", which has no codec; leave speech_model_params.codec_path NULL, not " + o.codec);
    }
    if (!family.voices && !o.voices.empty()) {
        throw std::invalid_argument(model + ", which takes no voices" +
                                    (family.task == SPEECH_TASK_SYNTHESIS ? " and speaks with its own" : "") +
                                    "; leave speech_model_params.voices empty");
    }
    if (!family.context && o.context != defaults.context) {
        throw std::invalid_argument(model + ", which has no context to set; leave speech_model_params.context at " +
                                    std::to_string(defaults.context));
    }
    if (!family.steps && o.steps != defaults.steps) {
        throw std::invalid_argument(model + ", which has no sampler steps; leave speech_model_params.steps 0");
    }
}

void require_task(const speech_model * model, speech_task task, const char * instead) {
    if (model->family->task != task) {
        throw std::invalid_argument(model->engine->info().name + " is a " + task_name(model->family->task) + " model; use " + instead);
    }
}

/**
 * Refuses a language a model whose language is only checked does not have. A model that is told its language
 * (Qwen3-TTS) checks the language itself.
 */
void check_language(const speech_model * model, const std::string & tag) {
    const EngineInfo & info = model->engine->info();
    if (info.language_selectable || tag.empty() || tag == "auto") return;
    std::string list;
    for (const std::string & l : info.languages) {
        if (bcp47_matches(tag, l)) return;
        list += (list.empty() ? "" : ", ") + l;
    }
    throw std::invalid_argument(info.name + (model->family->task == SPEECH_TASK_SYNTHESIS ? " speaks " : " recognizes ") + list +
                                ", not " + tag);
}

/** The backend of `params.device`, freed when it goes out of scope unless released. */
std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> start_device(const speech_model_params & params) {
    return {init_backend(text_or_empty(params.device)), ggml_backend_free};
}

}  // namespace

extern "C" {

int speech_api_version(void) {
    return SPEECH_API_VERSION;
}

const char * speech_version(void) {
    return SPEECH_VERSION;
}

const char * speech_last_error(void) {
    return last_error.c_str();
}

size_t speech_device_count(void) {
    configure_ggml();
    return ggml_backend_dev_count();
}

speech_status speech_device_get(size_t index, speech_device * device) {
    return guarded([&] {
        require(device, "device");
        configure_ggml();
        if (index >= ggml_backend_dev_count()) {
            throw std::out_of_range("there is no device " + std::to_string(index) + "; speech_device_count() gives " +
                                    std::to_string(ggml_backend_dev_count()));
        }
        ggml_backend_dev_t dev = ggml_backend_dev_get(index);
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        device->name = ggml_backend_dev_name(dev);
        device->description = ggml_backend_dev_description(dev);
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU: device->kind = SPEECH_DEVICE_CPU; break;
            case GGML_BACKEND_DEVICE_TYPE_GPU: device->kind = SPEECH_DEVICE_GPU; break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU: device->kind = SPEECH_DEVICE_IGPU; break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: device->kind = SPEECH_DEVICE_ACCEL; break;
            default: throw std::runtime_error(std::string("ggml lists the device ") + device->name + " as of a kind speech.cpp does not know");
        }
        device->memory_total = total;
        device->memory_free = free;
        return SPEECH_OK;
    });
}

speech_model_params speech_model_default_params(void) {
    speech_model_params p = {};
    p.context = 2048;
    return p;
}

speech_status speech_model_load(const speech_model_params * params, speech_model ** model) {
    if (model) *model = nullptr;
    return guarded([&] {
        require(params, "params");
        require(model, "model");
        const EngineOptions options = engine_options(*params);
        const Family & family = family_of(options.model);
        check_inputs(family, options);
        auto m = std::make_unique<speech_model>();
        m->family = &family;
        auto backend = start_device(*params);
        m->engine = family.make(options, backend.get());
        m->backend = backend.release();
        *model = m.release();
        return SPEECH_OK;
    });
}

void speech_model_free(speech_model * model) {
    delete model;
}

const char * speech_model_name(const speech_model * model) {
    return model->engine->info().name.c_str();
}

const char * speech_model_architecture(const speech_model * model) {
    return model->family->architecture;
}

speech_task speech_model_task(const speech_model * model) {
    return model->family->task;
}

int speech_model_sample_rate(const speech_model * model) {
    return model->engine->info().sample_rate;
}

speech_streaming speech_model_streaming(const speech_model * model) {
    return model->engine->info().streaming;
}

size_t speech_model_voice_count(const speech_model * model) {
    return model->engine->info().voices.size();
}

const char * speech_model_voice(const speech_model * model, size_t index) {
    const auto & voices = model->engine->info().voices;
    return index < voices.size() ? voices[index].c_str() : nullptr;
}

size_t speech_model_language_count(const speech_model * model) {
    return model->engine->info().languages.size();
}

const char * speech_model_language(const speech_model * model, size_t index) {
    const auto & languages = model->engine->info().languages;
    return index < languages.size() ? languages[index].c_str() : nullptr;
}

int speech_model_language_selectable(const speech_model * model) {
    return model->engine->info().language_selectable ? 1 : 0;
}

int speech_model_steps(const speech_model * model) {
    return model->engine->info().steps;
}

const char * speech_model_backend(const speech_model * model) {
    return ggml_backend_name(model->backend);
}

speech_request speech_request_default(void) {
    speech_request r = {};
    r.speed = 1;
    r.duration_scale = 1;
    return r;
}

speech_status speech_synthesize(speech_model * model, const speech_request * request, speech_audio_callback on_audio,
                                void * user_data) {
    return guarded([&] {
        require(model, "model");
        require(request, "request");
        require(on_audio, "on_audio");
        require(request->text, "speech_request.text");
        require(request->voice, "speech_request.voice");
        require_task(model, SPEECH_TASK_SYNTHESIS, "speech_transcribe() to recognize speech with it");
        EngineRequest r;
        r.text = request->text;
        r.voice = request->voice;
        r.language = text_or_empty(request->language);
        r.seed = request->seed;
        r.speed = request->speed;
        r.seconds = request->seconds;
        r.duration_scale = request->duration_scale;
        check_language(model, r.language);
        std::lock_guard<std::mutex> lock(model->speaking);
        model->cancelled = false;
        bool stopped = false;
        model->engine->speak(r, [&](const float * samples, size_t n) {
            stopped = stopped || model->cancelled || on_audio(samples, n, user_data) != 0;
            return !stopped;
        });
        return stopped ? SPEECH_STOPPED : SPEECH_OK;
    });
}

speech_transcription_request speech_transcription_request_default(void) {
    speech_transcription_request r = {};
    return r;
}

speech_status speech_transcribe(speech_model * model, const speech_transcription_request * request, speech_text_callback on_text,
                                void * user_data) {
    return guarded([&] {
        require(model, "model");
        require(request, "request");
        require(on_text, "on_text");
        require_task(model, SPEECH_TASK_RECOGNITION, "speech_synthesize() to speak with it");
        const EngineInfo & info = model->engine->info();
        if (request->n_samples == 0) throw std::invalid_argument("the audio has no samples; give at least one");
        require(request->samples, "speech_transcription_request.samples");
        // Resampling is left to the caller: a resampler's filter changes what the model hears, and the official
        // implementations differ in theirs, so the library takes only the rate the model was trained on.
        if (request->sample_rate != info.sample_rate) {
            throw std::invalid_argument("the audio is at " + std::to_string(request->sample_rate) + " Hz, and " + info.name +
                                        " takes " + std::to_string(info.sample_rate) + " Hz; resample it to " +
                                        std::to_string(info.sample_rate) + " Hz first, for example with ffmpeg -ar " +
                                        std::to_string(info.sample_rate));
        }
        check_language(model, text_or_empty(request->language));
        const std::vector<float> samples(request->samples, request->samples + request->n_samples);
        std::lock_guard<std::mutex> lock(model->speaking);
        model->cancelled = false;
        const std::optional<std::string> text = model->engine->transcribe(samples, [&] { return model->cancelled.load(); });
        if (!text || model->cancelled) return SPEECH_STOPPED;
        return on_text(text->c_str(), user_data) != 0 ? SPEECH_STOPPED : SPEECH_OK;
    });
}

void speech_cancel(speech_model * model) {
    model->cancelled = true;
}

speech_status speech_make_voice(const speech_model_params * params, const char * wave_path, const char * voice_path) {
    return guarded([&] {
        require(params, "params");
        require(wave_path, "wave_path");
        require(voice_path, "voice_path");
        const EngineOptions options = engine_options(*params);
        const Family & family = family_of(options.model);
        if (family.make != make_irodori_tts) {
            throw std::runtime_error(options.model + " is a model of " + family.name + "; only Irodori-TTS makes voice files");
        }
        check_inputs(family, options);
        auto backend = start_device(*params);
        irodori::Synthesizer synth(options.model, options.codec, backend.get());
        synth.save_voice(synth.load_voice(wave_path), voice_path);
        return SPEECH_OK;
    });
}

}
