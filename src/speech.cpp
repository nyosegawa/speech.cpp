#include "speech.h"

#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "backend.h"
#include "engine.h"
#include "irodori-tts/synthesizer.h"
#include "model-file.h"

struct speech_model {
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
    require(params.codec_path, "speech_model_params.codec_path");
    if (params.n_voices > 0) require(params.voices, "speech_model_params.voices");
    EngineOptions o;
    o.model = params.model_path;
    o.codec = params.codec_path;
    o.context = params.context;
    o.steps = params.steps;
    for (size_t i = 0; i < params.n_voices; i++) {
        require(params.voices[i].name, "a voice's name");
        require(params.voices[i].path, "a voice's path");
        o.voices.push_back({params.voices[i].name, params.voices[i].path});
    }
    return o;
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
        const std::string architecture = gguf_architecture(options.model);
        auto m = std::make_unique<speech_model>();
        auto backend = start_device(*params);
        if (architecture == "qwen3tts-talker") m->engine = make_qwen3_tts(options, backend.get());
        else if (architecture == "irodori-tts") m->engine = make_irodori_tts(options, backend.get());
        else throw std::runtime_error(options.model + " is a model of " + architecture + ", which speech.cpp does not run");
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
    return model->engine->info().architecture.c_str();
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

speech_status speech_synthesize(speech_model * model, const speech_request * request, speech_audio_callback on_audio,
                                void * user_data) {
    return guarded([&] {
        require(model, "model");
        require(request, "request");
        require(on_audio, "on_audio");
        require(request->text, "speech_request.text");
        require(request->voice, "speech_request.voice");
        EngineRequest r;
        r.text = request->text;
        r.voice = request->voice;
        r.language = text_or_empty(request->language);
        r.seed = request->seed;
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

void speech_cancel(speech_model * model) {
    model->cancelled = true;
}

speech_status speech_make_voice(const speech_model_params * params, const char * wave_path, const char * voice_path) {
    return guarded([&] {
        require(params, "params");
        require(wave_path, "wave_path");
        require(voice_path, "voice_path");
        const EngineOptions options = engine_options(*params);
        const std::string architecture = gguf_architecture(options.model);
        if (architecture != "irodori-tts") {
            throw std::runtime_error(options.model + " is a model of " + architecture + "; only Irodori-TTS makes voice files");
        }
        auto backend = start_device(*params);
        irodori::Synthesizer synth(options.model, options.codec, backend.get());
        synth.save_voice(synth.load_voice(wave_path), voice_path);
        return SPEECH_OK;
    });
}

}
