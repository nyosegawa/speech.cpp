#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "failure.h"
#include "speech.h"

// The objects of the C API as the subcommands hold them, and how they load a model: the device and threads of the
// command line, a warm-up for the programs that answer requests as they come, and the voices of --add-voice.

using Model = std::unique_ptr<speech_model, decltype(&speech_model_free)>;
using ModelInfo = std::unique_ptr<speech_model_info, decltype(&speech_model_info_free)>;
using Request = std::unique_ptr<speech_request, decltype(&speech_request_free)>;
using LoadParams = std::unique_ptr<speech_load_params, decltype(&speech_load_params_free)>;

/** What a subcommand loads a model with: the load parameters it sets, and the voices it adds, each a name and a file. */
struct Loading {
    std::optional<std::string> device;
    std::optional<int> threads;
    bool warmup = false;
    std::vector<std::pair<std::string, std::string>> voices;
};

/**
 * How a detection model loads beside the models `loading` loads: on the CPU with one thread. Silero VAD's graphs are
 * small enough that a GPU takes longer to start one than the CPU to compute it, and a thread pool longer to wake than one
 * thread: on an Apple M5 (2026-10-08) a minute in pieces of 20 ms took 0.12 s on one thread, 0.17 s on the performance
 * cores and 0.52 s on Metal, and a minute whole 0.087 s, 0.042 s and 0.11 s. One device for every detection also cuts a
 * file and the same audio streamed into the same regions, which another device's arithmetic could move where a
 * probability lies near a threshold.
 */
inline Loading detection_loading(Loading loading) {
    loading.device = std::string("cpu");
    loading.threads = 1;
    return loading;
}

inline LoadParams load_params(const Loading & loading) {
    speech_load_params * raw = nullptr;
    check(speech_load_params_new(&raw));
    LoadParams params(raw, speech_load_params_free);
    if (loading.device) check(speech_load_params_set_device(raw, loading.device->c_str()));
    if (loading.threads) check(speech_load_params_set_threads(raw, *loading.threads));
    check(speech_load_params_set_warmup(raw, loading.warmup ? 1 : 0));
    return params;
}

/** Loads the model file at `path` and adds the voices, in their order; the library's refusal throws a Failure. */
inline Model load_model(const std::string & path, const Loading & loading) {
    const LoadParams params = load_params(loading);
    speech_model * raw = nullptr;
    check(speech_model_load(path.c_str(), params.get(), &raw));
    Model model(raw, speech_model_free);
    for (const auto & [name, file] : loading.voices) check(speech_voice_add(raw, name.c_str(), file.c_str()));
    return model;
}

/** The information of a loaded model as it is now. */
inline ModelInfo model_info(const speech_model * model) {
    speech_model_info * raw = nullptr;
    check(speech_model_get_info(model, &raw));
    return ModelInfo(raw, speech_model_info_free);
}

/** The information of the model file at `path`, read without loading it. */
inline ModelInfo file_info(const std::string & path) {
    speech_model_info * raw = nullptr;
    check(speech_model_info_open(path.c_str(), &raw));
    return ModelInfo(raw, speech_model_info_free);
}

inline Request new_request(speech_model * model) {
    speech_request * raw = nullptr;
    check(speech_request_new(model, &raw));
    return Request(raw, speech_request_free);
}

/** "synthesis", "recognition" or "detection". */
inline const char * task_name(speech_task task) {
    switch (task) {
        case SPEECH_TASK_SYNTHESIS: return "synthesis";
        case SPEECH_TASK_RECOGNITION: return "recognition";
        case SPEECH_TASK_DETECTION: return "detection";
    }
    throw std::logic_error("a task speech.h does not have");
}

/** The kind of a device as the subcommands write it: "cpu", "gpu" or "igpu". */
inline const char * device_kind_name(size_t index) {
    speech_device_kind kind = SPEECH_DEVICE_CPU;
    check(speech_device_get_kind(index, &kind));
    return kind == SPEECH_DEVICE_GPU ? "gpu" : kind == SPEECH_DEVICE_IGPU ? "igpu" : "cpu";
}
