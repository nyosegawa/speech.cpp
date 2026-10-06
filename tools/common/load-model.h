#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "speech.h"

// How the tools load a model through the C API: the device a command line names, a warm-up, and the voices of
// --voice NAME=FILE added to it.

/** The last error of the library on this thread, thrown with its message. */
[[noreturn]] inline void throw_last_error() {
    throw std::runtime_error(speech_last_error());
}

using Model = std::unique_ptr<speech_model, decltype(&speech_model_free)>;
using ModelInfo = std::unique_ptr<speech_model_info, decltype(&speech_model_info_free)>;

/**
 * Loads the model file at `path` on `device`, or on the library's choice when it is empty, warmed up so that the GPU's
 * kernels are compiled before the first request, and adds each voice of `voices`, a name and a voice file or WAVE file.
 */
inline Model load_model(const std::string & path, const std::string & device,
                        const std::vector<std::pair<std::string, std::string>> & voices) {
    speech_load_params * raw = nullptr;
    if (speech_load_params_new(&raw) != SPEECH_OK) throw_last_error();
    const std::unique_ptr<speech_load_params, decltype(&speech_load_params_free)> params(raw, speech_load_params_free);
    if (!device.empty() && speech_load_params_set_device(params.get(), device.c_str()) != SPEECH_OK) throw_last_error();
    if (speech_load_params_set_warmup(params.get(), 1) != SPEECH_OK) throw_last_error();
    speech_model * model = nullptr;
    if (speech_model_load(path.c_str(), params.get(), &model) != SPEECH_OK) throw_last_error();
    Model owned(model, speech_model_free);
    for (const auto & [name, file] : voices) {
        if (speech_voice_add(model, name.c_str(), file.c_str()) != SPEECH_OK) throw_last_error();
    }
    return owned;
}

/** The information of a loaded model as it is now. */
inline ModelInfo model_info(const speech_model * model) {
    speech_model_info * info = nullptr;
    if (speech_model_get_info(model, &info) != SPEECH_OK) throw_last_error();
    return ModelInfo(info, speech_model_info_free);
}

/** The steps a request runs: those a command line gives, or the model's default; 0 for a model without steps. */
inline int64_t steps_in_effect(const speech_model_info * info, int given) {
    if (!speech_model_info_takes(info, SPEECH_OPT_STEPS)) return 0;
    if (given > 0) return given;
    int64_t steps = 0;
    if (speech_model_info_option_default_int(info, SPEECH_OPT_STEPS, &steps) != SPEECH_OK) throw_last_error();
    return steps;
}
