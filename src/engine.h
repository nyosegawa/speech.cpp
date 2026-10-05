#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ggml-backend.h"
#include "speech.h"

/** What a family reads of speech_model_params, copied out of the caller's memory. */
struct EngineOptions {
    std::string model, codec;
    int context = 2048;
    std::vector<std::pair<std::string, std::string>> voices;
    int steps = 0;
};

/** What a loaded model tells its callers, fixed from its load to its end. */
struct EngineInfo {
    std::string name, architecture;
    int sample_rate = 0;
    speech_streaming streaming = SPEECH_STREAMING_FRAME;
    std::vector<std::string> voices, languages;
    bool language_selectable = false;
    int steps = 0;
};

/** One request with its strings copied; an empty language leaves the choice to the model. */
struct EngineRequest {
    std::string text, voice, language;
    uint64_t seed = 0;
};

/**
 * Called with each piece of audio as it is made, and with none (null, 0) where the synthesis can stop before it
 * has audio; returning false stops the synthesis without more audio.
 */
using AudioCallback = std::function<bool(const float * samples, size_t n)>;

/** One family of models behind the C API. */
class Engine {
public:
    virtual ~Engine() = default;

    const EngineInfo & info() const { return info_; }

    /** Speaks one request. A request the family cannot take throws. */
    virtual void speak(const EngineRequest & request, const AudioCallback & on_audio) = 0;

protected:
    EngineInfo info_;
};

std::unique_ptr<Engine> make_qwen3_tts(const EngineOptions & options, ggml_backend_t backend);
std::unique_ptr<Engine> make_irodori_tts(const EngineOptions & options, ggml_backend_t backend);
