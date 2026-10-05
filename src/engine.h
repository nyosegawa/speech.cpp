#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ggml-backend.h"
#include "speech.h"

/**
 * What a family reads of speech_model_params, copied out of the caller's memory. src/speech.cpp has already refused
 * what the family does not take, so a family reads only its own fields.
 */
struct EngineOptions {
    std::string model;
    std::vector<std::pair<std::string, std::string>> voices;
    int steps = 0;
};

/** What a loaded model tells its callers, fixed from its load to its end. */
struct EngineInfo {
    std::string name;
    int sample_rate = 0;
    speech_streaming streaming = SPEECH_STREAMING_NONE;
    std::vector<std::string> voices, languages;
    bool language_selectable = false;
    int steps = 0;
};

/** One request with its strings copied; an empty language leaves the choice to the model. */
struct EngineRequest {
    std::string text, voice, language;
    uint64_t seed = 0;
    /** As speech_request has them: 1, 0 and 1 leave the rate and the length to the model. */
    double speed = 1, seconds = 0, duration_scale = 1;
};

/**
 * Called with each piece of audio as it is made, and with none (null, 0) where the synthesis can stop before it
 * has audio; returning false stops the synthesis without more audio.
 */
using AudioCallback = std::function<bool(const float * samples, size_t n)>;

/** Whether the request has been cancelled, asked where the recognition can stop. */
using StopCheck = std::function<bool()>;

/**
 * One family of models behind the C API. A synthesis family overrides speak() and a recognition family
 * transcribe(); src/speech.cpp calls only the one its table names for the family.
 */
class Engine {
public:
    virtual ~Engine() = default;

    const EngineInfo & info() const { return info_; }

    /** Speaks one request. A request the family cannot take throws. */
    virtual void speak(const EngineRequest & request, const AudioCallback & on_audio);

    /**
     * The text of mono samples, which src/speech.cpp has resampled to info().sample_rate and whose language it has
     * checked; nothing when `stopped` said so.
     */
    virtual std::optional<std::string> transcribe(const std::vector<float> & samples, const StopCheck & stopped);

protected:
    EngineInfo info_;
};

std::unique_ptr<Engine> make_qwen3_tts(const EngineOptions & options, ggml_backend_t backend);
std::unique_ptr<Engine> make_irodori_tts(const EngineOptions & options, ggml_backend_t backend);
std::unique_ptr<Engine> make_fastconformer(const EngineOptions & options, ggml_backend_t backend);
