#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "irodori-tts/synthesizer.h"

namespace {

class IrodoriTtsEngine : public Engine {
public:
    IrodoriTtsEngine(const EngineOptions & options, ggml_backend_t backend) : synth_(options.model, options.codec, backend), steps_(options.steps) {
        if (options.voices.empty()) {
            throw std::runtime_error("an Irodori-TTS model has no voices of its own; give it at least one, a WAVE or voice file");
        }
        for (const auto & [name, path] : options.voices) {
            if (voices_.count(name)) throw std::runtime_error("two voices are named " + name);
            voices_[name] = synth_.load_voice(path);
            info_.voices.push_back(name);
        }
        info_.name = synth_.model().str("general.name");
        info_.sample_rate = synth_.sample_rate();
        info_.streaming = SPEECH_STREAMING_SENTENCE;
        info_.languages = synth_.model().str_array("speech.languages");
        info_.language_selectable = false;
        info_.steps = steps_ > 0 ? steps_ : (int) synth_.model().u32("irodori.default_steps");
        // The first synthesis compiles the GPU kernels, which on Vulkan takes seconds; a short and a longer
        // text run through every stage and both sizes of the decoder's windows.
        for (const char * text : {"あ。", "明日の東京は晴れで、最高気温は二十四度の予報です。"}) {
            irodori::Request warmup;
            warmup.text = text;
            warmup.steps = steps_;
            synth_.synthesize(warmup, voices_.begin()->second, [](const float *, size_t) { return true; });
        }
    }

    void speak(const EngineRequest & request, const AudioCallback & on_audio) override {
        const auto voice = voices_.find(request.voice);
        if (voice == voices_.end()) throw std::runtime_error("no voice is named \"" + request.voice + "\"");
        irodori::Request r;
        r.text = request.text;
        r.seed = request.seed;
        r.steps = steps_;
        r.length.seconds = request.seconds;
        r.length.duration_scale = request.duration_scale;
        r.length.speed = request.speed;
        r.cancelled = [&] { return !on_audio(nullptr, 0); };
        synth_.synthesize(r, voice->second, on_audio);
    }

private:
    irodori::Synthesizer synth_;
    int steps_;
    std::map<std::string, irodori::Voice> voices_;
};

}  // namespace

std::unique_ptr<Engine> make_irodori_tts(const EngineOptions & options, ggml_backend_t backend) {
    return std::make_unique<IrodoriTtsEngine>(options, backend);
}
