#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "engine.h"
#include "qwen3-tts/synthesizer.h"

namespace {

class Qwen3TtsEngine : public Engine {
public:
    Qwen3TtsEngine(const EngineOptions & options, ggml_backend_t backend) : synth_(options.model, backend) {
        info_.name = synth_.model().str("general.name");
        info_.sample_rate = synth_.sample_rate();
        info_.streaming = SPEECH_STREAMING_FRAME;
        info_.voices = synth_.ids().voices;
        info_.languages = synth_.ids().languages;
        info_.language_selectable = synth_.model().str("speech.language_use") == "steers";
        // The first synthesis compiles the GPU kernels, which on Vulkan takes seconds for every new shape, so a
        // short and a longer text run through the prompt, the talker, the code predictor and the codec at the
        // sizes speech uses.
        for (const auto & [text, frames] : std::vector<std::pair<std::string, int>>{
                 {"あ", 4}, {"明日の東京は晴れで、最高気温は二十四度の予報です。", 40}}) {
            SynthesisRequest warmup;
            warmup.text = text;
            warmup.speaker = info_.voices[0];
            warmup.max_frames = frames;
            synth_.synthesize(warmup, [](const float *, size_t) { return true; });
        }
    }

    void speak(const EngineRequest & request, const AudioCallback & on_audio) override {
        // The official implementation has no control of the rate or the length: the talker decides when the
        // speech ends. Changing the audio's rate afterwards would change its pitch or add the artifacts of a
        // time stretch, so such a request is refused.
        if (request.speed != 1) {
            throw std::invalid_argument("Qwen3-TTS cannot change its speaking rate; leave speed out or give 1");
        }
        if (request.seconds != 0 || request.duration_scale != 1) {
            throw std::invalid_argument("Qwen3-TTS cannot set the length of its speech; leave seconds and the duration scale out");
        }
        SynthesisRequest r;
        r.text = request.text;
        r.speaker = request.voice;
        r.language = request.language.empty() ? "auto" : request.language;
        r.seed = request.seed;
        // Qwen3-TTS passes audio after its first frame and then every four frames, so it stops at the sink.
        synth_.synthesize(r, on_audio);
    }

private:
    Synthesizer synth_;
};

}  // namespace

std::unique_ptr<Engine> make_qwen3_tts(const EngineOptions & options, ggml_backend_t backend) {
    return std::make_unique<Qwen3TtsEngine>(options, backend);
}
