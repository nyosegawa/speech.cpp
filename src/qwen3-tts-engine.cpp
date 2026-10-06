#include <climits>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "engine.h"
#include "qwen3-tts/codec.h"
#include "qwen3-tts/synthesizer.h"

namespace {

/** The frames whose audio fits in `seconds`, counted in whole samples so that 0.24 s at 24 kHz is three frames. */
int frames_within(double seconds, int sample_rate, int samples_per_frame) {
    return (int) ((int64_t) std::floor(seconds * sample_rate) / samples_per_frame);
}

class Qwen3TtsEngine : public Engine {
public:
    Qwen3TtsEngine(const std::string & path, ggml_backend_t backend) : synth_(path, backend) {}

    speech_stop speak(const std::string & text, const RequestValues & values, Run & run) override {
        SynthesisRequest r;
        r.text = text;
        r.speaker = values.string(SPEECH_OPT_VOICE);
        r.language = values.string(SPEECH_OPT_LANGUAGE);
        r.seed = (uint64_t) values.integer(SPEECH_OPT_SEED);
        const bool capped = values.has(SPEECH_OPT_MAX_SECONDS);
        if (capped) r.max_frames = frames_within(values.number(SPEECH_OPT_MAX_SECONDS), synth_.sample_rate(), synth_.samples_per_frame());
        // Qwen3-TTS passes audio after its first frame and then every four frames, so it stops at the sink.
        const SynthesisOutcome outcome = synth_.synthesize(r, [&](const float * samples, size_t n) { return run.audio(samples, n); });
        if (outcome.ended) return SPEECH_STOP_COMPLETE;
        return capped && outcome.frames == r.max_frames ? SPEECH_STOP_MAX_SECONDS : SPEECH_STOP_MODEL_LIMIT;
    }

    void warm_up() override {
        // The first synthesis compiles the GPU kernels, which on Vulkan takes seconds for every new shape, so a short and
        // a longer text run through the prompt, the talker, the code predictor and the codec at the sizes speech uses.
        for (const auto & [text, frames] : std::vector<std::pair<std::string, int>>{
                 {"あ", 4}, {"明日の東京は晴れで、最高気温は二十四度の予報です。", 40}}) {
            SynthesisRequest warmup;
            warmup.text = text;
            warmup.speaker = synth_.ids().voices[0];
            warmup.max_frames = frames;
            synth_.synthesize(warmup, [](const float *, size_t) { return true; });
        }
    }

private:
    Synthesizer synth_;
};

}  // namespace

/**
 * The table of the options Qwen3-TTS takes. The official implementation has no control of the rate or the length, and
 * changing the audio's rate afterwards would change its pitch or add a time stretch's artifacts, so it takes neither:
 * its speech ends where the talker ends it, at the request's max_seconds, or at the model's limit of frames.
 */
FamilyInfo describe_qwen3_tts(const std::shared_ptr<const ModelFile> & file) {
    const ModelFile & m = *file;
    FamilyInfo info;
    info.incremental = true;
    const std::vector<std::string> names = m.str_array("speech.voices"), languages = m.str_array("speech.voice_languages"),
                                   genders = m.str_array("speech.voice_genders"), descriptions = m.str_array("speech.voice_descriptions");
    for (size_t i = 0; i < names.size(); i++) info.voices.push_back({names[i], languages[i], genders[i], descriptions[i]});
    info.max_text_tokens = (size_t) text_token_limit(m);
    const double longest = (double) m.u32("qwen3-tts.generation.max_frames") * codec_samples_per_frame(m) / m.u32("speech.sample_rate");
    info.options = {
        {SPEECH_OPT_VOICE, true, true},
        {SPEECH_OPT_LANGUAGE, false, true, std::string("auto")},
        {SPEECH_OPT_SEED, false, true, std::nullopt, 0, (double) kMaxSeed},
        {SPEECH_OPT_MAX_SECONDS, false, true, std::nullopt, 0, longest, true},
    };
    const auto tokenizer = std::make_shared<Lazy<Tokenizer>>(file);
    info.count_tokens = [tokenizer](const std::string & text) { return tokenizer->get().encode(text).size(); };
    return info;
}

std::unique_ptr<Engine> load_qwen3_tts(const std::string & path, ggml_backend_t backend) {
    return std::make_unique<Qwen3TtsEngine>(path, backend);
}
