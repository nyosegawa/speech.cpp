#include <climits>
#include <map>
#include <string>
#include <vector>

#include "engine.h"
#include "irodori-tts/synthesizer.h"
#include "irodori-tts/text-normalizer.h"

namespace {

/** The frames of the latent a warm-up speaks from, a second of silence, since a model may have no voice when it loads. */
constexpr int kWarmUpFrames = 25;

/**
 * The sampler's steps of a warm-up. Two run both of its graphs: RF guides its first step, at t 0.999, and not its
 * second, at 0.4995, below the irodori-tts.sampler.cfg_min_t of 0.5 that the RF file gives; MeanFlow has one graph.
 */
constexpr int kWarmUpSteps = 2;

class IrodoriTtsEngine : public Engine {
public:
    IrodoriTtsEngine(const std::string & path, ggml_backend_t backend) : synth_(path, backend) {}

    speech_stop speak(const std::string & text, const RequestValues & values, Run & run) override {
        irodori::Request r;
        r.text = text;
        r.seed = (uint64_t) values.integer(SPEECH_OPT_SEED);
        r.steps = (int) values.integer(SPEECH_OPT_STEPS);
        r.length.seconds = values.has(SPEECH_OPT_SECONDS) ? values.number(SPEECH_OPT_SECONDS) : 0;
        r.length.duration_scale = values.number(SPEECH_OPT_DURATION_SCALE);
        r.length.speed = values.number(SPEECH_OPT_SPEED);
        r.progress = [&](double done) { return run.progress(done); };
        synth_.synthesize(r, voices_.at(values.string(SPEECH_OPT_VOICE)), [&](const float * samples, size_t n) { return run.audio(samples, n); });
        return SPEECH_STOP_COMPLETE;
    }

    void add_voice(const std::string & name, const std::string & path) override {
        voices_.emplace(name, synth_.load_voice(path));
    }

    void warm_up() override {
        // The first synthesis compiles the GPU kernels, which on Vulkan takes seconds; a short and a longer text run
        // through every stage and both sizes of the decoder's windows.
        const irodori::Voice silence = synth_.voice_from_latent(std::vector<float>((size_t) kWarmUpFrames * synth_.codec().latent_dim(), 0.0f));
        for (const char * text : {"あ。", "明日の東京は晴れで、最高気温は二十四度の予報です。"}) {
            irodori::Request warmup;
            warmup.text = text;
            warmup.steps = kWarmUpSteps;
            synth_.synthesize(warmup, silence, [](const float *, size_t) { return true; });
        }
    }

private:
    irodori::Synthesizer synth_;
    std::map<std::string, irodori::Voice> voices_;
};

/** The tokens of a text as the synthesis counts them: those of its normalized form, <s> included. */
struct TokenCounter {
    explicit TokenCounter(const ModelFile & m) : tokenizer(m) {}
    size_t count(const std::string & text) const { return tokenizer.encode(irodori::normalize_text(text)).size(); }
    irodori::Tokenizer tokenizer;
};

}  // namespace

/**
 * The table of the options Irodori-TTS takes, with the bounds of the length and the speed and the sampler's steps its
 * file gives. It has no voices of its own; a request speaks in one added since loading. Its one language is checked and
 * not used. It fixes the length before it makes the speech, so it takes seconds and a scale of the predicted length
 * rather than max_seconds.
 */
FamilyInfo describe_irodori_tts(const std::shared_ptr<const ModelFile> & file) {
    const ModelFile & m = *file;
    FamilyInfo info;
    info.voice_codec = m.str("irodori-tts.codec.sha256");
    info.max_text_tokens = m.u32("irodori-tts.text.max_tokens");
    info.options = {
        {SPEECH_OPT_VOICE, true, true},
        {SPEECH_OPT_LANGUAGE, false, false, std::string("auto")},
        {SPEECH_OPT_SEED, false, true, std::nullopt, 0, (double) kMaxSeed},
        {SPEECH_OPT_SPEED, false, true, 1.0, m.f32("irodori-tts.length.min_speed"), m.f32("irodori-tts.length.max_speed")},
        {SPEECH_OPT_SECONDS, false, true, std::nullopt, m.f32("irodori-tts.length.min_seconds"), m.f32("irodori-tts.length.max_seconds")},
        {SPEECH_OPT_DURATION_SCALE, false, true, 1.0, 0, INFINITY, true},
        {SPEECH_OPT_STEPS, false, true, (int64_t) m.u32("irodori-tts.sampler.default_steps"), 1, (double) INT_MAX},
    };
    const auto counter = std::make_shared<Lazy<TokenCounter>>(file);
    info.count_tokens = [counter](const std::string & text) { return counter->get().count(text); };
    return info;
}

std::unique_ptr<Engine> load_irodori_tts(const std::string & path, ggml_backend_t backend) {
    return std::make_unique<IrodoriTtsEngine>(path, backend);
}
