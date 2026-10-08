#include <string>
#include <vector>

#include "engine.h"
#include "silero-vad/detector.h"
#include "silero-vad/regions.h"

namespace {

class SileroVadEngine : public Engine {
public:
    SileroVadEngine(const std::string & path, ggml_backend_t backend) : detector_(path, backend) {}

    /** The blocks of chunks report their progress as they finish; the regions follow from the probabilities at once. */
    std::vector<TimedText> detect(const std::vector<float> & samples, const RequestValues & values, Run & run) override {
        silero_vad::RegionRule rule = silero_vad::default_rule(detector_.model());
        rule.threshold = values.number(SPEECH_OPT_THRESHOLD);
        rule.min_speech_duration_ms = values.integer(SPEECH_OPT_MIN_SPEECH_DURATION_MS);
        rule.min_silence_duration_ms = values.integer(SPEECH_OPT_MIN_SILENCE_DURATION_MS);
        rule.speech_pad_ms = values.integer(SPEECH_OPT_SPEECH_PAD_MS);
        if (values.has(SPEECH_OPT_MAX_SPEECH_DURATION_S)) rule.max_speech_duration_s = values.number(SPEECH_OPT_MAX_SPEECH_DURATION_S);
        const std::vector<float> probs = detector_.probabilities(samples, [&](double done) { return run.progress(done); });
        if (run.stopped()) return {};
        const double rate = detector_.sample_rate();
        std::vector<TimedText> out;
        for (const silero_vad::Region & r : silero_vad::speech_regions(probs, (int64_t) samples.size(), detector_.sample_rate(), detector_.network().chunk(), rule)) {
            out.push_back({(double) r.start / rate, (double) r.end / rate, ""});
        }
        return out;
    }

    void warm_up() override {
        // The first graph builds Metal's pipelines and compiles Vulkan's shaders.
        detector_.probabilities(std::vector<float>((size_t) detector_.sample_rate(), 0.0f));
    }

private:
    silero_vad::Detector detector_;
};

}  // namespace

/**
 * The table of the options Silero VAD takes: those of get_speech_timestamps() that a caller tunes, under its names and
 * with its defaults, which the file gives. The model takes no language and no timestamps, whose neutral values every
 * model accepts. max_speech_duration_s has no default, as the official's infinity is no value a request sets.
 */
FamilyInfo describe_silero_vad(const std::shared_ptr<const ModelFile> & file) {
    const ModelFile & m = *file;
    FamilyInfo info;
    info.options = {
        {SPEECH_OPT_THRESHOLD, false, true, m.f64("silero-vad.threshold"), 0, 1},
        {SPEECH_OPT_MIN_SPEECH_DURATION_MS, false, true, (int64_t) m.u32("silero-vad.min_speech_duration_ms"), 0},
        {SPEECH_OPT_MIN_SILENCE_DURATION_MS, false, true, (int64_t) m.u32("silero-vad.min_silence_duration_ms"), 0},
        {SPEECH_OPT_SPEECH_PAD_MS, false, true, (int64_t) m.u32("silero-vad.speech_pad_ms"), 0},
        {SPEECH_OPT_MAX_SPEECH_DURATION_S, false, true, std::nullopt, 0, INFINITY, true},
    };
    return info;
}

std::unique_ptr<Engine> load_silero_vad(const std::string & path, ggml_backend_t backend) {
    return std::make_unique<SileroVadEngine>(path, backend);
}
