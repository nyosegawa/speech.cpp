#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "engine.h"
#include "error.h"
#include "resample.h"
#include "semamba/enhancer.h"

namespace {

/** What a request asks of an enhancement, checked against the model: the rate it enhances at and its chunks. */
struct Plan {
    int rate;
    double chunk, portion;
    size_t fewest;
};

class SemambaEngine : public Engine {
public:
    SemambaEngine(const std::string & path, ggml_backend_t backend) : enhancer_(path, backend) {}

    /** The chunks report their progress as they finish, and pass their audio once no later chunk changes it. */
    void enhance(const std::vector<float> & samples, int sample_rate, const RequestValues & values, Run & run) override {
        const Plan plan = plan_of(sample_rate, values);
        const std::vector<float> audio = plan.rate == sample_rate ? samples : Resampler(sample_rate, plan.rate)(samples);
        check_length(audio.size(), plan);
        enhancer_.enhance(audio, plan.rate, plan.chunk, plan.portion, [&](const float * s, size_t n) { return run.audio(s, n); },
                          [&](double done) { return run.progress(done); });
    }

    std::unique_ptr<EnhancementStream> start_enhancement(int sample_rate, const RequestValues & values, AudioSink sink) override {
        const Plan plan = plan_of(sample_rate, values);
        return std::make_unique<Stream>(*this, plan, sample_rate, std::move(sink));
    }

    void warm_up() override {
        // The first graph builds Metal's pipelines and compiles Vulkan's shaders.
        const std::vector<float> silence((size_t) enhancer_.lowest_rate() / 4, 0.0f);
        enhancer_.chunk(silence.data(), silence.size(), enhancer_.lowest_rate());
    }

private:
    /** Audio at `sample_rate` resampled where the request sets bwe, and enhanced as semamba::Stream enhances it. */
    class Stream : public EnhancementStream {
    public:
        Stream(SemambaEngine & engine, const Plan & plan, int sample_rate, AudioSink sink)
            : engine_(engine), plan_(plan), stream_(engine.enhancer_, plan.rate, plan.chunk, plan.portion, std::move(sink)) {
            if (plan.rate != sample_rate) resampler_.emplace(sample_rate, plan.rate);
        }

        bool push(const float * samples, size_t n) override {
            if (!resampler_) return stream_.push(samples, n);
            resampled_.clear();
            resampler_->push(samples, n, resampled_);
            return stream_.push(resampled_.data(), resampled_.size());
        }

        bool end() override {
            if (resampler_) {
                resampled_.clear();
                resampler_->end(resampled_);
                if (!stream_.push(resampled_.data(), resampled_.size())) return false;
            }
            engine_.check_length((size_t) stream_.received(), plan_);
            return stream_.end();
        }

    private:
        SemambaEngine & engine_;
        Plan plan_;
        semamba::Stream stream_;
        std::optional<ResampleStream> resampler_;
        std::vector<float> resampled_;
    };

    /** The rate a request enhances audio given at `sample_rate` at, and its chunks, refused where the model cannot take them. */
    Plan plan_of(int sample_rate, const RequestValues & values) const {
        const std::string name = enhancer_.model().str("general.name");
        const int lowest = enhancer_.lowest_rate(), highest = enhancer_.highest_rate();
        const int rate = values.has(SPEECH_OPT_BWE) ? (int) values.integer(SPEECH_OPT_BWE) : sample_rate;
        if (rate < lowest || rate > highest) {
            throw Error(Fault::OutOfRange,
                        name + " enhances audio at " + std::to_string(lowest) + " to " + std::to_string(highest) + " Hz, and this audio is at " +
                            std::to_string(sample_rate) + " Hz; set bwe to a rate in that range to resample it",
                        "audio");
        }
        const size_t fewest = enhancer_.fewest_samples(rate);
        const double chunk = values.number(SPEECH_OPT_CHUNK_SIZE_IN_SECONDS), portion = values.number(SPEECH_OPT_HOP_LENGTH_PORTION);
        const int64_t size = (int64_t) (chunk * rate);
        // Audio shorter than a chunk is one chunk of its own length, which the length's check covers.
        if (size < (int64_t) fewest) {
            throw Error(Fault::OutOfRange,
                        "chunk_size_in_seconds " + std::to_string(chunk) + " gives chunks of " + std::to_string(size) + " samples at " + std::to_string(rate) +
                            " Hz, and " + name + "'s STFT takes at least " + std::to_string(fewest) + "; give a longer chunk",
                        "chunk_size_in_seconds");
        }
        if ((int64_t) (portion * (double) size) < 1) {
            throw Error(Fault::OutOfRange, "hop_length_portion " + std::to_string(portion) + " of a chunk of " + std::to_string(size) +
                                               " samples is less than one sample; give a larger portion or a longer chunk",
                        "hop_length_portion");
        }
        return {rate, chunk, portion, fewest};
    }

    void check_length(size_t samples, const Plan & plan) const {
        if (samples < plan.fewest) {
            throw Error(Fault::OutOfRange,
                        "the audio has " + std::to_string(samples) + " samples at " + std::to_string(plan.rate) + " Hz, and " +
                            enhancer_.model().str("general.name") + "'s STFT takes at least " + std::to_string(plan.fewest) + "; give longer audio",
                        "audio");
        }
    }

    semamba::Enhancer enhancer_;
};

}  // namespace

/**
 * The table of the options RE-USE takes: those of inference_chunk.py, under its names and with the defaults of its script,
 * which the file gives, and its flag --BWE, whose range is the rates the model card lists. A chunk takes at most an hour.
 * The model takes no language and no timestamps, whose neutral values every model accepts.
 */
FamilyInfo describe_semamba(const std::shared_ptr<const ModelFile> & file) {
    const ModelFile & m = *file;
    FamilyInfo info;
    info.options = {
        {SPEECH_OPT_CHUNK_SIZE_IN_SECONDS, false, true, (double) m.f32("semamba.chunk_size_in_seconds"), 0, 3600, true},
        {SPEECH_OPT_HOP_LENGTH_PORTION, false, true, (double) m.f32("semamba.hop_length_portion"), 0, 1, true},
        {SPEECH_OPT_BWE, false, true, std::nullopt, (double) m.size("semamba.sample_rate_min"), (double) m.size("semamba.sample_rate_max")},
    };
    return info;
}

std::unique_ptr<Engine> load_semamba(const std::string & path, ggml_backend_t backend, int) {
    return std::make_unique<SemambaEngine>(path, backend);
}
