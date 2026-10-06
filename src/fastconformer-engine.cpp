#include <string>
#include <vector>

#include "engine.h"
#include "fastconformer/recognizer.h"

namespace {

class FastConformerEngine : public Engine {
public:
    FastConformerEngine(const std::string & path, ggml_backend_t backend) : recognizer_(path, backend) {}

    /**
     * The three stages report their progress in thirds, the decoding by the frames it has passed: the frontend takes
     * milliseconds, the encoder runs as one graph, and the decoding steps frame by frame.
     */
    Recognized transcribe(const std::vector<float> & samples, const RequestValues & values, Run & run) override {
        const fastconformer::Frontend & frontend = recognizer_.frontend();
        const std::vector<float> features = frontend.features(samples);
        if (!run.progress(1.0 / 3)) return {};
        const std::vector<float> projected = recognizer_.encode(features, frontend.frames(samples.size()));
        if (!run.progress(2.0 / 3)) return {};
        const fastconformer::Decoding decoding = recognizer_.decoding(projected, [&](double done) { return run.progress((2 + done) / 3); });
        if (run.stopped()) return {};
        const fastconformer::Detokenizer & detokenizer = recognizer_.detokenizer();
        Recognized out;
        out.text = detokenizer.text(decoding.ids);
        if (values.boolean(SPEECH_OPT_TIMESTAMPS)) {
            const std::vector<std::string> texts = detokenizer.token_texts(decoding.ids);
            const std::vector<fastconformer::Span> spans = fastconformer::token_spans(decoding, detokenizer);
            for (size_t i = 0; i < texts.size(); i++) {
                out.tokens.push_back({recognizer_.seconds(spans[i].start), recognizer_.seconds(spans[i].end), texts[i]});
            }
            for (const fastconformer::Segment & s : recognizer_.segments(decoding)) {
                out.segments.push_back({recognizer_.seconds(s.span.start), recognizer_.seconds(s.span.end), s.text});
            }
        }
        run.progress(1);
        return out;
    }

    void warm_up() override {
        // The first recognition builds Metal's pipelines and compiles Vulkan's shaders, which take seconds.
        recognizer_.recognize(std::vector<float>((size_t) recognizer_.sample_rate(), 0.0f));
    }

private:
    fastconformer::Recognizer recognizer_;
};

}  // namespace

/**
 * The table of the options FastConformer takes. The model has no input for a language, so a request's language is
 * checked against the model's and not used; parakeet-tdt-0.6b-v3 finds the language of the audio itself.
 */
FamilyInfo describe_fastconformer(const std::shared_ptr<const ModelFile> &) {
    FamilyInfo info;
    info.options = {
        {SPEECH_OPT_LANGUAGE, false, false, std::string("auto")},
        {SPEECH_OPT_TIMESTAMPS, false, true, false},
    };
    return info;
}

std::unique_ptr<Engine> load_fastconformer(const std::string & path, ggml_backend_t backend) {
    return std::make_unique<FastConformerEngine>(path, backend);
}
