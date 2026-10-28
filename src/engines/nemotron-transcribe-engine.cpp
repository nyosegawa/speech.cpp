#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "nemotron-transcribe/recognizer.h"

namespace {

class NemotronTranscribeEngine : public Engine {
public:
    NemotronTranscribeEngine(const std::string & path, ggml_backend_t backend) : recognizer_(path, backend) {}

    Recognized transcribe(const std::vector<float> & samples, const RequestValues & values, Run & run) override {
        nemotron_transcribe::RecognitionRequest request;
        request.context = values.string(SPEECH_OPT_PROMPT);
        const std::optional<nemotron_transcribe::Recognition> found =
            recognizer_.recognize(samples, request, [&](double done) { return run.progress(done); });
        if (!found || run.stopped()) return {};
        run.progress(1);
        Recognized out;
        out.text = found->text;
        out.stop = found->limited ? SPEECH_STOP_MODEL_LIMIT : SPEECH_STOP_COMPLETE;
        return out;
    }

    void warm_up() override {
        // The first recognition builds Metal's pipelines and compiles Vulkan's shaders, which take seconds.
        recognizer_.recognize(std::vector<float>((size_t) recognizer_.sample_rate(), 0.0f), {}, {});
    }

private:
    nemotron_transcribe::Recognizer recognizer_;
};

}  // namespace

/**
 * The table of the options Nemotron 3.5 Transcribe takes. The language is checked against the model's and steers
 * nothing: the model card asks for the language as spoken. The prompt is the context terms of the card's contextual
 * transcription. The port computes no times, so a request takes timestamps only as false, the neutral value.
 */
FamilyInfo describe_nemotron_transcribe(const std::shared_ptr<const ModelFile> &) {
    FamilyInfo info;
    info.options = {
        {SPEECH_OPT_LANGUAGE, false, false, std::string("auto")},
        {SPEECH_OPT_PROMPT, false, true, std::string()},
    };
    return info;
}

std::unique_ptr<Engine> load_nemotron_transcribe(const std::string & path, ggml_backend_t backend, int) {
    return std::make_unique<NemotronTranscribeEngine>(path, backend);
}
