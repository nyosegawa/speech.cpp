#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "expert-placement.h"
#include "nemotron-transcribe/layout.h"
#include "nemotron-transcribe/recognizer.h"

namespace {

class NemotronTranscribeEngine : public Engine {
public:
    NemotronTranscribeEngine(const std::string & path, ggml_backend_t backend, int threads)
        : placement_(place_experts(path, backend, threads)), recognizer_(path, backend, &placement_) {}

    bool computes_on_cpu() const override { return placement_.cpu() != nullptr; }

    Recognized transcribe(const std::vector<float> & samples, const RequestValues & values, Run & run) override {
        placement_.allow(values.boolean(SPEECH_OPT_EXPERTS_ON_CPU));
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
        // The first recognition builds Metal's pipelines and compiles Vulkan's shaders, which take seconds. It stops after
        // the first token it decodes, which runs every graph: the model may write a turn of its own about silence.
        recognizer_.recognize(std::vector<float>((size_t) recognizer_.sample_rate(), 0.0f), {}, [](double done) { return done <= 2.0 / 3; });
    }

private:
    /** Every layer's experts on the CPU, and on a GPU those of as many layers as its free memory holds beside the rest. */
    static ExpertPlacement place_experts(const std::string & path, ggml_backend_t backend, int threads) {
        const ModelFile file(path, nemotron_transcribe::layout);
        return ExpertPlacement(file, (int) file.str_array("nemotron-transcribe.llm.layer_types").size(), nemotron_transcribe::Recognizer::working_bytes(file),
                               backend, threads, "Nemotron 3.5 Transcribe", nemotron_transcribe::Recognizer::reads);
    }

    ExpertPlacement placement_;
    nemotron_transcribe::Recognizer recognizer_;
};

}  // namespace

/**
 * The table of the options Nemotron 3.5 Transcribe takes. The language is checked against the model's and steers
 * nothing: the model card asks for the language as spoken. The prompt is the context terms of the card's contextual
 * transcription. The port computes no times, so a request takes timestamps only as false, the neutral value. A GPU that
 * cannot hold every expert leaves the experts of its later layers to the CPU, which experts_on_cpu false refuses.
 */
FamilyInfo describe_nemotron_transcribe(const std::shared_ptr<const ModelFile> &) {
    FamilyInfo info;
    info.options = {
        {SPEECH_OPT_LANGUAGE, false, false, std::string("auto")},
        {SPEECH_OPT_PROMPT, false, true, std::string()},
        {SPEECH_OPT_EXPERTS_ON_CPU, false, true, true},
    };
    return info;
}

std::unique_ptr<Engine> load_nemotron_transcribe(const std::string & path, ggml_backend_t backend, int threads) {
    return std::make_unique<NemotronTranscribeEngine>(path, backend, threads);
}
