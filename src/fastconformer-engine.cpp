#include <optional>
#include <string>
#include <vector>

#include "engine.h"
#include "fastconformer/recognizer.h"

namespace {

class FastConformerEngine : public Engine {
public:
    FastConformerEngine(const EngineOptions & options, ggml_backend_t backend) : recognizer_(options.model, backend) {
        info_.name = recognizer_.model().str("general.name");
        info_.sample_rate = recognizer_.sample_rate();
        info_.languages = recognizer_.model().str_array("speech.languages");
        // The CTC head has no input for a language, so a request's language is only checked against the model's.
        info_.language_selectable = false;
        // The first recognition builds Metal's pipelines and compiles Vulkan's shaders, which take seconds.
        recognizer_.recognize(std::vector<float>((size_t) recognizer_.sample_rate(), 0.0f));
    }

    std::optional<std::string> transcribe(const std::vector<float> & samples, const StopCheck & stopped) override {
        if (stopped()) return std::nullopt;
        const std::vector<float> features = recognizer_.frontend().features(samples);
        if (stopped()) return std::nullopt;
        const std::vector<float> logits = recognizer_.logits(features, recognizer_.frontend().frames(samples.size()));
        if (stopped()) return std::nullopt;
        return recognizer_.detokenizer().text(recognizer_.ctc().greedy(logits));
    }

private:
    fastconformer::Recognizer recognizer_;
};

}  // namespace

std::unique_ptr<Engine> make_fastconformer(const EngineOptions & options, ggml_backend_t backend) {
    return std::make_unique<FastConformerEngine>(options, backend);
}
