#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ctc.h"
#include "encoder.h"
#include "frontend.h"
#include "model-file.h"

namespace fastconformer {

/** What recognize() found: the CTC head's greedy tokens and their text. */
struct Transcript {
    std::vector<int32_t> ids;
    std::string text;
};

/** A FastConformer checkpoint with a CTC head, from audio to text on one backend. */
class Recognizer {
public:
    Recognizer(const std::string & path, ggml_backend_t backend);
    ~Recognizer();
    Recognizer(const Recognizer &) = delete;
    Recognizer & operator=(const Recognizer &) = delete;

    /** The text of mono samples at sample_rate(). */
    Transcript recognize(const std::vector<float> & samples);

    /** The CTC head's logits for `features` ([frames, mels] row-major), [T, classes] row-major. */
    std::vector<float> logits(const std::vector<float> & features, int64_t frames);

    int sample_rate() const { return frontend_.sample_rate(); }
    const ModelFile & model() const { return model_; }
    const Frontend & frontend() const { return frontend_; }
    const Encoder & encoder() const { return encoder_; }
    const CtcHead & ctc() const { return ctc_; }
    const Detokenizer & detokenizer() const { return detokenizer_; }

private:
    ggml_backend_t backend_;
    ModelFile model_;
    Frontend frontend_;
    Encoder encoder_;
    CtcHead ctc_;
    Detokenizer detokenizer_;
    ggml_gallocr_t allocr_;
};

}  // namespace fastconformer
