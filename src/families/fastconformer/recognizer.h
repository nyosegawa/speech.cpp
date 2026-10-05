#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "decoder.h"
#include "detokenizer.h"
#include "encoder.h"
#include "frontend.h"
#include "model-file.h"
#include "transducer.h"

namespace fastconformer {

/** What recognize() found: the decoding's tokens and their text. */
struct Transcript {
    std::vector<int32_t> ids;
    std::string text;
};

/** A FastConformer checkpoint with a TDT or an RNN-T decoder, from audio to text on one backend. */
class Recognizer {
public:
    Recognizer(const std::string & path, ggml_backend_t backend);
    ~Recognizer();
    Recognizer(const Recognizer &) = delete;
    Recognizer & operator=(const Recognizer &) = delete;

    /** The text of mono samples at sample_rate(). */
    Transcript recognize(const std::vector<float> & samples);

    /** The encoder's output for `features` ([frames, mels] row-major) projected for the joint, [T, hidden] row-major. */
    std::vector<float> encode(const std::vector<float> & features, int64_t frames);

    /** The token ids of encode()'s output. */
    std::vector<int32_t> decode(const std::vector<float> & projected) const { return decoder_->decode(projected, backend_); }

    int sample_rate() const { return frontend_.sample_rate(); }
    const ModelFile & model() const { return model_; }
    const Frontend & frontend() const { return frontend_; }
    const Encoder & encoder() const { return encoder_; }
    const PredictionNetwork & prediction() const { return prediction_; }
    const Joint & joint() const { return joint_; }
    const Decoder & decoder() const { return *decoder_; }
    const Detokenizer & detokenizer() const { return detokenizer_; }

private:
    ggml_backend_t backend_;
    ModelFile model_;
    Frontend frontend_;
    Encoder encoder_;
    PredictionNetwork prediction_;
    Joint joint_;
    std::unique_ptr<Decoder> decoder_;
    Detokenizer detokenizer_;
    ggml_gallocr_t allocr_;
};

}  // namespace fastconformer
