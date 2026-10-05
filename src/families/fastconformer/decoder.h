#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "transducer.h"

namespace fastconformer {

/** A decoding of the transducer's outputs into tokens, the one NeMo's transcribe() runs for the model by default. */
class Decoder {
public:
    virtual ~Decoder() = default;

    /** The token ids for `projected`, the encoder's output through Joint::project_encoder(), [T, hidden] row-major. */
    virtual std::vector<int32_t> decode(const std::vector<float> & projected, ggml_backend_t backend) const = 0;

    virtual int blank() const = 0;
};

/**
 * The decoding fastconformer.decoder.kind names: greedy TDT ("tdt", TdtDecoder) or the alignment-length synchronous beam
 * search over RNN-T ("rnnt", AlsdDecoder).
 */
std::unique_ptr<Decoder> make_decoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint);

}  // namespace fastconformer
