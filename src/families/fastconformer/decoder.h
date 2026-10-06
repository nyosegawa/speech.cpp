#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "transducer.h"

namespace fastconformer {

/** The tokens a decoding emitted, in order, with the encoder frames it emitted them on. */
struct Decoding {
    std::vector<int32_t> ids;
    /** The encoder frame each token was emitted on. */
    std::vector<int64_t> frames;
    /** The duration in frames TDT predicted with each token; empty for a decoding that predicts none. */
    std::vector<int32_t> durations;
};

/**
 * Told the fraction of the encoder's frames a decoding has passed, as it goes; answering false stops the decoding,
 * which returns the tokens it has found.
 */
using DecodingProgress = std::function<bool(double done)>;

/** A decoding of the transducer's outputs into tokens, the one NeMo's transcribe() runs for the model by default. */
class Decoder {
public:
    virtual ~Decoder() = default;

    /**
     * The tokens of `projected`, the encoder's output through Joint::project_encoder(), [T, hidden] row-major, telling
     * `progress`, when it is given, how far the decoding has come.
     */
    virtual Decoding decode(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress) const = 0;

    virtual int blank() const = 0;
};

/**
 * The decoding fastconformer.decoder.kind names: greedy TDT ("tdt", TdtDecoder) or the alignment-length synchronous beam
 * search over RNN-T ("rnnt", AlsdDecoder).
 */
std::unique_ptr<Decoder> make_decoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint);

}  // namespace fastconformer
