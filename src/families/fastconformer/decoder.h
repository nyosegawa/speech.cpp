#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
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
    /** The graphs the decoding computed. Each is a round trip to the backend, so that on a GPU its time grows with them. */
    size_t graphs = 0;
};

/**
 * Told the fraction of the encoder's frames a decoding has passed, as it goes; answering false stops the decoding,
 * which returns the tokens it has found.
 */
using DecodingProgress = std::function<bool(double done)>;

/** A decoding of the transducer's outputs into tokens. */
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
 * The decodings of a model with the file `m`, by the names of the request option decoding, the one NeMo's transcribe()
 * runs by default first: "greedy" for fastconformer.decoder.kind "tdt" (TdtDecoder), and "beam", the beam search its
 * checkpoint configures (AlsdDecoder), and "greedy" (RnntGreedyDecoder) for "rnnt".
 */
std::vector<std::string> decodings(const ModelFile & m);

/** The decoding of decodings(m) named `name`. */
std::unique_ptr<Decoder> make_decoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint, const std::string & name);

/** The index of the largest of `n` values, the first of equal ones, as torch.max() and argmax() choose it. */
int first_argmax(const float * v, int n);

}  // namespace fastconformer
