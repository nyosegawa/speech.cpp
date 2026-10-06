#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"

namespace qwen3_asr {

/** One attention window of an utterance: its feature frames and its tokens. */
struct EncoderWindow {
    int64_t first_frame = 0, frames = 0;
    int64_t first_token = 0, tokens = 0;
};

/** Intermediate results of one window, for the checks. */
struct EncoderStages {
    /** The output of each convolution after its GELU for each of the window's chunks, [time, mels, channels, chunks]. */
    std::vector<ggml_tensor *> convolutions;
    /** conv_out's output with the positions added, the window's tokens, [d_model, tokens]. */
    ggml_tensor * input = nullptr;
    /** The output of each layer, [d_model, tokens] each. */
    std::vector<ggml_tensor *> layers;
    /** The output of ln_post, [d_model, tokens]. */
    ggml_tensor * output = nullptr;
};

/**
 * Qwen3-ASR's audio encoder and its projector, as transformers 5.18 runs them (Qwen3ASREncoder and
 * Qwen3ASRMultiModalProjector in transformers/models/qwen3_asr/modeling_qwen3_asr.py).
 *
 * The features are cut into chunks of qwen3-asr.encoder.chunk_frames (100 frames, 1 s), the last padded with zeros to a
 * whole chunk as the feature extractor pads it, an utterance shorter than a chunk included. Each chunk goes through
 * three 3x3 convolutions of stride 2, each followed by GELU, and conv_out to d_model, giving a token per 8 frames of
 * it, to which the sinusoid of the token's position within its chunk is added; the tokens of the padding are dropped,
 * so a last chunk of r frames gives ⌈⌈⌈r/2⌉/2⌉/2⌉ tokens. The layers, each a pre-LayerNorm attention and a pre-LayerNorm
 * GELU feed-forward with biases, attend within windows of qwen3-asr.encoder.window_frames (800 frames, 8 s, 104
 * tokens) from the first token, the last window holding the rest. LayerNorm (ln_post) and the projector, a linear
 * layer, GELU and a linear layer to the decoder's width, follow.
 *
 * Every chunk and every window is computed on its own, so a graph of a few consecutive windows, each attending within
 * itself, gives what the official computes over the whole utterance, and the memory of a graph does not grow with the
 * utterance.
 */
class Encoder {
public:
    /** The encoder of the model file `m`, which outlives it, computing on `backend`. */
    Encoder(const ModelFile & m, ggml_backend_t backend);
    ~Encoder();
    Encoder(const Encoder &) = delete;
    Encoder & operator=(const Encoder &) = delete;

    /** The tokens of an utterance of `frames` feature frames. */
    int64_t tokens(int64_t frames) const;

    /** The attention windows of an utterance of `frames` feature frames, in order. */
    std::vector<EncoderWindow> windows(int64_t frames) const;

    /**
     * Builds the projector's output for the consecutive windows `windows` of the utterance whose features are
     * `features`, [frames, mels] row-major: [output_dim, their tokens]. `stages` takes those of a single window.
     */
    ggml_tensor * build(Graph & g, const std::vector<float> & features, const std::vector<EncoderWindow> & windows,
                        EncoderStages * stages = nullptr) const;

    /**
     * The projector's output for every token of the utterance whose features are `features`, [frames, mels]
     * row-major, computed a few windows at a time: [tokens, output_dim] row-major. `keep_going`, when given, hears the
     * windows done after each graph, and false stops the encoder there, which then returns none.
     */
    std::optional<std::vector<float>> encode(const std::vector<float> & features,
                                             const std::function<bool(size_t windows)> & keep_going = nullptr);

    int d_model() const { return d_model_; }
    int output_dim() const { return output_dim_; }
    int64_t channels() const { return channels_; }
    int chunk_frames() const { return chunk_frames_; }

private:
    ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;
    ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;
    ggml_tensor * convolution(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;
    ggml_tensor * layer(ggml_context * ctx, ggml_tensor * x, ggml_tensor * mask, const std::string & name) const;

    const ModelFile & m_;
    ggml_backend_t backend_;
    int mels_, d_model_, heads_, layers_, chunk_frames_, window_chunks_, output_dim_;
    int64_t channels_, chunk_tokens_;
    float eps_;
    /** The sinusoids of a chunk's positions, [d_model, chunk tokens]: sines in the first half of the channels, cosines in the second. */
    std::vector<float> positions_;
    ggml_gallocr_t allocr_ = nullptr;
};

}  // namespace qwen3_asr
