#pragma once

#include <vector>

#include "graph.h"
#include "model-file.h"

namespace silero_vad {

/**
 * The 16 kHz network of Silero VAD as the JIT model's _model computes one chunk (vad_annotator.py in the package's
 * silero_vad.jit): the chunk with the samples before it as context, reflected at its end; the magnitude of an STFT
 * computed as a convolution with a basis of real and imaginary parts; an encoder of convolutions, each with a ReLU;
 * an LSTM cell, whose state carries from one chunk to the next; and a decoder of a ReLU, a convolution of width 1 and a
 * sigmoid, whose mean over the encoder's one frame is the chunk's speech probability.
 *
 * Every stage but the LSTM cell depends on its chunk alone, so the graph computes those of many chunks at once, the
 * chunks on the last axis, and steps the cell through them in order.
 */
class Network {
public:
    explicit Network(const ModelFile & m);

    /** The magnitude of the STFT of each chunk's input [context + chunk, n], [bins, frames, n]. */
    ggml_tensor * stft(ggml_context * ctx, ggml_tensor * input) const;

    /** Encoder block `i`, its convolution and ReLU, on [channels, frames, n]. */
    ggml_tensor * block(ggml_context * ctx, int i, ggml_tensor * x) const;

    /** The encoder's output for the STFT's magnitude, [channels, n], with each block's output in `blocks` when asked. */
    ggml_tensor * encode(ggml_context * ctx, ggml_tensor * magnitude, std::vector<ggml_tensor *> * blocks = nullptr) const;

    /** The LSTM cell stepped through n chunks in order. */
    struct Steps {
        /** h after each chunk, the cell's output, [hidden, n], and c after each when asked, or nullptr. */
        ggml_tensor * h;
        ggml_tensor * c;
        /** The state after the last chunk, [hidden] each. */
        ggml_tensor * last_h;
        ggml_tensor * last_c;
    };

    /** The cell over the encoder's output [channels, n], from the state `h0` and `c0` ([hidden] each). */
    Steps lstm(ggml_context * ctx, ggml_tensor * encoded, ggml_tensor * h0, ggml_tensor * c0, bool keep_c = false) const;

    /** The speech probability of each chunk of the cell's outputs [hidden, n], [n]. */
    ggml_tensor * decode(ggml_context * ctx, ggml_tensor * h) const;

    int chunk() const { return chunk_; }
    int context() const { return context_; }
    int bins() const { return n_fft_ / 2 + 1; }
    int frames() const { return frames_; }
    int hidden() const { return hidden_; }
    int blocks() const { return (int) strides_.size(); }

private:
    const ModelFile & m_;
    int chunk_, context_, n_fft_, hop_, reflect_, frames_, hidden_;
    std::vector<int32_t> strides_, padding_;
};

}  // namespace silero_vad
