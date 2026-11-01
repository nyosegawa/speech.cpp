#pragma once

#include <string>
#include <vector>

#include "graph.h"
#include "model-file.h"

namespace semamba {

/**
 * The SEMamba network of the official models/generator_SEMamba_time_d4.py, from a chunk's compressed magnitude and
 * phase to the magnitude decoder's output and the phase decoder's real and imaginary parts, before atan2 and the crop:
 * the input padded with zeros, the dense encoder, the TFMamba blocks and the two decoders. Activations of the codec
 * are [bins, frames, channels], the bins fastest, and those of the blocks [channels, frames, bins] over time and
 * [channels, bins, frames] over frequency, each Mamba's sequences on the last axis.
 */
class Network {
public:
    explicit Network(const ModelFile & m);

    /** The inputs one graph's sequences of `length` steps and `count` sequences take: their flip and their zero states. */
    struct Sequences {
        int64_t length, count;
        /** For each step of each sequence, the row of the same sequence `length` - 1 - step: [length * count]. */
        ggml_tensor * reverse;
        /** The convolution's zero history, [d_conv - 1, d_inner, count], and the scan's zero state and its slot. */
        ggml_tensor * history;
        ggml_tensor * state;
        ggml_tensor * slots;
    };

    /** The inputs of sequences of `length` steps and `count` sequences in `g`. */
    Sequences sequences(Graph & g, int64_t length, int64_t count) const;

    /** The encoder's output for the magnitude and phase [bins, frames, 2], [bins', frames', hidden]. */
    ggml_tensor * encode(ggml_context * ctx, ggml_tensor * input) const;

    /**
     * The bidirectional Mamba named `prefix` ("blk.0.time") on [hidden, length, count]: Mamba forward and over the flipped
     * sequences, each with its input added, joined, projected and normalized, as MambaBlock.forward() computes.
     */
    ggml_tensor * bidirectional(ggml_context * ctx, const std::string & prefix, ggml_tensor * x, const Sequences & s) const;

    /**
     * TFMamba block `b` on [hidden, frames', bins']: over time and then over frequency, each with its input added. `over_bins`
     * are the sequences over frequency, `over_frames` over time.
     */
    ggml_tensor * block(ggml_context * ctx, int b, ggml_tensor * x, const Sequences & over_frames, const Sequences & over_bins) const;

    /** A decoder ("mag" or "pha") on the blocks' output [bins', frames', hidden], before its last convolution. */
    ggml_tensor * decode(ggml_context * ctx, const std::string & decoder, ggml_tensor * x) const;

    /** A last convolution of one channel ("mag.out", "pha.r" or "pha.i") of a decoder's output. */
    ggml_tensor * project(ggml_context * ctx, const std::string & name, ggml_tensor * x) const;

    /** What forward() gives: [bins'', frames'', 1] each, the decoders' outputs before the crop. */
    struct Outputs {
        ggml_tensor * magnitude, * real, * imag;
        /** The encoder's output and each block's, [bins', frames', hidden], when asked. */
        ggml_tensor * encoded = nullptr;
        std::vector<ggml_tensor *> blocks;
    };

    /**
     * The network on the magnitude and phase of `frames` frames of `bins` bins, [bins, frames] each, with each block's
     * output in Outputs::blocks when `keep_blocks` asks.
     */
    Outputs forward(Graph & g, const std::vector<float> & magnitude, const std::vector<float> & phase, int64_t frames, int64_t bins,
                    bool keep_blocks = false) const;

    int hidden() const { return hidden_; }
    int blocks() const { return blocks_; }
    /** The frames and bins the encoder leaves of `frames` and `bins`, which the decoders multiply back. */
    int64_t encoded_frames(int64_t frames) const { return (frames + padding_ - 1) / frame_stride_ + 1; }
    int64_t encoded_bins(int64_t bins) const { return (bins + padding_ - 3) / bin_stride_ + 1; }

private:
    ggml_tensor * w(const std::string & name) const { return m_.tensor(name); }
    /** A convolution of the layout's `name` with its bias. */
    ggml_tensor * conv(ggml_context * ctx, const std::string & name, ggml_tensor * x, int s0, int s1, int p0, int p1, int d1) const;
    /** The instance normalization and the PReLU of `name` after its convolution. */
    ggml_tensor * norm_prelu(ggml_context * ctx, const std::string & name, ggml_tensor * x) const;
    ggml_tensor * dense(ggml_context * ctx, const std::string & name, ggml_tensor * x) const;
    /** An SPConvTranspose2d along the bins with its norm and PReLU: `r` times the bins of [bins, frames, hidden]. */
    ggml_tensor * up(ggml_context * ctx, const std::string & name, ggml_tensor * x, int r) const;
    /** One direction's Mamba-1 on [hidden, length, count]. */
    ggml_tensor * mamba(ggml_context * ctx, const std::string & prefix, ggml_tensor * x, const Sequences & s) const;
    /** [hidden, length, count] with each sequence's steps in reverse. */
    ggml_tensor * flip(ggml_context * ctx, ggml_tensor * x, const Sequences & s) const;

    const ModelFile & m_;
    int hidden_, inner_, state_, width_, rank_;
    int depth_, frame_stride_, bin_stride_, blocks_, padding_;
    float instance_eps_, layer_eps_;
};

}  // namespace semamba
