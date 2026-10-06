#pragma once

#include <cstdint>

#include "model-file.h"

namespace qwen3_asr {

/**
 * Layout 1 of a Qwen3-ASR model file, which reference/qwen3-asr/convert.py writes: the frontend's window and mel
 * filterbank, the encoder's convolutions and layers, the projector, and the decoder with its tokenizer, every key of
 * qwen3-asr.* and speech.* that the family reads, and the tensors the keys call for.
 */
extern const Layout layout;

/**
 * The width and height of every convolution of the encoder, which pads by 1 on each side and has a stride of 2: the
 * official module fixes them in its code (nn.Conv2d(..., 3, 2, padding=1)), and the converter refuses others.
 */
constexpr int64_t kConvWidth = 3;

/** The convolutions of the encoder, each halving the time and the mel axes, rounding up. */
constexpr int kConvLayers = 3;

/** The length of an axis of `n` after the encoder's convolutions; an empty axis stays empty. */
constexpr int64_t after_convolutions(int64_t n) {
    for (int i = 0; i < kConvLayers && n > 0; i++) n = (n - 1) / 2 + 1;
    return n;
}

}  // namespace qwen3_asr
