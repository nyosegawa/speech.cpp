#pragma once

#include "model-file.h"

namespace silero_vad {

/**
 * Layout 1 of a Silero VAD model file, which reference/silero-vad/convert.py writes: the 16 kHz model's STFT basis, the
 * convolutions of its encoder, its LSTM cell and its decoder, the sizes of its chunks, context and STFT and its
 * encoder's strides and padding, the defaults of the options get_speech_timestamps() takes and the constants of its rule
 * for regions, every key of silero-vad.* and speech.* that the family reads, and the tensors the keys call for. The file
 * names no language, as a detection model takes none.
 */
extern const Layout layout;

}  // namespace silero_vad
