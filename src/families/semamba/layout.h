#pragma once

#include "model-file.h"

namespace semamba {

/**
 * Layout 1 of a SEMamba model file, which reference/semamba/convert.py writes: the STFT's sizes at its reference rate, the
 * rates the model takes, the settings and constants of the official chunked inference, the dense encoder, the TFMamba
 * blocks with a Mamba-1 to each direction over time and over frequency, the magnitude and phase decoders, every key of
 * semamba.* and speech.* that the family reads, and the tensors the keys call for. The file names no language, as the
 * model enhances speech in any.
 */
extern const Layout layout;

}  // namespace semamba
