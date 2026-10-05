#pragma once

#include <string>
#include <vector>

#include "codec.h"

namespace irodori {

/** The loudness the official runtime gives a reference before encoding it, in LUFS. */
constexpr double kReferenceLufs = -16.0;

/**
 * The latent of a reference voice in a WAVE file, row-major [frames, latent_dim], as the official runtime
 * makes it: the channels averaged, resampled to the codec's rate, the loudness normalized, then encoded. The
 * runtime resamples with torchaudio's defaults and this with resample.h's filter, so a file at another rate
 * gives a latent slightly different from the runtime's. A file longer than `max_seconds`, which the runtime
 * would cut, throws.
 */
std::vector<float> encode_reference(Codec & codec, const std::string & wav_path, double max_seconds);

}  // namespace irodori
