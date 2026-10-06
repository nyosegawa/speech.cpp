#pragma once

#include <cstdint>
#include <vector>

#include "model-file.h"

namespace qwen3_asr {

/**
 * Where qwen-asr's split_audio_into_chunks() (qwen_asr/inference/utils.py) cuts audio too long for the model to take at
 * once, with the limits of the file: while more than qwen3-asr.audio.max_samples are left after the last cut, the next
 * cut goes into the quietest window of qwen3-asr.audio.split_window_samples (the smallest sum of magnitudes, the first of
 * equal ones) within qwen3-asr.audio.split_search_samples on either side of max_samples past the last cut, at its
 * quietest sample. Audio of at most max_samples is not cut. The parts are recognized one by one and their texts joined.
 *
 * The official sums the magnitudes of each window in float32 through numpy's convolution, which BLAS computes in an
 * order of its own; this sums them in double precision, so two windows whose sums are equal to float32's precision may
 * be told apart differently.
 */
class Splitter {
public:
    explicit Splitter(const ModelFile & m);

    /**
     * The bounds of the parts of `samples`, the audio as qwen-asr's normalization leaves it: the first part's start, 0,
     * each cut, and the end, so that part i is [bounds[i], bounds[i + 1]).
     */
    std::vector<int64_t> bounds(const std::vector<float> & samples) const;

private:
    int64_t max_samples_, search_samples_, window_samples_;
};

}  // namespace qwen3_asr
