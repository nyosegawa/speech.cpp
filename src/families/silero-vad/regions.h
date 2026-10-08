#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "model-file.h"

namespace silero_vad {

/**
 * What turns the chunks' speech probabilities into regions: the options of get_speech_timestamps() a request sets, and
 * the constants of its rule, which the model file gives.
 */
struct RegionRule {
    double threshold;
    int64_t min_speech_duration_ms, min_silence_duration_ms, speech_pad_ms;
    /** Infinity for no limit, get_speech_timestamps()'s default. */
    double max_speech_duration_s = INFINITY;
    int64_t min_silence_at_max_speech_ms;
    /** The threshold below which a chunk ends a region is the threshold less the offset, but at least the floor. */
    double neg_threshold_offset, neg_threshold_floor;
};

/** The rule with the defaults of the model file `m`. */
RegionRule default_rule(const ModelFile & m);

/** A region where someone speaks, in samples from the start of the audio: [start, end). */
struct Region {
    int64_t start, end;
};

/**
 * The regions that silero-vad 6.2.3's get_speech_timestamps_from_probs() gives for the speech probability of each chunk
 * of `chunk` samples of audio of `samples` samples at `sample_rate`, with use_max_poss_sil_at_max_speech at its
 * default, true: a region longer than max_speech_duration_s is cut at the longest of its silences longer than
 * min_silence_at_max_speech_ms, or where it reaches the limit when it has none. The sample counts it derives from
 * milliseconds and seconds are doubles and its positions integers, as Python computes them, so that every comparison
 * gives what the official gives.
 */
std::vector<Region> speech_regions(const std::vector<float> & probs, int64_t samples, int sample_rate, int chunk, const RegionRule & rule);

}  // namespace silero_vad
