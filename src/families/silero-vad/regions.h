#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>
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

/** A region begun and not given yet: its padded start, and whether it is certain to be given. */
struct Begun {
    int64_t start;
    bool kept;
};

/**
 * The regions that silero-vad 6.2.3's get_speech_timestamps_from_probs() gives for the speech probability of each chunk
 * of `chunk` samples of audio at `sample_rate`, with use_max_poss_sil_at_max_speech at its default, true: a region longer
 * than max_speech_duration_s is cut at the longest of its silences longer than min_silence_at_max_speech_ms, or where it
 * reaches the limit when it has none. The sample counts it derives from milliseconds and seconds are doubles and its
 * positions integers, as Python computes them, so that every comparison gives what the official gives.
 *
 * The official walks the chunks once, left to right, with a small state, and pads the regions it found at the end: a
 * region's padded end depends on how far the next region starts, and its padded start on how far the one before ended.
 * Here the walk takes one chunk at a time, add(), and each region is padded as soon as the region after it is found, or,
 * through settle(), as soon as the chunks so far make its padded end certain; end() pads the rest. A region once given
 * never changes, and the regions given after end() are the official's however the chunks came.
 */
class RegionStream {
public:
    RegionStream(const RegionRule & rule, int sample_rate, int chunk);

    /** Takes the speech probability of the next chunk. */
    void add(double prob);

    /**
     * Gives each region whose padded bounds the chunks so far make certain, knowing that the audio holds at least `heard`
     * samples. With min_silence_duration_ms of at least twice speech_pad_ms, a region that silence ends is given at the
     * chunk that ends it. Otherwise its padded end waits on whether a region starts within twice speech_pad_ms of it,
     * and it is given once that many samples pass without one or once the region that started is certain to be kept,
     * longer than min_speech_duration_ms: at most 2 · speech_pad_ms + min_speech_duration_ms + min_silence_duration_ms
     * and two chunks after its end, or later where max_speech_duration_s cuts it at a silence before the limit.
     */
    void settle(int64_t heard);

    /** Ends the audio, of `samples` samples, after its last chunk, and gives the regions left. */
    void end(int64_t samples);

    /** The regions given so far, in order. */
    const std::vector<Region> & regions() const { return given_; }

    /**
     * The earliest region begun but not given yet: one whose end waits on what follows, which is kept, or else the region
     * under way, kept once no end it can still take leaves it no longer than min_speech_duration_ms; none when there is
     * neither. A region under way not kept is dropped if its speech ends that soon, and no region is then given from its
     * start. With what settle() or end() was last told of the audio heard.
     */
    std::optional<Begun> open() const;

private:
    /**
     * The earliest end the region under way can still take: where its probability fell below neg_threshold, unless it
     * rose above the threshold since, or else the next chunk, whose probability may fall, or the end of the audio heard.
     */
    int64_t earliest_end() const;
    /** Appends a region of the official's list, before padding, and gives the one before it. */
    void append(Region raw);
    /** The padded start of a region whose unpadded start is `start`, after the regions appended so far. */
    int64_t padded_start(int64_t start) const;
    /**
     * The padded end of the region pending, followed by a region starting at `next`, unpadded, or by none, in audio of
     * `length` samples, or of at least `length` samples where the region's end and padding lie within them.
     */
    int64_t padded_end(std::optional<int64_t> next, double length) const;
    void give_pending(std::optional<int64_t> next, double length);

    double threshold_, neg_threshold_, min_speech_, pad_, max_speech_, min_silence_, min_silence_at_max_speech_;
    int chunk_;
    /** The chunks taken, and the samples settle() or end() was last told the audio holds. */
    int64_t chunks_ = 0, heard_ = 0;
    bool triggered_ = false;
    // The start of the region under way while triggered, and where its probability first fell below neg_threshold, 0
    // while it has not, as the official's temp_end.
    int64_t start_ = 0, temp_end_ = 0;
    /** Each silence within the region under way that lasted longer than min_silence_at_max_speech: where it began and its length. */
    std::vector<std::pair<int64_t, int64_t>> possible_ends_;
    /** The unpadded end of the last region appended. */
    std::optional<int64_t> last_end_;
    /** The last region appended while it is not given: its padded start and its unpadded end. */
    std::optional<Region> pending_;
    std::vector<Region> given_;
};

/** The regions of the official rule for the probabilities of the whole audio, of `samples` samples. */
std::vector<Region> speech_regions(const std::vector<float> & probs, int64_t samples, int sample_rate, int chunk, const RegionRule & rule);

}  // namespace silero_vad
