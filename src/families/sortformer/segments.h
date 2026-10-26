#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace sortformer {

/**
 * The postprocessing diarize() runs on each speaker's probabilities (PostProcessingParams of
 * nemo/collections/asr/parts/utils/vad_utils.py): a segment starts above onset and ends below offset, keeping its state
 * between them; it is padded by pad_onset before and pad_offset after, and padded segments that meet are joined; then a
 * segment shorter than min_duration_on is dropped and a pause shorter than min_duration_off between two that are left is
 * filled. Times are in seconds.
 */
struct Postprocessing {
    double onset = 0.5, offset = 0.5, pad_onset = 0, pad_offset = 0, min_duration_on = 0, min_duration_off = 0;
};

/** A segment of one speaker, in seconds from the start of the audio, rounded to 10 ms as diarize() rounds it. */
struct Segment {
    double start = 0, end = 0;
    int speaker = 0;
};

/**
 * The segments of probabilities, [frames, speakers] row-major, one row every `frame_seconds`, as predlist_to_timestamps()
 * gives them with `post`: each speaker's in order, the speakers in order, and the times computed in float32 as torch
 * computes them.
 */
std::vector<Segment> segments(const std::vector<float> & probs, int speakers, double frame_seconds, const Postprocessing & post);

/**
 * The segments of probabilities that arrive a few frames at a time, given once nothing later can change them: once the
 * probabilities have ended, they are those segments() gives for all of them. Before that, everything earlier than
 * settled() is final: each segment that ends before it has been given, and the segment of a speaker that began before
 * it and goes on past it keeps its start. That is pad_onset, min_duration_on and min_duration_off and a frame before the
 * last frame heard, the reach of what a later frame can join, fill or keep.
 */
class SegmentStream {
public:
    SegmentStream(int speakers, double frame_seconds, const Postprocessing & post);

    /** Takes `frames` more rows of probabilities, [frames, speakers] row-major. */
    void push(const float * probs, int64_t frames);
    /** Ends the probabilities and gives the segments left. */
    void end();

    /** The segments given so far, in the order they were given. */
    const std::vector<Segment> & given() const { return given_; }
    /** The time before which every segment is final, in seconds; +inf once the probabilities have ended. */
    double settled() const;
    /** The start of the segment of `speaker` that began before settled() and goes on past it, if there is one. */
    std::optional<double> speaking(int speaker) const;

private:
    /** A speaker's state: whether it is above, and since which frame; its segments not yet settled, as frames. */
    struct Track {
        bool above = false;
        int64_t since = 0;
        std::vector<std::pair<int64_t, int64_t>> raw;
        std::optional<double> speaking;
    };

    void settle(bool ended);

    int speakers_;
    double frame_seconds_;
    Postprocessing post_;
    int64_t frames_ = 0;
    bool ended_ = false;
    std::vector<Track> tracks_;
    std::vector<Segment> given_;
};

}  // namespace sortformer
