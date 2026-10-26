#include "segments.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>

namespace sortformer {

namespace {

using Span = std::pair<float, float>;

/**
 * merge_overlap_segment(): spans sorted by their starts, each run whose spans meet or overlap joined from the first's
 * start to the last's end.
 */
std::vector<Span> merge_overlaps(std::vector<Span> spans) {
    if (spans.size() < 2) return spans;
    std::sort(spans.begin(), spans.end(), [](const Span & a, const Span & b) { return a.first < b.first; });
    std::vector<Span> out = {spans[0]};
    for (size_t i = 1; i < spans.size(); i++) {
        if (spans[i - 1].second >= spans[i].first) out.back().second = spans[i].second;
        else out.push_back(spans[i]);
    }
    return out;
}

/**
 * binarization_vectorized()'s padding and joining and filtering() with filter_speech_first, of one speaker's segments in
 * frames [start, end), in float32 as torch computes them.
 */
std::vector<Span> process(const std::vector<std::pair<int64_t, int64_t>> & raw, double frame_seconds, const Postprocessing & post) {
    const float frame = (float) frame_seconds, pad_onset = (float) post.pad_onset, pad_offset = (float) post.pad_offset;
    std::vector<Span> spans;
    for (const auto & [first, last] : raw) {
        // The family is compiled without fused multiply-adds, which would round an end once where torch rounds it
        // twice and decide otherwise whether two padded segments meet.
        const float start = std::max((float) first * frame - pad_onset, 0.0f), end = (float) last * frame + pad_offset;
        if (end > start) spans.push_back({start, end});
    }
    if (post.pad_onset > 0 || post.pad_offset > 0) spans = merge_overlaps(std::move(spans));
    if (spans.empty()) return spans;
    if (post.min_duration_on > 0) {
        const float shortest = (float) post.min_duration_on;
        spans.erase(std::remove_if(spans.begin(), spans.end(), [&](const Span & s) { return !(s.second - s.first >= shortest); }), spans.end());
    }
    if (post.min_duration_off > 0 && spans.size() > 1) {
        const float shortest = (float) post.min_duration_off;
        std::vector<Span> filled = spans;
        for (size_t i = 1; i < spans.size(); i++) {
            const Span gap = {spans[i - 1].second, spans[i].first};
            if (!(gap.second - gap.first >= shortest)) filled.push_back(gap);
        }
        spans = merge_overlaps(std::move(filled));
    }
    return spans;
}

/** A time rounded to 10 ms as Python's round(x, 2) rounds the double of it: correctly, the halves to even. */
double centiseconds(float t) {
    char text[64];
    std::snprintf(text, sizeof text, "%.2f", (double) t);
    return std::strtod(text, nullptr);
}

/** One speaker's state from one frame's probability: binarization_vectorized()'s hysteresis, or its toggling. */
bool next_state(bool above, float p, const Postprocessing & post) {
    const float onset = (float) post.onset, offset = (float) post.offset;
    if (onset >= offset) return p > onset ? true : p < offset ? false : above;
    return p >= offset ? true : p <= onset ? false : !above;
}

}  // namespace

std::vector<Segment> segments(const std::vector<float> & probs, int speakers, double frame_seconds, const Postprocessing & post) {
    const int64_t frames = (int64_t) probs.size() / speakers;
    std::vector<Segment> out;
    for (int s = 0; s < speakers; s++) {
        std::vector<std::pair<int64_t, int64_t>> raw;
        bool above = false;
        int64_t since = 0;
        for (int64_t t = 0; t < frames; t++) {
            const bool next = next_state(above, probs[(size_t) (t * speakers + s)], post);
            if (next && !above) since = t;
            if (!next && above) raw.push_back({since, t});
            above = next;
        }
        if (above) raw.push_back({since, frames});
        for (const Span & span : process(raw, frame_seconds, post)) out.push_back({centiseconds(span.first), centiseconds(span.second), s});
    }
    return out;
}

SegmentStream::SegmentStream(int speakers, double frame_seconds, const Postprocessing & post)
    : speakers_(speakers), frame_seconds_(frame_seconds), post_(post), tracks_((size_t) speakers) {}

void SegmentStream::push(const float * probs, int64_t frames) {
    for (int64_t i = 0; i < frames; i++, frames_++) {
        for (int s = 0; s < speakers_; s++) {
            Track & track = tracks_[(size_t) s];
            const bool next = next_state(track.above, probs[(size_t) (i * speakers_ + s)], post_);
            if (next && !track.above) track.since = frames_;
            if (!next && track.above) track.raw.push_back({track.since, frames_});
            track.above = next;
        }
    }
    settle(false);
}

void SegmentStream::end() {
    for (Track & track : tracks_) {
        if (track.above) track.raw.push_back({track.since, frames_});
        track.above = false;
    }
    settle(true);
    ended_ = true;
}

double SegmentStream::settled() const {
    if (ended_) return std::numeric_limits<double>::infinity();
    // What a later frame can still change reaches back pad_onset (its padded start), min_duration_on (a segment it keeps
    // from being dropped) and min_duration_off (a pause it fills), and one frame more for float32's rounding.
    const int64_t reach = (int64_t) std::ceil((post_.pad_onset + post_.min_duration_on + post_.min_duration_off) / frame_seconds_ - 1e-9) + 1;
    return (double) std::max<int64_t>(frames_ - reach, 0) * frame_seconds_;
}

std::optional<double> SegmentStream::speaking(int speaker) const {
    return ended_ ? std::nullopt : tracks_.at((size_t) speaker).speaking;
}

void SegmentStream::settle(bool ended) {
    const double horizon = ended ? std::numeric_limits<double>::infinity() : settled();
    for (int s = 0; s < speakers_; s++) {
        Track & track = tracks_[(size_t) s];
        std::vector<std::pair<int64_t, int64_t>> raw = track.raw;
        if (track.above) raw.push_back({track.since, frames_});
        const std::vector<Span> spans = process(raw, frame_seconds_, post_);
        size_t final = 0;
        while (final < spans.size() && (double) spans[final].second < horizon) {
            given_.push_back({centiseconds(spans[final].first), centiseconds(spans[final].second), s});
            final++;
        }
        track.speaking.reset();
        double open = std::numeric_limits<double>::infinity();
        if (final < spans.size()) {
            open = spans[final].first;
            if (open < horizon) track.speaking = centiseconds(spans[final].first);
        }
        // A segment that ends before the horizon and before the first segment not given is part of one given or dropped
        // for good.
        const float frame = (float) frame_seconds_, pad_offset = (float) post_.pad_offset;
        track.raw.erase(std::remove_if(track.raw.begin(), track.raw.end(),
                                       [&](const std::pair<int64_t, int64_t> & r) {
                                           const double end = (double) ((float) r.second * frame + pad_offset);
                                           return end < horizon && end < open;
                                       }),
                        track.raw.end());
    }
}

}  // namespace sortformer
