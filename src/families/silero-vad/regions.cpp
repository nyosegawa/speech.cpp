#include "regions.h"

#include <algorithm>
#include <utility>

namespace silero_vad {

namespace {

/** Python's x // 2 of an integer, which rounds toward negative infinity where C++'s division rounds toward zero. */
int64_t floor_half(int64_t x) {
    return x >= 0 ? x / 2 : -((-x + 1) / 2);
}

}  // namespace

RegionRule default_rule(const ModelFile & m) {
    RegionRule r;
    r.threshold = m.f64("silero-vad.threshold");
    r.min_speech_duration_ms = m.u32("silero-vad.min_speech_duration_ms");
    r.min_silence_duration_ms = m.u32("silero-vad.min_silence_duration_ms");
    r.speech_pad_ms = m.u32("silero-vad.speech_pad_ms");
    r.min_silence_at_max_speech_ms = m.u32("silero-vad.min_silence_at_max_speech_ms");
    r.neg_threshold_offset = m.f64("silero-vad.neg_threshold_offset");
    r.neg_threshold_floor = m.f64("silero-vad.neg_threshold_floor");
    return r;
}

// Python's int * int / 1000 divides the exact product; the double product is exact up to 2^53, past which a duration is
// longer than any audio either way.
RegionStream::RegionStream(const RegionRule & rule, int sample_rate, int chunk)
    : threshold_(rule.threshold),
      neg_threshold_(std::max(rule.threshold - rule.neg_threshold_offset, rule.neg_threshold_floor)),
      min_speech_((double) sample_rate * (double) rule.min_speech_duration_ms / 1000),
      pad_((double) sample_rate * (double) rule.speech_pad_ms / 1000),
      max_speech_((double) sample_rate * rule.max_speech_duration_s - chunk - 2 * pad_),
      min_silence_((double) sample_rate * (double) rule.min_silence_duration_ms / 1000),
      min_silence_at_max_speech_((double) sample_rate * (double) rule.min_silence_at_max_speech_ms / 1000),
      chunk_(chunk) {}

void RegionStream::add(double p) {
    const int64_t cur = (int64_t) chunk_ * chunks_++;
    if (p >= threshold_ && temp_end_ != 0) {
        const int64_t silence = cur - temp_end_;
        if ((double) silence > min_silence_at_max_speech_) possible_ends_.push_back({temp_end_, silence});
        temp_end_ = 0;
    }
    if (p >= threshold_ && !triggered_) {
        triggered_ = true;
        start_ = cur;
        return;
    }
    if (triggered_ && (double) (cur - start_) > max_speech_) {
        if (!possible_ends_.empty()) {
            // Python's max() gives the first of equal silences.
            std::pair<int64_t, int64_t> longest = possible_ends_.front();
            for (const auto & e : possible_ends_) {
                if (e.second > longest.second) longest = e;
            }
            append({start_, longest.first});
            const int64_t next_start = longest.first + longest.second;
            // The official's comparison as it writes it, which holds unless the silence is as long as the audio before
            // this chunk.
            if (next_start < longest.first + cur) start_ = next_start;
            else triggered_ = false;
            temp_end_ = 0;
            possible_ends_.clear();
        } else {
            append({start_, cur});
            temp_end_ = 0;
            triggered_ = false;
            possible_ends_.clear();
            return;
        }
    }
    if (p < neg_threshold_ && triggered_) {
        if (temp_end_ == 0) temp_end_ = cur;
        if ((double) (cur - temp_end_) < min_silence_) return;
        if ((double) (temp_end_ - start_) > min_speech_) append({start_, temp_end_});
        temp_end_ = 0;
        triggered_ = false;
        possible_ends_.clear();
    }
}

int64_t RegionStream::earliest_end() const {
    return temp_end_ != 0 ? temp_end_ : std::min((int64_t) chunk_ * chunks_, heard_);
}

void RegionStream::settle(int64_t heard) {
    heard_ = heard;
    if (!pending_) return;
    const int64_t end = pending_->end, next_chunk = (int64_t) chunk_ * chunks_;
    // Every region still to come starts at the region under way's start or, with none under way, at a chunk not yet taken.
    // Where it starts twice the padding or more after the pending region's end, the pending region is padded by the
    // padding, which ends within the audio heard: before that start, or within `heard` as the last test asks.
    if (triggered_) {
        // The region under way is kept, as it is cut at max_speech or ends longer than min_speech.
        if ((double) (earliest_end() - start_) > min_speech_) give_pending(start_, (double) heard);
        else if ((double) (start_ - end) >= 2 * pad_) give_pending(std::nullopt, (double) heard);
    } else if ((double) (next_chunk - end) >= 2 * pad_ && (double) end + pad_ <= (double) heard) {
        give_pending(std::nullopt, (double) heard);
    }
}

void RegionStream::end(int64_t samples) {
    heard_ = samples;
    if (triggered_ && (double) (samples - start_) > min_speech_) append({start_, samples});
    triggered_ = false;
    if (pending_) give_pending(std::nullopt, (double) samples);
}

std::optional<Begun> RegionStream::open() const {
    if (pending_) return Begun{pending_->start, true};
    if (triggered_) return Begun{padded_start(start_), (double) (earliest_end() - start_) > min_speech_};
    return std::nullopt;
}

void RegionStream::append(Region raw) {
    const int64_t start = padded_start(raw.start);
    // A region found ends before the chunk under way, which lies within the audio.
    if (pending_) give_pending(raw.start, (double) raw.start);
    pending_ = Region{start, raw.end};
    last_end_ = raw.end;
}

// The padding: Python's int() of a float rounds toward zero, as the casts do.
int64_t RegionStream::padded_start(int64_t start) const {
    if (last_end_) {
        const int64_t silence = start - *last_end_;
        if ((double) silence < 2 * pad_) return std::max<int64_t>(0, start - floor_half(silence));
    }
    return (int64_t) std::max(0.0, (double) start - pad_);
}

int64_t RegionStream::padded_end(std::optional<int64_t> next, double length) const {
    const int64_t end = pending_->end;
    if (next) {
        const int64_t silence = *next - end;
        if ((double) silence < 2 * pad_) return end + floor_half(silence);
    }
    return (int64_t) std::min(length, (double) end + pad_);
}

void RegionStream::give_pending(std::optional<int64_t> next, double length) {
    given_.push_back({pending_->start, padded_end(next, length)});
    pending_.reset();
}

std::vector<Region> speech_regions(const std::vector<float> & probs, int64_t samples, int sample_rate, int chunk, const RegionRule & rule) {
    RegionStream regions(rule, sample_rate, chunk);
    for (const float p : probs) regions.add(p);
    regions.end(samples);
    return regions.regions();
}

}  // namespace silero_vad
