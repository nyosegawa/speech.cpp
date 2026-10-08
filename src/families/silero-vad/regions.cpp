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

std::vector<Region> speech_regions(const std::vector<float> & probs, int64_t samples, int sample_rate, int chunk, const RegionRule & rule) {
    // Python's int * int / 1000 divides the exact product; the double product is exact up to 2^53, past which a duration
    // is longer than any audio either way.
    const double rate = sample_rate;
    const double min_speech = rate * (double) rule.min_speech_duration_ms / 1000;
    const double pad = rate * (double) rule.speech_pad_ms / 1000;
    const double max_speech = rate * rule.max_speech_duration_s - chunk - 2 * pad;
    const double min_silence = rate * (double) rule.min_silence_duration_ms / 1000;
    const double min_silence_at_max_speech = rate * (double) rule.min_silence_at_max_speech_ms / 1000;
    const double neg_threshold = std::max(rule.threshold - rule.neg_threshold_offset, rule.neg_threshold_floor);

    std::vector<Region> speeches;
    bool triggered = false;
    // The start of the region under way while triggered, and where its probability first fell below neg_threshold, 0
    // while it has not, as the official's temp_end.
    int64_t start = 0, temp_end = 0;
    // Each silence within the region under way that lasted longer than min_silence_at_max_speech: where it began and
    // its length.
    std::vector<std::pair<int64_t, int64_t>> possible_ends;
    for (size_t i = 0; i < probs.size(); i++) {
        const double p = probs[i];
        const int64_t cur = (int64_t) chunk * (int64_t) i;
        if (p >= rule.threshold && temp_end != 0) {
            const int64_t silence = cur - temp_end;
            if ((double) silence > min_silence_at_max_speech) possible_ends.push_back({temp_end, silence});
            temp_end = 0;
        }
        if (p >= rule.threshold && !triggered) {
            triggered = true;
            start = cur;
            continue;
        }
        if (triggered && (double) (cur - start) > max_speech) {
            if (!possible_ends.empty()) {
                // Python's max() gives the first of equal silences.
                std::pair<int64_t, int64_t> longest = possible_ends.front();
                for (const auto & e : possible_ends) {
                    if (e.second > longest.second) longest = e;
                }
                speeches.push_back({start, longest.first});
                const int64_t next_start = longest.first + longest.second;
                // The official's comparison as it writes it, which holds unless the silence is as long as the audio
                // before this chunk.
                if (next_start < longest.first + cur) start = next_start;
                else triggered = false;
                temp_end = 0;
                possible_ends.clear();
            } else {
                speeches.push_back({start, cur});
                temp_end = 0;
                triggered = false;
                possible_ends.clear();
                continue;
            }
        }
        if (p < neg_threshold && triggered) {
            if (temp_end == 0) temp_end = cur;
            if ((double) (cur - temp_end) < min_silence) continue;
            if ((double) (temp_end - start) > min_speech) speeches.push_back({start, temp_end});
            temp_end = 0;
            triggered = false;
            possible_ends.clear();
            continue;
        }
    }
    if (triggered && (double) (samples - start) > min_speech) speeches.push_back({start, samples});

    // The padding: Python's int() of a float rounds toward zero, as the casts do.
    for (size_t i = 0; i < speeches.size(); i++) {
        Region & s = speeches[i];
        if (i == 0) s.start = (int64_t) std::max(0.0, (double) s.start - pad);
        if (i + 1 != speeches.size()) {
            Region & next = speeches[i + 1];
            const int64_t silence = next.start - s.end;
            if ((double) silence < 2 * pad) {
                s.end += floor_half(silence);
                next.start = std::max<int64_t>(0, next.start - floor_half(silence));
            } else {
                s.end = (int64_t) std::min((double) samples, (double) s.end + pad);
                next.start = (int64_t) std::max(0.0, (double) next.start - pad);
            }
        } else {
            s.end = (int64_t) std::min((double) samples, (double) s.end + pad);
        }
    }
    return speeches;
}

}  // namespace silero_vad
