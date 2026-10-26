#include "cache.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace sortformer {

namespace {

constexpr float kInfinity = std::numeric_limits<float>::infinity();

/**
 * The `k` largest of `values`, by their indices: the larger first, and of equal values the earlier. `ties` counts a
 * choice whose last value equals the first left out, other than -inf, which nothing chooses by.
 */
std::vector<int64_t> largest(const std::vector<float> & values, int64_t k, int * ties) {
    std::vector<int64_t> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    k = std::min<int64_t>(k, (int64_t) order.size());
    const auto before = [&](int64_t a, int64_t b) {
        return values[(size_t) a] > values[(size_t) b] || (values[(size_t) a] == values[(size_t) b] && a < b);
    };
    std::partial_sort(order.begin(), order.begin() + std::min<int64_t>(k + 1, (int64_t) order.size()), order.end(), before);
    if (ties && k > 0 && k < (int64_t) order.size() && values[(size_t) order[(size_t) (k - 1)]] == values[(size_t) order[(size_t) k]] &&
        values[(size_t) order[(size_t) k]] != -kInfinity) {
        (*ties)++;
    }
    order.resize((size_t) k);
    return order;
}

/**
 * _boost_topk_scores(): raises each speaker's `k` highest scores by scale · -log(0.5), which torch subtracts as a float32
 * of scale · log(0.5), so that a disabled score stays -inf.
 */
void boost(std::vector<float> & scores, int64_t n, int speakers, int64_t k, double scale, int * ties) {
    const float step = (float) (scale * std::log(0.5));
    std::vector<float> column((size_t) n);
    for (int s = 0; s < speakers; s++) {
        for (int64_t f = 0; f < n; f++) column[(size_t) f] = scores[(size_t) (f * speakers + s)];
        for (const int64_t f : largest(column, k, ties)) scores[(size_t) (f * speakers + s)] -= step;
    }
}

}  // namespace

Compression compression_of(const ModelFile & m) {
    Compression c;
    c.speakers = (int) m.u32("sortformer.speakers");
    c.silence_frames = (int) m.u32("sortformer.cache.silence_frames");
    c.score_threshold = m.f32("sortformer.cache.score_threshold");
    c.boost_latest = m.f32("sortformer.cache.boost_latest");
    c.strong_boost_rate = m.f32("sortformer.cache.strong_boost_rate");
    c.weak_boost_rate = m.f32("sortformer.cache.weak_boost_rate");
    c.min_pos_scores_rate = m.f32("sortformer.cache.min_pos_scores_rate");
    return c;
}

std::vector<int64_t> compress(const std::vector<float> & probs, int64_t n, const Streaming & streaming, const Compression & c, int * ties) {
    const int speakers = c.speakers;
    // The shares are floored in double precision, as math.floor() of a Python float.
    const int64_t per_speaker = streaming.spkcache_len / speakers - c.silence_frames;
    const int64_t strong = (int64_t) std::floor((double) per_speaker * (double) c.strong_boost_rate);
    const int64_t weak = (int64_t) std::floor((double) per_speaker * (double) c.weak_boost_rate);
    const int64_t min_positive = (int64_t) std::floor((double) per_speaker * (double) c.min_pos_scores_rate);

    // _get_log_pred_scores(), in float32: high where one speaker speaks alone and surely. The sum over the speakers adds
    // them in order, where torch's kernel may pair them, which can move a score by a unit in its last place.
    const float threshold = c.score_threshold, log_half = (float) std::log(0.5);
    std::vector<float> scores((size_t) (n * speakers)), absent((size_t) speakers);
    for (int64_t f = 0; f < n; f++) {
        const float * p = &probs[(size_t) (f * speakers)];
        float sum = 0;
        for (int s = 0; s < speakers; s++) {
            absent[(size_t) s] = std::log(std::max(1.0f - p[s], threshold));
            sum += absent[(size_t) s];
        }
        for (int s = 0; s < speakers; s++) {
            scores[(size_t) (f * speakers + s)] = std::log(std::max(p[s], threshold)) - absent[(size_t) s] + sum - log_half;
        }
    }
    // _disable_low_scores(): no score where a speaker is silent, and where a speaker has enough frames of speech alone,
    // none where it speaks with others.
    std::vector<int64_t> positive((size_t) speakers, 0);
    for (int64_t i = 0; i < n * speakers; i++) {
        if (!(probs[(size_t) i] > 0.5f)) scores[(size_t) i] = -kInfinity;
        if (scores[(size_t) i] > 0) positive[(size_t) (i % speakers)]++;
    }
    for (int64_t i = 0; i < n * speakers; i++) {
        if (!(scores[(size_t) i] > 0) && probs[(size_t) i] > 0.5f && positive[(size_t) (i % speakers)] >= min_positive) scores[(size_t) i] = -kInfinity;
    }
    // The rows the queue has just given score a little more.
    if (c.boost_latest > 0) {
        for (int64_t i = (int64_t) streaming.spkcache_len * speakers; i < n * speakers; i++) scores[(size_t) i] += c.boost_latest;
    }
    // A strong boost keeps a few frames of every speaker, and a weak one keeps a speaker from taking the whole cache.
    boost(scores, n, speakers, strong, 2.0, ties);
    boost(scores, n, speakers, weak, 1.0, ties);

    // _get_topk_indices() over each speaker's frames and its silence frames, which score +inf, one speaker after
    // another: the spkcache_len highest, in that order, those of -inf disabled and moved to the end.
    const int64_t padded = n + c.silence_frames;
    std::vector<float> flat((size_t) (speakers * padded));
    for (int s = 0; s < speakers; s++) {
        for (int64_t f = 0; f < padded; f++) flat[(size_t) (s * padded + f)] = f < n ? scores[(size_t) (f * speakers + s)] : kInfinity;
    }
    std::vector<int64_t> chosen = largest(flat, streaming.spkcache_len, ties);
    const int64_t disabled = (int64_t) flat.size();
    for (int64_t & i : chosen) {
        if (flat[(size_t) i] == -kInfinity) i = disabled;
    }
    std::sort(chosen.begin(), chosen.end());
    std::vector<int64_t> rows;
    rows.reserve(chosen.size());
    for (const int64_t i : chosen) rows.push_back(i == disabled || i % padded >= n ? -1 : i % padded);
    return rows;
}

void update(State & state, const Streaming & streaming, const Compression & compression, const std::vector<float> & silence, const float * chunk,
            int64_t frames, int64_t first, const float * probs, int * ties) {
    const int speakers = compression.speakers;
    const int64_t d = (int64_t) silence.size(), cache_n = state.cache_rows(), fifo_n = state.fifo_rows();
    // The queue's rows take the probabilities this step gave them, and the chunk's rows follow.
    state.fifo_probs.assign(probs + cache_n * speakers, probs + (cache_n + fifo_n + frames) * speakers);
    state.fifo.insert(state.fifo.end(), chunk, chunk + frames * d);
    for (int64_t i = 0; i < frames; i++) state.fifo_frames.push_back(first + i);
    if (fifo_n + frames <= streaming.fifo_len) return;

    int64_t pop = streaming.spkcache_update_period;
    pop = std::max(pop, frames - streaming.fifo_len + fifo_n);
    pop = std::min(pop, fifo_n + frames);
    state.cache.insert(state.cache.end(), state.fifo.begin(), state.fifo.begin() + pop * d);
    state.cache_frames.insert(state.cache_frames.end(), state.fifo_frames.begin(), state.fifo_frames.begin() + pop);
    // Until its first compression, the cache's rows take the probabilities this step gave them.
    if (!state.compressed) state.cache_probs.assign(probs, probs + cache_n * speakers);
    state.cache_probs.insert(state.cache_probs.end(), state.fifo_probs.begin(), state.fifo_probs.begin() + pop * speakers);
    state.fifo.erase(state.fifo.begin(), state.fifo.begin() + pop * d);
    state.fifo_probs.erase(state.fifo_probs.begin(), state.fifo_probs.begin() + pop * speakers);
    state.fifo_frames.erase(state.fifo_frames.begin(), state.fifo_frames.begin() + pop);
    if (state.cache_rows() <= streaming.spkcache_len) return;

    const std::vector<int64_t> rows = compress(state.cache_probs, state.cache_rows(), streaming, compression, ties);
    std::vector<float> cache, cache_probs;
    std::vector<int64_t> cache_frames;
    cache.reserve(rows.size() * (size_t) d);
    for (const int64_t r : rows) {
        const float * row = r < 0 ? silence.data() : &state.cache[(size_t) (r * d)];
        cache.insert(cache.end(), row, row + d);
        for (int s = 0; s < speakers; s++) cache_probs.push_back(r < 0 ? 0.0f : state.cache_probs[(size_t) (r * speakers + s)]);
        cache_frames.push_back(r < 0 ? -1 : state.cache_frames[(size_t) r]);
    }
    state.cache = std::move(cache);
    state.cache_probs = std::move(cache_probs);
    state.cache_frames = std::move(cache_frames);
    state.compressed = true;
}

}  // namespace sortformer
