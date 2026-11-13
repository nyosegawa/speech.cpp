#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace haru {

Sampling file_sampling(const ModelFile & m) {
    Sampling s;
    s.temperature = m.f32("haru.sampling.temperature");
    s.top_k = (int) m.u32("haru.sampling.top_k");
    s.top_p = m.f32("haru.sampling.top_p");
    return s;
}

Candidates candidates(const std::vector<float> & logits, const Sampling & s) {
    // Minus infinity is a token the session bans, which the softmax gives no chance; NaN and plus infinity can only
    // come from the device.
    for (float l : logits) {
        if (std::isnan(l) || l == INFINITY) throw Error(Fault::Device, "the model computed a logit that is not a number; run it on another device");
    }
    const int64_t v = (int64_t) logits.size();
    Candidates c;
    if (s.temperature <= 0.0f) {
        c.tokens = {(int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin())};
        c.weights = {1.0};
        return c;
    }
    std::vector<float> scaled(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        scaled[i] = logits[i] / s.temperature;
        if (std::isfinite(logits[i]) && !std::isfinite(scaled[i])) throw temperature_overflow(s.temperature);
    }
    const int64_t k = s.top_k > 0 && s.top_k < v ? s.top_k : v;
    std::vector<int32_t> order((size_t) v);
    std::iota(order.begin(), order.end(), 0);
    const auto falling = [&](int32_t a, int32_t b) { return scaled[a] > scaled[b] || (scaled[a] == scaled[b] && a < b); };
    std::partial_sort(order.begin(), order.begin() + k, order.end(), falling);
    order.resize((size_t) k);
    // torch's softmax in float32: the exponentials of the values less the largest, divided by their sum.
    std::vector<float> p((size_t) k);
    float sum = 0.0f;
    for (int64_t i = 0; i < k; i++) {
        p[i] = std::exp(scaled[order[i]] - scaled[order[0]]);
        sum += p[i];
    }
    for (float & x : p) x /= sum;
    c.tokens = order;
    c.weights.assign(p.begin(), p.end());
    if (s.top_p > 0.0f && s.top_p < 1.0f) {
        // torch keeps a token where its cumulative sum less its own probability is below top_p, both in float32.
        float cumulative = 0.0f;
        for (int64_t i = 0; i < k; i++) {
            cumulative += p[i];
            if (!(cumulative - p[i] < s.top_p)) c.weights[i] = 0.0;
        }
    }
    return c;
}

DryConfig file_dry(const ModelFile & m) {
    const std::string s = "haru.sampling.";
    DryConfig d;
    d.multiplier = m.f64(s + "dry_multiplier");
    d.base = m.f64(s + "dry_base");
    d.max_penalty = m.f64(s + "dry_max_penalty");
    d.allowed_length = (int) m.u32(s + "dry_allowed_length");
    d.lookback = (int) m.u32(s + "dry_lookback");
    d.max_match_length = m.size(s + "dry_max_match_length");
    return d;
}

TextIds file_text_ids(const ModelFile & m) {
    TextIds ids;
    ids.pad = (int32_t) m.u32("haru.pad_token_id");
    ids.eop = (int32_t) m.u32("haru.eop_token_id");
    ids.eos = (int32_t) m.u32("haru.eos_token_id");
    return ids;
}

Dry::Dry(const DryConfig & config, std::vector<int32_t> exempt) : config_(config), exempt_(std::move(exempt)) {}

bool Dry::exempt(int32_t token) const { return std::find(exempt_.begin(), exempt_.end(), token) != exempt_.end(); }

void Dry::update(int32_t token) {
    if (exempt(token)) return;
    history_.push_back(token);
    const size_t size = (size_t) config_.lookback + (size_t) config_.max_match_length + 1;
    if (history_.size() > size) history_.erase(history_.begin(), history_.begin() + (std::ptrdiff_t) (history_.size() - size));
    penalties_.clear();
    const int64_t n = (int64_t) history_.size();
    if (n < std::max(1, config_.allowed_length)) return;
    std::map<int32_t, int> longest;
    const int32_t last = history_[(size_t) (n - 1)];
    for (int64_t j = n - 2; j >= std::max<int64_t>(0, n - 1 - config_.lookback); j--) {
        if (history_[(size_t) j] != last) continue;
        int length = 1;
        while (length < config_.max_match_length && j - length >= 0 && history_[(size_t) (j - length)] == history_[(size_t) (n - 1 - length)]) length++;
        if (length >= config_.allowed_length) {
            const int32_t candidate = history_[(size_t) (j + 1)];
            auto it = longest.find(candidate);
            if (it == longest.end() || length > it->second) longest[candidate] = length;
        }
    }
    for (const auto & [t, length] : longest) {
        if (!exempt(t)) penalties_.push_back({t, std::min(config_.max_penalty, config_.multiplier * std::pow(config_.base, length - config_.allowed_length))});
    }
}

void Dry::apply(std::vector<float> & logits) const {
    for (const auto & [t, penalty] : penalties_) logits[(size_t) t] -= (float) penalty;
}

void shape_logits(std::vector<float> & logits, const Dry & dry, const float silent[2], const TextIds & ids, float eop_bias, bool must_speak) {
    dry.apply(logits);
    logits[(size_t) ids.pad] = silent[0];
    logits[(size_t) ids.eop] = silent[1];
    if (eop_bias != 0.0f) logits[(size_t) ids.eop] += eop_bias;
    if (must_speak) {
        for (int32_t t : {ids.pad, ids.eop, ids.eos}) logits[(size_t) t] = -INFINITY;
    }
}

}  // namespace haru
