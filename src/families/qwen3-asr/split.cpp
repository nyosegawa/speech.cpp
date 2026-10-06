#include "split.h"

#include <algorithm>
#include <cmath>

namespace qwen3_asr {

Splitter::Splitter(const ModelFile & m)
    : max_samples_(m.u32("qwen3-asr.audio.max_samples")),
      search_samples_(m.u32("qwen3-asr.audio.split_search_samples")),
      window_samples_(m.u32("qwen3-asr.audio.split_window_samples")) {}

std::vector<int64_t> Splitter::bounds(const std::vector<float> & samples) const {
    const int64_t total = (int64_t) samples.size();
    std::vector<int64_t> out = {0};
    std::vector<double> sums;
    for (int64_t start = 0; total - start > max_samples_;) {
        const int64_t cut = start + max_samples_, left = std::max(start, cut - search_samples_), right = std::min(total, cut + search_samples_);
        int64_t boundary = cut;
        if (right - left > window_samples_) {
            // The sum of the magnitudes of each window of window_samples_ that fits in [left, right), as
            // np.convolve(seg_abs, ones(win), mode="valid") gives them. Each is summed whole, so that windows of the same
            // samples, silent ones among them, get the same sum, which a running sum's rounding would tell apart.
            const int64_t windows = right - left - window_samples_ + 1;
            sums.assign((size_t) windows, 0.0);
            for (int64_t w = 0; w < windows; w++) {
                double sum = 0;
                for (int64_t i = 0; i < window_samples_; i++) sum += std::fabs(samples[(size_t) (left + w + i)]);
                sums[(size_t) w] = sum;
            }
            const int64_t quietest = (int64_t) (std::min_element(sums.begin(), sums.end()) - sums.begin());
            const auto first = samples.begin() + (std::ptrdiff_t) (left + quietest);
            const auto inner = std::min_element(first, first + (std::ptrdiff_t) window_samples_, [](float a, float b) { return std::fabs(a) < std::fabs(b); });
            boundary = left + quietest + (int64_t) (inner - first);
        }
        boundary = std::min(std::max(boundary, start + 1), total);
        out.push_back(boundary);
        start = boundary;
    }
    out.push_back(total);
    return out;
}

}  // namespace qwen3_asr
