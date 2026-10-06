#include "frontend.h"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

#include "error.h"
namespace fastconformer {

namespace {

std::vector<double> read_tensor(const ModelFile & m, const std::string & name, int64_t n) {
    ggml_tensor * t = m.tensor(name);
    if (t->type != GGML_TYPE_F32 || ggml_nelements(t) != n) {
        throw Error(Fault::File, name + " in " + m.path() + " is not float32 with " + std::to_string(n) + " values");
    }
    std::vector<float> f(n);
    ggml_backend_tensor_get(t, f.data(), 0, ggml_nbytes(t));
    return std::vector<double>(f.begin(), f.end());
}

/** An in-place radix-2 FFT; the size is a power of two and `twiddles` holds exp(-2 pi i k / size) for k < size / 2. */
void fft(std::vector<std::complex<double>> & x, const std::vector<std::complex<double>> & twiddles) {
    const size_t n = x.size();
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const size_t stride = n / len;
        for (size_t i = 0; i < n; i += len) {
            for (size_t k = 0; k < len / 2; k++) {
                const std::complex<double> u = x[i + k], v = x[i + k + len / 2] * twiddles[k * stride];
                x[i + k] = u + v;
                x[i + k + len / 2] = u - v;
            }
        }
    }
}

}  // namespace

Frontend::Frontend(const ModelFile & m)
    : sample_rate_((int) m.u32("speech.sample_rate")),
      n_fft_((int) m.u32("fastconformer.frontend.n_fft")),
      hop_((int) m.u32("fastconformer.frontend.hop_length")),
      mels_((int) m.u32("fastconformer.frontend.n_mels")),
      preemphasis_(m.f32("fastconformer.frontend.preemphasis")),
      log_guard_(m.f32("fastconformer.frontend.log_guard")),
      std_guard_(m.f32("fastconformer.frontend.std_guard")) {
    if (n_fft_ <= 0 || (n_fft_ & (n_fft_ - 1)) != 0) throw Error(Fault::File, "fastconformer.frontend.n_fft of " + m.path() + " is not a power of two");
    const int64_t window = ggml_nelements(m.tensor("frontend.window"));
    if (window > n_fft_) throw Error(Fault::File, "frontend.window of " + m.path() + " is longer than fastconformer.frontend.n_fft");
    const std::vector<double> w = read_tensor(m, "frontend.window", window);
    window_.assign(n_fft_, 0.0);
    // torch.stft() centres a window shorter than n_fft in it.
    const int64_t left = (n_fft_ - window) / 2;
    for (int64_t i = 0; i < window; i++) window_[left + i] = w[i];
    filterbank_ = read_tensor(m, "frontend.filterbank", (int64_t) mels_ * (n_fft_ / 2 + 1));
    const int bins = n_fft_ / 2 + 1;
    for (int b = 0; b < mels_; b++) {
        int first = 0, last = bins;
        while (first < bins && filterbank_[(size_t) b * bins + first] == 0) first++;
        while (last > first && filterbank_[(size_t) b * bins + last - 1] == 0) last--;
        bands_.push_back({first, last});
    }
    for (int k = 0; k < n_fft_ / 2; k++) twiddles_.push_back(std::polar(1.0, -2 * 3.14159265358979323846 * k / n_fft_));
}

int64_t Frontend::frames(size_t samples) const {
    // FilterbankFeatures.get_seq_len(): the STFT gives one frame more, which the official code masks out.
    return (int64_t) samples / hop_;
}

std::vector<float> Frontend::features(const std::vector<float> & samples) const {
    const int64_t frames = this->frames(samples.size());
    // normalize_batch() divides by frames - 1.
    if (frames < 2) {
        throw Error(Fault::OutOfRange, "the audio has " + std::to_string(samples.size()) + " samples at " + std::to_string(sample_rate_) +
                                           " Hz; recognition needs at least " + std::to_string(2 * hop_),
                    "audio");
    }
    // Pre-emphasis in float32, as FilterbankFeatures.forward() computes it.
    std::vector<float> x(samples.size());
    x[0] = samples[0];
    for (size_t i = 1; i < samples.size(); i++) x[i] = samples[i] - preemphasis_ * samples[i - 1];

    const int bins = n_fft_ / 2 + 1;
    const int64_t n = (int64_t) x.size();
    std::vector<double> mel((size_t) (frames * mels_));
    std::vector<std::complex<double>> buffer(n_fft_);
    std::vector<double> power(bins);
    for (int64_t f = 0; f < frames; f++) {
        // center=True with pad_mode="constant": the signal has n_fft / 2 zeros before it.
        const int64_t start = f * hop_ - n_fft_ / 2;
        for (int k = 0; k < n_fft_; k++) {
            const int64_t i = start + k;
            buffer[k] = i >= 0 && i < n ? x[i] * window_[k] : 0.0;
        }
        fft(buffer, twiddles_);
        for (int k = 0; k < bins; k++) power[k] = std::norm(buffer[k]);
        for (int m = 0; m < mels_; m++) {
            double sum = 0;
            for (int k = bands_[m].first; k < bands_[m].second; k++) sum += filterbank_[(size_t) m * bins + k] * power[k];
            mel[(size_t) (f * mels_ + m)] = std::log(sum + (double) log_guard_);
        }
    }

    // normalize_batch(normalize_type="per_feature"): the unbiased standard deviation, plus a guard.
    std::vector<float> out(mel.size());
    for (int m = 0; m < mels_; m++) {
        double mean = 0;
        for (int64_t f = 0; f < frames; f++) mean += mel[(size_t) (f * mels_ + m)];
        mean /= (double) frames;
        double var = 0;
        for (int64_t f = 0; f < frames; f++) {
            const double d = mel[(size_t) (f * mels_ + m)] - mean;
            var += d * d;
        }
        const double std = std::sqrt(var / (double) (frames - 1)) + (double) std_guard_;
        for (int64_t f = 0; f < frames; f++) {
            out[(size_t) (f * mels_ + m)] = (float) ((mel[(size_t) (f * mels_ + m)] - mean) / std);
        }
    }
    return out;
}

}  // namespace fastconformer
