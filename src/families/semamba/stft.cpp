#include "stft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>

namespace semamba {

namespace {

constexpr double kPi = 3.14159265358979323846;

}  // namespace

int scaled(int reference, int rate, int reference_rate) {
    const int64_t value = (int64_t) reference * rate / reference_rate;
    return (int) (value % 2 == 0 ? value : value + 1);
}

Stft::Stft(int n_fft, int hop) : n_fft_(n_fft), hop_(hop), fft_(n_fft), window_((size_t) n_fft) {
    if (hop <= 0 || hop > n_fft) throw std::logic_error("an STFT's hop is not within its FFT");
    for (int i = 0; i < n_fft; i++) window_[(size_t) i] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / n_fft);
}

void Stft::forward(const float * samples, size_t n, std::vector<float> & magnitude, std::vector<float> & phase) const {
    const int half = n_fft_ / 2, count = frames(n), b = bins();
    if (n <= (size_t) half) throw std::logic_error("an STFT's input is not longer than half its FFT");
    magnitude.assign((size_t) count * b, 0.0f);
    phase.assign((size_t) count * b, 0.0f);
    std::vector<double> frame((size_t) n_fft_);
    std::vector<std::complex<double>> spectrum((size_t) b);
    const int64_t last = (int64_t) n - 1;
    for (int t = 0; t < count; t++) {
        for (int i = 0; i < n_fft_; i++) {
            // The sample of the padded signal: the input reflected about its first and its last sample.
            int64_t at = (int64_t) t * hop_ + i - half;
            if (at < 0) at = -at;
            if (at > last) at = 2 * last - at;
            frame[(size_t) i] = samples[at] * window_[(size_t) i];
        }
        fft_.transform(frame.data(), spectrum.data());
        for (int k = 0; k < b; k++) {
            const std::complex<double> x = spectrum[(size_t) k];
            magnitude[(size_t) t * b + k] = (float) std::log1p(std::abs(x));
            phase[(size_t) t * b + k] = (float) std::atan2(x.imag(), x.real());
        }
    }
}

std::vector<float> Stft::inverse(const std::vector<float> & magnitude, const std::vector<float> & phase, int frames) const {
    const int half = n_fft_ / 2, b = bins();
    const size_t length = (size_t) n_fft_ + (size_t) hop_ * (size_t) (frames - 1);
    std::vector<double> sum(length, 0.0), envelope(length, 0.0), real((size_t) n_fft_), frame((size_t) n_fft_);
    std::vector<std::complex<double>> spectrum((size_t) b);
    for (int t = 0; t < frames; t++) {
        // A real inverse FFT through the forward one: for X Hermitian, x[n] = (Re F[n] + Im F[n]) / N, where F is the
        // transform of r[k] = Re X[k] + Im X[k] over the whole circle.
        for (int k = 0; k <= half; k++) {
            const double m = std::expm1(std::max(0.0f, magnitude[(size_t) t * b + k]));
            const double p = phase[(size_t) t * b + k];
            const double re = m * std::cos(p), im = k == 0 || k == half ? 0.0 : m * std::sin(p);
            real[(size_t) k] = re + im;
            if (k != 0 && k != half) real[(size_t) (n_fft_ - k)] = re - im;
        }
        fft_.transform(real.data(), spectrum.data());
        for (int i = 0; i < n_fft_; i++) {
            const std::complex<double> f = i <= half ? spectrum[(size_t) i] : std::conj(spectrum[(size_t) (n_fft_ - i)]);
            frame[(size_t) i] = (f.real() + f.imag()) / n_fft_;
        }
        for (int i = 0; i < n_fft_; i++) {
            sum[(size_t) t * hop_ + i] += frame[(size_t) i] * window_[(size_t) i];
            envelope[(size_t) t * hop_ + i] += window_[(size_t) i] * window_[(size_t) i];
        }
    }
    std::vector<float> out((size_t) hop_ * (size_t) (frames - 1));
    for (size_t i = 0; i < out.size(); i++) out[i] = (float) (sum[i + half] / envelope[i + half]);
    return out;
}

}  // namespace semamba
