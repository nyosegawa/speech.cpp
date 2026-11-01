#pragma once

#include <cstddef>
#include <vector>

#include "fft.h"

namespace semamba {

/**
 * The STFT of the official models/stfts.py at one rate, which scales the reference sizes as the scripts do
 * (make_even(n_fft * rate // sampling_rate)): torch.stft() with a periodic Hann window as long as the FFT, centred with
 * the signal reflected by half an FFT at each end, its magnitude compressed by log1p; and torch.istft() of a compressed
 * magnitude, expanded by expm1 after a ReLU, and a phase, whose frames are added up under the window and divided by the
 * sum of its squares, half an FFT cut from each end. Spectrograms are [frames, bins], the bins fastest. It computes in
 * double precision and rounds each value it gives to float.
 */
class Stft {
public:
    Stft(int n_fft, int hop);

    int n_fft() const { return n_fft_; }
    int hop() const { return hop_; }
    int bins() const { return n_fft_ / 2 + 1; }
    /** The frames of `samples` samples, which must be more than half an FFT for the reflection. */
    int frames(size_t samples) const { return 1 + (int) (samples / (size_t) hop_); }

    /** The compressed magnitude and the phase of each bin of each frame of `n` samples. */
    void forward(const float * samples, size_t n, std::vector<float> & magnitude, std::vector<float> & phase) const;

    /**
     * The samples of `frames` frames of a compressed magnitude and a phase, hop() * (frames - 1) of them. The imaginary
     * parts of the first and the last bin are left out, as a real inverse FFT leaves them.
     */
    std::vector<float> inverse(const std::vector<float> & magnitude, const std::vector<float> & phase, int frames) const;

private:
    int n_fft_, hop_;
    RealFft fft_;
    std::vector<double> window_;
};

/** The size the scripts give a reference size at `rate`: reference * rate / reference_rate, rounded down, made even. */
int scaled(int reference, int rate, int reference_rate);

}  // namespace semamba
