#pragma once

#include <vector>

/**
 * Changes the sample rate of mono audio as torchaudio.functional.resample() does with the parameters its
 * documentation gives for librosa's kaiser_best: a rational polyphase windowed sinc with 64 zero crossings on each
 * side, cut off at 0.9475937167399596 of the lower rate's Nyquist frequency, under a Kaiser window of beta
 * 14.769656459379492, which torchaudio rounds to float32 together with the window's peak i0(beta), and otherwise
 * computed in double precision. The output has torchaudio's length, ceil(n · to / from) as torchaudio rounds it,
 * and the audio is padded with zeros at both ends as torchaudio pads it. At equal rates the samples pass through
 * unchanged.
 */
class Resampler {
public:
    /**
     * Prepares the filter from `from_rate` to `to_rate` Hz. Throws for a rate that is not positive and for two rates
     * whose ratio in lowest terms has a term above 4096.
     */
    Resampler(int from_rate, int to_rate);

    /** The samples at the new rate, each rounded to float; at equal rates, `samples` themselves. */
    std::vector<float> operator()(std::vector<float> samples) const;

    /** The samples at the new rate in double precision, before the rounding to float. */
    std::vector<double> in_double(const std::vector<float> & samples) const;

private:
    template <typename T>
    std::vector<T> apply(const std::vector<float> & samples) const;

    /** The ratio in lowest terms: `from_` input samples become `to_` output samples. */
    int from_ = 1, to_ = 1;
    /** torchaudio's padding of the input on the left, in input samples. */
    long long width_ = 0;
    /** The taps of each of the `to_` phases: phase j starts at input offset first_[j] - width_. */
    int taps_ = 0;
    std::vector<long long> first_;
    /** [to_][taps_], zero past the end of a phase's taps. */
    std::vector<double> kernel_;
};
