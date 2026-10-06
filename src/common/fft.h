#pragma once

#include <complex>
#include <vector>

/**
 * The discrete Fourier transform of one size, X[k] = Σ x[n] exp(-2πi k n / size), in double precision: Cooley and
 * Tukey's decimation in time over the size's prime factors, one radix per factor. It takes any size, and is fast for
 * sizes whose prime factors are small, such as NeMo's 512 = 2⁹ and Whisper's 400 = 2⁴ × 5².
 */
class Fft {
public:
    explicit Fft(int size);

    int size() const { return size_; }

    /** Transforms the size() values of `in` into `out`, which holds as many; the two do not overlap. */
    void transform(const std::complex<double> * in, std::complex<double> * out) const;

private:
    /**
     * The transform of the n = size() / stride values in[0], in[stride], ... into out[0, n), whose radix is the
     * factor `level`.
     */
    void work(const std::complex<double> * in, std::complex<double> * out, size_t stride, size_t level) const;

    int size_;
    std::vector<int> factors_;
    /** exp(-2πi t / size) for t < size. */
    std::vector<std::complex<double>> twiddles_;
};
