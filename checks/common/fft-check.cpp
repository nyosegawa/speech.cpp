// Checks the mixed-radix FFT against a direct double-precision DFT, including the power-of-two and mixed sizes
// used by speech frontends, complex inputs, impulses and constants.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

#include "fft.h"

int main() {
    bool ok = true;
    for (int n : {1, 2, 7, 16, 25, 400, 512}) {
        Fft fft(n);
        for (int pattern = 0; pattern < 3; pattern++) {
            std::vector<std::complex<double>> input(n), output(n);
            for (int i = 0; i < n; i++) {
                if (pattern == 0) input[i] = {std::sin(i * 1.37), std::cos(i * 0.79)};
                if (pattern == 1) input[i] = i == n / 3 ? 1.0 : 0.0;
                if (pattern == 2) input[i] = {0.75, -0.25};
            }
            fft.transform(input.data(), output.data());
            double largest = 0;
            for (int k = 0; k < n; k++) {
                std::complex<double> expected = 0;
                for (int i = 0; i < n; i++) {
                    expected += input[i] * std::polar(1.0, -2 * 3.14159265358979323846 * i * k / n);
                }
                largest = std::max(largest, std::abs(output[k] - expected));
            }
            std::printf("size %d pattern %d: largest error %.3g\n", n, pattern, largest);
            ok = ok && largest < 1e-9;
        }
    }
    return ok ? 0 : 1;
}
