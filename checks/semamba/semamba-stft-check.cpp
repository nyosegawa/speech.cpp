// Checks SEMamba's stages on the host against RE-USE's official code, on every chunk of every input of
// reference/semamba/dump.py, each from the chunk's own dumped input: the STFT of the chunk's samples, its compressed
// magnitude and its complex values; the decoding of the network's dumped outputs, cropped, given their angle and their
// frames zeroed; and the inverse STFT of the dumped zeroed magnitude and angle, padded to the chunk. The dumps are those
// in <reference out dir>/<the file's general.name>/.
//
// usage: semamba-stft-check <model.gguf> <reference out dir>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "semamba-dumps.h"
#include "semamba/enhancer.h"
#include "semamba/layout.h"

using namespace semamba_dumps;

namespace {

/** The real and imaginary parts of each bin, from a compressed magnitude and a phase. */
std::vector<float> complex_values(const std::vector<float> & magnitude, const std::vector<float> & phase) {
    std::vector<float> out(2 * magnitude.size());
    for (size_t i = 0; i < magnitude.size(); i++) {
        const double m = std::expm1((double) magnitude[i]);
        out[2 * i] = (float) (m * std::cos((double) phase[i]));
        out[2 * i + 1] = (float) (m * std::sin((double) phase[i]));
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir>\n", args[0].c_str());
        return 2;
    }
    try {
        start_ggml();
        const ModelFile model(args[1], semamba::layout);
        const int reference = model.size("semamba.stft.sampling_rate"), n_fft = model.size("semamba.stft.n_fft"), hop = model.size("semamba.stft.hop_size");
        const float zero_portion = model.f32("semamba.zero_portion"), pad_value = model.f32("semamba.pad_value");
        bool ok = true;
        for (const auto & dump : dumps(args[2], model)) {
            const Meta meta = read_meta(dump);
            const semamba::Stft stft(semamba::scaled(n_fft, meta.rate, reference), semamba::scaled(hop, meta.rate, reference));
            std::printf("%s (%d Hz, FFT %d, hop %d, %zu chunks)\n", dump.filename().u8string().c_str(), meta.rate, stft.n_fft(), stft.hop(),
                        meta.starts.size());
            ok = ok && stft.n_fft() == meta.n_fft && stft.hop() == meta.hop_size;
            Diff magnitude{}, values{}, inverse{};
            bool zeroed = true, decoded = true;
            Diff decoded_magnitude{}, decoded_values{};
            for (size_t k = 0; k < meta.starts.size(); k++) {
                const auto folder = chunk(dump, k);
                const Npy input = read(folder, "input"), mag = read(folder, "mag"), pha = read(folder, "pha");
                std::vector<float> got_mag, got_pha;
                stft.forward(input.f32.data(), input.f32.size(), got_mag, got_pha);
                ok = ok && got_mag.size() == mag.f32.size();
                keep_worst(magnitude, compare(got_mag, mag.f32));
                // The phase of a bin whose value is near the negative reals is +pi or -pi by a rounding, so its complex
                // value is compared rather than its angle.
                keep_worst(values, compare(complex_values(got_mag, got_pha), complex_values(mag.f32, pha.f32)));

                const Npy raw = read(folder, "mag_raw"), real = read(folder, "phase_r"), imag = read(folder, "phase_i");
                const Npy amp = read(folder, "amp"), angle = read(folder, "angle"), zero = read(folder, "zeroed");
                const int64_t frames = mag.shape[0], bins = mag.shape[1];
                const semamba::Decoded d = semamba::decode(raw.f32, real.f32, imag.f32, raw.shape[1], frames, bins, zero_portion);
                zeroed = zeroed && d.zeroed == zero.u8;
                keep_worst(decoded_magnitude, compare(d.magnitude, amp.f32));
                keep_worst(decoded_values, compare(complex_values(d.magnitude, d.phase), complex_values(amp.f32, angle.f32)));
                decoded = decoded && d.magnitude.size() == amp.f32.size();

                const Npy output = read(folder, "output");
                std::vector<float> got = stft.inverse(amp.f32, angle.f32, (int) frames);
                got.resize(output.f32.size(), pad_value);
                keep_worst(inverse, compare(got, output.f32));
            }
            print_diff("  STFT magnitude, compressed", magnitude);
            print_diff("  STFT values", values);
            print_diff("  decoded magnitude", decoded_magnitude);
            print_diff("  decoded values", decoded_values);
            std::printf("  zeroed frames: %s\n", zeroed ? "the official's" : "DIFFERENT");
            print_diff("  inverse STFT", inverse);
            // Double precision against cuFFT's float32: 130 to 141 dB on these inputs on an Apple M5 (2026-11-01), the decoded
            // magnitude a copy of the official's. A wrong window, padding, scaling of the sizes or bin of an inverse falls far
            // below.
            ok = ok && zeroed && decoded && magnitude.snr_db > 100 && values.snr_db > 100 && decoded_magnitude.snr_db > 140 &&
                 decoded_values.snr_db > 120 && inverse.snr_db > 100;
        }
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
