#include "reference.h"

#include <stdexcept>

#include "loudness.h"
#include "resample.h"
#include "wav.h"

namespace irodori {

std::vector<float> encode_reference(Codec & codec, const std::string & wav_path, double max_seconds) {
    const Wav wav = read_wav(wav_path);
    const Resampler resample = [&] {
        try {
            return Resampler(wav.sample_rate, codec.sample_rate());
        } catch (const std::invalid_argument & e) {
            throw std::invalid_argument(wav_path + ": " + e.what());
        }
    }();
    std::vector<float> mono = wav.mono();
    const double seconds = (double) mono.size() / wav.sample_rate;
    if (seconds > max_seconds) {
        throw std::runtime_error(wav_path + " is " + std::to_string(seconds) + " s long; a reference voice is at most " +
                                 std::to_string(max_seconds) + " s");
    }
    return codec.encode(normalize_loudness(resample(std::move(mono)), codec.sample_rate(), kReferenceLufs));
}

}  // namespace irodori
