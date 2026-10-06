#include "reference.h"

#include <stdexcept>

#include "error.h"
#include "loudness.h"
#include "resample.h"
#include "wav.h"

namespace irodori {

ReferenceRules::ReferenceRules(const ModelFile & m)
    : max_seconds(m.f32("irodori-tts.reference.max_seconds")), lufs(m.f32("irodori-tts.reference.lufs")) {}

EncodedReference encode_reference(Codec & codec, const std::string & wav_path, const ReferenceRules & rules) {
    const Wav wav = read_wav(wav_path);
    const Resampler resample = [&] {
        try {
            return Resampler(wav.sample_rate, codec.sample_rate());
        } catch (const std::invalid_argument & e) {
            throw Error(Fault::InvalidArgument, wav_path + ": " + e.what());
        }
    }();
    std::vector<float> mono = wav.mono();
    const double seconds = (double) mono.size() / wav.sample_rate;
    if (seconds > rules.max_seconds) {
        throw Error(Fault::OutOfRange, wav_path + " is " + std::to_string(seconds) + " s long; a reference voice is at most " +
                                           std::to_string(rules.max_seconds) + " s");
    }
    return {codec.encode(normalize_loudness(resample(std::move(mono)), codec.sample_rate(), rules.lufs)), seconds, wav.sample_rate};
}

}  // namespace irodori
