#include "reference.h"

#include <cmath>
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
    std::vector<float> audio = resample(std::move(mono));
    audio = rules.lufs ? normalize_loudness(audio, codec.sample_rate(), *rules.lufs) : bound_peak(std::move(audio));
    return {codec.encode(audio), seconds, wav.sample_rate};
}

std::vector<float> join_references(const std::vector<EncodedReference> & references, const Codec & codec, const ReferenceRules & rules,
                                   const char * input) {
    std::vector<float> latent;
    for (const EncodedReference & r : references) latent.insert(latent.end(), r.latent.begin(), r.latent.end());
    // The runtime cuts the joined latent at the frames that hold max_ref_seconds.
    const int64_t most = (int64_t) std::ceil(rules.max_seconds * codec.sample_rate() / codec.hop());
    const int64_t frames = (int64_t) latent.size() / codec.latent_dim();
    if (frames > most) {
        throw Error(Fault::OutOfRange, "the references are " + std::to_string(frames) + " frames of the codec together and a voice takes at most " +
                                           std::to_string(most) + ", " + std::to_string(rules.max_seconds) + " s; give shorter references",
                    input);
    }
    return latent;
}

}  // namespace irodori
