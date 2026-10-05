#pragma once

#include <string>
#include <vector>

#include "codec.h"
#include "model-file.h"

namespace irodori {

/** How the official runtime takes a reference recording, from the model file: its longest length and the loudness it is brought to. */
struct ReferenceRules {
    double max_seconds = 0;
    /** In LUFS. */
    double lufs = 0;

    explicit ReferenceRules(const ModelFile & m);
};

/** The latent of a reference recording, row-major [frames, latent_dim], and what the recording was. */
struct EncodedReference {
    std::vector<float> latent;
    double seconds = 0;
    int sample_rate = 0;
};

/**
 * The latent of a reference voice in a WAVE file as the official runtime makes it: the channels averaged, the
 * loudness normalized, then encoded. The file must be at the codec's sample rate and at most `rules.max_seconds`
 * long; the runtime would resample or cut it, and this throws.
 */
EncodedReference encode_reference(Codec & codec, const std::string & wav_path, const ReferenceRules & rules);

}  // namespace irodori
