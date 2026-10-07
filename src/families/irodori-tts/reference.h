#pragma once

#include <optional>
#include <string>
#include <vector>

#include "codec.h"
#include "model-file.h"

namespace irodori {

/**
 * How the official runtime takes a reference recording: its longest length, and the loudness it is brought to, the
 * model file's unless a voice is made with another.
 */
struct ReferenceRules {
    double max_seconds = 0;
    /**
     * In LUFS; none keeps the recording's loudness and scales down one whose peak exceeds 1, as the runtime's
     * ref_normalize_db of None with ref_ensure_max.
     */
    std::optional<double> lufs;

    explicit ReferenceRules(const ModelFile & m);
};

/** The latent of a reference recording, row-major [frames, latent_dim], and what the recording was. */
struct EncodedReference {
    std::vector<float> latent;
    double seconds = 0;
    int sample_rate = 0;
};

/**
 * The latent of a reference voice in a WAVE file as the official runtime makes it: the channels averaged, resampled
 * to the codec's rate, the loudness brought to the rules', then encoded; `sample_rate` is the file's rate before
 * resampling. The runtime resamples with torchaudio's defaults and this with resample.h's filter, so a file at another
 * rate gives a latent slightly different from the runtime's. A file longer than `rules.max_seconds`, which the runtime
 * would cut, throws.
 */
EncodedReference encode_reference(Codec & codec, const std::string & wav_path, const ReferenceRules & rules);

/**
 * The latent of a voice of several reference recordings as the runtime makes one of its ref_wavs: each encoded on its
 * own, as encode_reference() encodes it, and joined in order. Latents longer together than `rules.max_seconds` allows,
 * which the runtime would cut, throw, naming `input`.
 */
std::vector<float> join_references(const std::vector<EncodedReference> & references, const Codec & codec, const ReferenceRules & rules,
                                   const char * input);

}  // namespace irodori
