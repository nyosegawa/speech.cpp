#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "codec.h"
#include "model-file.h"

namespace haru {

/**
 * A voice a conversation speaks with: the latent means of its reference, [latent, frames], as the codec encodes the
 * reference whole, and the index of the reference's quietest whole frame, the speaker's noise floor, or -1 where it has
 * none. A conversation draws each latent about its mean when it starts, as the official session does.
 */
struct Voice {
    std::vector<float> means;
    int64_t quietest = -1;
};

/** The general.architecture of a Haru voice file. */
constexpr const char * kVoiceArchitecture = "haru-voice";

/**
 * Layout 1 of a voice file, which make_voice_file() writes: the key haru-voice.codec_sha256, which must be the model's
 * haru.codec.sha256, haru-voice.quietest_frame, haru-voice.reference.seconds and haru-voice.reference.sample_rate, the
 * length and the rate of the recording it was made of, and the tensor latent, [latent, frames].
 */
Layout voice_layout(const ModelFile & model);

/**
 * The voice of a recording, mono or mixed to mono, at any rate: resampled to the model's rate, prepared as
 * prepare_reference() prepares it, and encoded by `codec` whole from a state of zeros, which it leaves at zeros. A
 * recording with less than haru.dialogue.reference_min_seconds of sound is refused, naming `input`.
 */
Voice voice_of_recording(const ModelFile & model, Codec & codec, const std::vector<float> & mono, int sample_rate, const std::string & input);

/** The voice a voice file holds, read with voice_layout(), or that of a WAVE file. */
Voice load_voice(const ModelFile & model, Codec & codec, ggml_backend_t backend, const std::string & path);

/** Writes the voice of the WAVE file `wav_path` to a voice file at `voice_path`, for the model at `model_path`. */
void make_voice_file(const std::string & model_path, const std::string & wav_path, const std::string & voice_path, ggml_backend_t backend);

}  // namespace haru
