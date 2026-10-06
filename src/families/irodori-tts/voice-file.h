#pragma once

#include <string>
#include <vector>

#include "ggml-backend.h"

namespace irodori {

/** What a voice file says of the recording its latent was encoded from and of the device that encoded it. */
struct VoiceOrigin {
    double reference_seconds = 0;
    /** The recording's rate before it was resampled. */
    int reference_sample_rate = 0;
    /** "cpu", "gpu" or "igpu". */
    std::string device_kind;
};

/** The kind of device `backend` runs on, as a voice file names it. */
std::string device_kind(ggml_backend_t backend);

/**
 * Writes a voice file of voice_layout: the latent, row-major [frames, latent_dim], bound to the codec whose hash is
 * `codec_sha256`.
 */
void write_voice_file(const std::string & path, const std::vector<float> & latent, int latent_dim, const std::string & codec_sha256,
                      const VoiceOrigin & origin);

/**
 * Makes a voice file at `voice_path` for the model file at `model_path` from the reference recording in the WAVE file at
 * `reference_path`, as Synthesizer::load_voice() encodes a reference, reading from the model file the codec's encoder
 * alone and running it on `backend`.
 */
void make_voice_file(const std::string & model_path, const std::string & reference_path, const std::string & voice_path,
                     ggml_backend_t backend);

}  // namespace irodori
