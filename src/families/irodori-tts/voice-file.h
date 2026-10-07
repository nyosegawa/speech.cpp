#pragma once

#include <optional>
#include <string>
#include <vector>

#include "ggml-backend.h"

namespace irodori {

/** One recording a voice's latent was encoded from: its length, and its rate before it was resampled. */
struct Recording {
    double seconds = 0;
    int sample_rate = 0;
};

/** What a voice file says of the recordings its latent was encoded from and of the device that encoded them. */
struct VoiceOrigin {
    std::vector<Recording> recordings;
    /** The loudness in LUFS each recording was brought to, or none for recordings kept as they were. */
    std::optional<double> lufs;
    /** "cpu", "gpu" or "igpu". */
    std::string device_kind;
};

/**
 * How the recordings of a voice are brought before they are encoded: to the model's loudness
 * (irodori-tts.reference.lufs) when `normalize` is set without `lufs`, to `lufs` when it is given, or kept as recorded
 * without `normalize`, as the runtime's ref_normalize_db of None.
 */
struct Loudness {
    bool normalize = true;
    std::optional<double> lufs;
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
 * Makes a voice file at `voice_path` for the model file at `model_path` from the reference recordings in the WAVE files
 * at `reference_paths`, each encoded on its own at `loudness` as Synthesizer::load_voice() encodes a reference and
 * joined in order, as the runtime joins its ref_wavs, reading from the model file the codec's encoder alone and running
 * it on `backend`.
 */
void make_voice_file(const std::string & model_path, const std::vector<std::string> & reference_paths, const Loudness & loudness,
                     const std::string & voice_path, ggml_backend_t backend);

}  // namespace irodori
