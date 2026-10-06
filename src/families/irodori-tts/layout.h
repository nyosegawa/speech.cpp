#pragma once

#include "model-file.h"

namespace irodori {

/**
 * Layout 1 of an Irodori-TTS model file, which reference/irodori-tts/convert.py writes: the tokenizer, ModernBERT-ja
 * and its projector, the speaker encoder, the duration predictor, the DiT and the DACVAE codec, every key of
 * irodori-tts.* and speech.* that the family reads, and the tensors the keys call for.
 */
extern const Layout model_layout;

/** The general.architecture and the speech.layout of a voice file, which write_voice_file() writes. */
constexpr const char * kVoiceArchitecture = "irodori-tts-voice";
constexpr uint32_t kVoiceLayout = 1;

/** The release whose reader first takes voice files of kVoiceLayout, which a voice file names as speech.requires. */
constexpr const char * kVoiceLayoutRequires = "0.7.0";

/**
 * Layout 1 of a voice file for the model file `model`, which outlives it: the codec latent of a reference recording,
 * F32 [latent_dim, frames], the hash of the codec that encoded it, and the recording's length and rate and the kind of
 * device that encoded it. A voice file of a codec other than the model's is refused as the caller's mistake, before
 * its latent is checked.
 */
Layout voice_layout(const ModelFile & model);

}  // namespace irodori
