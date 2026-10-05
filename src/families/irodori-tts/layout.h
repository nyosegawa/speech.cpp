#pragma once

#include "model-file.h"

namespace irodori {

/**
 * Layout 1 of an Irodori-TTS model file, which reference/irodori-tts/convert.py writes: the tokenizer, ModernBERT-ja
 * and its projector, the speaker encoder, the duration predictor, the DiT and the DACVAE codec, every key of
 * irodori-tts.* and speech.* that the family reads, and the tensors the keys call for.
 */
extern const Layout model_layout;

/**
 * Layout 1 of a voice file, which Synthesizer::save_voice() writes: the codec latent of a reference recording, the
 * hash of the codec that encoded it, and the recording's length and rate and the kind of device that encoded it.
 */
extern const Layout voice_layout;

/** The release whose reader first takes voice files of voice_layout, which a voice file names as speech.requires. */
constexpr const char * kVoiceLayoutRequires = "0.7.0";

}  // namespace irodori
