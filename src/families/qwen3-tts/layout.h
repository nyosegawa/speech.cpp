#pragma once

#include "model-file.h"

/**
 * Layout 1 of a Qwen3-TTS model file, which reference/qwen3-tts/convert.py writes: the talker, the code predictor,
 * the text embedding and the tokenizer of a CustomVoice checkpoint with the 12Hz codec's decoder, every key of
 * qwen3-tts.* and speech.* that the family reads, and the tensors the keys call for.
 */
extern const Layout qwen3_tts_layout;
