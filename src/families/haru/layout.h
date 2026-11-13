#pragma once

#include "model-file.h"

namespace haru {

/**
 * Layout 1 of a Haru model file, which reference/haru/convert.py writes: the language model, a Qwen3.5 text model of
 * Gated DeltaNet and gated attention layers whose token embedding is its output head, the projections of the model's
 * and the user's audio latents into it, the MeanFlow head that draws a frame's audio latent from its hidden state, the
 * VibeVoice acoustic codec, the tokenizer and the character card's wording in each language; every key of haru.* and
 * speech.* that the family reads, and the tensors the keys call for.
 */
extern const Layout model_layout;

}  // namespace haru
