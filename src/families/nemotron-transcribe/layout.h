#pragma once

#include "model-file.h"

namespace nemotron_transcribe {

/**
 * Layout 1 of a Nemotron 3.5 Transcribe model file, which reference/nemotron-transcribe/convert.py writes: NeMo's
 * frontend; the perception module, its ASR encoder and its speaker branch (Nemotron 3 Diarization's encoder and head
 * with its silence embedding and the settings of its speaker cache and of its streaming), the windows the parallel
 * expert encoder takes audio in, the fusion's norms and sinusoidal speaker kernel and the projection to the language
 * model; the language model's Mamba-2, attention and MoE layers with their embeddings, final norm and head, and its
 * multi-token prediction layers; the tokenizer, the prompt and the generation's settings; the CTC timestamp head with
 * its vocabulary and the settings of its alignment; every key of nemotron-transcribe.* and speech.* that the family
 * reads; and the tensors the keys call for.
 */
extern const Layout layout;

}  // namespace nemotron_transcribe
