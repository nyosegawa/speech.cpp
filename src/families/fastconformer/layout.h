#pragma once

#include "model-file.h"

namespace fastconformer {

/**
 * Layout 1 of a FastConformer model file, which reference/fastconformer/convert.py writes: the frontend's window and
 * filterbank, the subsampling, the conformer layers, the prediction network and the joint of a NeMo checkpoint with
 * the settings of the decoding transcribe() runs and the SentencePiece pieces, every key of fastconformer.* and
 * speech.* that the family reads, and the tensors the keys call for.
 */
extern const Layout layout;

}  // namespace fastconformer
