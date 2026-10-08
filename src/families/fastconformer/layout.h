#pragma once

#include "model-file.h"

namespace fastconformer {

/**
 * Layout 2 of a FastConformer model file, which reference/fastconformer/convert.py writes: the frontend's window and
 * filterbank, the subsampling, the conformer layers, the prediction network and the joint of a NeMo checkpoint with
 * the settings of the decoding transcribe() runs, for RNN-T also the limit of greedy decoding, and the SentencePiece
 * pieces, every key of fastconformer.* and speech.* that the family reads, and the tensors the keys call for. A file
 * of layout 1 reads as one of layout 2, an RNN-T file of it with reazonspeech-nemo-v2's limit of 10.
 */
extern const Layout layout;

}  // namespace fastconformer
