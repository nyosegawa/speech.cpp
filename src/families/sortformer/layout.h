#pragma once

#include "model-file.h"

namespace sortformer {

/**
 * Layout 1 of a Sortformer model file, which reference/sortformer/convert.py writes: NeMo's frontend, the feature
 * stacking's projection, the Transformer encoder with rotary positions, encoder_proj, the upsampling convolution to
 * 10 ms frames and the speaker head, the learnable silence embedding of the speaker cache and the settings of its
 * compression, the streaming configurations of the model card by name, the defaults of the postprocessing that turns
 * the probabilities into segments, every key of sortformer.* and speech.* that the family reads, and the tensors the
 * keys call for. The file names no language, as a diarization model takes none.
 */
extern const Layout layout;

}  // namespace sortformer
