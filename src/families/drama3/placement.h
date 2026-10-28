#pragma once

#include <cstdint>

#include "model-file.h"

namespace drama3 {

/**
 * The memory a device needs beside the weights it holds to serve every request the model file `m` takes: the keys and
 * values of the text model's attention layers and of the audio decoder for the most rows a chunk's prompt and speech
 * reach, in float32 as the caches hold them, and what the graphs and the backend allocate while they compute.
 */
uint64_t working_bytes(const ModelFile & m);

}  // namespace drama3
