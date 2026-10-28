#include "placement.h"

#include <algorithm>
#include <string>
#include <vector>

namespace drama3 {

namespace {

/**
 * What the graphs and the backend allocate while a model computes, beyond its weights and its caches: the buffers of
 * the text model's, the audio decoder's and the codec's graphs at their largest, a block of 512 prompt rows attending
 * to the most rows and a window of 128 frames of the codec, the Gated DeltaNet's states and their copy at the end of a
 * voice's head of the prompt, and the backend's own. For a prompt of 8,018 rows and 327 frames of speech with the Q4_K
 * file on an Apple M5 the three buffers took 271, 275 and 497 MiB and the states 64 MiB, as their copy does; on an RTX
 * 2080 with Vulkan the backend's own took about 0.3 GB more (2026-10-15).
 */
constexpr uint64_t kComputeBytes = (uint64_t) 1536 << 20;

}  // namespace

uint64_t working_bytes(const ModelFile & m) {
    const std::string x = "drama3.text.", a = "drama3.audio_decoder.";
    const uint64_t rows = (uint64_t) m.size("drama3.generation.max_prompt_rows") + (uint64_t) m.size("drama3.generation.max_frames");
    const std::vector<std::string> types = m.str_array(x + "layer_types");
    const uint64_t attention_layers = (uint64_t) std::count(types.begin(), types.end(), "full_attention");
    const uint64_t decoder_layers = (uint64_t) m.count(a + "shared_enrichment_layers") + (uint64_t) m.count(a + "num_codebooks") - 1;
    // A key and a value of float32 for each kv head of each row.
    const uint64_t text = 2 * sizeof(float) * (uint64_t) m.size(x + "head_dim") * (uint64_t) m.size(x + "num_key_value_heads");
    const uint64_t decoder = 2 * sizeof(float) * (uint64_t) m.size(a + "head_dim") * (uint64_t) m.size(a + "n_local_heads");
    return rows * (attention_layers * text + decoder_layers * decoder) + kComputeBytes;
}

}  // namespace drama3
