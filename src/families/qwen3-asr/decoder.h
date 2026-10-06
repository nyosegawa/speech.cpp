#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "model-file.h"
#include "prompt.h"
#include "qwen3-decoder.h"

namespace qwen3_asr {

/** The ids a greedy decoding wrote, the end token included when one ended it, and whether it reached the limit. */
struct Generation {
    std::vector<int32_t> ids;
    bool limited = false;
};

/**
 * Qwen3-ASR's decoder as transformers' Qwen3ASRForConditionalGeneration runs it under generate() with the checkpoint's
 * generation_config.json: the token embeddings of a prompt with the projector's output in the rows of the audio's
 * tokens, the shared Qwen3 stack (dec.*) with its output tied to the token embeddings, and greedy decoding, the first
 * of equal logits taken as torch.argmax() takes it, until one of qwen3-asr.generation.eos_ids or
 * qwen3-asr.generation.max_new_tokens tokens.
 */
class Decoder {
public:
    /** The decoder of the model file `m`, which outlives it, computing on `backend` with a cache of `cache_type`. */
    Decoder(const ModelFile & m, ggml_backend_t backend, ggml_type cache_type);
    ~Decoder();
    Decoder(const Decoder &) = delete;
    Decoder & operator=(const Decoder &) = delete;

    /**
     * The decoder's input for `prompt`: the token embeddings of its ids with `audio`, the projector's output
     * [audio tokens, hidden] row-major, in the rows of the audio's tokens; [ids, hidden] row-major.
     */
    std::vector<float> embeddings(const PromptIds & prompt, const std::vector<float> & audio);

    /**
     * Starts a sequence with the `n` rows of `embeds`, [n, hidden] row-major, which it runs through the stack a block of
     * kQwen3BlockRows rows at a time, leaving the logits of the last row. `keep_going` hears the rows done after each
     * block, and false stops the prefill there, which then returns false.
     */
    bool prefill(const std::vector<float> & embeds, int64_t n, const std::function<bool(int64_t rows)> & keep_going);

    /** Feeds the token `id` after the sequence, leaving the logits of the next. */
    void step(int32_t id);

    /** The logits of the next token, [vocabulary]. */
    const std::vector<float> & logits() const { return stack_.logits(); }

    /**
     * Decodes greedily from the prefill's logits. `keep_going` hears the number of tokens written after each, and false
     * stops the decoding there.
     */
    Generation generate(const std::function<bool(size_t tokens)> & keep_going);

    /** The positions a sequence with a prompt of `n` rows takes: the prompt and the most tokens the model writes. */
    int64_t positions(int64_t n) const { return n + max_new_tokens_; }
    int64_t max_positions() const { return stack_.max_positions(); }
    int64_t max_new_tokens() const { return max_new_tokens_; }
    int hidden() const { return stack_.shape().hidden; }

private:
    const ModelFile & m_;
    ggml_backend_t backend_;
    Qwen3Decoder stack_;
    int64_t max_new_tokens_;
    std::vector<int32_t> eos_;
    /** The allocator of the embedding graphs; the stack has its own. */
    ggml_gallocr_t allocr_ = nullptr;
};

}  // namespace qwen3_asr
