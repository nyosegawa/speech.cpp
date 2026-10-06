#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "model-file.h"

/** The shape of one Qwen3 decoder stack (the talker's or the code predictor's). */
struct DecoderShape {
    int hidden = 0;
    int ffn = 0;
    int n_layer = 0;
    int n_head = 0;
    int n_kv_head = 0;
    int head_dim = 0;
    float rms_eps = 0;
    float rope_theta = 0;
};

/**
 * The talker and its code predictor. The talker reads one embedding per frame and predicts the frame's
 * first code; the code predictor, conditioned on the talker's last hidden state, predicts the other
 * fifteen one by one. Both keep a key/value cache: the talker's spans the utterance and grows with it, the code
 * predictor's is rebuilt every frame.
 */
class Talker {
public:
    /** The talker of the model file `m`, which outlives it. */
    Talker(const ModelFile & m, ggml_backend_t backend);
    ~Talker();

    const ModelFile & model() const { return m_; }
    int num_code_groups() const { return n_groups_; }
    int hidden() const { return talker_.hidden; }
    int vocab() const { return vocab_; }
    int cp_vocab() const { return cp_vocab_; }
    /** The positions an utterance has at most, its prompt's and its frames' together. */
    int max_positions() const { return max_positions_; }

    /** Projected text embeddings of `ids`, row-major [ids.size(), hidden]. */
    std::vector<float> text_embeddings(const std::vector<int32_t> & ids);
    /** Talker codec embeddings of `ids`, row-major [ids.size(), hidden]. */
    std::vector<float> codec_embeddings(const std::vector<int32_t> & ids);

    /**
     * Starts an utterance of at most `positions` positions, the prompt's and the frames': gives the cache room for the
     * prompt and its first frames, and runs `embeds` ([n, hidden], row-major) through the talker.
     */
    void prefill(const std::vector<float> & embeds, int n, int64_t positions);
    /** Feeds one frame: the sum of its 16 code embeddings plus `extra` ([hidden]). */
    void step(const int32_t * codes, const std::vector<float> & extra);

    /** The codec head's logits for the next frame's first code. */
    const std::vector<float> & logits() const { return logits_; }
    /** The talker's last hidden state after the final norm, which conditions the code predictor. */
    const std::vector<float> & hidden_state() const { return hidden_; }

    /** Code predictor: starts a frame from the talker's hidden state and the first code, and returns the logits of code 1. */
    const std::vector<float> & cp_begin(int32_t code0);
    /** Feeds code `group` (1..14) and returns the logits of code `group + 1`. */
    const std::vector<float> & cp_next(int group, int32_t code);

    int64_t n_past() const { return n_past_; }
    /** The positions the talker's cache holds before it grows. */
    int64_t cache_capacity() const;

private:
    struct Cache;

    ggml_tensor * run_stack(struct GraphCtx & g, const std::string & prefix, const DecoderShape & s, ggml_tensor * x,
                            ggml_tensor * pos, ggml_tensor * mask, Cache & cache, int64_t n_past, int64_t n_tokens);
    void run_talker(const std::vector<float> * embeds, const int32_t * codes, const std::vector<float> * extra, int64_t n);
    const std::vector<float> & run_cp(int32_t code, int group);
    /** Moves the talker's cache to one with room for `positions` positions, keeping the n_past() it holds. */
    void resize_cache(int64_t positions);

    ggml_backend_t backend_;
    const ModelFile & m_;
    DecoderShape talker_, cp_;
    int n_groups_ = 0;
    int vocab_ = 0;
    int cp_vocab_ = 0;
    int max_positions_ = 0;
    bool cp_projected_ = false;

    std::unique_ptr<Cache> cache_, cp_cache_;
    ggml_gallocr_t allocr_ = nullptr;

    int64_t positions_ = 0;
    int64_t n_past_ = 0;
    int64_t cp_past_ = 0;
    std::vector<float> logits_, hidden_, cp_logits_;
};
