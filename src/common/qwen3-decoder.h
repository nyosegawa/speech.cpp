#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"

/** The shape of one Qwen3 decoder stack. */
struct Qwen3Shape {
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
 * The shape that the keys `<prefix>.hidden_size`, `.intermediate_size`, `.num_hidden_layers`, `.num_attention_heads`,
 * `.num_key_value_heads`, `.head_dim`, `.rms_norm_eps` and `.rope_theta` of `m` give, named as Qwen3's configuration
 * names them.
 */
Qwen3Shape read_qwen3_shape(const ModelFile & m, const std::string & prefix);

/** Builds rows [from, from + rows) of a run's input, [hidden, rows], in the graph `g` that computes them. */
using Qwen3Rows = std::function<ggml_tensor *(Graph & g, int64_t from, int64_t rows)>;

/**
 * A Qwen3 decoder stack of a model file: in each layer, RMSNorm, attention with grouped key/value heads, RMSNorm of
 * each head's queries and keys and rotate-half RoPE, and RMSNorm and a SwiGLU feed-forward; and RMSNorm at the end.
 * Its tensors are `<tensors>.blk.<layer>.` attn_norm, attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_o,
 * ffn_norm, ffn_gate, ffn_up and ffn_down, and `<tensors>.norm`.
 *
 * It runs one sequence at a time and keeps the keys and values of the sequence's positions in a cache that grows with
 * the sequence. The caller builds a run's input rows, embeddings it computed or rows the graph looks up from ids,
 * and names the output matrix whose logits the run's last row gives: a head of the model's own, or the input
 * embeddings where the model ties its head to them.
 */
class Qwen3Decoder {
public:
    /**
     * The stack `tensors` of `m`, which outlives it, of shape `shape`, for sequences of at most `max_positions`
     * positions, with a cache of `cache_type`.
     */
    Qwen3Decoder(const ModelFile & m, ggml_backend_t backend, std::string tensors, const Qwen3Shape & shape,
                 int64_t max_positions, ggml_type cache_type);
    ~Qwen3Decoder();
    Qwen3Decoder(const Qwen3Decoder &) = delete;
    Qwen3Decoder & operator=(const Qwen3Decoder &) = delete;

    const Qwen3Shape & shape() const { return shape_; }
    int64_t max_positions() const { return max_positions_; }
    /** The positions of the current sequence that the cache holds. */
    int64_t n_past() const { return n_past_; }
    /** The positions the cache holds before it grows. */
    int64_t cache_capacity() const;

    /**
     * Starts a sequence of at most `positions` positions whose first run has `first` rows. The cache gets room for
     * those rows and one step of its growth more, up to `positions`, and a cache that a longer sequence grew is given
     * back. More positions than max_positions() throw.
     */
    void start(int64_t positions, int64_t first);

    /**
     * Runs `n` rows, which `rows` builds, through the stack after the positions the sequence has so far, growing the
     * cache when it has no room for them. With a `head`, an output matrix [hidden, vocabulary], the last row's hidden
     * state after the final norm and its logits under `head` are read for hidden() and logits().
     */
    void run(int64_t n, const Qwen3Rows & rows, ggml_tensor * head = nullptr);

    const std::vector<float> & hidden() const { return hidden_; }
    const std::vector<float> & logits() const { return logits_; }

private:
    struct Cache;

    /** Builds the layers over the `rows` rows of `x` that follow the n_past() positions, writing their keys and values. */
    ggml_tensor * layers(Graph & g, ggml_tensor * x, int64_t rows);
    /** Moves the cache to one with room for `positions` positions, keeping the n_past() it holds. */
    void resize_cache(int64_t positions);

    ggml_backend_t backend_;
    const ModelFile & m_;
    const std::string tensors_;
    const Qwen3Shape shape_;
    const int64_t max_positions_;
    const ggml_type cache_type_;

    std::unique_ptr<Cache> cache_;
    ggml_gallocr_t allocr_ = nullptr;
    int64_t positions_ = 0;
    int64_t n_past_ = 0;
    std::vector<float> hidden_, logits_;
};
