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
 * names them. A size of 0, heads that are no multiple of the key/value heads and an odd head width, which RoPE cannot
 * turn in pairs, throw.
 */
Qwen3Shape read_qwen3_shape(const ModelFile & m, const std::string & prefix);

/**
 * Appends to `t` the tensors of the stack `tensors` of shape `s` that Qwen3Decoder reads, its matrices in one of
 * `matrix_types` and its norms in F32.
 */
void add_qwen3_tensors(std::vector<TensorSpec> & t, const std::string & tensors, const Qwen3Shape & s,
                       const std::vector<ggml_type> & matrix_types);

/** Builds rows [from, from + rows) of a run's input, [hidden, rows], in the graph `g` of the block that holds them. */
using Qwen3Rows = std::function<ggml_tensor *(Graph & g, int64_t from, int64_t rows)>;

/**
 * The rows one graph of a run computes at most. Each row of a block holds a score for every position before it, so a
 * block's scores take block rows × positions × heads × 4 bytes, against the square of the positions in one graph: for
 * the longest prompt of Qwen3-TTS's talker, 24,576 rows, 0.8 GB where one graph needs 39 GB. On an Apple M5, blocks of
 * 128 to 1024 rows prefilled 5,137 rows equally fast, in 7.0 s with the 1.7B talker on Metal and in 39 to 42 s with
 * the 0.6B on the CPU, both in Q8_0, against 9.7 s and 86 to 99 s in one graph (2026-10-06). ggml_flash_attn_ext would
 * hold no scores at all, but it reads the values a row per position, where the cache keeps them transposed for the
 * steps.
 */
constexpr int64_t kQwen3BlockRows = 512;

/**
 * A Qwen3 decoder stack of a model file: in each layer, RMSNorm, attention with grouped key/value heads, RMSNorm of
 * each head's queries and keys and rotate-half RoPE, and RMSNorm and a SwiGLU feed-forward; and RMSNorm at the end.
 * Its tensors are `<tensors>.blk.<layer>.` attn_norm, attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_o,
 * ffn_norm, ffn_gate, ffn_up and ffn_down, and `<tensors>.norm`.
 *
 * It runs one sequence at a time and keeps the keys and values of the sequence's positions in a cache that grows with
 * the sequence. The rows of a run go through in blocks, each block attending to the positions before it and causally
 * to its own rows, so that the memory of a long run grows with its rows rather than with their square. The caller
 * builds a run's input rows, embeddings it computed or rows the graph looks up from ids, and names the output matrix
 * whose logits the run's last row gives: a head of the model's own, or the input embeddings where the model ties its
 * head to them.
 */
class Qwen3Decoder {
public:
    /**
     * The stack `tensors` of `m`, which outlives it, of shape `shape`, for sequences of at most `max_positions`
     * positions, with a cache of `cache_type`, running blocks of at most `block_rows` rows.
     */
    Qwen3Decoder(const ModelFile & m, ggml_backend_t backend, std::string tensors, const Qwen3Shape & shape,
                 int64_t max_positions, ggml_type cache_type, int64_t block_rows = kQwen3BlockRows);
    ~Qwen3Decoder();
    Qwen3Decoder(const Qwen3Decoder &) = delete;
    Qwen3Decoder & operator=(const Qwen3Decoder &) = delete;

    const Qwen3Shape & shape() const { return shape_; }
    int64_t max_positions() const { return max_positions_; }
    /** The positions of the current sequence that the cache holds. */
    int64_t n_past() const { return n_past_; }
    /** The positions the cache holds before it grows. */
    int64_t cache_capacity() const;
    /** The bytes the device holds for computing a graph: what the largest graph so far took. */
    size_t graph_bytes() const;

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

    /**
     * The keys and the values the cache holds for layer `layer` at the n_past() positions, as float32 a row per
     * position, [n_past, kv heads * head dim] each: what a check compares row by row.
     */
    void read_cache(int layer, std::vector<float> & keys, std::vector<float> & values) const;

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
    const int64_t block_rows_;

    std::unique_ptr<Cache> cache_;
    ggml_gallocr_t allocr_ = nullptr;
    int64_t positions_ = 0;
    int64_t n_past_ = 0;
    std::vector<float> hidden_, logits_;
};
