#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "keys-values.h"
#include "model-file.h"

/**
 * What the layers of a run computed on the way, by layer, each [width, rows] with the channel fastest: the attention's
 * or the Gated DeltaNet's output, the router's logits (none for dense layers), the feed-forward output, the experts'
 * with the shared expert's for a mixture of experts, and the layer's output.
 */
struct Qwen35Stages {
    std::vector<std::vector<float>> mixer, router, ffn, layers;
};

/**
 * The layers of a Qwen3.5 text model, or a run of them: in each, RMSNorm and a Gated DeltaNet or gated attention, then
 * RMSNorm and a feed-forward layer, a dense SwiGLU or, where the file has `<prefix>num_experts`, experts of which a
 * router takes some, with a shared expert whose sigmoid gate scales it; after the last, the final norm. It reads the
 * keys `<prefix>...` of the text model and the tensors blk.N.* and output_norm, and runs one sequence at a time,
 * keeping for each attention layer the keys and values of its rows, and for each Gated DeltaNet layer its recurrent state
 * and the inputs of its convolution's last rows. Drama 3's backbone is a mixture of experts and Haru's is dense.
 */
class Qwen35Decoder {
public:
    /**
     * Layers [first, last) of the text model of the model file `m`, which outlives it, and the final norm where `last`
     * is the model's last; -1 takes every layer from `first`. Only their tensors need be loaded. Where `m` holds some
     * of their tensors in the CPU's memory, `cpu` is the CPU's backend, which outlives it and computes what reads them,
     * the rest computing on `backend`.
     */
    Qwen35Decoder(const ModelFile & m, const std::string & prefix, ggml_backend_t backend, int first = 0, int last = -1,
                  ggml_backend_t cpu = nullptr);
    ~Qwen35Decoder();
    Qwen35Decoder(const Qwen35Decoder &) = delete;
    Qwen35Decoder & operator=(const Qwen35Decoder &) = delete;

    int hidden() const { return h_; }
    /** The layers of the model, whichever this one runs. */
    int layers() const { return (int) types_.size(); }
    bool attention(int l) const { return types_[(size_t) l]; }
    /** The rows run so far. */
    int64_t n_past() const { return n_past_; }
    /** The allocator of the layers' graphs on the model's device, which a caller's own graphs of the model share. */
    ggml_gallocr_t allocator() const { return allocr_; }

    /** Starts a sequence of at most `rows` rows: the attention layers get room for them, and every state starts at 0. */
    void start(int64_t rows);

    /** Keeps a copy of every Gated DeltaNet layer's state as it stands after n_past() rows, for rewind(). */
    void keep_states();

    /**
     * Goes back to the row after the first `rows` rows of the sequence, where keep_states() kept the states: the
     * attention layers keep the keys and values of those rows, and the states return to the copy.
     */
    void rewind(int64_t rows);

    /**
     * Allocates now the memory of a sequence of `rows` rows whose prompt takes `prompt_rows`, which its largest graphs
     * need later: the attention layers' room for the rows, `also`, a caller's graph computed with allocator(), and a
     * block of the longest prompt's rows, so that no memory taken after it, by another model, leaves such a sequence
     * short. It ends with a sequence started.
     */
    void reserve(int64_t rows, int64_t prompt_rows, ggml_cgraph * also = nullptr);

    /**
     * Runs `n` rows of input `x`, [hidden, n], after n_past() through the layers, and returns the last layer's output,
     * after the final norm where it is the model's last. `stages`, when given, receives what each layer computed.
     */
    std::vector<float> run(const std::vector<float> & x, int64_t n, Qwen35Stages * stages = nullptr);

    /**
     * Builds in `g` the layers on the rows of `x`, [hidden, rows], after n_past(), and the final norm where the run ends
     * with the model's last layer, for a caller that computes them in a graph of its own and then calls advance(). `kept`,
     * when given, receives each layer's mixer, router (null for a dense layer), feed-forward and output.
     */
    ggml_tensor * build_rows(Graph & g, ggml_tensor * x, std::vector<std::vector<ggml_tensor *>> * kept = nullptr);

    /** Counts `n` rows run by a caller's graph of build_rows(). */
    void advance(int64_t n);

    /**
     * Gated DeltaNet layer `l`'s recurrent state, [key dim, value dim, value heads] with the key channel fastest, and the
     * inputs of its convolution's last rows, [width - 1, channels] with the row fastest: what a check compares with the
     * official ones.
     */
    std::vector<float> delta_state(int l) const;
    std::vector<float> conv_state(int l) const;

private:
    struct States;

    ggml_tensor * attention_mixer(Graph & g, int l, ggml_tensor * x, int64_t first, ggml_tensor * pos, ggml_tensor * mask);
    ggml_tensor * delta_mixer(Graph & g, int l, ggml_tensor * x);
    /** The experts and the shared expert on `x`, and the router's logits in `router`. */
    ggml_tensor * experts(Graph & g, int l, ggml_tensor * x, ggml_tensor ** router);
    ggml_tensor * dense(Graph & g, int l, ggml_tensor * x);

    ggml_backend_t backend_;
    const ModelFile & m_;
    ggml_gallocr_t allocr_ = nullptr;
    /** The scheduler of the layers' graphs over `backend` and the CPU, where the CPU holds some of their tensors. */
    ggml_backend_sched_t sched_ = nullptr;
    int first_ = 0, last_ = 0;
    std::vector<bool> types_;
    int h_ = 0, heads_ = 0, kv_heads_ = 0, head_dim_ = 0, rotary_dim_ = 0;
    int key_heads_ = 0, value_heads_ = 0, key_dim_ = 0, value_dim_ = 0, conv_width_ = 0;
    /** The experts of a mixture of experts and those the router takes, 0 for dense layers. */
    int experts_ = 0, used_ = 0;
    bool norm_topk_ = false;
    float eps_ = 0, rope_theta_ = 0;
    std::unique_ptr<KeysValues> keys_values_;
    std::unique_ptr<States> states_;
    /**
     * The copy keep_states() makes of states_, allocated with them so that the model holds its memory from its load on,
     * and the rows run when it was made.
     */
    std::unique_ptr<States> kept_;
    int64_t kept_rows_ = -1;
    /** The index of each layer among the attention layers or the Gated DeltaNet layers of the run. */
    std::vector<int> slot_;
    int64_t capacity_ = 0, n_past_ = 0;
};
