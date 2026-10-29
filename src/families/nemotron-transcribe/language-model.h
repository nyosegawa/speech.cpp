#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "keys-values.h"
#include "model-file.h"

namespace nemotron_transcribe {

/**
 * What the layers of a run computed on the way, by layer, each [width, rows] with the channel fastest: the router's
 * logits of a MoE layer, empty for the others, and each layer's output.
 */
struct LanguageModelStages {
    std::vector<std::vector<float>> router, layers;
};

/**
 * Nemotron 3.5 Lightning, the language model of Nemotron 3.5 Transcribe, as Transformers' NemotronHForCausalLM computes
 * it, or a run of its layers: in each, RMSNorm and a Mamba-2 mixer, a grouped-query attention without positions, or
 * experts of which a sigmoid router takes some, with a shared expert, each with ReLU² between its projections; after the
 * last, RMSNorm and the head's logits. It runs one sequence at a time and keeps, for each attention layer, the keys and
 * values of its rows, and for each Mamba layer its recurrent state and the inputs of its convolution's last rows.
 */
class LanguageModel {
public:
    /**
     * Layers [first, last) of the language model of the model file `m`, which outlives it; -1 takes every layer from
     * `first`. Only their tensors need be loaded, and the final norm's and the head's for logits(). Where `m` holds some of
     * their tensors in the CPU's memory, `cpu` is the CPU's backend, which outlives it and computes what reads them, the
     * rest computing on `backend`.
     */
    LanguageModel(const ModelFile & m, ggml_backend_t backend, int first = 0, int last = -1, ggml_backend_t cpu = nullptr);
    ~LanguageModel();
    LanguageModel(const LanguageModel &) = delete;
    LanguageModel & operator=(const LanguageModel &) = delete;

    int hidden() const { return h_; }
    /** The layers of the model, whichever this one runs, and the type of each: "mamba", "attention" or "moe". */
    int layers() const { return (int) types_.size(); }
    const std::string & type(int l) const { return types_[(size_t) l]; }
    /** The rows run so far. */
    int64_t n_past() const { return n_past_; }

    /** Starts a sequence of at most `rows` rows: the attention layers get room for them, and every state starts at 0. */
    void start(int64_t rows);
    /**
     * Starts a sequence of at most `rows` rows and allocates the memory of the largest graph a run of it builds, so
     * that a run takes no memory beyond what the model holds from then on.
     */
    void reserve(int64_t rows);

    /** The embeddings of the tokens `ids`, [hidden, n]. */
    std::vector<float> embed(const std::vector<int32_t> & ids);

    /**
     * Runs `n` rows of input `x`, [hidden, n], after n_past() through the layers, and returns the last layer's output.
     * `stages`, when given, receives what each layer computed. `proceed`, when given, hears the rows run so far after
     * each block of them; false stops the run, which then returns no rows.
     */
    std::vector<float> run(const std::vector<float> & x, int64_t n, LanguageModelStages * stages = nullptr,
                           const std::function<bool(int64_t done)> & proceed = {});

    /** The head's logits of `hidden`, [hidden] the last layer's output of a row, after the final norm, over the vocabulary. */
    std::vector<float> logits(const std::vector<float> & hidden);

private:
    struct States;

    ggml_tensor * mamba(Graph & g, int l, ggml_tensor * x);
    ggml_tensor * attention(Graph & g, int l, ggml_tensor * x, ggml_tensor * mask);
    /** The experts and the shared expert on `x`, and the router's logits in `router`. */
    ggml_tensor * experts(Graph & g, int l, ggml_tensor * x, ggml_tensor ** router);
    /**
     * The graph of the layers on the rows of `x`, [hidden, rows], after n_past(); `kept`, when given, receives each
     * layer's router logits and output.
     */
    ggml_tensor * build_rows(Graph & g, ggml_tensor * x, std::vector<std::vector<ggml_tensor *>> * kept);

    ggml_backend_t backend_;
    const ModelFile & m_;
    ggml_gallocr_t allocr_ = nullptr;
    /** The scheduler of the layers' graphs over `backend` and the CPU, where the CPU holds some of their tensors. */
    ggml_backend_sched_t sched_ = nullptr;
    int first_ = 0, last_ = 0;
    std::vector<std::string> types_;
    /** Each layer's index among the layers of its kind that this model runs, -1 for the others. */
    std::vector<int> slot_;
    int h_ = 0, mamba_heads_ = 0, mamba_head_dim_ = 0, groups_ = 0, state_ = 0, conv_width_ = 0;
    int heads_ = 0, kv_heads_ = 0, head_dim_ = 0, experts_ = 0, used_ = 0;
    bool norm_topk_ = false;
    float eps_ = 0, routed_scale_ = 0;
    std::unique_ptr<States> states_;
    std::unique_ptr<KeysValues> keys_values_;
    int64_t capacity_ = 0, n_past_ = 0;
};

}  // namespace nemotron_transcribe
