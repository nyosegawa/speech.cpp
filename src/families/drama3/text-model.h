#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"
#include "qwen35-decoder.h"

namespace drama3 {

/**
 * What the layers of a run computed on the way, by layer: the attention's or the Gated DeltaNet's output, the router's
 * logits, the experts' output with the shared expert's, and the layer's output.
 */
using TextModelStages = Qwen35Stages;

/**
 * Drama 3's text model, a Qwen3.5-MoE backbone, or a run of its layers: the layers, 256 experts of which the router takes
 * 8 with a shared expert, as Qwen35Decoder runs them, with Drama 3's inputs, a frame's main token and the codes the text
 * model sees, and its head of the main token's logits.
 */
class TextModel {
public:
    /**
     * Layers [first, last) of the text model of the model file `m`, which outlives it, and the final norm where `last`
     * is the model's last; -1 takes every layer from `first`. Only their tensors need be loaded. Where `m` holds some
     * of their tensors in the CPU's memory, `cpu` is the CPU's backend, which outlives it and computes what reads them,
     * the rest computing on `backend`.
     */
    TextModel(const ModelFile & m, ggml_backend_t backend, int first = 0, int last = -1, ggml_backend_t cpu = nullptr);

    int hidden() const { return decoder_.hidden(); }
    /** The layers of the model, whichever this one runs. */
    int layers() const { return decoder_.layers(); }
    bool attention(int l) const { return decoder_.attention(l); }
    /** The rows run so far. */
    int64_t n_past() const { return decoder_.n_past(); }

    /** Starts a sequence of at most `rows` rows: the attention layers get room for them, and every state starts at 0. */
    void start(int64_t rows) { decoder_.start(rows); }

    /** Keeps a copy of every Gated DeltaNet layer's state as it stands after n_past() rows, for rewind(). */
    void keep_states() { decoder_.keep_states(); }

    /**
     * Goes back to the row after the first `rows` rows of the sequence, where keep_states() kept the states: the
     * attention layers keep the keys and values of those rows, and the states return to the copy.
     */
    void rewind(int64_t rows) { decoder_.rewind(rows); }

    /**
     * Allocates now the memory of a sequence of `rows` rows whose prompt takes `prompt_rows`, which its largest graphs
     * need later: the attention layers' room for the rows and the buffers of the graphs of the longest prompt, so that
     * no memory taken after it, by another model, leaves such a sequence short. It ends with a sequence started.
     */
    void reserve(int64_t rows, int64_t prompt_rows);

    /**
     * The inputs of `n` frames of 1 + codebooks ids, row-major, [hidden, n]: the main token's embedding, and for a frame
     * whose main token is a semantic id, the mean of it and the embeddings of the codes the text model sees. It reads
     * token_embd and the audio decoder's ad.codebook_embd.
     */
    std::vector<float> embed(const std::vector<int32_t> & frames, int64_t n);

    /**
     * Runs `n` rows of input `x`, [hidden, n], after n_past() through the layers, and returns the last layer's output,
     * after the final norm where it is the model's last. `stages`, when given, receives what each layer computed.
     */
    std::vector<float> run(const std::vector<float> & x, int64_t n, TextModelStages * stages = nullptr) { return decoder_.run(x, n, stages); }

    /** The head's logits of `hidden`, [hidden] after the final norm, over drama3.output.token_ids. */
    std::vector<float> logits(const std::vector<float> & hidden);

    /**
     * Gated DeltaNet layer `l`'s recurrent state, [key dim, value dim, value heads] with the key channel fastest, and the
     * inputs of its convolution's last rows, [width - 1, channels] with the row fastest: what a check compares with the
     * official ones.
     */
    std::vector<float> delta_state(int l) const { return decoder_.delta_state(l); }
    std::vector<float> conv_state(int l) const { return decoder_.conv_state(l); }

private:
    /** The graph of the embeddings of `n` frames, which embed() computes. */
    ggml_tensor * build_embed(Graph & g, const std::vector<int32_t> & frames, int64_t n);

    ggml_backend_t backend_;
    const ModelFile & m_;
    Qwen35Decoder decoder_;
    int h_ = 0, codebooks_ = 0, codes_ = 0, seen_ = 0;
    int32_t semantic_begin_ = 0, semantic_end_ = 0;
};

}  // namespace drama3
