#pragma once

#include <cstdint>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"

namespace haru {

/**
 * The MeanFlow head: one step of the mean flow from noise to a frame's audio latent, conditioned on the language
 * model's hidden state. Its condition is the projected hidden state plus the embeddings of four scalars, the time 0,
 * the interval embedding, the step 1 and the guidance as 1 - 1 / guidance, which a request fixes, so that the head
 * computes them once for a guidance and takes them as inputs of each frame's graph.
 */
class Head {
public:
    /** The head of the model file `m`, which outlives it, with the file's guidance. */
    Head(const ModelFile & m, ggml_backend_t backend);
    ~Head();
    Head(const Head &) = delete;
    Head & operator=(const Head &) = delete;

    /** Sets the guidance of the frames to come, which the file's guidance_min and guidance_max bound. */
    void set_guidance(float guidance);
    float guidance() const { return guidance_; }

    /**
     * Builds in `g` the normalized latents of `hidden`, [hidden, n], from `noise`, [latent, n]: noise plus the
     * velocity the head gives. `layers`, when given, receives each layer's output, and `cond` the condition after its
     * SiLU.
     */
    ggml_tensor * build(Graph & g, ggml_tensor * hidden, ggml_tensor * noise, std::vector<ggml_tensor *> * layers = nullptr,
                        ggml_tensor ** cond = nullptr) const;

    /** A normalized latent turned into the codec's, as the official unscale_latents(): x / scale - bias, in float. */
    float unscale(float x) const { return x / latent_scale_ - latent_bias_; }
    /** A codec's latent normalized for the head and the projections, as scale_latents(): (x + bias) x scale. */
    float scale(float x) const { return (x + latent_bias_) * latent_scale_; }

private:
    const ModelFile & m_;
    ggml_backend_t backend_;
    ggml_gallocr_t allocr_ = nullptr;
    int64_t hidden_ = 0, frequencies_ = 0;
    float eps_ = 0.0f, interval_ = 0.0f, frequency_base_ = 0.0f, guidance_ = 0.0f, latent_scale_ = 0.0f, latent_bias_ = 0.0f;
    /** The embeddings of the four scalars, each [hidden]. */
    std::vector<std::vector<float>> constants_;
};

/** Builds in `g` the text logits of `hidden`, [hidden, n], from the token embedding the output shares: [vocab, n]. */
ggml_tensor * build_logits(Graph & g, const ModelFile & m, ggml_tensor * hidden);

/** Builds in `g` the logits of PAD and EOP, from their rows kept in float32: [2, n]. */
ggml_tensor * build_silent_logits(Graph & g, const ModelFile & m, ggml_tensor * hidden);

}  // namespace haru
