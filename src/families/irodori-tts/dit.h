#pragma once

#include <vector>

#include "graph.h"
#include "model-file.h"

namespace irodori {

/**
 * What the DiT attends to besides the latent: the text and speaker conditions, channel-first, and the noise that takes
 * the speaker condition's place in a branch of RF's guidance, of the speaker condition's shape, when the request asks
 * for it.
 */
struct Conditions {
    std::vector<float> text;
    int text_tokens = 0;
    std::vector<float> speaker;
    int speaker_tokens = 0;
    std::vector<float> speaker_noise;
};

/**
 * What one latent of the DiT's batch attends to besides itself. RF's guidance runs the latent again without the text,
 * without the speaker, or with noise in the speaker's place.
 */
struct Branch {
    enum class Speaker { Kept, Left, Noise };
    bool text = true;
    Speaker speaker = Speaker::Kept;
};

/**
 * The text-to-latent DiT of Irodori-TTS: blocks of joint attention over the latent and the conditions,
 * each modulated by the step's time through a low-rank AdaLN. A MeanFlow model also takes the step's
 * interval.
 */
class Dit {
public:
    explicit Dit(const ModelFile & m);

    bool meanflow() const { return meanflow_; }
    int latent_dim() const { return latent_dim_; }

    /**
     * The velocity [latent_dim, frames, branches] of the latent x (row-major [frames, latent_dim]) at time t, with
     * `delta` the step's interval for a MeanFlow model, for each of `branches`, which form the batch. `blocks`, when
     * given, receives each block's output and `cond` the timestep condition.
     */
    ggml_tensor * build(Graph & g, const std::vector<float> & x, int frames, float t, float delta, const Conditions & c,
                        const std::vector<Branch> & branches, std::vector<ggml_tensor *> * blocks = nullptr, ggml_tensor ** cond = nullptr) const;

private:
    const ModelFile & m_;
    bool meanflow_;
    int dim_, heads_, layers_, latent_dim_, timestep_dim_;
    float eps_, theta_;
};

/** The official get_timestep_embedding(): cos then sin of t at 256 frequencies, in float32. */
std::vector<float> timestep_embedding(float t, int dim);

}  // namespace irodori
