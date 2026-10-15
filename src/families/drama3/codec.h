#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"

namespace drama3 {

/** What a codec computed on the way, by stage name, each in ggml order ([channels, time] with the channel fastest). */
using CodecStages = std::map<std::string, std::vector<float>>;

/**
 * The codec of Drama 3, Fish Audio S2-Pro's: a causal DAC at 44.1 kHz. Its encoder turns 512 samples into a vector of
 * 1,024, and its quantizer halves the rate twice, runs a transformer and draws a semantic code from 4,096 and 9
 * residual codes from 1,024 each for every 2,048 samples; its decoder turns the codes back into samples the other way.
 * Every convolution is causal, padded on the left alone, and every attention is causal within a window.
 */
class Codec {
public:
    /** The codec of the model file `m`, which outlives it. */
    Codec(const ModelFile & m, ggml_backend_t backend);
    ~Codec();
    Codec(const Codec &) = delete;
    Codec & operator=(const Codec &) = delete;

    int sample_rate() const { return sample_rate_; }
    /** The samples of a frame of codes. */
    int hop() const { return hop_; }
    /** The codes of a frame: the semantic code, then the residual ones. */
    int codebooks() const { return 1 + residual_codebooks_; }

    /**
     * The codes of `audio`, mono at sample_rate(), as [codebooks, frames] with the frame fastest, the audio padded with
     * silence on the right to whole frames as the official encoder pads it. `stages`, when given, receives the encoder
     * blocks' outputs (encoder_0 to encoder_6), the downsampling's (downsample) and the transformer's before the
     * quantizer (pre_module).
     */
    std::vector<int32_t> encode(const std::vector<float> & audio, CodecStages * stages = nullptr);

    /**
     * The codes of `latent`, [1024, frames] with the channel fastest as the transformer before the quantizer gives it
     * (encode()'s stage pre_module), as encode() gives them: the semantic code of each frame, then each residual code
     * of what the codes before it leave.
     */
    std::vector<int32_t> quantize(const std::vector<float> & latent, int64_t frames);

    /**
     * The audio of `codes`, [codebooks, frames] with the frame fastest as encode() gives them, frames * hop() samples.
     * A code past its codebook is taken as the codebook's last, as the official decoder clamps it. `stages`, when given,
     * receives the codebooks' vectors summed (quantized), the transformer's output after the quantizer (post_module),
     * the upsampling's (upsample) and the decoder blocks' outputs (decoder_0 to decoder_7).
     */
    std::vector<float> decode(const std::vector<int32_t> & codes, int64_t frames, CodecStages * stages = nullptr);

private:
    ggml_backend_t backend_;
    const ModelFile & m_;
    ggml_gallocr_t allocr_ = nullptr;
    int sample_rate_ = 0, hop_ = 0, latent_ = 0, residual_codebooks_ = 0, semantic_size_ = 0, codebook_size_ = 0;
    std::vector<int32_t> encoder_rates_, decoder_rates_, factors_;
};

}  // namespace drama3
