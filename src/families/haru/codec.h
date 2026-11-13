#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"

namespace haru {

/** What the codec computed on the way, by stage name, each in ggml order ([channels, time] with the channel fastest). */
using CodecStages = std::map<std::string, std::vector<float>>;

/**
 * The VibeVoice acoustic codec of Haru: its encoder turns each frame of audio, 3200 samples at 24 kHz, into the mean of
 * a latent of 64 values, and its decoder turns latents back into audio. Both are causal stacks of convolutions and
 * blocks of a depthwise convolution and an MLP, and every convolution starts from the inputs of its last calls, which
 * the codec keeps as its state: encoding or decoding a few frames at a time gives what encoding or decoding them all at
 * once gives, but for the rounding of other shapes of products. A codec starts with a state of zeros, as the official
 * one does.
 */
class Codec {
public:
    /** The codec of the model file `m`, which outlives it. */
    Codec(const ModelFile & m, ggml_backend_t backend);
    ~Codec();
    Codec(const Codec &) = delete;
    Codec & operator=(const Codec &) = delete;

    /** The samples of a frame, which one latent encodes. */
    int64_t frame_samples() const { return frame_samples_; }
    int64_t latent_dim() const { return latent_; }

    /**
     * The latent means of `audio`, mono at 24 kHz, continuing the encoder's state, as [latent_dim, frames] with the
     * value fastest: one frame for each frame_samples() samples, the last one padded as the official encoder pads a clip
     * of whole frames and a part, each strided convolution taking zeros past its input's end. `stages`, when given,
     * receives the output of each stage of channels (encoder_0 to encoder_6) and the means (encoder_head).
     */
    std::vector<float> encode(const std::vector<float> & audio, CodecStages * stages = nullptr);

    /**
     * The audio of `latents`, [latent_dim, frames] with the value fastest, continuing the decoder's state:
     * frames * frame_samples() samples. `stages`, when given, receives the output of each stage of channels
     * (decoder_0 to decoder_6) and the audio (decoder_head).
     */
    std::vector<float> decode(const std::vector<float> & latents, int64_t frames, CodecStages * stages = nullptr);

    /** Sets the encoder's state to zeros, as before a conversation's first frame. */
    void reset_encoder();
    /** Sets the decoder's state to zeros. */
    void reset_decoder();

private:
    struct States;
    struct Net;

    const ModelFile & m_;
    ggml_backend_t backend_;
    ggml_gallocr_t allocr_ = nullptr;
    int64_t frame_samples_ = 0, latent_ = 0, base_ = 0, width_ = 0;
    float eps_ = 0.0f;
    std::vector<int32_t> ratios_, encoder_depths_, decoder_depths_;
    States * encoder_state_ = nullptr;
    States * decoder_state_ = nullptr;
};

}  // namespace haru
