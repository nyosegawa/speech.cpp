#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "codec.h"
#include "detokenizer.h"
#include "head.h"
#include "model-file.h"
#include "prompt.h"
#include "qwen2-tokenizer.h"
#include "qwen35-decoder.h"
#include "sampler.h"

namespace haru {

/**
 * Where a conversation's random draws come from, in the order it takes them: when it starts, the scale and the noise of
 * the reference's latents; for each frame it gives, the head's noise and then the text token; for each frame of the
 * user's, the scale and the noise of its latent. The official session takes the token before the head's noise from one
 * torch generator, which nothing outside torch reproduces, so a conversation draws from its own source.
 */
class Randomness {
public:
    virtual ~Randomness() = default;
    /** `n` draws of the standard normal distribution. */
    virtual void normal(float * out, size_t n) = 0;
    /** A token of `c`, drawn by its weights. */
    virtual int32_t token(const Candidates & c) = 0;
};

/** Draws from a 64-bit Mersenne Twister of a seed: normals by Box and Muller's transform of uniform(), tokens by pick(). */
class SeededRandomness : public Randomness {
public:
    explicit SeededRandomness(uint64_t seed) : rng_(seed) {}
    void normal(float * out, size_t n) override;
    int32_t token(const Candidates & c) override;

private:
    std::mt19937_64 rng_;
};

/** A request's settings of a conversation, the file's where it sets none. */
struct DialogueSettings {
    Sampling sampling;
    float guidance = 0.0f;
    float eop_bias = 0.0f;
    bool speak_first = false;
    int64_t max_frames = 0;
};

DialogueSettings file_dialogue_settings(const ModelFile & m);

/** A frame the model gives: 133 ms of its audio, clamped to [-1, 1] as the official session passes it, and its text. */
struct DialogueFrame {
    std::vector<float> audio;
    int32_t token = 0;
    std::string text;
};

/** What a conversation computed on the way, for a check: each in ggml order, appended frame by frame. */
struct DialogueStages {
    std::vector<int32_t> prompt_tokens;
    std::vector<float> prompt_model, prompt_user, prompt_model_mask, prompt_user_mask;
    /** The input rows: the prompt's, then each step's, [hidden, rows]. */
    std::vector<float> embeddings;
    /** The hidden state each frame comes from, [hidden, frames]. */
    std::vector<float> hidden;
    std::vector<float> logits, silent_logits, head_latents, latents, codec_audio;
    std::vector<float> user_means, user_latents;
};

/** The seconds a conversation's frames have spent in each part, summed from its start. */
struct DialogueTimes {
    /** The codec's encoding of the user's frames. */
    double encode = 0;
    /** The graphs of the language model's rows, the logits and the head. */
    double model = 0;
    /** The shaping of the logits and the draw of the tokens, on the host. */
    double sampling = 0;
    /** The codec's decoding of the model's frames. */
    double decode = 0;
};

/**
 * One full-duplex conversation with Haru, as haru.session's DuplexSession holds it: start() runs the prompt, the card
 * and the voice reference, and gives the model's first frame; each step() takes 133 ms of the user's audio and gives the
 * model's next frame, so that the model stays one frame ahead. Each frame's text token and audio latent come from the
 * hidden state of the row before it: the token from the logits the session shapes and draws from, the latent from the
 * MeanFlow head, decoded to audio at once. The codec, the language model's keys, values and states, DRY and the text's
 * stream carry over from frame to frame.
 */
class Dialogue {
public:
    /** A conversation of the model file `m`, which outlives it, its tensors on `backend`. */
    Dialogue(const ModelFile & m, ggml_backend_t backend);
    ~Dialogue();
    Dialogue(const Dialogue &) = delete;
    Dialogue & operator=(const Dialogue &) = delete;

    int64_t frame_samples() const { return codec_.frame_samples(); }
    const Qwen2Tokenizer & tokenizer() const { return *tokenizer_; }
    const AddedVocabulary & added() const { return added_; }

    /**
     * The latent means of a reference, mono at 24 kHz as prepare_reference() leaves it, [latent, frames]: encoded whole
     * from a codec state of zeros, as the official encode() does.
     */
    std::vector<float> encode_reference(const std::vector<float> & reference);

    /**
     * Starts a conversation of the card `card` (its ids, card_ids()), a reference of the latent means `reference`,
     * [latent, frames] (none for no reference), whose quietest whole frame is `quietest` (-1 for none), and gives the
     * model's first frame. Every state starts again.
     */
    DialogueFrame start(const std::vector<int32_t> & card, const std::vector<float> & reference, int64_t quietest, const DialogueSettings & s,
                        Randomness & r, DialogueStages * stages = nullptr);

    /** Takes the user's next frame, frame_samples() samples at 24 kHz, and gives the model's next frame. */
    DialogueFrame step(const std::vector<float> & user, Randomness & r, DialogueStages * stages = nullptr);

    /** Whether the conversation has given its max_frames frames. */
    bool done() const { return frames_ >= settings_.max_frames; }
    /** The time spent in each part of the frames since the conversation started, its first frame's included. */
    const DialogueTimes & times() const { return times_; }

private:
    /** The latent of digital silence: the last of haru.dialogue.silence_frames frames of zeros, encoded at once. */
    const std::vector<float> & silence();
    /** A latent drawn about `mean`, [latent, n]: mean + s x noise_std x noise, one s for all of them. */
    std::vector<float> sample(const std::vector<float> & mean, Randomness & r);
    /** The codec's latents normalized for the projections, as scale_latents() gives them. */
    std::vector<float> scaled(const std::vector<float> & latents) const;
    /** Builds the input rows of `tokens` with the model's and the user's latents, scaled, each masked where a mask is given. */
    ggml_tensor * build_embed(Graph & g, const std::vector<int32_t> & tokens, const std::vector<float> & model, const std::vector<float> & user,
                              const std::vector<float> * model_mask, const std::vector<float> * user_mask);
    ggml_tensor * project(Graph & g, const std::string & side, ggml_tensor * latents);
    /** Gives the frame of the hidden state `hidden`, [hidden, 1], computing the outputs of `g` with it. */
    DialogueFrame emit(Graph & g, ggml_tensor * hidden, Randomness & r, DialogueStages * stages);

    const ModelFile & m_;
    ggml_backend_t backend_;
    std::unique_ptr<Qwen2Tokenizer> tokenizer_;
    AddedVocabulary added_;
    Codec codec_;
    Head head_;
    Qwen35Decoder decoder_;
    TextIds ids_;
    DryConfig dry_config_;
    std::unique_ptr<Dry> dry_;
    std::unique_ptr<StreamDetokenizer> detokenizer_;
    DialogueSettings settings_;
    int64_t latent_ = 0, hidden_ = 0, frames_ = 0;
    float noise_std_ = 0.0f, norm_eps_ = 0.0f;
    bool must_speak_ = false;
    std::vector<float> silence_;
    /** The last frame's token and latent, which the next step's row takes. */
    int32_t token_ = 0;
    std::vector<float> latent_last_;
    DialogueTimes times_;
};

}  // namespace haru
