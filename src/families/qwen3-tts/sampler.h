#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "model-file.h"

/** Sampling settings of one stack; the defaults leave the distribution as it is. */
struct SamplingParams {
    bool greedy = false;
    float temperature = 1.0f;
    /** 0 keeps every token. */
    int top_k = 0;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
};

/**
 * How the official generate() samples a model, read from its file: each stack's settings, the frames made before the
 * end of speech may come and at most, and the ids at the end of the talker's vocabulary that are never sampled, the
 * end of speech excepted.
 */
struct Generation {
    SamplingParams talker, code_predictor;
    int min_frames = 0, max_frames = 0;
    int suppressed_tokens = 0;

    explicit Generation(const ModelFile & m);

    /**
     * The talker's ids that the frame after `frames` frames may not take: the suppressed ones, and the end of speech
     * before min_frames.
     */
    std::vector<bool> banned(int vocab, int32_t end_of_speech, int frames) const;
};

/**
 * Picks a token the way transformers' generate() does: the repetition penalty on the tokens in
 * `history`, the tokens in `banned` removed, then argmax, or temperature, top-k and top-p before a
 * draw.
 */
int32_t sample(std::vector<float> logits, const SamplingParams & p, const std::vector<int32_t> & history,
               const std::vector<bool> & banned, std::mt19937_64 & rng);
