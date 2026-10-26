#pragma once

#include <cstdint>
#include <vector>

#include "model-file.h"

namespace sortformer {

/**
 * A streaming configuration in 80 ms frames, as SortformerModules names its settings: the rows the speaker cache keeps,
 * the rows the FIFO queue holds before it gives rows to the cache, the frames of a chunk and of the right context the
 * encoder hears after it, and the rows the queue gives the cache at once.
 */
struct Streaming {
    int spkcache_len = 0, fifo_len = 0, chunk_len = 0, chunk_right_context = 0, spkcache_update_period = 0;
};

/** The settings of the speaker cache's compression, as the model file gives them. */
struct Compression {
    int speakers = 0, silence_frames = 0;
    float score_threshold = 0, boost_latest = 0, strong_boost_rate = 0, weak_boost_rate = 0, min_pos_scores_rate = 0;
};

Compression compression_of(const ModelFile & m);

/**
 * The speaker cache and the FIFO queue of a stream: their rows, pre-encoded 80 ms frames or the silence embedding,
 * [rows, d_model] row-major, the speakers' probabilities of each row, [rows, speakers] row-major, and the frame each row
 * is a copy of, -1 for the silence embedding; and whether the cache has been compressed, after which it keeps the
 * probabilities its rows had when it took them.
 */
struct State {
    std::vector<float> cache, cache_probs, fifo, fifo_probs;
    std::vector<int64_t> cache_frames, fifo_frames;
    bool compressed = false;

    int64_t cache_rows() const { return (int64_t) cache_frames.size(); }
    int64_t fifo_rows() const { return (int64_t) fifo_frames.size(); }
};

/**
 * SortformerModules.streaming_update() of one step, synchronous and without a left context: the queue takes the probabilities
 * the step gave its rows and the chunk's `frames` rows, [frames, d_model] row-major, the first of which is the 80 ms frame
 * `first`; once it holds more than fifo_len rows, it gives its oldest rows to the cache, spkcache_update_period of them or
 * as many more as it must, and a cache that then holds more than spkcache_len rows is compressed. `probs` are the 80 ms
 * probabilities of the step's whole input, the cache's rows, the queue's, the chunk's and its right context's, [rows,
 * speakers] row-major. `ties` counts the compression's choices among equal scores, as compress() says.
 */
void update(State & state, const Streaming & streaming, const Compression & compression, const std::vector<float> & silence, const float * chunk,
            int64_t frames, int64_t first, const float * probs, int * ties = nullptr);

/**
 * SortformerModules._compress_spkcache() without its training's permutation of speakers: the spkcache_len rows of the n
 * rows whose probabilities are `probs` [n, speakers], of which those from spkcache_len on came last, that best hold each
 * speaker's speech alone, as their scores choose them, the frames of each speaker in order, each speaker's after the one
 * before, with silence_frames rows of the silence embedding after each speaker's and as many more as no frame scores for
 * at the end. Returns the chosen rows, -1 for the silence embedding. Of rows of equal scores at the edge of a choice,
 * the earlier is chosen, where torch.topk() leaves it to its C++ library's std::nth_element(); `ties` counts such
 * choices, for a check to tell them from a defect.
 */
std::vector<int64_t> compress(const std::vector<float> & probs, int64_t n, const Streaming & streaming, const Compression & compression,
                              int * ties = nullptr);

}  // namespace sortformer
