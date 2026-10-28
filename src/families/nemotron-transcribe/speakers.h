#pragma once

#include <cstdint>
#include <vector>

#include "ggml-backend.h"
#include "model-file.h"
#include "sortformer-cache.h"
#include "sortformer-network.h"

namespace nemotron_transcribe {

/**
 * The speaker branch of the parallel expert encoder: Nemotron 3 Diarization streaming one step per window of the ASR
 * encoder, as ParallelExpertEncoder._forward_online_impl() runs it, synchronously with the encoder's FIFO and speaker
 * cache settings. A step hears the window's features with the left and right context the window has, updates the
 * speaker cache and the FIFO queue with the window's own frames, but after the last window, and gives the window's
 * speaker probabilities every 80 ms, the average of the eight 10 ms probabilities of each frame.
 */
class SpeakerBranch {
public:
    /** The branch of the model file `m`, which outlives it, computing on `backend`. */
    SpeakerBranch(const ModelFile & m, ggml_backend_t backend);

    /** Starts a recording: an empty speaker cache and FIFO queue. */
    void start();

    /**
     * The step of a window whose core starts at the 80 ms frame `first` of the recording: `frames` frames of log-mel
     * features, [frames, mels] row-major, of which the first `left` and the last `right` are the window's context, and
     * whether the window is the recording's last, whose step updates nothing. Returns the core's probabilities,
     * [ceil((frames - left - right) / factor) at most, speakers] row-major, of the core's frames that the features reach.
     */
    std::vector<float> step(const float * features, int64_t frames, int64_t left, int64_t right, int64_t first, bool last);

    int speakers() const { return network_.speakers(); }
    int factor() const { return network_.factor(); }
    const sortformer::State & state() const { return state_; }

private:
    sortformer::Network network_;
    sortformer::Streaming streaming_;
    sortformer::Compression compression_;
    std::vector<float> silence_;
    sortformer::State state_;
};

}  // namespace nemotron_transcribe
