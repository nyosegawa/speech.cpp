#pragma once

#include <functional>
#include <string>
#include <vector>

#include "model-file.h"
#include "network.h"
#include "regions.h"

namespace silero_vad {

/**
 * A Silero VAD model on one backend, from audio to the speech probability of each chunk and to the regions where
 * someone speaks, as silero-vad's get_speech_timestamps() runs it: the audio cut into chunks, the last padded with zeros,
 * each given the samples before it as context, zeros before the first, and the LSTM cell's state from zeros.
 */
class Detector {
public:
    /** Loads the model file at `path`; a backend that cannot compute an operation of the network throws a device error. */
    Detector(const std::string & path, ggml_backend_t backend);
    ~Detector();
    Detector(const Detector &) = delete;
    Detector & operator=(const Detector &) = delete;

    /**
     * Each chunk's input: the context before it and the chunk, [context + chunk, n] in ggml order, for the mono samples
     * at sample_rate(), of at least one sample.
     */
    std::vector<float> inputs(const std::vector<float> & samples) const;

    /**
     * The speech probability of each chunk of the samples, the chunks computed in blocks of kBlock, after each of which
     * `progress` is told the share done; it returns early, with the probabilities computed so far, once `progress`
     * returns false.
     */
    std::vector<float> probabilities(const std::vector<float> & samples, const std::function<bool(double done)> & progress = {});

    /** The regions of the samples by `rule`, in samples. */
    std::vector<Region> regions(const std::vector<float> & samples, const RegionRule & rule) {
        return speech_regions(probabilities(samples), (int64_t) samples.size(), sample_rate_, network_.chunk(), rule);
    }

    int sample_rate() const { return sample_rate_; }
    const ModelFile & model() const { return model_; }
    const Network & network() const { return network_; }

    /**
     * The chunks one graph computes, 16 s, in a graph of under 8000 nodes, most of them the LSTM cell's steps. On an Apple
     * M5 a minute of audio took 40 to 48 ms with blocks of 128 to 2048 chunks, on the CPU and on Metal alike
     * (2026-10-08), so the block bounds the graph's size and how long a cancel waits, not the speed.
     */
    static constexpr int64_t kBlock = 512;

private:
    ggml_backend_t backend_;
    ModelFile model_;
    Network network_;
    int sample_rate_;
    ggml_gallocr_t allocr_;
};

}  // namespace silero_vad
