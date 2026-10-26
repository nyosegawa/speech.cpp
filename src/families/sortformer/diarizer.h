#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "cache.h"
#include "ggml-backend.h"
#include "model-file.h"
#include "nemo-frontend.h"
#include "network.h"
#include "segments.h"

namespace sortformer {

/** A Sortformer model on its device: its frontend, its network, its cache's settings and the file's configurations. */
class Diarizer {
public:
    Diarizer(const std::string & path, ggml_backend_t backend);

    const ModelFile & model() const { return model_; }
    const NemoFrontend & frontend() const { return frontend_; }
    Network & network() { return network_; }
    const Compression & compression() const { return compression_; }
    /** The learnable silence embedding, which the cache holds in place of a frame. */
    const std::vector<float> & silence() const { return silence_; }
    int sample_rate() const { return frontend_.sample_rate(); }
    int speakers() const { return compression_.speakers; }
    /** The seconds between two rows of probabilities: a hop of the frontend. */
    double frame_seconds() const { return (double) frontend_.hop_length() / frontend_.sample_rate(); }

    /** The names of the file's streaming configurations, the offline one first. */
    const std::vector<std::string> & modes() const { return modes_; }
    /** The configuration named `name`, one of modes(). */
    Streaming mode(const std::string & name) const;
    /** The postprocessing's defaults, from the file. */
    Postprocessing postprocessing() const;

private:
    ModelFile model_;
    NemoFrontend frontend_;
    Network network_;
    Compression compression_;
    std::vector<float> silence_;
    std::vector<std::string> modes_;
};

/**
 * SortformerEncLabelModel.forward() in streaming mode, for audio that arrives a piece at a time: the frontend computes
 * each frame once the samples of its window have arrived, and a step runs once its chunk's features and its right
 * context's have, or, once the audio has ended, with the features left, the right context cut short as the official
 * streaming_feat_loader() cuts the last ones. The probabilities, one row every 10 ms, are those forward() gives the whole
 * audio, however it was cut into pieces, since every frame and every step is computed as it would be from the whole.
 */
class ChunkStream {
public:
    /**
     * A stream of `diarizer`, which outlives it, in `streaming`. `on_step`, when given, hears each step's number once it
     * has run, and false stops the stream before its next step: the stream then runs no more and gives no more rows.
     */
    ChunkStream(Diarizer & diarizer, const Streaming & streaming, std::function<bool(int64_t step)> on_step = {});

    /** Takes `n` more samples at the model's rate and appends the rows of the steps it computes to `probs`. */
    void push(const float * samples, size_t n, std::vector<float> & probs);
    /** Ends the audio, runs the steps left and appends their rows, the last cut to the audio's frames. */
    void end(std::vector<float> & probs);

    const State & state() const { return state_; }

private:
    /** Computes the frames whose windows the samples received cover, or all of them once the audio has ended. */
    void compute_frames(bool ended);
    /** Runs the steps whose features are there, all of them once the audio has ended. */
    void run_steps(bool ended, std::vector<float> & probs);

    Diarizer & diarizer_;
    Streaming streaming_;
    std::function<bool(int64_t)> on_step_;
    bool stopped_ = false;
    State state_;
    /** The samples received, the pre-emphasized ones from x_begin_ on, and the last sample, which pre-emphasis reads. */
    int64_t received_ = 0, x_begin_ = 0;
    std::vector<float> x_;
    float last_ = 0;
    /** The frames computed, those from features_begin_ on in features_, [frames, mels] row-major. */
    int64_t computed_ = 0, features_begin_ = 0;
    std::vector<float> features_;
    int64_t step_ = 0;
};

}  // namespace sortformer
