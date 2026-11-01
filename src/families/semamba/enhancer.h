#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "model-file.h"
#include "network.h"
#include "stft.h"

namespace semamba {

/**
 * The spectrogram a chunk's network outputs give, as inference_chunk.py makes it before the inverse STFT: the magnitude
 * decoder's output and atan2 of the phase decoder's imaginary and real parts, cropped to the input's frames and bins,
 * [frames, bins] each, with every frame zeroed of whose bins more than `zero_portion` are at zero or below.
 */
struct Decoded {
    std::vector<float> magnitude, phase;
    /** 1 for each frame that was zeroed. */
    std::vector<uint8_t> zeroed;
};

/**
 * Crops and zeroes the network's outputs, [bins'', frames''] each as forward() gives them, to `frames` frames of `bins`
 * bins.
 */
Decoded decode(const std::vector<float> & magnitude, const std::vector<float> & real, const std::vector<float> & imag, int64_t out_bins,
               int64_t frames, int64_t bins, float zero_portion);

/**
 * A SEMamba model on one backend, enhancing audio as RE-USE's inference_chunk.py does: chunks of chunk_size_in_seconds
 * that start every hop_length_portion of a chunk, the last one ending with the audio, each through the STFT of its rate,
 * the network, the zeroing of frames and the inverse STFT, then added up under a Hann window as long as a chunk and
 * divided by the sum of the windows.
 */
class Enhancer {
public:
    /** Loads the model file at `path`; a backend that cannot compute an operation of the network throws a device error. */
    Enhancer(const std::string & path, ggml_backend_t backend);
    ~Enhancer();
    Enhancer(const Enhancer &) = delete;
    Enhancer & operator=(const Enhancer &) = delete;

    /** The STFT at `rate`. */
    Stft stft(int rate) const;

    /** The network's outputs for a chunk's spectrogram of `frames` frames of `bins` bins, [bins'', frames''] each. */
    struct Raw {
        std::vector<float> magnitude, real, imag;
        int64_t bins, frames;
    };
    Raw network(const std::vector<float> & magnitude, const std::vector<float> & phase, int64_t frames, int64_t bins);

    /** One chunk of `n` samples at `rate` enhanced: as many samples, the inverse STFT padded with pad_value or cut. */
    std::vector<float> chunk(const float * samples, size_t n, int rate);

    /**
     * Enhances `samples` at `rate` as a Stream given them at once, telling `progress` the share of chunks done; it stops,
     * and returns false, once `emit` or `progress` returns false.
     */
    bool enhance(const std::vector<float> & samples, int rate, double chunk_size_in_seconds, double hop_length_portion,
                 const std::function<bool(const float * samples, size_t n)> & emit, const std::function<bool(double done)> & progress);

    const ModelFile & model() const { return model_; }
    const Network & net() const { return network_; }
    int lowest_rate() const { return lowest_; }
    int highest_rate() const { return highest_; }
    /** The fewest samples a chunk at `rate` takes: more than half an FFT, for the STFT's reflection. */
    size_t fewest_samples(int rate) const;

private:
    ggml_backend_t backend_;
    ModelFile model_;
    Network network_;
    int reference_rate_, n_fft_, hop_, lowest_, highest_;
    float window_floor_, zero_portion_, pad_value_;
    ggml_gallocr_t allocr_;

    friend class Stream;
};

/**
 * The chunks of inference_chunk.py over audio at one rate that arrives a piece at a time: each chunk that starts a
 * hop_length_portion of a chunk after the one before is computed once its samples have arrived, and the last, which ends
 * with the audio, once the audio has ended. A sample is passed on once no later chunk can reach it: once a chunk that
 * starts after it has been computed, between a chunk and a chunk and a hop of audio after the sample, or at the end. The
 * chunks are the whole audio's and are summed in the same order, so that the samples are the same bit for bit however the
 * audio was cut.
 */
class Stream {
public:
    /** What enhances one chunk of `n` samples into as many. */
    using Chunk = std::function<std::vector<float>(const float * samples, size_t n)>;

    /**
     * A stream of chunks of `chunk_size_in_seconds` at `rate` on `enhancer`, passing its samples to `emit`, and, where
     * `total` gives the audio's length in advance, telling `progress` the share of chunks done.
     */
    Stream(Enhancer & enhancer, int rate, double chunk_size_in_seconds, double hop_length_portion,
           std::function<bool(const float * samples, size_t n)> emit, std::function<bool(double done)> progress = {}, int64_t total = 0);

    /** The same stream with each chunk enhanced by `chunk` and its window kept above `window_floor`. */
    Stream(Chunk chunk, float window_floor, int rate, double chunk_size_in_seconds, double hop_length_portion,
           std::function<bool(const float * samples, size_t n)> emit, std::function<bool(double done)> progress = {}, int64_t total = 0);

    /** Takes the next `n` samples; false once `emit` or `progress` has returned false, after which it takes no more. */
    bool push(const float * samples, size_t n);

    /** Ends the audio and passes the samples left; false as push() is. */
    bool end();

    /** The samples received. */
    int64_t received() const { return received_; }

private:
    /** Adds the chunk of `length` samples from `start`. */
    bool compute(int64_t start, int64_t length);
    /** Passes the samples before `settled` that are not passed yet. */
    bool pass(int64_t settled);

    Chunk chunk_;
    int64_t size_, hop_, total_chunks_ = 0;
    std::vector<float> window_;
    std::function<bool(const float *, size_t)> emit_;
    std::function<bool(double)> progress_;
    /** The audio from sample `first_` on, and the samples received. */
    std::vector<float> audio_;
    int64_t first_ = 0, received_ = 0;
    /** The chunks computed, those that start every hop and, at the end, the last. */
    int64_t computed_ = 0;
    /** The sums of the samples under their windows and of the windows, from the first sample not yet passed. */
    std::vector<float> sum_, weight_, out_;
    int64_t passed_ = 0;
    bool stopped_ = false;
};

}  // namespace semamba
