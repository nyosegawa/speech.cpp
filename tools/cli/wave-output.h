#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "failure.h"

// The WAVE files that speech tts and speech enhance write: 16-bit mono PCM, written as the audio comes, so that a player
// reading stdout starts before the rest is made. In a regular file that the WAVE starts at the beginning of, its RIFF
// and data sizes are set once the audio is complete; anywhere else (a pipe, a console, a file appended to) they stay
// 0xFFFFFFFF, as ffmpeg writes them to a pipe.

/** A failure to write, the io category with no input at fault. */
Failure io_failure(const std::string & message);

/** Writes 16-bit mono PCM as a WAVE stream, converting the samples as the worker does. */
class WavWriter {
public:
    /** Writes the header of a stream of unknown length at `sample_rate` to `file`. */
    WavWriter(FILE * file, int sample_rate);

    void write(const float * s, size_t n);

    /** Sets the sizes where the header can be written again in place, and flushes. */
    void finish();

private:
    void put(const void * data, size_t n);
    void put32(uint32_t v);
    void put16(uint16_t v);
    void header(uint32_t riff_size, uint32_t data_size);

    FILE * file_;
    int sample_rate_;
    bool rewritable_ = false;
    uint64_t bytes_ = 0;
    std::vector<int16_t> pcm_;
};

/** The WAVE file named by -o, or stdout for "-", removed again unless keep() is called, so that a failed run leaves none behind. */
class Output {
public:
    Output(const std::string & path, FILE * out);
    ~Output();
    Output(const Output &) = delete;
    Output & operator=(const Output &) = delete;

    FILE * file() const { return file_; }
    void keep() { kept_ = true; }

private:
    std::string path_;
    FILE * file_ = nullptr;
    bool kept_ = false;
};
