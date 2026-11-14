#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>

// The system's default audio output, through miniaudio, for speech talk --live.

/** The default playback device, started, playing mono 32-bit float samples at a rate it is given, and silence while it has none. */
class Speaker {
public:
    /**
     * Opens and starts the default playback device for samples at `rate`, which miniaudio resamples to the device's own;
     * one that does not open or start throws a Failure of io.
     */
    explicit Speaker(int rate);
    /** Stops the device. */
    ~Speaker();
    Speaker(const Speaker &) = delete;
    Speaker & operator=(const Speaker &) = delete;

    const std::string & name() const { return name_; }
    /** Plays `n` samples after those queued before. */
    void play(const float * samples, size_t n);
    /** The samples queued that have not been played. */
    size_t queued();

private:
    struct Device;

    void fill(float * out, size_t n);

    std::unique_ptr<Device> device_;
    std::string name_;
    std::mutex mutex_;
    std::deque<float> queue_;
};
