#pragma once

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// The system's default microphone, through miniaudio, for speech asr --live and speech talk --live.

/** The default capture device, started, giving mono 32-bit float samples at the device's own rate. */
class Microphone {
public:
    /** Opens and starts the default capture device; one that does not open or start throws a Failure of io. */
    Microphone();
    /** Stops the device. */
    ~Microphone();
    Microphone(const Microphone &) = delete;
    Microphone & operator=(const Microphone &) = delete;

    int rate() const { return rate_; }
    const std::string & name() const { return name_; }
    /**
     * The samples captured since the last call, waiting up to `wait` while there are none. Once they have been nothing
     * but zeros for 3 s, it says on stderr, once, what keeps the microphone from the program.
     */
    std::vector<float> take(std::chrono::milliseconds wait);

private:
    struct Device;

    void captured(const float * samples, size_t n);

    std::unique_ptr<Device> device_;
    int rate_ = 0;
    std::string name_;
    std::mutex mutex_;
    std::condition_variable arrived_;
    std::vector<float> captured_;
    double silent_ = 0;
    bool warned_ = false;
};
