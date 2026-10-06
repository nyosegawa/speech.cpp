#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "library.h"

// How the server runs requests on its one model: each set up by its HTTP handler as it arrives, so that the library
// refuses a value at once, then run in the order the requests arrived, each on a thread of its own that hands what it
// makes to the handler, and stopped when its client goes away.

namespace server {

using Clock = std::chrono::steady_clock;

/** The audio of a request as the worker sends it: 16-bit little-endian samples, clamped to [-1, 1] and rounded. */
inline void append_pcm(std::string & out, const float * s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const int16_t v = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s[i])) * 32767.0f);
        out += (char) (v & 0xFF);
        out += (char) ((uint16_t) v >> 8);
    }
}

/**
 * Lets requests take the model in the order they arrive. The library serializes concurrent calls by itself, but in
 * no particular order, and a request must know when it is its turn to tell a client that went away while waiting
 * from one whose request is under way.
 */
class Turns {
public:
    uint64_t take() {
        std::lock_guard<std::mutex> lock(mutex_);
        return next_++;
    }
    void wait(uint64_t ticket) {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return serving_ == ticket; });
    }
    void pass() {
        std::lock_guard<std::mutex> lock(mutex_);
        serving_++;
        changed_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    uint64_t next_ = 0, serving_ = 0;
};

/** What every request's run shares: the library's request, its state, which the HTTP handler waits on, and its client's departure. */
struct Job {
    /** The request as its handler set it up, which the run spends and a client that goes away cancels. */
    Request request{nullptr, speech_request_free};
    std::mutex mutex;
    std::condition_variable changed;
    bool finished = false;
    bool abandoned = false;
    speech_status status = SPEECH_OK;
    /** The library's failure of a run that failed. */
    std::optional<Failure> failure;
    Clock::time_point arrived = Clock::now(), gone;

    /** Stops the request for a client that went away, whether it waits or runs. */
    void abandon() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!abandoned) gone = Clock::now();
        abandoned = true;
        speech_request_cancel(request.get());
    }
};

/** One request's synthesis, which hands its audio to the HTTP handler as it is made. */
struct SpeechJob : Job {
    /** The model's sample rate, which the log gives the length in. */
    int sample_rate = 0;
    /** The PCM made and not yet sent. */
    std::string pending;
    /** A callback has been called, so the library has accepted the request and its work has begun. */
    bool accepted = false;
    uint64_t samples = 0;
    int64_t seed = -1;
    speech_stop stop = SPEECH_STOP_COMPLETE;
    Clock::time_point first_audio;
};

/** One request's recognition, whose result the HTTP handler reads from the request once it is done. */
struct TranscriptionJob : Job {
    /** The length of the audio in seconds and its rate, for the log and verbose_json. */
    double duration = 0;
    int sample_rate = 0;
};

inline int on_audio(const float * s, size_t n, void * user_data) {
    SpeechJob & job = *static_cast<SpeechJob *>(user_data);
    std::lock_guard<std::mutex> lock(job.mutex);
    job.accepted = true;
    if (job.samples == 0) job.first_audio = Clock::now();
    append_pcm(job.pending, s, n);
    job.samples += n;
    job.changed.notify_all();
    return 0;
}

/** The library reports progress once it has checked the request, so a request that reports any is accepted. */
inline int on_progress(double, void * user_data) {
    SpeechJob & job = *static_cast<SpeechJob *>(user_data);
    std::lock_guard<std::mutex> lock(job.mutex);
    job.accepted = true;
    job.changed.notify_all();
    return 0;
}

inline double seconds_between(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

/**
 * Runs a request on the model in its turn, on a thread of its own, unless its client went away while it waited, and
 * logs how it went: `name` begins the line, `run` runs the request, and `outcome`, called with the job locked, says
 * what the run made.
 */
template <typename Run, typename Outcome>
void run_in_turn(Job & job, Turns & turns, uint64_t ticket, const std::string & name, Run run, Outcome outcome) {
    turns.wait(ticket);
    const Clock::time_point started = Clock::now();
    bool ran = false;
    {
        std::lock_guard<std::mutex> lock(job.mutex);
        ran = !job.abandoned;
    }
    speech_status status = SPEECH_CANCELLED;
    std::optional<Failure> failure;
    if (ran) {
        status = run(job.request.get());
        if (status < 0) failure = library_failure(status);
    }
    {
        std::lock_guard<std::mutex> lock(job.mutex);
        job.finished = true;
        job.status = status;
        job.failure = failure;
        job.changed.notify_all();
        const Clock::time_point now = Clock::now();
        char line[256];
        std::snprintf(line, sizeof line, ": waited %.3f s", seconds_between(job.arrived, started));
        std::string log = name + line;
        if (!ran) {
            log += ", not started: the client went away while it waited";
        } else {
            log += outcome(started);
            std::snprintf(line, sizeof line, " in %.3f s", seconds_between(started, now));
            log += line;
            if (failure) log += ", failed: " + failure->code() + ": " + failure->what();
            if (job.abandoned) {
                std::snprintf(line, sizeof line, ", stopped %.3f s after the client went away", seconds_between(job.gone, now));
                log += line;
            }
        }
        std::fprintf(stderr, "%s\n", log.c_str());
    }
    turns.pass();
}

inline void synthesize(const std::shared_ptr<SpeechJob> & job, Turns & turns, uint64_t ticket, const std::string & name) {
    run_in_turn(*job, turns, ticket, name, [&](speech_request * r) {
        const speech_status s = speech_synthesize(r, on_audio, job.get());
        if (s >= 0) {
            const speech_result * result = speech_request_result(r);
            std::lock_guard<std::mutex> lock(job->mutex);
            job->seed = speech_result_seed(result);
            job->stop = speech_result_stop(result);
        }
        return s;
    }, [&](Clock::time_point started) {
        if (!job->samples) return std::string();
        char line[256];
        std::snprintf(line, sizeof line, ", first audio after %.3f s, %.2f s of audio", seconds_between(started, job->first_audio),
                      (double) job->samples / job->sample_rate);
        return std::string(line);
    });
}

inline void transcribe(const std::shared_ptr<TranscriptionJob> & job, Turns & turns, uint64_t ticket) {
    char name[256];
    std::snprintf(name, sizeof name, "transcription: %.2f s of audio at %d Hz", job->duration, job->sample_rate);
    run_in_turn(*job, turns, ticket, name, [&](speech_request * r) { return speech_transcribe(r); }, [&](Clock::time_point) {
        const speech_result * result = speech_request_result(job->request.get());
        return result ? ", " + std::to_string(std::string(speech_result_text(result)).size()) + " bytes of text" : std::string();
    });
}

}  // namespace server
