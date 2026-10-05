#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>

#include "openai-api.h"
#include "speech.h"

// How the server runs requests on its one model: in the order they arrive, each on a thread of its own that hands
// what it makes to the HTTP handler, and stopped when its client goes away.

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

/** What every request's run shares: its state, which the HTTP handler waits on, and its client's departure. */
struct Job {
    speech_model * model;
    std::mutex mutex;
    std::condition_variable changed;
    bool running = false;
    bool finished = false;
    bool abandoned = false;
    speech_status status = SPEECH_OK;
    std::string error;
    Clock::time_point arrived = Clock::now(), gone;

    /** Stops the request for a client that went away, whether it waits or runs. */
    void abandon() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!abandoned) gone = Clock::now();
        abandoned = true;
        // Only while this request runs: once it has returned, speech_cancel() would stop the next one.
        if (running) speech_cancel(model);
    }
};

/** One request's synthesis, which hands its audio to the HTTP handler as it is made. */
struct SpeechJob : Job {
    SpeechRequest request;
    /** The PCM made and not yet sent. */
    std::string pending;
    /** The callback has been called, so the library has accepted the request. */
    bool accepted = false;
    size_t samples = 0;
    Clock::time_point first_audio;
};

/** One request's recognition, whose text the HTTP handler sends once it is done. */
struct TranscriptionJob : Job {
    TranscriptionRequest request;
    std::string text;
};

inline int on_audio(const float * s, size_t n, void * user_data) {
    SpeechJob & job = *static_cast<SpeechJob *>(user_data);
    std::lock_guard<std::mutex> lock(job.mutex);
    if (job.abandoned) return 1;
    job.accepted = true;
    if (n > 0 && job.samples == 0) job.first_audio = Clock::now();
    append_pcm(job.pending, s, n);
    job.samples += n;
    job.changed.notify_all();
    return 0;
}

inline int on_text(const char * text, void * user_data) {
    TranscriptionJob & job = *static_cast<TranscriptionJob *>(user_data);
    std::lock_guard<std::mutex> lock(job.mutex);
    job.text += text;
    return 0;
}

inline double seconds_since(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

/**
 * Runs a request on the model in its turn, on a thread of its own, unless its client went away while it waited, and
 * logs how it went: `name` begins the line, and `outcome`, called with the job locked, says what the run made.
 */
template <typename Run, typename Outcome>
inline void run_in_turn(Job & job, Turns & turns, uint64_t ticket, const std::string & name, Run run, Outcome outcome) {
    turns.wait(ticket);
    const Clock::time_point started = Clock::now();
    {
        std::lock_guard<std::mutex> lock(job.mutex);
        job.running = !job.abandoned;
    }
    const bool ran = job.running;
    speech_status status = SPEECH_STOPPED;
    std::string error;
    if (ran) {
        status = run();
        if (status == SPEECH_ERROR) error = speech_last_error();
    }
    {
        std::lock_guard<std::mutex> lock(job.mutex);
        job.running = false;
        job.finished = true;
        job.status = status;
        job.error = error;
        job.changed.notify_all();
        const Clock::time_point now = Clock::now();
        char line[256];
        std::snprintf(line, sizeof line, ": waited %.3f s", seconds_since(job.arrived, started));
        std::string log = name + line;
        if (!ran) {
            log += ", not started: the client went away while it waited";
        } else {
            log += outcome(started);
            std::snprintf(line, sizeof line, " in %.3f s", seconds_since(started, now));
            log += line;
            if (status == SPEECH_ERROR) log += ", failed: " + error;
            if (job.abandoned) {
                std::snprintf(line, sizeof line, ", stopped %.3f s after the client went away", seconds_since(job.gone, now));
                log += line;
            }
        }
        std::fprintf(stderr, "%s\n", log.c_str());
    }
    turns.pass();
}

inline void synthesize(const std::shared_ptr<SpeechJob> & job, Turns & turns, uint64_t ticket) {
    char name[256];
    std::snprintf(name, sizeof name, "speech: voice %s, seed %llu", job->request.voice.c_str(), (unsigned long long) job->request.seed);
    run_in_turn(*job, turns, ticket, name, [&] {
        speech_request r = speech_request_default();
        r.text = job->request.input.c_str();
        r.voice = job->request.voice.c_str();
        r.language = job->request.language.c_str();
        r.seed = job->request.seed;
        r.speed = job->request.speed;
        r.seconds = job->request.seconds;
        r.duration_scale = job->request.duration_scale;
        return speech_synthesize(job->model, &r, on_audio, job.get());
    }, [&](Clock::time_point started) {
        if (!job->samples) return std::string();
        char line[256];
        std::snprintf(line, sizeof line, ", first audio after %.3f s, %.2f s of audio", seconds_since(started, job->first_audio),
                      (double) job->samples / speech_model_sample_rate(job->model));
        return std::string(line);
    });
}

inline void transcribe(const std::shared_ptr<TranscriptionJob> & job, Turns & turns, uint64_t ticket) {
    char name[256];
    std::snprintf(name, sizeof name, "transcription: %.2f s of audio at %d Hz", (double) job->request.samples.size() / job->request.sample_rate,
                  job->request.sample_rate);
    run_in_turn(*job, turns, ticket, name, [&] {
        speech_transcription_request r = speech_transcription_request_default();
        r.samples = job->request.samples.data();
        r.n_samples = job->request.samples.size();
        r.sample_rate = job->request.sample_rate;
        r.language = job->request.language.c_str();
        return speech_transcribe(job->model, &r, on_text, job.get());
    }, [&](Clock::time_point) { return ", " + std::to_string(job->text.size()) + " bytes of text"; });
}
