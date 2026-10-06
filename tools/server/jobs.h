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
    /** The steps of every speech request, or 0 for the model's own. */
    int steps = 0;
    std::mutex mutex;
    std::condition_variable changed;
    /** The library's request while it runs, which a client that goes away cancels. */
    speech_request * request = nullptr;
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
        if (request) speech_request_cancel(request);
    }
};

/** One request's synthesis, which hands its audio to the HTTP handler as it is made. */
struct SpeechJob : Job {
    SpeechRequest asked;
    /** The model's sample rate, which the log gives the length in. */
    int sample_rate = 0;
    /** The PCM made and not yet sent. */
    std::string pending;
    /** The callback has been called, so the library has accepted the request. */
    bool accepted = false;
    size_t samples = 0;
    Clock::time_point first_audio;
};

/** One request's recognition, whose text the HTTP handler sends once it is done. */
struct TranscriptionJob : Job {
    TranscriptionRequest asked;
    std::string text;
};

inline int on_audio(const float * s, size_t n, void * user_data) {
    SpeechJob & job = *static_cast<SpeechJob *>(user_data);
    std::lock_guard<std::mutex> lock(job.mutex);
    if (job.abandoned) return 1;
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
    if (job.abandoned) return 1;
    job.accepted = true;
    job.changed.notify_all();
    return 0;
}

inline double seconds_since(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

/**
 * Runs a request on the model in its turn, on a thread of its own, unless its client went away while it waited, and
 * logs how it went: `name` begins the line, `prepare` sets the library's request up and `run` runs it, and `outcome`,
 * called with the job locked, says what the run made.
 */
template <typename Prepare, typename Run, typename Outcome>
inline void run_in_turn(Job & job, Turns & turns, uint64_t ticket, const std::string & name, Prepare prepare, Run run, Outcome outcome) {
    turns.wait(ticket);
    const Clock::time_point started = Clock::now();
    speech_request * request = nullptr;
    speech_status status = speech_request_new(job.model, &request);
    const std::unique_ptr<speech_request, decltype(&speech_request_free)> owned(request, speech_request_free);
    if (status == SPEECH_OK) status = prepare(request);
    bool ran = false;
    {
        std::lock_guard<std::mutex> lock(job.mutex);
        ran = !job.abandoned;
        if (ran && status == SPEECH_OK) job.request = request;
    }
    std::string error;
    if (ran && status == SPEECH_OK) status = run(request);
    if (status < 0) error = speech_last_error();
    if (!ran) status = SPEECH_CANCELLED;
    {
        std::lock_guard<std::mutex> lock(job.mutex);
        job.request = nullptr;
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
            log += outcome(started, request);
            std::snprintf(line, sizeof line, " in %.3f s", seconds_since(started, now));
            log += line;
            if (status < 0) log += ", failed: " + error;
            if (job.abandoned) {
                std::snprintf(line, sizeof line, ", stopped %.3f s after the client went away", seconds_since(job.gone, now));
                log += line;
            }
        }
        std::fprintf(stderr, "%s\n", log.c_str());
    }
    turns.pass();
}

/** Sets one option, returning the library's status. */
inline speech_status set_float_if(speech_request * r, speech_option option, const std::optional<double> & v) {
    return v ? speech_request_set_float(r, option, *v) : SPEECH_OK;
}

inline void synthesize(const std::shared_ptr<SpeechJob> & job, Turns & turns, uint64_t ticket) {
    char name[256];
    std::snprintf(name, sizeof name, "speech: voice %s, seed %llu", job->asked.voice.c_str(), (unsigned long long) job->asked.seed);
    const SpeechRequest & a = job->asked;
    run_in_turn(*job, turns, ticket, name, [&](speech_request * r) {
        speech_status s = speech_request_set_text(r, a.input.c_str());
        if (s == SPEECH_OK) s = speech_request_set_string(r, SPEECH_OPT_VOICE, a.voice.c_str());
        if (s == SPEECH_OK && !a.language.empty()) s = speech_request_set_string(r, SPEECH_OPT_LANGUAGE, a.language.c_str());
        if (s == SPEECH_OK) s = speech_request_set_int(r, SPEECH_OPT_SEED, (int64_t) a.seed);
        if (s == SPEECH_OK && job->steps > 0) s = speech_request_set_int(r, SPEECH_OPT_STEPS, job->steps);
        if (s == SPEECH_OK) s = set_float_if(r, SPEECH_OPT_SPEED, a.speed);
        // A length of 0 leaves it to the model.
        if (s == SPEECH_OK && a.seconds.value_or(0) != 0) s = set_float_if(r, SPEECH_OPT_SECONDS, a.seconds);
        if (s == SPEECH_OK) s = set_float_if(r, SPEECH_OPT_DURATION_SCALE, a.duration_scale);
        if (s == SPEECH_OK) s = speech_request_set_progress(r, on_progress, job.get());
        return s;
    }, [&](speech_request * r) { return speech_synthesize(r, on_audio, job.get()); },
    [&](Clock::time_point started, speech_request *) {
        if (!job->samples) return std::string();
        char line[256];
        std::snprintf(line, sizeof line, ", first audio after %.3f s, %.2f s of audio", seconds_since(started, job->first_audio),
                      (double) job->samples / job->sample_rate);
        return std::string(line);
    });
}

inline void transcribe(const std::shared_ptr<TranscriptionJob> & job, Turns & turns, uint64_t ticket) {
    char name[256];
    const TranscriptionRequest & a = job->asked;
    std::snprintf(name, sizeof name, "transcription: %.2f s of audio at %d Hz", (double) a.samples.size() / a.sample_rate, a.sample_rate);
    run_in_turn(*job, turns, ticket, name, [&](speech_request * r) {
        speech_status s = speech_request_set_audio(r, a.samples.data(), a.samples.size(), a.sample_rate);
        if (s == SPEECH_OK && !a.language.empty()) s = speech_request_set_string(r, SPEECH_OPT_LANGUAGE, a.language.c_str());
        return s;
    }, [&](speech_request * r) {
        const speech_status s = speech_transcribe(r);
        if (s == SPEECH_OK) job->text = speech_result_text(speech_request_result(r));
        return s;
    }, [&](Clock::time_point, speech_request *) { return ", " + std::to_string(job->text.size()) + " bytes of text"; });
}
