#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "httplib.h"

#include "jobs.h"

// The audio a request makes as the server answers it, which /v1/audio/speech and /v1/audio/enhancements share: 16-bit
// PCM gathered as the library passes it, sent whole as a WAV file or streamed as it comes.

namespace server {

/** A request whose run passes audio, which hands it to the HTTP handler as it is made. */
struct AudioJob : Job {
    /** The rate of the audio, which the log gives its length in. */
    int sample_rate = 0;
    /** The PCM made and not yet sent. */
    std::string pending;
    /** A callback has been called, so the library has accepted a request and its work has begun. */
    bool accepted = false;
    uint64_t samples = 0;
    Clock::time_point first_audio;
};

/** The audio callback of a request run for an AudioJob, its `user_data`: the PCM, as the worker sends it, for the handler. */
int on_job_audio(const float * s, size_t n, void * user_data);

/** The progress callback of a request run for an AudioJob: the library reports progress once it has checked a request. */
int on_job_progress(double done, void * user_data);

/** A WAV file's header for `data_bytes` of 16-bit mono PCM at `sample_rate`. */
std::string wav_header(size_t data_bytes, int sample_rate);

/**
 * Sends the job's audio as it is made, each piece through `frame` (the PCM itself, or an SSE event of it), and once the
 * run has ended, what `ending` gives for its last message, or nothing. A failure after the stream has begun ends it
 * without `ending` and its last chunk, which the client reads as a broken transfer, unless `ending` gives an error event
 * for it; a client that goes away stops the run.
 */
bool stream_audio(AudioJob & job, httplib::DataSink & sink, const std::function<std::string(const std::string & pcm)> & frame,
                  const std::function<std::string()> & ending);

}  // namespace server
