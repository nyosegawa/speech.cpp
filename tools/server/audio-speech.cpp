#include "audio-speech.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "audio-stream.h"
#include "openai-api.h"
#include "sentences.h"

namespace server {

namespace {

using openai::ApiError;
using openai::send_error;

/** One speech, which hands its audio to the HTTP handler as it is made. */
struct SpeechJob : AudioJob {
    /** The text and the options, the seed among them, which the handler has had the library check. */
    std::string text;
    std::vector<RequestOption> options;
    int64_t seed = -1;
    speech_stop stop = SPEECH_STOP_COMPLETE;
    size_t requests = 0;
};

void synthesize(const std::shared_ptr<SpeechJob> & job, Turns & turns, uint64_t ticket, const std::string & name) {
    run_in_turn(*job, turns, ticket, name, [&] {
        const std::optional<Spoken> spoken =
            speak_text(job->served->get(), job->text, job->options, Split::Sentences, on_job_audio, on_job_progress, static_cast<AudioJob *>(job.get()),
                       job->cancellation);
        if (!spoken) return SPEECH_CANCELLED;
        std::lock_guard<std::mutex> lock(job->mutex);
        job->seed = spoken->seed;
        job->stop = spoken->stop;
        job->requests = spoken->requests;
        return SPEECH_OK;
    }, [&](Clock::time_point started) {
        if (!job->samples) return std::string();
        char line[256];
        std::snprintf(line, sizeof line, ", time to first audio %.3f s, %.2f s of audio", seconds_between(started, job->first_audio),
                      (double) job->samples / job->sample_rate);
        return std::string(line) + (job->requests > 1 ? " from " + std::to_string(job->requests) + " requests" : "");
    });
}

/** A seed from 0 to 2^53 - 1, the range the library draws from, for a request that sets none. */
int64_t draw_seed() {
    std::random_device device;
    return (int64_t) (((uint64_t) device() << 32 | device()) & ((1ull << 53) - 1));
}

}  // namespace

void answer_speech(std::shared_ptr<Served> served, const httplib::Request & req, httplib::Response & res) {
    const auto info = served->info();
    const int sample_rate = speech_model_info_sample_rate(info.get());
    openai::SpeechRequest asked;
    try {
        asked = openai::read_speech_request(req.body, speech_model_info_name(info.get()));
    } catch (const ApiError & e) {
        send_error(res, e);
        return;
    }
    asked.options = served->with_defaults(asked.options);
    auto job = std::make_shared<SpeechJob>();
    job->sample_rate = sample_rate;
    std::optional<int64_t> seed;
    std::string voice;
    for (const RequestOption & o : asked.options) {
        if (o.option == SPEECH_OPT_SEED) seed = std::get<int64_t>(o.value);
        if (o.option == SPEECH_OPT_VOICE) voice = std::get<std::string>(o.value);
    }
    // A pcm stream's headers leave before its result, so the server draws the seed the library would draw.
    if (!seed && speech_model_info_takes(info.get(), SPEECH_OPT_SEED)) {
        seed = draw_seed();
        asked.options.push_back({SPEECH_OPT_SEED, *seed});
    }
    job->served = served;
    try {
        // The library checks each value as it is set, so a request it refuses is answered before it waits for its turn.
        const Request checked = new_request(served->get());
        check(speech_request_set_text(checked.get(), asked.input.c_str()));
        apply_options(checked.get(), asked.options);
    } catch (const Failure & e) {
        send_error(res, openai::library_error(e));
        return;
    }
    job->text = std::move(asked.input);
    job->options = asked.options;
    char name[256];
    std::snprintf(name, sizeof name, "voice %s, seed %lld", voice.c_str(), (long long) seed.value_or(-1));
    job->method = req.method;
    job->path = req.path;
    job->client = req.remote_addr;
    const uint64_t ticket = served->turns().take();
    std::thread(synthesize, job, std::ref(served->turns()), ticket, std::string(name)).detach();
    const LogLine log_line(*job);

    // A wav is sent whole, so it waits for the end; a stream waits until the library has begun a request's work, so
    // that a request it refuses is answered with an error status rather than a stream that breaks off.
    const bool whole = asked.format == "wav";
    {
        std::unique_lock<std::mutex> lock(job->mutex);
        if (!wait_for(*job, lock, [&] { return !whole && job->accepted; }, [&] { return req.is_connection_closed(); })) return;
        if (job->finished && job->status == SPEECH_CANCELLED) return;
        if (job->finished && job->failure && (whole || !job->accepted)) {
            send_error(res, openai::library_error(*job->failure));
            return;
        }
    }
    res.set_header("X-Sample-Rate", std::to_string(sample_rate));
    if (seed) res.set_header("X-Speech-Seed", std::to_string(*seed));
    if (whole) {
        res.set_header("X-Speech-Stop", speech_stop_name(job->stop));
        res.set_content(wav_header(job->pending.size(), sample_rate) + job->pending, "audio/wav");
        return;
    }
    const bool sse = asked.sse;
    if (sse) res.set_header("Cache-Control", "no-cache");
    // An SSE stream ends with its done event, or with an error event for a failure after it has begun.
    const auto frame = [sse](const std::string & pcm) { return sse ? openai::sse_delta(pcm) : pcm; };
    const auto ending = [job, sse]() -> std::string {
        if (!sse) return "";
        std::lock_guard<std::mutex> lock(job->mutex);
        if (job->status == SPEECH_OK) return openai::sse_done(job->seed, job->samples, speech_stop_name(job->stop));
        if (job->failure) return openai::sse_error(openai::library_error(*job->failure));
        return "";
    };
    res.set_chunked_content_provider(
        sse ? "text/event-stream" : "audio/pcm", [job, frame, ending](size_t, httplib::DataSink & sink) { return stream_audio(*job, sink, frame, ending); },
        [job](bool success) {
            if (!success) job->abandon();
        });
}

}  // namespace server
