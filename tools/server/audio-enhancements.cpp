#include "audio-enhancements.h"

#include <cstdio>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "audio-stream.h"
#include "enhancement.h"
#include "openai-api.h"

namespace server {

namespace {

using openai::ApiError;
using openai::send_error;

/** One enhancement, which hands its audio to the HTTP handler as it is made. */
struct EnhancementJob : AudioJob {
    speech_stop stop = SPEECH_STOP_COMPLETE;
    /** The length of the audio given, for the log. */
    double duration = 0;
};

void enhance(const std::shared_ptr<EnhancementJob> & job, Turns & turns, uint64_t ticket, const std::string & name) {
    run_in_turn(*job, turns, ticket, name, [&] {
        speech_request * r = job->request.get();
        check(speech_request_set_progress(r, on_job_progress, static_cast<AudioJob *>(job.get())));
        if (!enhance_audio(r, on_job_audio, static_cast<AudioJob *>(job.get()), job->cancellation)) return SPEECH_CANCELLED;
        std::lock_guard<std::mutex> lock(job->mutex);
        job->stop = speech_result_stop(speech_request_result(r));
        return SPEECH_OK;
    }, [&](Clock::time_point started) {
        if (!job->samples) return std::string();
        char line[128];
        std::snprintf(line, sizeof line, ", time to first audio %.3f s, %.2f s of audio at %d Hz", seconds_between(started, job->first_audio),
                      (double) job->samples / job->sample_rate, job->sample_rate);
        return std::string(line);
    });
}

}  // namespace

void answer_enhancement(std::shared_ptr<Served> served, const httplib::Request & req, httplib::Response & res) {
    const auto info = served->info();
    openai::EnhancementRequest asked;
    try {
        std::vector<openai::FormPart> parts;
        for (const auto & [field_name, field] : req.form.fields) parts.push_back({field_name, field.content, "", false});
        for (const auto & [file_name, file] : req.form.files) parts.push_back({file_name, file.content, file.filename, true});
        asked = openai::read_enhancement_request(req.is_multipart_form_data(), parts, speech_model_info_name(info.get()));
    } catch (const ApiError & e) {
        send_error(res, e);
        return;
    }
    asked.options = served->with_defaults(asked.options);
    auto job = std::make_shared<EnhancementJob>();
    job->served = served;
    job->sample_rate = enhanced_rate(info.get(), asked.sample_rate, asked.options);
    job->duration = (double) asked.samples.size() / asked.sample_rate;
    try {
        // The library checks each value as it is set, so a request it refuses is answered before it waits for its turn.
        job->request = enhancement_request(served->get(), asked.samples, asked.sample_rate, asked.options);
    } catch (const Failure & e) {
        send_error(res, openai::library_error(e));
        return;
    }
    char name[96];
    std::snprintf(name, sizeof name, "%.2f s of audio at %d Hz", job->duration, asked.sample_rate);
    job->method = req.method;
    job->path = req.path;
    job->client = req.remote_addr;
    const uint64_t ticket = served->turns().take();
    std::thread(enhance, job, std::ref(served->turns()), ticket, std::string(name)).detach();
    const LogLine log_line(*job);

    // A wav is sent whole, so it waits for the end; a stream waits until the library has begun the request's work, so
    // that a request it refuses, such as one of audio at a rate the model does not take, is answered with an error status
    // rather than a stream that breaks off.
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
    res.set_header("X-Sample-Rate", std::to_string(job->sample_rate));
    if (whole) {
        res.set_header("X-Speech-Stop", speech_stop_name(job->stop));
        res.set_content(wav_header(job->pending.size(), job->sample_rate) + job->pending, "audio/wav");
        return;
    }
    res.set_chunked_content_provider(
        "audio/pcm",
        [job](size_t, httplib::DataSink & sink) {
            return stream_audio(*job, sink, [](const std::string & pcm) { return pcm; }, [] { return std::string(); });
        },
        [job](bool success) {
            if (!success) job->abandon();
        });
}

}  // namespace server
