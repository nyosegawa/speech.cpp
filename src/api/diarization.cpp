#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "api.h"
#include "resample.h"
#include "speech.h"

// Diarizations of audio that arrives a piece at a time: the options of a request checked once, each push resampled to
// the model's rate and given to the family's stream while the model is held, and the segments read as the stream gives
// them.

struct speech_diarization {
    speech_diarization(speech_model * model, ResampleStream resampler, std::unique_ptr<DiarizationStream> stream)
        : model(model), resampler(std::move(resampler)), stream(std::move(stream)) {}

    speech_model * model;
    ResampleStream resampler;
    std::unique_ptr<DiarizationStream> stream;
    /** Whether the audio has ended, and whether a push or the end failed, leaving the stream part of the way through. */
    bool ended = false, failed = false;
    std::vector<float> resampled;
};

namespace {

/** Throws unless the diarization takes more audio. */
void require_open(const speech_diarization & diarization) {
    if (diarization.failed) {
        throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the diarization failed in an earlier call and takes no more audio; free it and start a new one");
    }
    if (diarization.ended) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the diarization has ended; start a new one for more audio");
}

/** Runs `body` on the diarization's stream while the model is held, marking the diarization failed when it throws. */
template <typename Body>
void feed(speech_diarization & diarization, Body && body) {
    std::lock_guard<std::mutex> lock(diarization.model->busy);
    try {
        diarization.resampled.clear();
        body();
    } catch (...) {
        diarization.failed = true;
        throw;
    }
}

}  // namespace

extern "C" {

speech_status speech_diarization_start(const speech_request * request, int sample_rate, speech_diarization ** diarization) {
    if (diarization) *diarization = nullptr;
    return guarded([&] {
        require(request, "request");
        require(diarization, "diarization");
        const FileInfo & file = *request->model->file;
        if (file.family->task != SPEECH_TASK_DIARIZATION) {
            throw ApiError(SPEECH_ERROR_UNSUPPORTED, file.identity.name + " is a model of speech " + task_name(file.family->task) +
                                                         " and diarizes no speakers; use " + call_of(file.family->task));
        }
        if (!request->audio.empty()) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT,
                           "the request has audio, which a diarization of audio given a piece at a time leaves unused; give the audio to "
                           "speech_diarization_push(), or run speech_diarize() on the request",
                           "audio");
        }
        const RequestValues values = run_values(*request);
        std::optional<ResampleStream> resampler;
        try {
            resampler.emplace(sample_rate, file.sample_rate);
        } catch (const std::invalid_argument & e) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, e.what(), "audio");
        }
        std::unique_ptr<DiarizationStream> stream;
        {
            std::lock_guard<std::mutex> lock(request->model->busy);
            stream = request->model->engine->start_diarization(values);
        }
        *diarization = new speech_diarization(request->model, std::move(*resampler), std::move(stream));
        return SPEECH_OK;
    });
}

void speech_diarization_free(speech_diarization * diarization) {
    delete diarization;
}

speech_status speech_diarization_push(speech_diarization * diarization, const float * samples, size_t n_samples) {
    return guarded([&] {
        require(diarization, "diarization");
        if (n_samples > 0) require(samples, "samples", "audio");
        require_open(*diarization);
        feed(*diarization, [&] {
            diarization->resampler.push(samples, n_samples, diarization->resampled);
            diarization->stream->push(diarization->resampled.data(), diarization->resampled.size());
        });
        return SPEECH_OK;
    });
}

speech_status speech_diarization_end(speech_diarization * diarization) {
    return guarded([&] {
        require(diarization, "diarization");
        require_open(*diarization);
        feed(*diarization, [&] {
            diarization->resampler.end(diarization->resampled);
            diarization->stream->push(diarization->resampled.data(), diarization->resampled.size());
            diarization->stream->end();
        });
        diarization->ended = true;
        return SPEECH_OK;
    });
}

size_t speech_diarization_segment_count(const speech_diarization * diarization) {
    return diarization ? diarization->stream->segments().size() : 0;
}

speech_status speech_diarization_segment(const speech_diarization * diarization, size_t index, double * start, double * end, int * speaker) {
    return guarded([&] {
        require(diarization, "diarization");
        const std::vector<SpeakerSegment> & segments = diarization->stream->segments();
        if (index >= segments.size()) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT,
                           "there is no segment " + std::to_string(index) + "; the diarization has given " + std::to_string(segments.size()));
        }
        if (start) *start = segments[index].start;
        if (end) *end = segments[index].end;
        if (speaker) *speaker = segments[index].speaker;
        return SPEECH_OK;
    });
}

double speech_diarization_settled(const speech_diarization * diarization) {
    return diarization ? diarization->stream->settled() : 0.0;
}

int speech_diarization_speaking(const speech_diarization * diarization, int speaker, double * start) {
    if (!diarization || speaker < 0 || speaker >= diarization->model->file->described.speakers) return 0;
    const std::optional<double> begun = diarization->stream->speaking(speaker);
    if (begun && start) *start = *begun;
    return begun ? 1 : 0;
}

}  // extern "C"
