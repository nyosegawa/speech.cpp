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

// Detections of audio that arrives a piece at a time: the options of a request checked once, each push resampled to the
// model's rate and given to the family's stream while the model is held, and the regions read as the stream gives them.

struct speech_detection {
    speech_detection(speech_model * model, ResampleStream resampler, std::unique_ptr<DetectionStream> stream)
        : model(model), resampler(std::move(resampler)), stream(std::move(stream)) {}

    speech_model * model;
    ResampleStream resampler;
    std::unique_ptr<DetectionStream> stream;
    /** Whether the audio has ended, and whether a push or the end failed, leaving the stream part of the way through. */
    bool ended = false, failed = false;
    std::vector<float> resampled;
};

namespace {

/** Throws unless the detection takes more audio. */
void require_open(const speech_detection & detection) {
    if (detection.failed) {
        throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the detection failed in an earlier call and takes no more audio; free it and start a new one");
    }
    if (detection.ended) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the detection has ended; start a new one for more audio");
}

/** Runs `body` on the detection's stream while the model is held, marking the detection failed when it throws. */
template <typename Body>
void feed(speech_detection & detection, Body && body) {
    std::lock_guard<std::mutex> lock(detection.model->busy);
    try {
        detection.resampled.clear();
        body();
    } catch (...) {
        detection.failed = true;
        throw;
    }
}

}  // namespace

extern "C" {

speech_status speech_detection_start(const speech_request * request, int sample_rate, speech_detection ** detection) {
    if (detection) *detection = nullptr;
    return guarded([&] {
        require(request, "request");
        require(detection, "detection");
        const FileInfo & file = *request->model->file;
        if (file.family->task != SPEECH_TASK_DETECTION) {
            throw ApiError(SPEECH_ERROR_UNSUPPORTED, file.identity.name + " is a model of speech " + task_name(file.family->task) +
                                                         " and detects no speech; use " + call_of(file.family->task));
        }
        if (!request->audio.empty()) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT,
                           "the request has audio, which a detection of audio given a piece at a time leaves unused; give the audio to "
                           "speech_detection_push(), or run speech_detect() on the request",
                           "audio");
        }
        const RequestValues values = run_values(*request);
        std::optional<ResampleStream> resampler;
        try {
            resampler.emplace(sample_rate, file.sample_rate);
        } catch (const std::invalid_argument & e) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, e.what(), "audio");
        }
        std::unique_ptr<DetectionStream> stream;
        {
            std::lock_guard<std::mutex> lock(request->model->busy);
            stream = request->model->engine->start_detection(values);
        }
        *detection = new speech_detection(request->model, std::move(*resampler), std::move(stream));
        return SPEECH_OK;
    });
}

void speech_detection_free(speech_detection * detection) {
    delete detection;
}

speech_status speech_detection_push(speech_detection * detection, const float * samples, size_t n_samples) {
    return guarded([&] {
        require(detection, "detection");
        if (n_samples > 0) require(samples, "samples", "audio");
        require_open(*detection);
        feed(*detection, [&] {
            detection->resampler.push(samples, n_samples, detection->resampled);
            detection->stream->push(detection->resampled.data(), detection->resampled.size());
        });
        return SPEECH_OK;
    });
}

speech_status speech_detection_end(speech_detection * detection) {
    return guarded([&] {
        require(detection, "detection");
        require_open(*detection);
        feed(*detection, [&] {
            detection->resampler.end(detection->resampled);
            detection->stream->push(detection->resampled.data(), detection->resampled.size());
            detection->stream->end();
        });
        detection->ended = true;
        return SPEECH_OK;
    });
}

size_t speech_detection_region_count(const speech_detection * detection) {
    return detection ? detection->stream->regions().size() : 0;
}

speech_status speech_detection_region(const speech_detection * detection, size_t index, double * start, double * end) {
    return guarded([&] {
        require(detection, "detection");
        const std::vector<TimedText> & regions = detection->stream->regions();
        if (index >= regions.size()) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT,
                           "there is no region " + std::to_string(index) + "; the detection has given " + std::to_string(regions.size()));
        }
        if (start) *start = regions[index].start;
        if (end) *end = regions[index].end;
        return SPEECH_OK;
    });
}

int speech_detection_speaking(const speech_detection * detection, double * start) {
    if (!detection) return 0;
    const std::optional<double> open = detection->stream->open();
    if (open && start) *start = *open;
    return open ? 1 : 0;
}

}  // extern "C"
