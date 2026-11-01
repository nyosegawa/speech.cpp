#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "api.h"
#include "error.h"
#include "speech.h"

// Enhancements of audio that arrives a piece at a time: the options of a request checked once, each push given to the
// family's stream while the model is held, and the enhanced audio passed to the caller's callback as the stream settles
// it.

struct speech_enhancement {
    speech_model * model = nullptr;
    speech_audio_callback on_audio = nullptr;
    void * user_data = nullptr;
    std::unique_ptr<EnhancementStream> stream;
    /** Whether the audio has ended, the callback stopped the enhancement, or a push or the end failed. */
    bool ended = false, stopped = false, failed = false;
};

namespace {

/** Throws unless the enhancement takes more audio. */
void require_open(const speech_enhancement & enhancement) {
    if (enhancement.failed) {
        throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the enhancement failed in an earlier call and takes no more audio; free it and start a new one");
    }
    if (enhancement.stopped) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the callback stopped the enhancement; start a new one for more audio");
    if (enhancement.ended) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the enhancement has ended; start a new one for more audio");
}

/**
 * Runs `body` on the enhancement's stream while the model is held, marking the enhancement failed when it throws and
 * stopped when the callback asked it to stop, which returns SPEECH_CANCELLED.
 */
template <typename Body>
speech_status feed(speech_enhancement & enhancement, Body && body) {
    std::lock_guard<std::mutex> lock(enhancement.model->busy);
    try {
        if (!body()) {
            enhancement.stopped = true;
            return SPEECH_CANCELLED;
        }
    } catch (...) {
        enhancement.failed = true;
        throw;
    }
    return SPEECH_OK;
}

}  // namespace

extern "C" {

speech_status speech_enhancement_start(const speech_request * request, int sample_rate, speech_audio_callback on_audio, void * user_data,
                                       speech_enhancement ** enhancement) {
    if (enhancement) *enhancement = nullptr;
    return guarded([&] {
        require(request, "request");
        require(enhancement, "enhancement");
        if (!on_audio) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "on_audio is NULL; give a callback that takes the audio");
        const FileInfo & file = *request->model->file;
        if (file.family->task != SPEECH_TASK_ENHANCEMENT) {
            throw ApiError(SPEECH_ERROR_UNSUPPORTED, file.identity.name + " is a model of speech " + task_name(file.family->task) +
                                                         " and enhances no speech; use " + call_of(file.family->task));
        }
        if (!request->audio.empty()) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT,
                           "the request has audio, which an enhancement of audio given a piece at a time leaves unused; give the audio to "
                           "speech_enhancement_push(), or run speech_enhance() on the request",
                           "audio");
        }
        if (sample_rate <= 0) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the sample rate is " + std::to_string(sample_rate) + "; give a positive rate", "audio");
        const RequestValues values = run_values(*request);
        auto e = std::make_unique<speech_enhancement>();
        e->model = request->model;
        e->on_audio = on_audio;
        e->user_data = user_data;
        speech_enhancement * self = e.get();
        // Every sample passes here, so no enhancement passes audio that is not finite.
        AudioSink sink = [self](const float * samples, size_t n) {
            for (size_t i = 0; i < n; i++) {
                if (!std::isfinite(samples[i])) {
                    throw Error(Fault::OutOfRange, "the enhanced audio came out as numbers that are not finite, which the model's float32 cannot hold");
                }
            }
            return self->on_audio(samples, n, self->user_data) == 0;
        };
        {
            std::lock_guard<std::mutex> lock(request->model->busy);
            e->stream = request->model->engine->start_enhancement(sample_rate, values, std::move(sink));
        }
        *enhancement = e.release();
        return SPEECH_OK;
    });
}

void speech_enhancement_free(speech_enhancement * enhancement) {
    delete enhancement;
}

speech_status speech_enhancement_push(speech_enhancement * enhancement, const float * samples, size_t n_samples) {
    return guarded([&] {
        require(enhancement, "enhancement");
        if (n_samples > 0) require(samples, "samples", "audio");
        require_open(*enhancement);
        return feed(*enhancement, [&] { return enhancement->stream->push(samples, n_samples); });
    });
}

speech_status speech_enhancement_end(speech_enhancement * enhancement) {
    return guarded([&] {
        require(enhancement, "enhancement");
        require_open(*enhancement);
        const speech_status status = feed(*enhancement, [&] { return enhancement->stream->end(); });
        enhancement->ended = true;
        return status;
    });
}

}  // extern "C"
