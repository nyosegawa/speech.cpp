#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "api.h"
#include "error.h"
#include "resample.h"
#include "speech.h"

// Conversations with a dialogue model: the options of a request checked once, the audio heard resampled to the model's
// rate and given to the family's stream while the model is held, and each frame the model gives passed to the caller's
// callback.

struct speech_dialogue {
    speech_model * model = nullptr;
    speech_dialogue_callback on_frame = nullptr;
    void * user_data = nullptr;
    std::optional<ResampleStream> resampler;
    std::unique_ptr<DialogueStream> stream;
    std::vector<float> resampled;
    /** Whether the conversation has ended, the callback stopped it, or a call failed. */
    bool ended = false, stopped = false, failed = false;
};

namespace {

/** Throws unless the conversation takes more audio. */
void require_open(const speech_dialogue & dialogue) {
    if (dialogue.failed) {
        throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the conversation failed in an earlier call and takes no more audio; free it and start a new one");
    }
    if (dialogue.stopped) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the callback stopped the conversation; start a new one to talk again");
    if (dialogue.ended) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the conversation has ended; start a new one to talk again");
}

/**
 * Runs `body` while the model is held, marking the conversation failed when it throws, and returns SPEECH_CANCELLED once
 * the callback asked to stop.
 */
template <typename Body>
speech_status feed(speech_dialogue & dialogue, Body && body) {
    std::lock_guard<std::mutex> lock(dialogue.model->busy);
    try {
        dialogue.resampled.clear();
        body();
    } catch (...) {
        dialogue.failed = true;
        throw;
    }
    return dialogue.stopped ? SPEECH_CANCELLED : SPEECH_OK;
}

}  // namespace

extern "C" {

speech_status speech_dialogue_start(const speech_request * request, int sample_rate, speech_dialogue_callback on_frame, void * user_data,
                                    speech_dialogue ** dialogue) {
    if (dialogue) *dialogue = nullptr;
    return guarded([&] {
        require(request, "request");
        require(dialogue, "dialogue");
        if (!on_frame) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "on_frame is NULL; give a callback that takes the model's frames");
        const FileInfo & file = *request->model->file;
        if (file.family->task != SPEECH_TASK_DIALOGUE) {
            throw ApiError(SPEECH_ERROR_UNSUPPORTED, file.identity.name + " is a model of speech " + task_name(file.family->task) +
                                                         " and holds no conversation; use " + call_of(file.family->task));
        }
        if (!request->audio.empty()) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT,
                           "the request has audio, which a conversation leaves unused; give the audio the model hears to speech_dialogue_push()", "audio");
        }
        const RequestValues values = run_values(*request);
        auto d = std::make_unique<speech_dialogue>();
        d->model = request->model;
        d->on_frame = on_frame;
        d->user_data = user_data;
        try {
            d->resampler.emplace(sample_rate, file.sample_rate);
        } catch (const std::invalid_argument & e) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, e.what(), "audio");
        }
        speech_dialogue * self = d.get();
        // Every frame passes here, so that no conversation passes audio that is not finite or a callback that stopped.
        DialogueSink sink = [self](const float * samples, size_t n, const std::string & text) {
            for (size_t i = 0; i < n; i++) {
                if (!std::isfinite(samples[i])) throw Error(Fault::Device, "the model's audio came out as numbers that are not finite; run it on another device");
            }
            if (self->on_frame(samples, n, text.c_str(), self->user_data) != 0) self->stopped = true;
            return !self->stopped;
        };
        const speech_status status = feed(*d, [&] { d->stream = request->model->engine->start_dialogue(values, std::move(sink)); });
        *dialogue = d.release();
        return status;
    });
}

void speech_dialogue_free(speech_dialogue * dialogue) {
    delete dialogue;
}

speech_status speech_dialogue_push(speech_dialogue * dialogue, const float * samples, size_t n_samples) {
    return guarded([&] {
        require(dialogue, "dialogue");
        if (n_samples > 0) require(samples, "samples", "audio");
        require_open(*dialogue);
        return feed(*dialogue, [&] {
            dialogue->resampler->push(samples, n_samples, dialogue->resampled);
            dialogue->stream->push(dialogue->resampled.data(), dialogue->resampled.size());
        });
    });
}

speech_status speech_dialogue_end(speech_dialogue * dialogue) {
    return guarded([&] {
        require(dialogue, "dialogue");
        require_open(*dialogue);
        const speech_status status = feed(*dialogue, [&] {
            dialogue->resampler->end(dialogue->resampled);
            dialogue->stream->push(dialogue->resampled.data(), dialogue->resampled.size());
        });
        dialogue->ended = true;
        return status;
    });
}

}  // extern "C"
