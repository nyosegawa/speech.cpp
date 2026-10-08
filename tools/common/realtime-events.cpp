#include "realtime-events.h"

#include <cmath>
#include <cstdio>
#include <random>

#include "json.h"

namespace realtime {

using openai::ApiError;

ApiError error_of(std::exception_ptr failure) {
    try {
        std::rethrow_exception(failure);
    } catch (const ApiError & e) {
        return e;
    } catch (const Failure & failure) {
        ApiError e = openai::library_error(failure);
        if (failure.option() == speech_option_name(SPEECH_OPT_LANGUAGE) || failure.option() == speech_option_name(SPEECH_OPT_PROMPT)) {
            e.param = std::string(kTranscription) + "." + failure.option();
        }
        return e;
    } catch (const std::exception & e) {
        return {500, e.what(), "", speech_status_name(SPEECH_ERROR_INTERNAL)};
    }
}

Events::Events(std::function<void(const std::string &)> send) : send_(std::move(send)) {
    std::random_device device;
    char unique[17];
    std::snprintf(unique, sizeof unique, "%08x%08x", (unsigned) device(), (unsigned) device());
    unique_ = unique;
    session_id_ = "sess_" + unique_;
}

std::string Events::new_id(const char * prefix) {
    char n[24];
    std::snprintf(n, sizeof n, "%06llx", (unsigned long long) counter_++);
    return prefix + unique_ + n;
}

void Events::emit(const std::string & type, const std::string & members) {
    std::lock_guard<std::mutex> lock(mutex_);
    send_("{\"event_id\":\"" + new_id("event_") + "\",\"type\":\"" + type + "\"" + members + "}");
}

void Events::error(const ApiError & e, const std::string & event_id) {
    std::string object = openai::error_object(e);
    object.insert(object.size() - 1, ",\"event_id\":" + (event_id.empty() ? std::string("null") : json_string(event_id)));
    emit("error", ",\"error\":" + object);
}

std::string Events::item(uint64_t utterance) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = items_.find(utterance);
    if (found == items_.end()) found = items_.emplace(utterance, new_id("item_")).first;
    return found->second;
}

void Events::utterance(const utterances::Event & e) {
    using Kind = utterances::Event::Kind;
    const std::string id = item(e.utterance);
    const std::string item_id = ",\"item_id\":" + json_string(id);
    const std::string content = item_id + ",\"content_index\":0";
    const auto ms = [](double seconds) { return std::to_string(std::llround(seconds * 1000)); };
    switch (e.kind) {
        case Kind::SpeechStarted:
            emit("input_audio_buffer.speech_started", ",\"audio_start_ms\":" + ms(e.at) + item_id);
            return;
        case Kind::SpeechStopped:
            emit("input_audio_buffer.speech_stopped", ",\"audio_end_ms\":" + ms(e.at) + item_id);
            return;
        case Kind::Committed: {
            std::string previous;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                previous = previous_item_;
                previous_item_ = id;
                if (!e.recognized) items_.erase(e.utterance);
            }
            emit("input_audio_buffer.committed", ",\"previous_item_id\":" + (previous.empty() ? std::string("null") : json_string(previous)) + item_id);
            return;
        }
        case Kind::Delta:
            emit("conversation.item.input_audio_transcription.delta", content + ",\"delta\":" + json_string(e.delta));
            return;
        case Kind::Completed: {
            const Transcript & t = e.transcript;
            std::string languages;
            for (const std::string & tag : t.languages) languages += (languages.empty() ? "" : ",") + std::string("{\"code\":") + json_string(tag) + "}";
            emit("conversation.item.input_audio_transcription.completed",
                 content + ",\"transcript\":" + json_string(t.text) + ",\"usage\":{\"type\":\"duration\",\"seconds\":" + json_number(e.duration) + "}" +
                     (languages.empty() ? "" : ",\"languages\":[" + languages + "]") + ",\"stop\":\"" + speech_stop_name(t.stop) + "\"");
            break;
        }
        case Kind::Failed:
            emit("conversation.item.input_audio_transcription.failed", content + ",\"error\":" + openai::error_object(error_of(e.failure)));
            break;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    items_.erase(e.utterance);
}

}  // namespace realtime
