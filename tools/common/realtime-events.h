#pragma once

#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <string>

#include "openai-error.h"
#include "utterances.h"

// The server events of OpenAI's Realtime transcription, as its API reference gives the GA events at
// github.com/openai/openai-openapi commit 234829e (2026-10-07), each one JSON text with an event_id of the session's
// own: the events of an utterance (input_audio_buffer.speech_started, .speech_stopped and .committed, and
// conversation.item.input_audio_transcription.delta, .completed and .failed) under the item id of the utterance, and
// those a session writes itself. The session over the WebSocket and `speech asr --format json` write the same events.

namespace realtime {

/** The sample rate of the audio of a session, the one rate OpenAI's audio/pcm takes. */
constexpr int kRate = 24000;

/** The member of the session that sets the transcription's options. */
constexpr const char * kTranscription = "session.audio.input.transcription";

/**
 * The error of what a transcription or the check of its options threw: an ApiError as it is, the library's Failure by
 * its category, its param the member of the session that set the option at fault, and anything else as internal.
 */
openai::ApiError error_of(std::exception_ptr failure);

class Events {
public:
    /** `send` takes each event's JSON text, one at a time. */
    explicit Events(std::function<void(const std::string &)> send);

    /** The session's id: "sess_" and 16 hexadecimal digits that also begin every other id of the session. */
    const std::string & session_id() const { return session_id_; }

    /** Sends an event of `type` with `members`, each after a comma. */
    void emit(const std::string & type, const std::string & members);
    /** Sends the error event of `e`, about the client event `event_id`, or "" for none. */
    void error(const openai::ApiError & e, const std::string & event_id);
    /** Sends the event of an utterance, under the item id the utterance's first event took. */
    void utterance(const utterances::Event & e);
    /** The item id of an utterance, which its first event or this takes, until its last event. */
    std::string item(uint64_t utterance);

private:
    std::string new_id(const char * prefix);

    std::function<void(const std::string &)> send_;
    std::mutex mutex_;
    std::string unique_, session_id_;
    std::atomic<uint64_t> counter_{0};
    std::map<uint64_t, std::string> items_;
    std::string previous_item_;
};

}  // namespace realtime
