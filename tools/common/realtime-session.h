#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "cancellation.h"
#include "json-reader.h"
#include "openai-error.h"
#include "request-options.h"
#include "transcript.h"

// A session of OpenAI's Realtime transcription, as its API reference gives the GA events at github.com/openai/
// openai-openapi commit 234829e (2026-10-07): the client's session.update, input_audio_buffer.append, .commit and
// .clear, and the server's session.created, session.updated, input_audio_buffer.committed and .cleared,
// conversation.item.input_audio_transcription.delta, .completed and .failed, and error. The session reads and checks
// each client event, keeps its configuration and the audio appended, transcribes each committed buffer on a thread of
// its own in the order of the commits, and writes the server's events as JSON text; whoever carries the messages (the
// server's WebSocket) hands it the client's and sends its own. A member or a value speech.cpp does not take is answered
// with an error event rather than ignored.

namespace realtime {

/** The sample rate of the audio of a session, the one rate OpenAI's audio/pcm takes. */
constexpr int kRate = 24000;

/** The recognition model a session transcribes with, as its host holds it. */
class Recognizer {
public:
    virtual ~Recognizer() = default;
    /** The name of the recognition model held now, or "" while there is none. */
    virtual std::string held() = 0;
    /**
     * Checks `options` against the model held, named `model` unless that is "": a model of another name or none throws an
     * ApiError, and an option the model does not take the library's Failure.
     */
    virtual void check(const std::string & model, const std::vector<RequestOption> & options) = 0;
    /**
     * Recognizes `samples` at kRate on the model held, named `model` unless that is "", with `options`, and returns the
     * recognition, or nothing once `cancellation` stops it; it throws as check() does. `item` names the item for a log.
     */
    virtual std::optional<Transcript> transcribe(const std::string & model, const std::vector<float> & samples,
                                                 const std::vector<RequestOption> & options, Cancellation & cancellation,
                                                 const std::string & item) = 0;
};

class Session {
public:
    /** `send` takes each server event as one JSON text, one at a time, from the thread that calls the session or its own. */
    Session(Recognizer & recognizer, std::function<void(const std::string &)> send);
    /** Stops the transcription under way and drops those that wait. */
    ~Session();
    Session(const Session &) = delete;
    Session & operator=(const Session &) = delete;

    /** Sends session.created, the configuration the session starts with. */
    void open();
    /** Handles a client event: one JSON object in a text message; a binary message is refused. */
    void receive(const std::string & message, bool text);

private:
    /** The configuration a session.update changes. */
    struct Config {
        /** The transcription model the client named, or nothing for the one held. */
        std::optional<std::string> model;
        /** Whether a committed buffer is transcribed: audio.input.transcription is not null. */
        bool transcribe = true;
        std::optional<std::string> language;
        std::optional<std::string> prompt;

        std::vector<RequestOption> options() const;
    };

    /** A committed buffer that waits for its transcription. */
    struct Commit {
        std::string item;
        std::vector<float> samples;
        Config config;
    };

    void update(const JsonValue & event);
    void append(const JsonValue & event);
    void commit();
    void clear();
    /** Sends an event of `type` with `members`, each after a comma, and an event_id of the session's own. */
    void emit(const std::string & type, const std::string & members);
    void error(const openai::ApiError & e, const std::string & event_id);
    std::string session_json();
    std::string new_id(const char * prefix);
    void work();

    Recognizer & recognizer_;
    std::function<void(const std::string &)> send_;
    std::mutex send_mutex_;
    std::string id_;
    std::string unique_;
    std::atomic<uint64_t> counter_{0};
    Config config_;
    /** The audio appended since the last commit or clear, as 16-bit little-endian PCM. */
    std::string buffer_;
    std::string previous_item_;

    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<Commit> waiting_;
    bool closing_ = false;
    std::shared_ptr<Cancellation> running_;
    std::thread worker_;
};

}  // namespace realtime
