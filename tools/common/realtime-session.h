#pragma once

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
#include "realtime-events.h"
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

/**
 * A commit's place on the recognition model held when the session accepted it: the model, which it keeps until it has
 * run, and its turn among the requests on that model, given up if it never runs.
 */
class Reservation {
public:
    virtual ~Reservation() = default;
};

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
    /** Takes a place on the model held, named `model` unless that is "", for a commit; it throws as check() does. */
    virtual std::unique_ptr<Reservation> reserve(const std::string & model) = 0;
    /**
     * Recognizes `samples` at kRate in `reservation`'s turn on its model, with `options`, and returns the recognition, or
     * nothing once `cancellation` stops it; an option the model does not take throws the library's Failure. `item` names
     * the item for a log.
     */
    virtual std::optional<Transcript> transcribe(Reservation & reservation, const std::vector<float> & samples,
                                                 const std::vector<RequestOption> & options, Cancellation & cancellation,
                                                 const std::string & item) = 0;
};

class Session {
public:
    /**
     * `send` takes each server event as one JSON text, one at a time, from the thread that calls the session or its own.
     * `limit` bounds the bytes of 16-bit PCM the session holds before they are transcribed, appended or committed, and
     * `model` names the recognition model the session starts with, or is "" for the one held.
     */
    Session(Recognizer & recognizer, std::function<void(const std::string &)> send, size_t limit, const std::string & model);
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

    /** A committed buffer that waits for its transcription, with its place on the model or why it has none. */
    struct Commit {
        uint64_t number = 0;
        std::vector<float> samples;
        Config config;
        std::unique_ptr<Reservation> reservation;
        std::optional<openai::ApiError> refused;
    };

    void update(const JsonValue & event);
    void append(const JsonValue & event);
    void commit();
    void clear();
    std::string session_json();
    void work();

    Recognizer & recognizer_;
    Events events_;
    Config config_;
    /** The audio appended since the last commit or clear, as 16-bit little-endian PCM. */
    std::string buffer_;
    uint64_t commits_ = 0;

    const size_t limit_;

    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<Commit> waiting_;
    /** The bytes of 16-bit PCM of the commits that wait or run. */
    size_t committed_ = 0;
    bool closing_ = false;
    std::shared_ptr<Cancellation> running_;
    std::thread worker_;
};

}  // namespace realtime
