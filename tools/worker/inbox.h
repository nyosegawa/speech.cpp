#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "base64.h"
#include "flat-json.h"
#include "speech.h"

// The worker's side of reading the protocol: what a line on stdin may hold for the worker's task, the requests and
// cancels it queues, and the audio of a recognition request gathered from its chunks.

inline std::string value(const FlatJson & request, const char * key) {
    const auto it = request.find(key);
    return it == request.end() ? "" : it->second.text;
}

/** A number member of a request, or `absent` when the request leaves it out; a string or a non-finite number throws. */
inline double number(const FlatJson & request, const char * key, double absent) {
    const auto it = request.find(key);
    if (it == request.end()) return absent;
    const std::string & text = it->second.text;
    char * end = nullptr;
    const double v = std::strtod(text.c_str(), &end);
    if (it->second.kind != FlatValue::NUMBER || end != text.c_str() + text.size() || !std::isfinite(v)) {
        throw std::invalid_argument(std::string("\"") + key + "\" is " + json_string(text) + "; give it as a JSON number");
    }
    return v;
}

/**
 * A member that holds a whole number, such as a chunk's "seq" or an end's "sampleRate", or nothing when it is left
 * out; a string, a fraction, a sign or an exponent throws.
 */
inline bool whole_number(const FlatJson & message, const char * key, uint64_t & out) {
    const auto it = message.find(key);
    if (it == message.end()) return false;
    const std::string & t = it->second.text;
    const bool digits = it->second.kind == FlatValue::NUMBER && !t.empty() && t.size() <= 15 &&
                        std::all_of(t.begin(), t.end(), [](char c) { return c >= '0' && c <= '9'; });
    if (!digits) throw std::invalid_argument(std::string("\"") + key + "\" is " + json_string(t) + "; give it as a whole number");
    out = std::stoull(t);
    return true;
}

/** A request taken off stdin: a request to speak, or the end of a recognition request with the audio of its chunks. */
struct Request {
    FlatJson message;
    std::vector<int16_t> pcm;
};

/** A recognition request whose end has not arrived. */
struct Collecting {
    uint64_t next_seq = 0;
    std::vector<int16_t> pcm;
    /** A chunk was refused and answered with an error, so the request's other lines are dropped. */
    bool failed = false;
};

struct Inbox {
    speech_task task;
    speech_model * model;
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<Request> requests;
    std::set<std::string> cancelled;
    std::map<std::string, Collecting> collecting;
    /** The recognition request under way, which a cancel also stops in the library. */
    std::string running;
    bool closed = false;

    bool is_cancelled(const std::string & id) {
        std::lock_guard<std::mutex> lock(mutex);
        return cancelled.count(id) > 0;
    }
    void forget(const std::string & id) {
        std::lock_guard<std::mutex> lock(mutex);
        cancelled.erase(id);
    }
};

/** Why a message is not one the worker can serve for its task, or nothing when it is one. */
inline std::string unreadable(const FlatJson & message, speech_task task) {
    for (const auto & [key, v] : message) {
        if (v.kind == FlatValue::OTHER) {
            return "the member " + json_string(key) + " is " + v.text + "; the protocol's members are strings and numbers";
        }
    }
    if (!message.count("id")) return "the message has no \"id\"; every request and cancel needs one";
    const auto type = message.find("type");
    if (task == SPEECH_TASK_SYNTHESIS) {
        if (type != message.end() && type->second.text != "cancel") {
            return "the message's \"type\" is " + json_string(type->second.text) + "; a request has none and a cancel has \"cancel\"";
        }
        return "";
    }
    const char * takes = "a recognition worker takes \"chunk\", \"end\" and \"cancel\"";
    if (type == message.end()) {
        return std::string("the message has no \"type\", as a request to speak has; ") + takes + ", and speech is spoken by a synthesis model";
    }
    if (type->second.text != "chunk" && type->second.text != "end" && type->second.text != "cancel") {
        return "the message's \"type\" is " + json_string(type->second.text) + "; " + takes;
    }
    return "";
}

/**
 * Takes a recognition worker's chunk, end or cancel, with the inbox locked, and returns why the line is refused, or
 * nothing. A request with a refused line is answered by that refusal alone and drops its other lines.
 */
inline std::string take_recognition(Inbox & inbox, const FlatJson & message) {
    const std::string id = value(message, "id"), type = value(message, "type");
    if (type == "cancel") {
        inbox.cancelled.insert(id);
        inbox.collecting.erase(id);
        if (inbox.running == id) speech_cancel(inbox.model);
        return "";
    }
    if (inbox.cancelled.count(id)) {
        // The end of a request cancelled while it took chunks closes it.
        if (type == "end") inbox.cancelled.erase(id);
        return "";
    }
    Collecting & c = inbox.collecting[id];
    if (type == "chunk") {
        if (c.failed) return "";
        std::string problem;
        try {
            uint64_t seq = 0;
            if (!whole_number(message, "seq", seq)) throw std::invalid_argument("the chunk has no \"seq\"");
            if (seq != c.next_seq) {
                throw std::invalid_argument("the chunk's \"seq\" is " + std::to_string(seq) + " where " + std::to_string(c.next_seq) +
                                            " comes next; send a request's chunks in order from 0");
            }
            if (!message.count("pcm")) throw std::invalid_argument("the chunk has no \"pcm\"");
            std::string bytes;
            try {
                bytes = base64_decode(value(message, "pcm"));
            } catch (const std::exception & e) {
                throw std::invalid_argument(std::string("the chunk's \"pcm\" is not base64: ") + e.what());
            }
            if (bytes.size() % 2 != 0) throw std::invalid_argument("the chunk's \"pcm\" holds an odd number of bytes; it is 16-bit samples");
            const size_t at = c.pcm.size();
            c.pcm.resize(at + bytes.size() / 2);
            for (size_t i = 0; i < bytes.size() / 2; i++) {
                c.pcm[at + i] = (int16_t) (uint16_t) ((uint8_t) bytes[2 * i] | (uint8_t) bytes[2 * i + 1] << 8);
            }
            c.next_seq++;
        } catch (const std::exception & e) {
            problem = e.what();
        }
        if (!problem.empty()) {
            c.failed = true;
            c.pcm = {};
        }
        return problem;
    }
    Request request{message, std::move(c.pcm)};
    const bool failed = c.failed;
    inbox.collecting.erase(id);
    if (failed) return "";
    uint64_t rate = 0;
    try {
        if (!whole_number(message, "sampleRate", rate)) throw std::invalid_argument("the end has no \"sampleRate\"");
        if (rate == 0 || rate > INT_MAX) throw std::invalid_argument("the end's \"sampleRate\" is " + std::to_string(rate));
    } catch (const std::exception & e) {
        return std::string(e.what()) + "; the end of a request gives the rate of its PCM in Hz";
    }
    inbox.requests.push_back(std::move(request));
    inbox.ready.notify_one();
    return "";
}

/**
 * Reads stdin line by line until it closes, queueing the requests and taking the cancels, and answers a line it
 * refuses at once through `emit`.
 */
inline void read_requests(Inbox & inbox, void (*emit)(const std::string & json)) {
    std::string line;
    while (std::getline(std::cin, line)) {
        FlatJson message;
        std::string problem;
        try {
            message = parse_flat_json(line);
            problem = unreadable(message, inbox.task);
        } catch (const std::exception & e) {
            problem = std::string("a line on stdin cannot be read: ") + e.what() + "; send one JSON object per line";
        }
        if (problem.empty()) {
            std::lock_guard<std::mutex> lock(inbox.mutex);
            if (inbox.task == SPEECH_TASK_RECOGNITION) {
                problem = take_recognition(inbox, message);
            } else if (value(message, "type") == "cancel") {
                inbox.cancelled.insert(value(message, "id"));
            } else {
                inbox.requests.push_back({message, {}});
                inbox.ready.notify_one();
            }
        }
        if (!problem.empty()) {
            const auto id = message.find("id");
            const bool has_id = id != message.end() && id->second.kind != FlatValue::OTHER;
            emit("{\"type\":\"error\"," + (has_id ? "\"id\":" + json_string(id->second.text) + "," : std::string()) +
                 "\"error\":" + json_string(problem) + "}");
        }
    }
    std::lock_guard<std::mutex> lock(inbox.mutex);
    inbox.closed = true;
    inbox.ready.notify_one();
}
