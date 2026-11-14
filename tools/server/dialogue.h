#pragma once

#include <atomic>
#include <optional>

#include "httplib.h"

#include "access.h"
#include "openai-error.h"
#include "served-models.h"

// speech.cpp's own conversation over a WebSocket at /v1/dialogue, held with the dialogue model the server holds. The
// client starts it with a text message {"type": "start", "sample_rate": R, ...request options by name}, sends the audio
// it hears as 16-bit little-endian PCM at R in binary messages, and ends it with {"type": "hang_up"}. The server answers
// {"type": "started", "sample_rate", "frame_samples"} once the prompt has run, then each frame of the model's audio,
// 16-bit PCM at the model's rate in a binary message, the text the model began to say in it going before it as
// {"type": "text", "text"}; the first frame answers the prompt and each later one a frame heard. A hang-up gets
// {"type": "end", "frames", "samples"} and a failure {"type": "error", "error": OpenAI's error object}, after which the
// server closes the connection. The server computes each frame once its audio has arrived and keeps no time; the client
// that plays the frames live judges whether they keep up. One conversation runs at a time.

namespace server {

class Dialogues {
public:
    Dialogues(ServedModels & models, const Access & access) : models_(models), access_(access) {}

    /**
     * The refusal of a request for /v1/dialogue before it is upgraded, as an HTTP error: a Host other than the server's
     * own on a loopback address, any query, no dialogue model or a 503 while the page loads one, or a conversation
     * already going on; otherwise nothing.
     */
    std::optional<openai::ApiError> refusal(const httplib::Request & req) const;

    void route(httplib::Server & http);

private:
    ServedModels & models_;
    const Access & access_;
    std::atomic<bool> talking_{false};
};

}  // namespace server
