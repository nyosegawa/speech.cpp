#pragma once

#include <cstddef>
#include <optional>

#include "httplib.h"

#include "access.h"
#include "openai-error.h"
#include "served-models.h"

// OpenAI's Realtime transcription over a WebSocket at /v1/realtime, the address that openai-python 3.26.0's
// client.realtime.connect() opens from a base URL of http://HOST:PORT/v1, with its query's `model`. The session of
// tools/common reads the client's events and writes the server's; this carries them over cpp-httplib's WebSocket and
// transcribes each commit on the recognition model held, in its turn among the HTTP requests on that model.

namespace server {

class Realtime {
public:
    /** `limit` bounds the bytes of 16-bit PCM a session holds before they are transcribed. */
    Realtime(ServedModels & models, const Access & access, size_t limit) : models_(models), access_(access), limit_(limit) {}

    /**
     * The refusal of a request for /v1/realtime before it is upgraded, as an HTTP error: a Host other than the server's
     * own on a loopback address, a query member other than `model`, a model other than the recognition model held, or
     * no recognition model, a 503 while the page loads one; otherwise nothing.
     */
    std::optional<openai::ApiError> refusal(const httplib::Request & req) const;

    void route(httplib::Server & http);

private:
    ServedModels & models_;
    const Access & access_;
    size_t limit_;
};

}  // namespace server
