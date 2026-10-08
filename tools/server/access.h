#pragma once

#include <optional>
#include <string>
#include <vector>

#include "httplib.h"

#include "openai-api.h"

// Who may reach the server. It has no authentication, and any web page the user has open can send a request to
// 127.0.0.1, so: a request whose Origin is neither the server's own nor one --cors-origin gives is refused at every
// endpoint, which leaves curl and scripts, which send none, as they are; and the page and its endpoints, which fetch,
// load and replace models, exist only while the server listens on a loopback address and answer only a Host of that
// address and the server's port, against DNS rebinding.

namespace server {

class Access {
public:
    /** For a server that listens on `host` at `port`, allowing the origins of --cors-origin, "*" for any. */
    Access(const std::string & host, int port, std::vector<std::string> cors_origins);

    /** Whether the server listens on 127.0.0.1, ::1 or localhost, where the page and its endpoints exist. */
    bool loopback() const { return loopback_; }
    /** The page's address. */
    std::string page_url() const;

    /** The refusal of a request from an origin that may not call the server, or nothing. */
    std::optional<openai::ApiError> origin_refusal(const httplib::Request & req) const;
    /** Whether the origin of a request is one --cors-origin gives, whose responses carry CORS headers. */
    bool cross_origin_allowed(const std::string & origin) const;
    bool any_origin() const;
    /** The refusal of a request for the page or its WebSocket whose Host is not the server's loopback address and port, or nothing. */
    std::optional<openai::ApiError> host_refusal(const httplib::Request & req) const;

private:
    std::string url_host_;
    int port_;
    bool loopback_;
    std::vector<std::string> cors_origins_;
    /** The Host values and the origins of the page: 127.0.0.1, localhost and [::1] with the port. */
    std::vector<std::string> hosts_, origins_;
};

}  // namespace server
