#include "access.h"

#include <algorithm>
#include <cctype>

#include "json.h"

namespace server {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

}  // namespace

Access::Access(const std::string & host, int port, std::vector<std::string> cors_origins)
    : url_host_(host == "::1" ? "[::1]" : host), port_(port), loopback_(host == "127.0.0.1" || host == "::1" || host == "localhost"),
      cors_origins_(std::move(cors_origins)) {
    if (!loopback_) return;
    for (const char * name : {"127.0.0.1", "localhost", "[::1]"}) {
        const std::string with_port = std::string(name) + ":" + std::to_string(port);
        hosts_.push_back(with_port);
        // A browser leaves out the port that is the scheme's default.
        if (port == 80) hosts_.push_back(name);
        origins_.push_back("http://" + (port == 80 ? std::string(name) : with_port));
    }
}

std::string Access::page_url() const {
    return "http://" + url_host_ + ":" + std::to_string(port_) + "/";
}

bool Access::any_origin() const {
    return std::find(cors_origins_.begin(), cors_origins_.end(), "*") != cors_origins_.end();
}

bool Access::cross_origin_allowed(const std::string & origin) const {
    return !origin.empty() && (any_origin() || std::find(cors_origins_.begin(), cors_origins_.end(), origin) != cors_origins_.end());
}

std::optional<openai::ApiError> Access::origin_refusal(const httplib::Request & req) const {
    if (!req.has_header("Origin")) return std::nullopt;
    const std::string origin = req.get_header_value("Origin");
    if (std::find(origins_.begin(), origins_.end(), lower(origin)) != origins_.end() || cross_origin_allowed(origin)) return std::nullopt;
    return openai::ApiError{403, "This server refuses requests that a web page at " + json_string(origin) +
                                     " sends. Start speech serve with --cors-origin " + origin + " to let that page call it.",
                            "", "origin_not_allowed"};
}

std::optional<openai::ApiError> Access::host_refusal(const httplib::Request & req) const {
    const std::string host = lower(req.get_header_value("Host"));
    if (std::find(hosts_.begin(), hosts_.end(), host) != hosts_.end()) return std::nullopt;
    return openai::ApiError{403, "The page and /v1/realtime answer requests for 127.0.0.1, localhost or [::1] at port " + std::to_string(port_) +
                                     " alone, and this one named the host " + json_string(host) + ". Use the address speech serve printed.",
                            "", "host_not_allowed"};
}

}  // namespace server
