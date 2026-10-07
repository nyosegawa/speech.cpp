#include "access.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <random>

#include "json.h"

namespace server {

namespace {

/** 128 bits from the system's random source, as 32 hexadecimal digits. */
std::string make_token() {
    std::random_device device;
    std::string out;
    for (int i = 0; i < 4; i++) {
        char part[9];
        std::snprintf(part, sizeof part, "%08x", (unsigned) device());
        out += part;
    }
    return out;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

/** Whether two strings are equal, in a time that does not tell how much of them agrees. */
bool same(const std::string & a, const std::string & b) {
    if (a.size() != b.size()) return false;
    unsigned char differ = 0;
    for (size_t i = 0; i < a.size(); i++) differ |= (unsigned char) (a[i] ^ b[i]);
    return differ == 0;
}

}  // namespace

Access::Access(const std::string & host, int port, std::vector<std::string> cors_origins)
    : url_host_(host == "::1" ? "[::1]" : host), port_(port), loopback_(host == "127.0.0.1" || host == "::1" || host == "localhost"),
      cors_origins_(std::move(cors_origins)), token_(make_token()) {
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
    return "http://" + url_host_ + ":" + std::to_string(port_) + "/#token=" + token_;
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
    return openai::ApiError{403, "The page answers requests for 127.0.0.1, localhost or [::1] at port " + std::to_string(port_) +
                                     " alone, and this one named the host " + json_string(host) + ". Open the address speech serve printed.",
                            "", "host_not_allowed"};
}

std::optional<openai::ApiError> Access::token_refusal(const httplib::Request & req) const {
    const std::string given = req.get_header_value("Authorization");
    const std::string prefix = "Bearer ";
    if (given.compare(0, prefix.size(), prefix) != 0) {
        return openai::ApiError{401, "This endpoint wants the token that speech serve printed with the page's address, as "
                                     "\"Authorization: Bearer TOKEN\". Open the page from that address.",
                                "", "missing_token"};
    }
    if (!same(given.substr(prefix.size()), token_)) {
        return openai::ApiError{403, "The token is not this server's. speech serve makes a new one each time it starts; open the address it "
                                     "printed this time.",
                                "", "invalid_token"};
    }
    return std::nullopt;
}

}  // namespace server
