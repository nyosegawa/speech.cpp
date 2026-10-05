#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

/** Standard base64 (RFC 4648, section 4) with padding, as the worker's chunks and the server's SSE events carry PCM. */
inline std::string base64(const uint8_t * data, size_t n) {
    static const char * table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((n + 2) / 3 * 4);
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t) data[i] << 16 | (i + 1 < n ? (uint32_t) data[i + 1] << 8 : 0) | (i + 2 < n ? data[i + 2] : 0);
        out += table[(v >> 18) & 63];
        out += table[(v >> 12) & 63];
        out += i + 1 < n ? table[(v >> 6) & 63] : '=';
        out += i + 2 < n ? table[v & 63] : '=';
    }
    return out;
}

/** The bytes of standard base64 with padding; anything else (a stray character, a length not a multiple of 4) throws. */
inline std::string base64_decode(const std::string & text) {
    auto value = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    if (text.size() % 4 != 0) throw std::invalid_argument("its length is not a multiple of 4");
    std::string out;
    out.reserve(text.size() / 4 * 3);
    for (size_t i = 0; i < text.size(); i += 4) {
        const bool last = i + 4 == text.size();
        const int pad = last ? (text[i + 3] == '=') + (text[i + 2] == '=' && text[i + 3] == '=') : 0;
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) {
            const int d = k >= 4 - pad ? 0 : value(text[i + k]);
            if (d < 0) throw std::invalid_argument(std::string("it holds the character '") + text[i + k] + "', which base64 does not use");
            v = v << 6 | (uint32_t) d;
        }
        out += (char) (v >> 16);
        if (pad < 2) out += (char) ((v >> 8) & 0xFF);
        if (pad < 1) out += (char) (v & 0xFF);
    }
    return out;
}
