#pragma once

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

/**
 * A member's value: a string's text, a number as it was written, or any other JSON value (an object, an array,
 * true, false or null) as it was written, which the protocol does not take but a reader can name.
 */
struct FlatValue {
    enum Kind { STRING, NUMBER, OTHER };
    std::string text;
    Kind kind = STRING;
};

/**
 * A JSON object read one level deep, which is all the worker protocol sends: its string and number members, and
 * any other member kept as OTHER so that the reader can refuse it by name. Text that is not one JSON object
 * throws.
 */
using FlatJson = std::map<std::string, FlatValue>;

namespace flat_json {

inline void append_utf8(std::string & out, uint32_t cp) {
    if (cp < 0x80) out += (char) cp;
    else if (cp < 0x800) { out += (char) (0xC0 | (cp >> 6)); out += (char) (0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { out += (char) (0xE0 | (cp >> 12)); out += (char) (0x80 | ((cp >> 6) & 0x3F)); out += (char) (0x80 | (cp & 0x3F)); }
    else { out += (char) (0xF0 | (cp >> 18)); out += (char) (0x80 | ((cp >> 12) & 0x3F)); out += (char) (0x80 | ((cp >> 6) & 0x3F)); out += (char) (0x80 | (cp & 0x3F)); }
}

inline std::string parse_string(const std::string & s, size_t & i) {
    if (s[i] != '"') throw std::runtime_error("expected a string");
    std::string out;
    for (i++; i < s.size(); i++) {
        const char c = s[i];
        if (c == '"') { i++; return out; }
        if (c != '\\') { out += c; continue; }
        if (++i >= s.size()) break;
        switch (s[i]) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                auto hex4 = [&](size_t at) { return (uint32_t) std::stoul(s.substr(at, 4), nullptr, 16); };
                uint32_t cp = hex4(i + 1);
                i += 4;
                if (cp >= 0xD800 && cp < 0xDC00 && i + 6 < s.size() && s[i + 1] == '\\' && s[i + 2] == 'u') {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (hex4(i + 3) - 0xDC00);
                    i += 6;
                }
                append_utf8(out, cp);
                break;
            }
            default: throw std::runtime_error("bad escape in a string");
        }
    }
    throw std::runtime_error("unterminated string");
}

inline void skip_space(const std::string & s, size_t & i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) i++;
}

/** Reads the JSON value at `i` and returns its kind; a string's text goes to `text`, a number's characters too. */
inline FlatValue::Kind parse_value(const std::string & s, size_t & i, std::string & text) {
    if (i >= s.size()) throw std::runtime_error("a value is missing");
    const char c = s[i];
    if (c == '"') {
        text = parse_string(s, i);
        return FlatValue::STRING;
    }
    if (c == '{' || c == '[') {
        const char close = c == '{' ? '}' : ']';
        i++;
        skip_space(s, i);
        if (i < s.size() && s[i] == close) {
            i++;
            return FlatValue::OTHER;
        }
        for (;;) {
            std::string ignored;
            if (c == '{') {
                skip_space(s, i);
                if (i >= s.size() || s[i] != '"') throw std::runtime_error("expected a member's name");
                parse_string(s, i);
                skip_space(s, i);
                if (i >= s.size() || s[i] != ':') throw std::runtime_error("expected ':'");
                i++;
            }
            skip_space(s, i);
            parse_value(s, i, ignored);
            skip_space(s, i);
            if (i < s.size() && s[i] == ',') { i++; continue; }
            if (i < s.size() && s[i] == close) { i++; return FlatValue::OTHER; }
            throw std::runtime_error(std::string("expected ',' or '") + close + "'");
        }
    }
    for (const char * word : {"true", "false", "null"}) {
        if (s.compare(i, std::strlen(word), word) == 0) {
            text = word;
            i += std::strlen(word);
            return FlatValue::OTHER;
        }
    }
    const size_t start = i;
    while (i < s.size() && (std::isdigit((unsigned char) s[i]) || s[i] == '-' || s[i] == '+' || s[i] == '.' || s[i] == 'e' || s[i] == 'E')) i++;
    if (i == start) throw std::runtime_error("expected a JSON value");
    text = s.substr(start, i - start);
    return FlatValue::NUMBER;
}

}  // namespace flat_json

inline FlatJson parse_flat_json(const std::string & s) {
    FlatJson out;
    size_t i = 0;
    flat_json::skip_space(s, i);
    if (i >= s.size() || s[i] != '{') throw std::runtime_error("the line is not a JSON object");
    std::string ignored;
    const size_t start = i;
    flat_json::parse_value(s, i, ignored);
    flat_json::skip_space(s, i);
    if (i != s.size()) throw std::runtime_error("the line goes on after its JSON object");
    // The whole object was read once above, so its members are read again without the checks.
    i = start + 1;
    for (;;) {
        flat_json::skip_space(s, i);
        if (s[i] == '}') return out;
        const std::string key = flat_json::parse_string(s, i);
        flat_json::skip_space(s, i);
        i++;
        flat_json::skip_space(s, i);
        const size_t value_start = i;
        FlatValue v;
        v.kind = flat_json::parse_value(s, i, v.text);
        if (v.kind == FlatValue::OTHER) v.text = s.substr(value_start, i - value_start);
        out[key] = v;
        flat_json::skip_space(s, i);
        if (s[i] == ',') i++;
    }
}

/** A JSON string literal of `s`, which is UTF-8. */
inline std::string json_string(const std::string & s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    return out + "\"";
}
