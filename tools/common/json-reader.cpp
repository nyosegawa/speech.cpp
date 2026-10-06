#include "json-reader.h"

#include <cstdint>
#include <stdexcept>

#include "json.h"

namespace {

/** The deepest nesting read; a line of the worker's protocol nests two or three deep. */
constexpr int kMaxDepth = 64;

class Reader {
public:
    explicit Reader(const std::string & s) : s_(s) {}

    JsonValue document() {
        space();
        JsonValue v = value(0);
        space();
        if (i_ != s_.size()) fail("the JSON value ends and more follows");
        return v;
    }

private:
    const std::string & s_;
    size_t i_ = 0;

    [[noreturn]] void fail(const std::string & what) const {
        throw std::invalid_argument(what + " at byte " + std::to_string(i_));
    }

    void space() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\r' || s_[i_] == '\n')) i_++;
    }

    bool literal(const char * word) {
        size_t n = 0;
        while (word[n]) n++;
        if (s_.compare(i_, n, word) != 0) return false;
        i_ += n;
        return true;
    }

    JsonValue value(int depth) {
        if (i_ >= s_.size()) fail("a value is missing");
        JsonValue v;
        const char c = s_[i_];
        if (c == '{' || c == '[') {
            if (depth >= kMaxDepth) fail("the values nest more than " + std::to_string(kMaxDepth) + " deep");
            return c == '{' ? object(depth) : array(depth);
        }
        if (c == '"') {
            v.kind = JsonValue::Kind::String;
            v.text = string();
        } else if (literal("true")) {
            v.kind = JsonValue::Kind::Bool;
            v.boolean = true;
        } else if (literal("false")) {
            v.kind = JsonValue::Kind::Bool;
        } else if (literal("null")) {
            v.kind = JsonValue::Kind::Null;
        } else {
            v.kind = JsonValue::Kind::Number;
            v.text = number();
        }
        return v;
    }

    JsonValue object(int depth) {
        JsonValue v;
        v.kind = JsonValue::Kind::Object;
        i_++;
        space();
        if (i_ < s_.size() && s_[i_] == '}') {
            i_++;
            return v;
        }
        for (;;) {
            space();
            if (i_ >= s_.size() || s_[i_] != '"') fail("a member's name is missing");
            std::string name = string();
            if (v.member(name)) fail("the member \"" + name + "\" appears twice");
            space();
            if (i_ >= s_.size() || s_[i_] != ':') fail("':' is missing after a member's name");
            i_++;
            space();
            v.members.emplace_back(std::move(name), value(depth + 1));
            space();
            if (i_ < s_.size() && s_[i_] == ',') {
                i_++;
                continue;
            }
            if (i_ < s_.size() && s_[i_] == '}') {
                i_++;
                return v;
            }
            fail("',' or '}' is missing");
        }
    }

    JsonValue array(int depth) {
        JsonValue v;
        v.kind = JsonValue::Kind::Array;
        i_++;
        space();
        if (i_ < s_.size() && s_[i_] == ']') {
            i_++;
            return v;
        }
        for (;;) {
            space();
            v.items.push_back(value(depth + 1));
            space();
            if (i_ < s_.size() && s_[i_] == ',') {
                i_++;
                continue;
            }
            if (i_ < s_.size() && s_[i_] == ']') {
                i_++;
                return v;
            }
            fail("',' or ']' is missing");
        }
    }

    std::string number() {
        const size_t start = i_;
        const auto digits = [&] {
            const size_t from = i_;
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') i_++;
            return i_ - from;
        };
        if (i_ < s_.size() && s_[i_] == '-') i_++;
        if (i_ < s_.size() && s_[i_] == '0') {
            i_++;
        } else if (digits() == 0) {
            fail("a value is not JSON");
        }
        if (i_ < s_.size() && s_[i_] == '.') {
            i_++;
            if (digits() == 0) fail("a number's fraction has no digits");
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            i_++;
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) i_++;
            if (digits() == 0) fail("a number's exponent has no digits");
        }
        return s_.substr(start, i_ - start);
    }

    uint32_t hex4() {
        if (i_ + 4 > s_.size()) fail("a \\u escape is cut short");
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) {
            const char c = s_[i_++];
            const int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
            if (d < 0) fail("a \\u escape has a character that is not hexadecimal");
            v = v << 4 | (uint32_t) d;
        }
        return v;
    }

    static void append_utf8(std::string & out, uint32_t cp) {
        if (cp < 0x80) {
            out += (char) cp;
        } else if (cp < 0x800) {
            out += (char) (0xC0 | (cp >> 6));
            out += (char) (0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char) (0xE0 | (cp >> 12));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        } else {
            out += (char) (0xF0 | (cp >> 18));
            out += (char) (0x80 | ((cp >> 12) & 0x3F));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        }
    }

    std::string string() {
        std::string out;
        i_++;
        while (i_ < s_.size()) {
            const unsigned char c = (unsigned char) s_[i_];
            if (c == '"') {
                i_++;
                return out;
            }
            if (c < 0x20) fail("a string holds a control character, which JSON writes as an escape");
            if (c != '\\') {
                out += (char) c;
                i_++;
                continue;
            }
            if (++i_ >= s_.size()) break;
            const char e = s_[i_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xDC00 && cp < 0xE000) fail("a \\u escape is the second half of a surrogate pair without the first");
                    if (cp >= 0xD800 && cp < 0xDC00) {
                        if (s_.compare(i_, 2, "\\u") != 0) fail("a \\u escape is the first half of a surrogate pair without the second");
                        i_ += 2;
                        const uint32_t low = hex4();
                        if (low < 0xDC00 || low >= 0xE000) fail("a surrogate pair's second half is not one");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: i_--; fail(std::string("a string has the escape \\") + e + ", which JSON does not have");
            }
        }
        fail("a string is not closed");
    }
};

}  // namespace

const JsonValue * JsonValue::member(const std::string & name) const {
    if (kind != Kind::Object) return nullptr;
    for (const auto & [key, value] : members) {
        if (key == name) return &value;
    }
    return nullptr;
}

bool JsonValue::is_integer() const {
    return kind == Kind::Number && text.find_first_of(".eE") == std::string::npos;
}

JsonValue parse_json(const std::string & text) {
    return Reader(text).document();
}

std::string to_json(const JsonValue & value) {
    switch (value.kind) {
        case JsonValue::Kind::Null: return "null";
        case JsonValue::Kind::Bool: return value.boolean ? "true" : "false";
        case JsonValue::Kind::Number: return value.text;
        case JsonValue::Kind::String: return json_string(value.text);
        case JsonValue::Kind::Array: {
            std::string out = "[";
            for (size_t i = 0; i < value.items.size(); i++) out += (i ? "," : "") + to_json(value.items[i]);
            return out + "]";
        }
        case JsonValue::Kind::Object: {
            std::string out = "{";
            for (size_t i = 0; i < value.members.size(); i++) {
                out += (i ? "," : "") + json_string(value.members[i].first) + ":" + to_json(value.members[i].second);
            }
            return out + "}";
        }
    }
    return "null";
}

std::string json_excerpt(const JsonValue & value) {
    std::string text = to_json(value);
    constexpr size_t kLength = 60;
    if (text.size() <= kLength) return text;
    // A cut inside a UTF-8 sequence would leave a broken character in the message.
    size_t cut = kLength;
    while (cut > 0 && ((unsigned char) text[cut] & 0xC0) == 0x80) cut--;
    return text.substr(0, cut) + "...";
}
