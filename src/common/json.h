#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

/** A JSON string literal of UTF-8 text. */
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

/**
 * The shortest decimal that reads back as `v` through `read`, at most `digits` significant digits, as JavaScript writes
 * a number: 655.36, 30, 1e-05. %g gives an exponent wherever it has fewer digits than the integer part ("3e+01"), so a
 * number from 0.0001 to below 1e17 is written in fixed notation instead. JSON has no infinity or NaN, so they are
 * written as null.
 */
template <typename T, typename Read>
std::string json_shortest(T v, int digits, Read read) {
    if (!std::isfinite(v)) return "null";
    char buf[64];
    for (int precision = 1; precision <= digits; precision++) {
        std::snprintf(buf, sizeof buf, "%.*g", precision, (double) v);
        if (read(buf) == v) break;
    }
    const double magnitude = std::fabs((double) v);
    if (std::strchr(buf, 'e') && magnitude >= 1e-4 && magnitude < 1e17) {
        for (int decimals = 0; decimals <= 20; decimals++) {
            std::snprintf(buf, sizeof buf, "%.*f", decimals, (double) v);
            if (read(buf) == v) break;
        }
    }
    return buf;
}

inline std::string json_number(double v) {
    return json_shortest(v, 17, [](const char * s) { return std::strtod(s, nullptr); });
}

/** The same for a float32, the shortest decimal that reads back as `v` in single precision. */
inline std::string json_number(float v) {
    return json_shortest(v, 9, [](const char * s) { return std::strtof(s, nullptr); });
}
