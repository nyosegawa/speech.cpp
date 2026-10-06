#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

/**
 * A JSON value as read, its numbers kept as they were written, so that a reader can tell an integer from a number with
 * a fraction or an exponent, and its object's members in their order.
 */
struct JsonValue {
    enum class Kind { Null, Bool, Number, String, Array, Object };
    Kind kind = Kind::Null;
    bool boolean = false;
    /** A string's text, or a number as it was written. */
    std::string text;
    std::vector<JsonValue> items;
    std::vector<std::pair<std::string, JsonValue>> members;

    /** The member of an object named `name`, or nullptr for an absent one and for a value that is not an object. */
    const JsonValue * member(const std::string & name) const;
    /** Whether the value is a number written without a fraction or an exponent. */
    bool is_integer() const;
};

/**
 * Reads `text`, which holds one JSON value (RFC 8259) with nothing but white space around it. Anything else throws
 * std::invalid_argument with what is wrong and at which byte: a value cut short, a number or an escape JSON does not
 * have, a control character in a string, a member that appears twice, or values nested more than 64 deep.
 */
JsonValue parse_json(const std::string & text);

/** The value as JSON text: strings escaped as json_string() escapes them, numbers as they were written. */
std::string to_json(const JsonValue & value);

/** The value as JSON text for a message, cut to about 60 bytes. */
std::string json_excerpt(const JsonValue & value);
