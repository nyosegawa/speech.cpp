#pragma once

#include <string>

#include "failure.h"

// OpenAI's error object, which the server answers a refused request with and puts in the error events of a Realtime
// session, with the library's categories mapped onto it.

namespace openai {

/** An error in OpenAI's shape: an HTTP status, its type following from the status, a message, the input at fault and a code. */
struct ApiError {
    int status;
    std::string message;
    std::string param;
    std::string code;
};

/**
 * The error of a failure of the library, by its category alone: invalid_argument, unsupported and out_of_range are
 * the request's (400, invalid_value, unsupported_parameter and unsupported_value), every other category the server's
 * (500, the category's name as the code). The param is the input at fault, "text" written as "input" and "audio" as
 * "file", the names OpenAI's requests give them.
 */
ApiError library_error(const Failure & failure);

/** The members of OpenAI's error object: {"message", "type", "param", "code"}. */
std::string error_object(const ApiError & e);

/** An error response's body: {"error": {...}}. */
std::string error_json(const ApiError & e);

}  // namespace openai
