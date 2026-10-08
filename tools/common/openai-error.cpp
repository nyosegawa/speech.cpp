#include "openai-error.h"

#include "json.h"

namespace openai {

ApiError library_error(const Failure & failure) {
    const std::string & option = failure.option();
    const std::string param = option == "text" ? "input" : option == "audio" ? "file" : option;
    const std::string & code = failure.code();
    if (code == speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT)) return {400, failure.what(), param, "invalid_value"};
    if (code == speech_status_name(SPEECH_ERROR_UNSUPPORTED)) return {400, failure.what(), param, "unsupported_parameter"};
    if (code == speech_status_name(SPEECH_ERROR_OUT_OF_RANGE)) return {400, failure.what(), param, "unsupported_value"};
    return {500, failure.what(), param, code};
}

std::string error_object(const ApiError & e) {
    const char * type = e.status >= 500 ? "server_error" : "invalid_request_error";
    return "{\"message\":" + json_string(e.message) + ",\"type\":\"" + type + "\",\"param\":" + (e.param.empty() ? "null" : json_string(e.param)) +
           ",\"code\":" + (e.code.empty() ? "null" : json_string(e.code)) + "}";
}

std::string error_json(const ApiError & e) {
    return "{\"error\":" + error_object(e) + "}";
}

}  // namespace openai
