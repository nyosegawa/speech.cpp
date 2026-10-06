#pragma once

#include <stdexcept>
#include <string>
#include <utility>

#include "speech.h"

/**
 * A failure in the library's terms, as every subcommand reports it: the name of its category (speech_status_name()),
 * the input it concerns or "" for none, and the message.
 */
class Failure : public std::runtime_error {
public:
    Failure(std::string code, std::string option, const std::string & message)
        : std::runtime_error(message), code_(std::move(code)), option_(std::move(option)) {}

    const std::string & code() const { return code_; }
    const std::string & option() const { return option_; }

private:
    std::string code_;
    std::string option_;
};

/** The failure the library's last call on this thread returned as `status`. */
inline Failure library_failure(speech_status status) {
    const char * name = speech_status_name(status);
    const char * option = speech_last_error_option();
    return Failure(name ? name : "internal", option ? option : "", speech_last_error());
}

/** Throws the library's failure when `status` is an error; SPEECH_OK and SPEECH_CANCELLED pass. */
inline speech_status check(speech_status status) {
    if (status < 0) throw library_failure(status);
    return status;
}

/** A command line that cannot be run, which exits with 2 and points to the subcommand's --help. */
struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
