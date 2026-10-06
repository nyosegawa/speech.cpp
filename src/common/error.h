#pragma once

#include <stdexcept>
#include <string>
#include <utility>

/** What kind of failure an Error is, which a program on the library reports as its category. */
enum class Fault {
    /** The caller's mistake: an empty text, a value that is not a number, a voice file of another codec. */
    InvalidArgument,
    /** A value the model does not take, though it takes the input: outside its range, or too long. */
    OutOfRange,
    /** A model, voice or audio file whose content cannot be used. */
    File,
    /** A device that is not there, does not start, or fails while it computes. */
    Device,
    /** Memory of the host or of a device that ran out. */
    OutOfMemory,
    /** A file that cannot be opened, read or written. */
    Io,
};

/**
 * A failure of a known kind with its message, naming the input it concerns where there is one: an option's name
 * ("seconds"), "text" or "audio". A failure that is not an Error is a defect of the library.
 */
class Error : public std::runtime_error {
public:
    Error(Fault fault, const std::string & message, std::string input = "")
        : std::runtime_error(message), fault_(fault), input_(std::move(input)) {}

    Fault fault() const { return fault_; }
    const std::string & input() const { return input_; }

private:
    Fault fault_;
    std::string input_;
};

/**
 * Runs `body` and names `input` in an Error it throws that names none, unless the Error is of a device or of memory,
 * which concerns no input: what a call that takes several files does to say which one failed.
 */
template <typename Body>
auto naming(const std::string & input, Body && body) -> decltype(body()) {
    try {
        return body();
    } catch (const Error & e) {
        if (!e.input().empty() || e.fault() == Fault::Device || e.fault() == Fault::OutOfMemory) throw;
        throw Error(e.fault(), e.what(), input);
    }
}
