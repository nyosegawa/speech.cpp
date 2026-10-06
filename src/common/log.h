#pragma once

#include <functional>
#include <string>

/** How much a log message matters, in the order of the C API's speech_log_level. */
enum class LogLevel { Debug = 0, Info = 1, Warn = 2, Error = 3 };

/** Receives one log message: a line or a piece of one, as ggml writes it. */
using LogSink = std::function<void(LogLevel level, const char * text)>;

/**
 * Sends every later message of the library and of ggml to `sink`, or drops them all when `sink` is empty, for the whole
 * process. Until it is called, warnings and errors go to stderr and the rest is dropped. A sink may be called from
 * several threads at once.
 */
void set_log_sink(LogSink sink);

/** Writes a message of the library through the sink. */
void log_message(LogLevel level, const std::string & text);

/** Routes ggml's messages through the sink; the first call that touches ggml makes it. */
void route_ggml_log();
