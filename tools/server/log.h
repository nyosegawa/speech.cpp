#pragma once

#include <string>

// The server's log on stderr once it listens: a line for each request, each realtime session and item, and each model
// or voice the page adds, after the time of day it was written at, so that a log kept in a file says when each happened.
// A request run by a job (jobs.h) has the line the job writes as it ends, which says what it made; every other request
// has the line of cpp-httplib's logger.

namespace server {

/** Writes `text` on stderr after the time of day: "10:42:13  text", the time in gray where colors are drawn. */
void log(const std::string & text);

/**
 * A request's line: its method and path, its status or how it ended ("gone" for a client that went away, "failed"),
 * colored by it where colors are drawn, then `what`, what it did, in gray.
 */
std::string request_text(const std::string & method, const std::string & path, const std::string & status, const std::string & what);

/**
 * Marks the request this thread handles as one whose job writes its line, so that the logger, which cpp-httplib calls
 * on the same thread once the handler has returned, leaves it out.
 */
void job_logs_request();

/** Whether job_logs_request() marked the request this thread handles, which it then clears. */
bool take_job_logs_request();

}  // namespace server
