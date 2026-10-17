#include "log.h"

#include <cstdio>
#include <ctime>

#include "terminal.h"

namespace server {

namespace {

thread_local bool t_job_logs = false;

std::string padded(const std::string & s, size_t width) {
    return s.size() >= width ? s : s + std::string(width - s.size(), ' ');
}

}  // namespace

void log(const std::string & text) {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char time[16];
    std::strftime(time, sizeof time, "%H:%M:%S", &local);
    // One write a line, so that the lines of requests that end together do not mix.
    std::fputs((terminal::paint(terminal::Color::Gray, time) + "  " + text + "\n").c_str(), stderr);
}

std::string request_text(const std::string & method, const std::string & path, const std::string & status, const std::string & what) {
    terminal::Color color = terminal::Color::Mint;
    if (status == "gone" || status[0] == '4') color = terminal::Color::Peach;
    else if (status == "failed" || status[0] == '5') color = terminal::Color::Red;
    else if (status[0] == '1' || status[0] == '3') color = terminal::Color::Sky;
    std::string out = padded(method, 4) + " " + padded(path, 26) + "  " + terminal::paint(color, what.empty() ? status : padded(status, 6));
    if (!what.empty()) out += "  " + terminal::paint(terminal::Color::Gray, what);
    return out;
}

void job_logs_request() {
    t_job_logs = true;
}

bool take_job_logs_request() {
    const bool marked = t_job_logs;
    t_job_logs = false;
    return marked;
}

}  // namespace server
