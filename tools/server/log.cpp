#include "log.h"

#include <cstdio>
#include <ctime>

#include "terminal.h"

namespace server {

namespace {

thread_local std::pair<std::string, bool> t_note;
bool g_clients = false;

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

void log_clients(bool name_them) {
    g_clients = name_them;
}

std::string request_text(const std::string & method, const std::string & path, const std::string & status, const std::string & what,
                         const std::string & client) {
    terminal::Color color = terminal::Color::Mint;
    if (status == "gone" || status[0] == '4') color = terminal::Color::Peach;
    else if (status == "failed" || status[0] == '5') color = terminal::Color::Red;
    else if (status[0] == '1' || status[0] == '3') color = terminal::Color::Sky;
    std::string rest = what;
    if (g_clients) rest += (rest.empty() ? "from " : ", from ") + client;
    std::string out = padded(method, 4) + " " + padded(path, 26) + "  " + terminal::paint(color, rest.empty() ? status : padded(status, 6));
    if (!rest.empty()) out += "  " + terminal::paint(terminal::Color::Gray, rest);
    return out;
}

void note_request(std::string what, bool skip) {
    t_note = {std::move(what), skip};
}

std::pair<std::string, bool> take_request_note() {
    std::pair<std::string, bool> note;
    note.swap(t_note);
    return note;
}

}  // namespace server
