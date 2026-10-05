#pragma once

#include <cstdio>
#include <stdexcept>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

/**
 * Keeps the caller's stdout for the program's own output alone, as a binary stream it returns. The stream
 * moves to a descriptor of its own and descriptor 1 then writes to stderr, so whatever ggml, a GPU driver or a
 * system framework prints to stdout lands among the logs instead of inside the output. On Windows, _dup2()
 * onto descriptor 1 also sets the process's standard output handle, which code writing with WriteFile() reads.
 */
inline FILE * take_stdout() {
    std::fflush(stdout);
#ifdef _WIN32
    const int fd = _dup(_fileno(stdout));
    if (fd < 0 || _dup2(_fileno(stderr), _fileno(stdout)) != 0) {
        throw std::runtime_error("cannot send stdout to stderr");
    }
    _setmode(fd, _O_BINARY);
    FILE * stream = _fdopen(fd, "wb");
#else
    const int fd = dup(STDOUT_FILENO);
    if (fd < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) throw std::runtime_error("cannot send stdout to stderr");
    FILE * stream = fdopen(fd, "w");
#endif
    if (!stream) throw std::runtime_error("cannot open a stream on stdout's descriptor");
    return stream;
}
