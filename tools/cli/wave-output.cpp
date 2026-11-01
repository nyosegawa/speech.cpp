#include "wave-output.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "ggml.h"
#include "speech.h"

namespace {

/**
 * Whether a write at the stream's offset 0 lands at the start of what it holds: a regular file not opened for
 * appending. Whether fseek() succeeds cannot tell: on a pipe on Windows, MSVC's fseek() returns 0 with an undefined
 * result, and a header written after it lands at the end of the stream.
 */
bool rewritable(FILE * file) {
#ifdef _WIN32
    const HANDLE handle = (HANDLE) _get_osfhandle(_fileno(file));
    return handle != INVALID_HANDLE_VALUE && GetFileType(handle) == FILE_TYPE_DISK;
#else
    struct stat st;
    const int fd = fileno(file);
    const int flags = fcntl(fd, F_GETFL);
    return fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && flags >= 0 && !(flags & O_APPEND);
#endif
}

}  // namespace

Failure io_failure(const std::string & message) {
    return Failure(speech_status_name(SPEECH_ERROR_IO), "", message);
}

WavWriter::WavWriter(FILE * file, int sample_rate) : file_(file), sample_rate_(sample_rate) {
    // A WAVE that starts past the file's beginning, after `>>` onto a file with content, has its header elsewhere than
    // offset 0.
    rewritable_ = rewritable(file_) && std::ftell(file_) == 0;
    header(UINT32_MAX, UINT32_MAX);
}

void WavWriter::write(const float * s, size_t n) {
    pcm_.resize(n);
    for (size_t i = 0; i < n; i++) pcm_[i] = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s[i])) * 32767.0f);
    put(pcm_.data(), n * sizeof(int16_t));
    std::fflush(file_);
    bytes_ += n * sizeof(int16_t);
}

void WavWriter::finish() {
    if (bytes_ > UINT32_MAX - 36) throw io_failure("the audio is longer than a WAVE file of 4 GiB holds; write less audio to one file");
    if (rewritable_) {
        if (std::fseek(file_, 0, SEEK_SET) != 0 || std::ftell(file_) != 0) throw io_failure("cannot seek back to the WAVE header to set its sizes");
        header((uint32_t) bytes_ + 36, (uint32_t) bytes_);
    }
    if (std::fflush(file_) != 0) throw io_failure("cannot write the WAVE file; check the space left on its disk");
}

void WavWriter::put(const void * data, size_t n) {
    if (std::fwrite(data, 1, n, file_) != n) throw io_failure("cannot write the WAVE file; check the space left on its disk");
}

void WavWriter::put32(uint32_t v) {
    const uint8_t b[4] = {(uint8_t) v, (uint8_t) (v >> 8), (uint8_t) (v >> 16), (uint8_t) (v >> 24)};
    put(b, 4);
}

void WavWriter::put16(uint16_t v) {
    const uint8_t b[2] = {(uint8_t) v, (uint8_t) (v >> 8)};
    put(b, 2);
}

void WavWriter::header(uint32_t riff_size, uint32_t data_size) {
    put("RIFF", 4);
    put32(riff_size);
    put("WAVEfmt ", 8);
    put32(16);
    put16(1);
    put16(1);
    put32((uint32_t) sample_rate_);
    put32((uint32_t) sample_rate_ * 2);
    put16(2);
    put16(16);
    put("data", 4);
    put32(data_size);
}

Output::Output(const std::string & path, FILE * out) : path_(path) {
    if (path == "-") {
        file_ = out;
        return;
    }
    file_ = ggml_fopen(path.c_str(), "wb");
    if (!file_) throw io_failure("cannot write " + path + "; check that its folder exists and is writable");
}

Output::~Output() {
    if (path_ == "-") return;
    std::fclose(file_);
    if (!kept_) {
        std::error_code ignored;
        std::filesystem::remove(std::filesystem::u8path(path_), ignored);
    }
}
