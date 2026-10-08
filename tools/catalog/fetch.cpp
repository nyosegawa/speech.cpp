#include "fetch.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <memory>
#include <optional>
#include <system_error>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

#include "cache.h"
#include "failure.h"
#include "ggml.h"
#include "sha256.h"
#include "speech.h"

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

Failure io_failure(const std::string & message) {
    return Failure(speech_status_name(SPEECH_ERROR_IO), "", message);
}

std::string megabytes(uint64_t bytes) {
    char s[32];
    std::snprintf(s, sizeof s, "%.0f MB", bytes / 1e6);
    return s;
}

/** A size in gigabytes, or in megabytes below 10 MB, which two decimals of a gigabyte would show as 0.00 GB. */
std::string gigabytes(uint64_t bytes) {
    char s[32];
    if (bytes < 10000000) std::snprintf(s, sizeof s, "%.1f MB", bytes / 1e6);
    else std::snprintf(s, sizeof s, "%.2f GB", bytes / 1e9);
    return s;
}

/** One end of a pipe from the child, read until the child closes it. */
#ifdef _WIN32
using PipeEnd = HANDLE;
#else
using PipeEnd = int;
#endif

/** Reads from a pipe into `buffer`, returning the bytes read, 0 at its end. */
size_t read_pipe(PipeEnd pipe, char * buffer, size_t size) {
#ifdef _WIN32
    DWORD n = 0;
    return ReadFile(pipe, buffer, (DWORD) size, &n, nullptr) ? n : 0;
#else
    for (;;) {
        const ssize_t n = ::read(pipe, buffer, size);
        if (n >= 0) return (size_t) n;
        if (errno != EINTR) return 0;
    }
#endif
}

void close_pipe(PipeEnd pipe) {
#ifdef _WIN32
    CloseHandle(pipe);
#else
    ::close(pipe);
#endif
}

/**
 * curl running with its stdout and stderr on pipes and its stdin on the null device, and no other handle or file
 * descriptor of this process, so that it holds no model file or socket open past its run.
 */
class Child {
public:
    explicit Child(const std::vector<std::string> & args) {
#ifdef _WIN32
        // System32's curl.exe, which Windows 10 1803 and later carry. A bare "curl.exe" would be looked for in the
        // current folder before System32.
        wchar_t system[MAX_PATH];
        const UINT length = GetSystemDirectoryW(system, MAX_PATH);
        if (length == 0 || length >= MAX_PATH) throw io_failure("cannot find the System32 folder, where curl.exe is");
        const std::wstring program = std::wstring(system) + L"\\curl.exe";
        std::wstring line = L"\"" + program + L"\"";
        for (const std::string & a : args) line += L" \"" + std::wstring(a.begin(), a.end()) + L"\"";

        SECURITY_ATTRIBUTES inherit = {sizeof inherit, nullptr, TRUE};
        HANDLE out_write = nullptr, err_write = nullptr;
        if (!CreatePipe(&out_, &out_write, &inherit, 0) || !CreatePipe(&err_, &err_write, &inherit, 0)) {
            throw io_failure("cannot make a pipe for curl (Windows error " + std::to_string(GetLastError()) + ")");
        }
        SetHandleInformation(out_, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(err_, HANDLE_FLAG_INHERIT, 0);
        HANDLE null = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING, 0, nullptr);
        HANDLE handles[3] = {null, out_write, err_write};
        SIZE_T bytes = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
        std::vector<char> list(bytes);
        STARTUPINFOEXW info = {};
        info.StartupInfo.cb = sizeof info;
        info.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        info.StartupInfo.hStdInput = null;
        info.StartupInfo.hStdOutput = out_write;
        info.StartupInfo.hStdError = err_write;
        info.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(list.data());
        InitializeProcThreadAttributeList(info.lpAttributeList, 1, 0, &bytes);
        UpdateProcThreadAttribute(info.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, handles, sizeof handles, nullptr, nullptr);
        PROCESS_INFORMATION process = {};
        // CREATE_NO_WINDOW keeps a speech without a console, started by a program that hid it, from opening a window.
        const BOOL started = CreateProcessW(program.c_str(), line.data(), nullptr, nullptr, TRUE, EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW,
                                            nullptr, nullptr, &info.StartupInfo, &process);
        const DWORD error = GetLastError();
        DeleteProcThreadAttributeList(info.lpAttributeList);
        CloseHandle(null);
        CloseHandle(out_write);
        CloseHandle(err_write);
        if (!started) {
            close_pipe(out_);
            close_pipe(err_);
            if (error == ERROR_FILE_NOT_FOUND) throw io_failure(program_missing());
            throw io_failure("cannot start curl (Windows error " + std::to_string(error) + ")");
        }
        CloseHandle(process.hThread);
        process_ = process.hProcess;
#else
        int out[2], err[2];
        if (pipe(out) != 0 || pipe(err) != 0) throw io_failure("cannot make a pipe for curl: " + std::generic_category().message(errno));
        fcntl(out[0], F_SETFD, FD_CLOEXEC);
        fcntl(err[0], F_SETFD, FD_CLOEXEC);
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
        posix_spawn_file_actions_adddup2(&actions, out[1], 1);
        posix_spawn_file_actions_adddup2(&actions, err[1], 2);
        posix_spawnattr_t attributes;
        posix_spawnattr_init(&attributes);
#ifdef __APPLE__
        posix_spawnattr_setflags(&attributes, POSIX_SPAWN_CLOEXEC_DEFAULT);
#else
        posix_spawn_file_actions_addclosefrom_np(&actions, 3);
#endif
        std::vector<char *> argv{const_cast<char *>("curl")};
        for (const std::string & a : args) argv.push_back(const_cast<char *>(a.c_str()));
        argv.push_back(nullptr);
        const int error = posix_spawnp(&pid_, "curl", &actions, &attributes, argv.data(), environ);
        posix_spawn_file_actions_destroy(&actions);
        posix_spawnattr_destroy(&attributes);
        ::close(out[1]);
        ::close(err[1]);
        out_ = out[0];
        err_ = err[0];
        if (error != 0) {
            ::close(out_);
            ::close(err_);
            if (error == ENOENT) throw io_failure(program_missing());
            throw io_failure("cannot start curl: " + std::generic_category().message(error));
        }
#endif
    }

    ~Child() {
#ifdef _WIN32
        if (process_) CloseHandle(process_);
#endif
    }

    PipeEnd out() const { return out_; }
    PipeEnd err() const { return err_; }

    /** Waits up to `ms` milliseconds for the child to end, and returns whether it has, with its exit code in `code`. */
    bool wait(int ms, int & code) {
#ifdef _WIN32
        if (WaitForSingleObject(process_, (DWORD) ms) != WAIT_OBJECT_0) return false;
        DWORD exit = 0;
        GetExitCodeProcess(process_, &exit);
        code = (int) exit;
        return true;
#else
        for (int waited = 0;; waited += 10) {
            int status = 0;
            if (waitpid(pid_, &status, WNOHANG) == pid_) {
                code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
                return true;
            }
            if (waited >= ms) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
#endif
    }

    void stop() {
#ifdef _WIN32
        TerminateProcess(process_, 1);
#else
        kill(pid_, SIGTERM);
#endif
    }

private:
    static std::string program_missing() {
        return "speech fetches models with the system's curl, which is not there; install curl, or fetch the model file yourself and give "
               "its path";
    }

    PipeEnd out_{}, err_{};
#ifdef _WIN32
    HANDLE process_ = nullptr;
#else
    pid_t pid_ = 0;
#endif
};

/** Hashes the first `size` bytes of `file`, the part a fetch resumes after. */
void hash_part(const fs::path & part, uint64_t size, Sha256 & hash) {
    const std::unique_ptr<FILE, int (*)(FILE *)> f(ggml_fopen(part.u8string().c_str(), "rb"), std::fclose);
    if (!f) throw io_failure("cannot read " + part.u8string());
    std::vector<char> buffer(1 << 20);
    for (uint64_t left = size; left > 0;) {
        const size_t n = std::fread(buffer.data(), 1, (size_t) std::min<uint64_t>(left, buffer.size()), f.get());
        if (n == 0) throw io_failure("cannot read " + part.u8string());
        hash.update(buffer.data(), n);
        left -= n;
    }
}

/** Removes a part whose bytes cannot become the file, and says so. */
Failure discarded(const fs::path & part, const std::string & why) {
    std::error_code ignored;
    fs::remove(part, ignored);
    return io_failure(why + "; the fetched bytes are removed, and the same command fetches the file again from its start");
}

/** The progress of a fetch on stderr: one line rewritten on a terminal, a line for each tenth elsewhere. */
class StderrProgress {
public:
    ~StderrProgress() {
        // A failure's message starts on a line of its own.
        if (open_) std::fprintf(stderr, "\n");
    }

    bool operator()(uint64_t done, uint64_t total) {
        const Clock::time_point now = Clock::now();
        if (!first_) first_ = {done, now};
        const double seconds = std::chrono::duration<double>(now - first_->second).count();
        const double rate = seconds > 0 ? (done - first_->first) / seconds / 1e6 : 0;
        const int tenth = (int) (done * 10 / total);
        char line[96];
        std::snprintf(line, sizeof line, "%s of %s, %d%%, %.1f MB/s", megabytes(done).c_str(), megabytes(total).c_str(), (int) (done * 100 / total), rate);
        if (terminal_) {
            std::fprintf(stderr, "\r  %-60s", line);
            open_ = done != total;
            if (!open_) std::fprintf(stderr, "\n");
        } else if (tenth > tenth_) {
            std::fprintf(stderr, "  %s\n", line);
        }
        tenth_ = tenth;
        return true;
    }

private:
#ifdef _WIN32
    const bool terminal_ = _isatty(_fileno(stderr));
#else
    const bool terminal_ = isatty(fileno(stderr));
#endif
    bool open_ = false;
    int tenth_ = 0;
    std::optional<std::pair<uint64_t, Clock::time_point>> first_;
};

}  // namespace

fs::path fetch_model(const CatalogChoice & choice, const FetchProgress & progress, const FetchNote & note) {
    const fs::path path = model_path(choice);
    std::error_code e;
    // A file is renamed into place only once it is whole and checked.
    if (fs::is_regular_file(path, e)) return path;
    fs::create_directories(path.parent_path(), e);
    if (e) throw io_failure("cannot make the folder " + path.parent_path().u8string() + ": " + e.message());
    const FileLock lock(path, [&] { note("waiting for another process that fetches or removes " + path.u8string()); });
    if (fs::is_regular_file(path, e)) return path;

    const CatalogFile & file = *choice.file;
    const fs::path part = partial_path(path);
    const uint64_t offset = fs::exists(part, e) ? (uint64_t) fs::file_size(part, e) : 0;
    if (e) throw io_failure("cannot read the size of " + part.u8string() + ": " + e.message());
    if (offset > file.size) throw discarded(part, part.u8string() + " holds " + std::to_string(offset) + " bytes, more than the file's " + std::to_string(file.size));
    Sha256 hash;
    if (offset > 0) {
        note("resuming at " + megabytes(offset) + " of " + megabytes(file.size));
        hash_part(part, offset, hash);
    }

    std::atomic<uint64_t> done{offset};
    if (offset < file.size) {
        const std::unique_ptr<FILE, int (*)(FILE *)> out(ggml_fopen(part.u8string().c_str(), "ab"), std::fclose);
        if (!out) throw io_failure("cannot write " + part.u8string() + "; check that its folder is writable");
        std::vector<std::string> args = {"--fail", "--location", "--silent", "--show-error", "--speed-limit", "1024", "--speed-time", "60"};
        if (offset > 0) args.insert(args.end(), {"--continue-at", std::to_string(offset)});
        args.push_back(file_url(choice));
        Child curl(args);

        bool too_long = false, unwritten = false;
        std::string curl_error;
        std::thread writer([&] {
            std::vector<char> buffer(1 << 20);
            while (const size_t n = read_pipe(curl.out(), buffer.data(), buffer.size())) {
                too_long = done + n > file.size;
                unwritten = !too_long && std::fwrite(buffer.data(), 1, n, out.get()) != n;
                if (too_long || unwritten) break;
                hash.update(buffer.data(), n);
                done += n;
            }
            // curl stops at its next write once the pipe has no reader.
            close_pipe(curl.out());
        });
        std::thread reader([&] {
            char buffer[4096];
            while (const size_t n = read_pipe(curl.err(), buffer, sizeof buffer)) curl_error.append(buffer, n);
            close_pipe(curl.err());
        });
        int code = 0;
        bool stopped = false;
        while (!curl.wait(250, code)) {
            if (!stopped && !progress(done, file.size)) {
                curl.stop();
                stopped = true;
            }
        }
        writer.join();
        reader.join();
        if (too_long) throw discarded(part, "the server sent more than the file's " + std::to_string(file.size) + " bytes");
        if (unwritten || std::fflush(out.get()) != 0) throw io_failure("cannot write " + part.u8string() + "; check the space left on its disk");
        if (stopped) throw io_failure("the fetch of " + choice_name(choice) + " was stopped; the same command resumes it");
        if (code != 0) {
            while (!curl_error.empty() && (curl_error.back() == '\n' || curl_error.back() == '\r')) curl_error.pop_back();
            throw io_failure("fetching " + file_url(choice) + " failed with curl's exit code " + std::to_string(code) +
                             (curl_error.empty() ? "" : " (" + curl_error + ")") + "; the same command resumes it");
        }
    }
    progress(done, file.size);
    if (done != file.size) {
        throw io_failure("curl ended after " + std::to_string(done) + " of the file's " + std::to_string(file.size) + " bytes; the same command resumes it");
    }
    const std::string sha256 = hash.hex();
    if (sha256 != file.sha256) {
        throw discarded(part, "the SHA-256 of the fetched " + file.file + " is " + sha256 + ", not the catalog's " + file.sha256);
    }
    fs::rename(part, path, e);
    if (e) throw io_failure("cannot rename " + part.u8string() + " to " + path.filename().u8string() + ": " + e.message());
    return path;
}

fs::path fetch_on_stderr(const CatalogChoice & choice) {
    const fs::path path = model_path(choice);
    std::fprintf(stderr, "fetching %s, %s (%s), from https://huggingface.co/%s into %s\n", choice_name(choice).c_str(), choice.file->file.c_str(),
                 gigabytes(choice.file->size).c_str(), choice.model->repository.c_str(), path.parent_path().u8string().c_str());
    const auto t0 = Clock::now();
    StderrProgress progress;
    fetch_model(choice, std::ref(progress), [](const std::string & note) { std::fprintf(stderr, "%s\n", note.c_str()); });
    std::fprintf(stderr, "fetched %s in %.0f s and checked its SHA-256\n", choice_name(choice).c_str(),
                 std::chrono::duration<double>(Clock::now() - t0).count());
    return path;
}

std::string model_file(const std::string & argument) {
    if (is_model_path(argument)) return argument;
    const CatalogChoice choice = find_model(argument);
    const fs::path path = model_path(choice);
    std::error_code e;
    return (fs::is_regular_file(path, e) ? path : fetch_on_stderr(choice)).u8string();
}
