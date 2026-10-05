// Speaks text with any model speech.cpp runs, through the C API, into a WAVE file or to stdout, and reports
// on stderr where the time went: the load, and for each text the first audio, the whole and the real-time
// factor. Without a text it speaks every line of stdin, one request per line, blank lines skipped, into the
// one WAVE in order; the seed rises by one from each line to the next, as in the worker.
//
// The WAVE is 16-bit mono PCM at the model's rate, written as the audio is made, so that a player reading
// stdout starts before the rest is made. In a regular file that the WAVE starts at the beginning of, its RIFF
// and data sizes are set once the audio is complete; anywhere else (a pipe, a console, a file appended to)
// they stay 0xFFFFFFFF, as ffmpeg writes them to a pipe.
//
// usage: speech-tts <model.gguf> -o <out.wav|-> [options] [text]
//          --device NAME|gpu|cpu   --seed n   --voice-name NAME   --language TAG
//          --voice NAME=FILE...  --steps n  --speed x  --seconds s | --duration-scale x  (Irodori-TTS)
//          -v
//        speech-tts make-voice <model.gguf> <reference.wav> <voice.gguf> [--device NAME|gpu|cpu]
//        speech-tts --devices

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "args.h"
#include "ggml.h"
#include "list-devices.h"
#include "speech.h"
#include "take-stdout.h"

namespace {

const char * const usage =
    "usage: speech-tts <model.gguf> -o <out.wav|-> [options] [text]\n"
    "       speech-tts make-voice <model.gguf> <reference.wav> <voice.gguf> [--device NAME]\n"
    "       speech-tts --devices\n"
    "\n"
    "Speaks the text, or without one every line of stdin, into one WAVE file; -o - writes it to stdout.\n"
    "\n"
    "  -o FILE                 the WAVE file to write, or - for stdout\n"
    "  --device NAME           a device as --devices names it, gpu (the first GPU, the default) or cpu\n"
    "  --seed n                the seed of the first text; each later line takes the next. Random unless given\n"
    "  --voice-name NAME       the voice to speak with: a Qwen3-TTS speaker or the NAME of a --voice.\n"
    "                          Needed unless the model has a single voice\n"
    "  --language TAG          a BCP 47 tag of one of the model's languages, or auto (the default)\n"
    "  --voice NAME=FILE       Irodori-TTS: a voice, a reference WAVE file or a voice file; repeat for more\n"
    "  --steps n               Irodori-TTS: the sampler's steps (the model's own by default)\n"
    "  --speed x               Irodori-TTS: the speaking rate, 0.25 to 4 (1)\n"
    "  --seconds s             Irodori-TTS: the length of each text's speech, instead of the predicted one\n"
    "  --duration-scale x      Irodori-TTS: the factor of the predicted length (1)\n"
    "  -v                      also report the model, its voices and languages, and each text's seed\n"
    "  --devices               list the devices and exit\n"
    "  make-voice              write an Irodori-TTS voice file from a reference WAVE file\n";

/** A command line that cannot be run, which exits with 2 and points to --help. */
struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

uint64_t whole_number(const std::string & option, const std::string & text) {
    errno = 0;
    char * end = nullptr;
    const unsigned long long v = std::strtoull(text.c_str(), &end, 10);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos || *end || errno == ERANGE) {
        throw UsageError(option + " takes a whole number, not \"" + text + "\"");
    }
    return v;
}

int small_number(const std::string & option, const std::string & text) {
    const uint64_t v = whole_number(option, text);
    if (v > INT_MAX) throw UsageError(option + " " + text + " is too large");
    return (int) v;
}

double real_number(const std::string & option, const std::string & text) {
    char * end = nullptr;
    const double v = std::strtod(text.c_str(), &end);
    if (text.empty() || *end || !std::isfinite(v)) throw UsageError(option + " takes a number, not \"" + text + "\"");
    return v;
}

struct Options {
    std::vector<std::string> positional;
    std::string output, device, voice, language;
    bool has_output = false;
    bool verbose = false;
    int steps = 0;
    std::vector<std::pair<std::string, std::string>> voices;
    bool has_seed = false;
    uint64_t seed = 0;
    speech_request request = speech_request_default();
};

Options parse_options(const std::vector<std::string> & a, size_t first) {
    Options o;
    bool options_end = false;
    for (size_t i = first; i < a.size(); i++) {
        const std::string & key = a[i];
        if (options_end || key.size() < 2 || key[0] != '-') {
            o.positional.push_back(key);
            continue;
        }
        if (key == "--") {
            options_end = true;
            continue;
        }
        if (key == "-v") {
            o.verbose = true;
            continue;
        }
        const auto value = [&]() -> const std::string & {
            if (i + 1 >= a.size()) throw UsageError(key + " needs a value");
            return a[++i];
        };
        if (key == "-o") {
            o.output = value();
            o.has_output = true;
        } else if (key == "--device") {
            o.device = value();
        } else if (key == "--seed") {
            o.seed = whole_number(key, value());
            o.has_seed = true;
        } else if (key == "--voice-name") {
            o.voice = value();
        } else if (key == "--language") {
            o.language = value();
        } else if (key == "--steps") {
            o.steps = small_number(key, value());
        } else if (key == "--voice") {
            const std::string & voice = value();
            const size_t eq = voice.find('=');
            if (eq == std::string::npos || eq == 0) throw UsageError("--voice takes NAME=FILE, not \"" + voice + "\"");
            o.voices.push_back({voice.substr(0, eq), voice.substr(eq + 1)});
        } else if (key == "--speed") {
            o.request.speed = real_number(key, value());
        } else if (key == "--seconds") {
            o.request.seconds = real_number(key, value());
        } else if (key == "--duration-scale") {
            o.request.duration_scale = real_number(key, value());
        } else {
            throw UsageError("unknown option " + key);
        }
    }
    return o;
}

/**
 * Whether a write at the stream's offset 0 lands at the start of what it holds: a regular file not opened for
 * appending. Whether fseek() succeeds cannot tell: on a pipe on Windows, MSVC's fseek() returns 0 with an
 * undefined result, and a header written after it lands at the end of the stream.
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

/** Writes 16-bit mono PCM as a WAVE stream, converting the samples as the worker does. */
class WavWriter {
public:
    WavWriter(FILE * file, int sample_rate) : file_(file), sample_rate_(sample_rate) {
        // A WAVE that starts past the file's beginning, after `>>` onto a file with content, has its header
        // elsewhere than offset 0.
        rewritable_ = rewritable(file_) && std::ftell(file_) == 0;
        header(UINT32_MAX, UINT32_MAX);
    }

    void write(const float * s, size_t n) {
        pcm_.resize(n);
        for (size_t i = 0; i < n; i++) pcm_[i] = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s[i])) * 32767.0f);
        put(pcm_.data(), n * sizeof(int16_t));
        std::fflush(file_);
        bytes_ += n * sizeof(int16_t);
    }

    /** Sets the sizes where the header can be written again in place, and flushes. */
    void finish() {
        if (bytes_ > UINT32_MAX - 36) throw std::runtime_error("the audio is longer than a WAVE file can hold; speak less text at a time");
        if (rewritable_) {
            if (std::fseek(file_, 0, SEEK_SET) != 0 || std::ftell(file_) != 0) {
                throw std::runtime_error("cannot seek back to the WAVE header to set its sizes");
            }
            header((uint32_t) bytes_ + 36, (uint32_t) bytes_);
        }
        if (std::fflush(file_) != 0) throw std::runtime_error("cannot write the WAVE file");
    }

private:
    void put(const void * data, size_t n) {
        if (std::fwrite(data, 1, n, file_) != n) throw std::runtime_error("cannot write the WAVE file");
    }
    void put32(uint32_t v) {
        const uint8_t b[4] = {(uint8_t) v, (uint8_t) (v >> 8), (uint8_t) (v >> 16), (uint8_t) (v >> 24)};
        put(b, 4);
    }
    void put16(uint16_t v) {
        const uint8_t b[2] = {(uint8_t) v, (uint8_t) (v >> 8)};
        put(b, 2);
    }
    void header(uint32_t riff_size, uint32_t data_size) {
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

    FILE * file_;
    int sample_rate_;
    bool rewritable_ = false;
    uint64_t bytes_ = 0;
    std::vector<int16_t> pcm_;
};

/** The WAVE file named by -o, removed again unless keep() is called, so that a failed run leaves none behind. */
class Output {
public:
    Output(const std::string & path, FILE * out) : path_(path) {
        if (path == "-") {
            file_ = out;
            return;
        }
        file_ = ggml_fopen(path.c_str(), "wb");
        if (!file_) throw std::runtime_error("cannot write " + path + "; check that its folder exists and is writable");
    }
    ~Output() {
        if (path_ == "-") return;
        std::fclose(file_);
        if (!kept_) {
            std::error_code ignored;
            std::filesystem::remove(std::filesystem::u8path(path_), ignored);
        }
    }
    FILE * file() const { return file_; }
    void keep() { kept_ = true; }

private:
    std::string path_;
    FILE * file_ = nullptr;
    bool kept_ = false;
};

int make_voice(const std::vector<std::string> & a) {
    const Options o = parse_options(a, 2);
    if (o.positional.size() != 3) throw UsageError("make-voice takes <model.gguf> <reference.wav> <voice.gguf>");
    if (o.has_output || o.has_seed || !o.voice.empty() || !o.language.empty() || !o.voices.empty() || o.steps || o.request.speed != 1 ||
        o.request.seconds != 0 || o.request.duration_scale != 1) {
        throw UsageError("make-voice takes no option but --device and -v");
    }
    speech_model_params params = speech_model_default_params();
    params.model_path = o.positional[0].c_str();
    params.device = o.device.c_str();
    const auto t0 = Clock::now();
    if (speech_make_voice(&params, o.positional[1].c_str(), o.positional[2].c_str()) != SPEECH_OK) {
        throw std::runtime_error(speech_last_error());
    }
    std::fprintf(stderr, "wrote %s in %.2f s\n", o.positional[2].c_str(), seconds_since(t0));
    return 0;
}

std::string joined(size_t count, const char * (*get)(const speech_model *, size_t), const speech_model * m) {
    std::string out;
    for (size_t i = 0; i < count; i++) out += (i ? ", " : "") + std::string(get(m, i));
    return out;
}

/** The voice to speak with: the one named, or the model's only one. */
std::string choose_voice(const speech_model * m, const std::string & named) {
    const size_t n = speech_model_voice_count(m);
    const std::string voices = joined(n, speech_model_voice, m);
    if (named.empty()) {
        if (n == 1) return speech_model_voice(m, 0);
        throw UsageError(std::string(speech_model_name(m)) + " has " + std::to_string(n) +
                         " voices; choose one with --voice-name: " + voices);
    }
    for (size_t i = 0; i < n; i++) {
        if (named == speech_model_voice(m, i)) return named;
    }
    throw UsageError(std::string(speech_model_name(m)) + " has no voice named \"" + named + "\"; its voices are " + voices);
}

/** What the audio callback needs of one text. */
struct Speaking {
    WavWriter & wav;
    Clock::time_point start;
    double first_audio = -1;
    size_t samples = 0;
};

int on_audio(const float * s, size_t n, void * user_data) {
    Speaking & r = *static_cast<Speaking *>(user_data);
    if (n == 0) return 0;
    if (r.first_audio < 0) r.first_audio = seconds_since(r.start);
    r.wav.write(s, n);
    r.samples += n;
    return 0;
}

/** The next line of stdin with text on it, without its line ending or a UTF-8 byte order mark, or false at the end. */
bool next_line(std::string & line, bool & first) {
    while (std::getline(std::cin, line)) {
        if (first && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
        first = false;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.find_first_not_of(" \t") != std::string::npos) return true;
    }
    return false;
}

int speak(const std::vector<std::string> & a, FILE * out) {
    Options o = parse_options(a, 1);
    if (o.positional.empty()) throw UsageError("give the model's GGUF file");
    if (o.positional.size() > 2) throw UsageError("give the model's one GGUF file, which holds its codec, and the text as one argument, in quotes");
    if (!o.has_output) throw UsageError("name the WAVE file to write with -o, or give -o - for stdout");
    const bool from_stdin = o.positional.size() == 1;
    const uint64_t first_seed = o.has_seed ? o.seed : std::random_device{}();

    std::vector<speech_voice_source> voices;
    for (const auto & [name, path] : o.voices) voices.push_back({name.c_str(), path.c_str()});
    speech_model_params params = speech_model_default_params();
    params.model_path = o.positional[0].c_str();
    params.device = o.device.c_str();
    params.voices = voices.data();
    params.n_voices = voices.size();
    params.steps = o.steps;
    speech_model * model = nullptr;
    auto t0 = Clock::now();
    if (speech_model_load(&params, &model) != SPEECH_OK) throw std::runtime_error(speech_last_error());
    std::unique_ptr<speech_model, decltype(&speech_model_free)> owned(model, speech_model_free);
    std::fprintf(stderr, "load %.2f s: %s on %s\n", seconds_since(t0), speech_model_name(model), speech_model_backend(model));
    if (o.verbose) {
        std::fprintf(stderr, "speech.cpp %s, %d Hz, voices: %s; languages: %s",
                     speech_version(), speech_model_sample_rate(model),
                     joined(speech_model_voice_count(model), speech_model_voice, model).c_str(),
                     joined(speech_model_language_count(model), speech_model_language, model).c_str());
        if (speech_model_steps(model) > 0) std::fprintf(stderr, "; %d steps", speech_model_steps(model));
        std::fprintf(stderr, "\n");
    }
    const std::string voice = choose_voice(model, o.voice);

    Output output(o.output, out);
    WavWriter wav(output.file(), speech_model_sample_rate(model));
    const double rate = speech_model_sample_rate(model);
    std::string text = from_stdin ? "" : o.positional[1];
    bool first_line = true;
    int count = 0;
    size_t samples = 0;
    double busy = 0;
    while (from_stdin ? next_line(text, first_line) : count == 0) {
        speech_request r = o.request;
        r.text = text.c_str();
        r.voice = voice.c_str();
        r.language = o.language.c_str();
        r.seed = first_seed + count;
        count++;
        if (o.verbose) std::fprintf(stderr, "%d: seed %llu: %s\n", count, (unsigned long long) r.seed, text.c_str());
        Speaking speaking{wav, Clock::now()};
        if (speech_synthesize(model, &r, on_audio, &speaking) != SPEECH_OK) {
            throw std::runtime_error((from_stdin ? "line " + std::to_string(count) + ": " : std::string()) + speech_last_error());
        }
        const double total = seconds_since(speaking.start), audio = speaking.samples / rate;
        std::fprintf(stderr, "%s%.2f s of audio: first audio %.3f s, total %.3f s, RTF %.3f\n",
                     from_stdin ? (std::to_string(count) + ": ").c_str() : "", audio, speaking.first_audio, total,
                     total / audio);
        samples += speaking.samples;
        busy += total;
    }
    if (count == 0) throw UsageError("stdin held no text; give the text as an argument or pipe lines of text in");
    wav.finish();
    output.keep();
    if (from_stdin) {
        std::fprintf(stderr, "%d lines, %.2f s of audio in %.3f s, RTF %.3f\n", count, samples / rate, busy,
                     busy / (samples / rate));
    }
    return 0;
}

int run(const std::vector<std::string> & a, FILE * out) {
    if (a.size() == 2 && (a[1] == "--help" || a[1] == "-h")) {
        std::fputs(usage, out);
        std::fflush(out);
        return 0;
    }
    if (a.size() == 2 && a[1] == "--devices") {
        list_devices(out);
        return 0;
    }
    if (a.size() >= 2 && a[1] == "make-voice") return make_voice(a);
    return speak(a, out);
}

}  // namespace

int main(int argc, char ** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    try {
        FILE * out = take_stdout();
        return run(utf8_args(argc, argv), out);
    } catch (const UsageError & e) {
        std::fprintf(stderr, "error: %s\nRun speech-tts --help for the options.\n", e.what());
        return 2;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
