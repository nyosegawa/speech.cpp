// Recognizes the speech in WAVE files with any recognition model speech.cpp runs, through the C API, and writes
// each file's text to stdout as one line, in the order the files are given. Reports on stderr where the time went:
// the load, and for each file its seconds of audio, the time to its text and the real-time factor.
//
// A WAVE file is 16-, 24- or 32-bit PCM or 32-bit float at the model's rate (16 kHz for FastConformer), its
// channels averaged; another rate is refused rather than resampled.
//
// usage: speech-asr <model.gguf> [options] <audio.wav>...
//          --device NAME|gpu|cpu   --language TAG   -v
//        speech-asr --devices

#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "args.h"
#include "list-devices.h"
#include "speech.h"
#include "take-stdout.h"
#include "wav.h"

namespace {

const char * const usage =
    "usage: speech-asr <model.gguf> [options] <audio.wav>...\n"
    "       speech-asr --devices\n"
    "\n"
    "Writes the text of each WAVE file to stdout, one line per file, in the order given.\n"
    "\n"
    "  --device NAME           a device as --devices names it, gpu (the first GPU, the default) or cpu\n"
    "  --language TAG          a BCP 47 tag of one of the model's languages, or auto (the default)\n"
    "  -v                      also report the model and its languages\n"
    "  --devices               list the devices and exit\n";

/** A command line that cannot be run, which exits with 2 and points to --help. */
struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct Options {
    std::vector<std::string> positional;
    std::string device, language;
    bool verbose = false;
};

Options parse_options(const std::vector<std::string> & a) {
    Options o;
    bool options_end = false;
    for (size_t i = 1; i < a.size(); i++) {
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
        if (key == "--device") o.device = value();
        else if (key == "--language") o.language = value();
        else throw UsageError("unknown option " + key);
    }
    return o;
}

int write_text(const char * text, void * user_data) {
    std::string & out = *static_cast<std::string *>(user_data);
    out += text;
    return 0;
}

int transcribe(const std::vector<std::string> & a, FILE * out) {
    const Options o = parse_options(a);
    if (o.positional.size() < 2) throw UsageError("give the model's GGUF file and at least one WAVE file");

    speech_model_params params = speech_model_default_params();
    params.model_path = o.positional[0].c_str();
    params.device = o.device.c_str();
    speech_model * model = nullptr;
    const auto t0 = Clock::now();
    if (speech_model_load(&params, &model) != SPEECH_OK) throw std::runtime_error(speech_last_error());
    std::unique_ptr<speech_model, decltype(&speech_model_free)> owned(model, speech_model_free);
    std::fprintf(stderr, "load %.2f s: %s on %s\n", seconds_since(t0), speech_model_name(model), speech_model_backend(model));
    if (o.verbose) {
        std::string languages;
        for (size_t i = 0; i < speech_model_language_count(model); i++) {
            languages += (i ? ", " : "") + std::string(speech_model_language(model, i));
        }
        std::fprintf(stderr, "speech.cpp %s, %d Hz, languages: %s\n", speech_version(), speech_model_sample_rate(model), languages.c_str());
    }

    const bool several = o.positional.size() > 2;
    double audio_total = 0, busy = 0;
    for (size_t i = 1; i < o.positional.size(); i++) {
        const std::string & path = o.positional[i];
        const Wav wav = read_wav(path);
        const std::vector<float> samples = wav.mono();
        speech_transcription_request r = speech_transcription_request_default();
        r.sample_rate = wav.sample_rate;
        r.samples = samples.data();
        r.n_samples = samples.size();
        r.language = o.language.c_str();
        std::string text;
        const auto start = Clock::now();
        if (speech_transcribe(model, &r, write_text, &text) != SPEECH_OK) {
            throw std::runtime_error(path + ": " + speech_last_error());
        }
        const double took = seconds_since(start), audio = (double) samples.size() / r.sample_rate;
        std::fprintf(out, "%s\n", text.c_str());
        std::fflush(out);
        std::fprintf(stderr, "%s: %.2f s of audio in %.3f s, RTF %.3f\n", path.c_str(), audio, took, took / audio);
        audio_total += audio;
        busy += took;
    }
    if (several) {
        std::fprintf(stderr, "%zu files, %.2f s of audio in %.3f s, RTF %.3f\n", o.positional.size() - 1, audio_total, busy,
                     busy / audio_total);
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
    return transcribe(a, out);
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
        std::fprintf(stderr, "error: %s\nRun speech-asr --help for the options.\n", e.what());
        return 2;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
