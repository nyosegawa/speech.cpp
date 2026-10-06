// Recognizes the speech in WAVE files with any recognition model speech.cpp runs, through the C API, and writes
// each file's text to stdout as one line, in the order the files are given. Reports on stderr where the time went:
// the load, and for each file its seconds of audio, the time to its text and the real-time factor.
//
// A WAVE file is 16-, 24- or 32-bit PCM or 32-bit float at any rate, its channels averaged; the library resamples
// it to the model's rate (16 kHz for FastConformer).
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
#include "load-model.h"
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

int transcribe(const std::vector<std::string> & a, FILE * out) {
    const Options o = parse_options(a);
    if (o.positional.size() < 2) throw UsageError("give the model's GGUF file and at least one WAVE file");

    const auto t0 = Clock::now();
    const Model model = load_model(o.positional[0], o.device, {});
    const ModelInfo info = model_info(model.get());
    const speech_model_info * m = info.get();
    std::fprintf(stderr, "load %.2f s: %s on %s\n", seconds_since(t0), speech_model_info_name(m), speech_model_info_device(m));
    if (o.verbose) {
        std::string languages;
        for (size_t i = 0; i < speech_model_info_language_count(m); i++) {
            languages += (i ? ", " : "") + std::string(speech_model_info_language(m, i));
        }
        std::fprintf(stderr, "speech.cpp %s, %d Hz, languages: %s\n", speech_version(), speech_model_info_sample_rate(m), languages.c_str());
    }

    const bool several = o.positional.size() > 2;
    double audio_total = 0, busy = 0;
    for (size_t i = 1; i < o.positional.size(); i++) {
        const std::string & path = o.positional[i];
        const Wav wav = read_wav(path);
        const std::vector<float> samples = wav.mono();
        speech_request * raw = nullptr;
        if (speech_request_new(model.get(), &raw) != SPEECH_OK) throw_last_error();
        const std::unique_ptr<speech_request, decltype(&speech_request_free)> r(raw, speech_request_free);
        const auto start = Clock::now();
        if (speech_request_set_audio(raw, samples.data(), samples.size(), wav.sample_rate) != SPEECH_OK ||
            (!o.language.empty() && speech_request_set_string(raw, SPEECH_OPT_LANGUAGE, o.language.c_str()) != SPEECH_OK) ||
            speech_transcribe(raw) != SPEECH_OK) {
            throw std::runtime_error(path + ": " + speech_last_error());
        }
        const std::string text = speech_result_text(speech_request_result(raw));
        const double took = seconds_since(start), audio = (double) samples.size() / wav.sample_rate;
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
