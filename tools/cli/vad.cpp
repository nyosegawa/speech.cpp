#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "commands.h"
#include "fetch.h"
#include "json.h"
#include "wave-input.h"

// speech vad: finds where someone speaks in WAVE files and writes each file's regions to stdout in the order the files
// are given, as text or as one JSON object per file, and reports on stderr the load and, for each file, its seconds of
// audio, the time to its regions and the real-time factor.

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

int run_vad(const CommandLine & line, FILE * out) {
    const std::string format = line.value("--format").value_or("text");
    if (format != "text" && format != "json") throw UsageError("--format takes text or json, not \"" + format + "\"");

    const auto t0 = Clock::now();
    const Model model = load_model(model_file(line.args[0]), line.loading(false));
    const ModelInfo info = model_info(model.get());
    const speech_model_info * m = info.get();
    std::fprintf(stderr, "load %.2f s: %s on %s\n", seconds_since(t0), speech_model_info_name(m), speech_model_info_device(m));
    if (line.has("-v")) std::fprintf(stderr, "speech.cpp %s, %d Hz\n", speech_version(), speech_model_info_sample_rate(m));

    double audio_total = 0, busy = 0;
    for (size_t i = 1; i < line.args.size(); i++) {
        const std::string & path = line.args[i];
        const Request request = new_request(model.get());
        double audio = 0;
        const auto start = Clock::now();
        try {
            const Wav wav = read_audio(path);
            const std::vector<float> samples = wav.mono();
            audio = (double) samples.size() / wav.sample_rate;
            check(speech_request_set_audio(request.get(), samples.data(), samples.size(), wav.sample_rate));
            apply_options(request.get(), line.options);
            check(speech_detect(request.get()));
        } catch (const Failure & e) {
            throw Failure(e.code(), e.option(), path + ": " + e.what());
        }
        const double took = seconds_since(start);
        const speech_result * result = speech_request_result(request.get());
        const size_t n = speech_result_segment_count(result);
        std::string json = "{\"file\":" + json_string(path) + ",\"regions\":[";
        for (size_t k = 0; k < n; k++) {
            double from = 0, to = 0;
            check(speech_result_segment(result, k, &from, &to, nullptr));
            if (format == "json") json += std::string(k ? "," : "") + "{\"start\":" + json_number(from) + ",\"end\":" + json_number(to) + "}";
            else std::fprintf(out, "%s\t%.3f\t%.3f\n", path.c_str(), from, to);
        }
        if (format == "json") std::fprintf(out, "%s]}\n", json.c_str());
        std::fflush(out);
        std::fprintf(stderr, "%s: %.2f s of audio in %.3f s, RTF %.4f, %zu regions\n", path.c_str(), audio, took, took / audio, n);
        audio_total += audio;
        busy += took;
    }
    if (line.args.size() > 2) {
        std::fprintf(stderr, "%zu files, %.2f s of audio in %.3f s, RTF %.4f\n", line.args.size() - 1, audio_total, busy, busy / audio_total);
    }
    return 0;
}

}  // namespace

Command vad_command() {
    Command c;
    c.name = "vad";
    c.usage = "vad MODEL [options] AUDIO.wav...";
    c.summary = "write where someone speaks in WAVE files";
    c.description =
        "Finds the regions where someone speaks in each WAVE file (16-, 24- or 32-bit PCM or 32-bit float, any rate,\n"
        "channels averaged) with a detection model and writes them to stdout in the order given: with --format text\n"
        "one line per region, FILE<TAB>START<TAB>END, times in seconds; with --format json one object per file,\n"
        "{\"file\", \"regions\": [{\"start\", \"end\"}]}. A file without speech has no line and no region. The options\n"
        "--threshold, --min-speech-duration-ms, --min-silence-duration-ms, --speech-pad-ms and\n"
        "--max-speech-duration-s are Silero VAD's get_speech_timestamps(), with its defaults.";
    c.flags = {
        {"--format", "text|json", false, "text (the default) or one JSON object per file and line"},
        device_flag("auto (the first GPU, or the CPU without one), gpu, cpu or a name `speech devices` lists"),
        threads_flag(),
        verbose_flag("also report the release and the sample rate"),
    };
    c.request_options = true;
    c.model = ModelKind::Detection;
    c.min_args = 2;
    c.max_args = SIZE_MAX;
    c.run = run_vad;
    return c;
}
