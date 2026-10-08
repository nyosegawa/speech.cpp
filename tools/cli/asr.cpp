#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "commands.h"
#include "fetch.h"
#include "json.h"
#include "wave-input.h"

// speech asr: recognizes the speech in WAVE files and writes each file's text to stdout in the order the files are
// given, as text or as one JSON object per file, and reports on stderr the load and, for each file, its seconds of
// audio, the time to its text and the real-time factor.

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

int run_asr(const CommandLine & line, FILE * out) {
    const std::string format = line.value("--format").value_or("text");
    if (format != "text" && format != "json") throw UsageError("--format takes text or json, not \"" + format + "\"");

    const auto t0 = Clock::now();
    const Model model = load_model(model_file(line.args[0]), line.loading(false));
    const ModelInfo info = model_info(model.get());
    const speech_model_info * m = info.get();
    std::fprintf(stderr, "load %.2f s: %s on %s\n", seconds_since(t0), speech_model_info_name(m), speech_model_info_device(m));
    if (line.has("-v")) {
        std::string languages;
        for (size_t i = 0; i < speech_model_info_language_count(m); i++) languages += (i ? ", " : "") + std::string(speech_model_info_language(m, i));
        std::fprintf(stderr, "speech.cpp %s, %d Hz, languages: %s\n", speech_version(), speech_model_info_sample_rate(m), languages.c_str());
    }
    const bool timestamps = timestamps_in_effect(m, line.options);

    double audio_total = 0, busy = 0;
    bool model_limit = false;
    for (size_t i = 1; i < line.args.size(); i++) {
        const std::string & path = line.args[i];
        const speech_result * result = nullptr;
        const Request request = new_request(model.get());
        double audio = 0;
        const auto start = Clock::now();
        try {
            const Wav wav = read_audio(path);
            const std::vector<float> samples = wav.mono();
            audio = (double) samples.size() / wav.sample_rate;
            check(speech_request_set_audio(request.get(), samples.data(), samples.size(), wav.sample_rate));
            apply_options(request.get(), line.options);
            check(speech_transcribe(request.get()));
            result = speech_request_result(request.get());
        } catch (const Failure & e) {
            throw Failure(e.code(), e.option(), path + ": " + e.what());
        }
        const double took = seconds_since(start);
        if (format == "json") {
            std::fprintf(out, "{\"file\":%s%s}\n", json_string(path).c_str(), recognition_members(result, timestamps).c_str());
        } else if (!timestamps) {
            std::fprintf(out, "%s\n", speech_result_text(result));
        } else {
            for (size_t k = 0; k < speech_result_segment_count(result); k++) {
                double from = 0, to = 0;
                const char * text = nullptr;
                check(speech_result_segment(result, k, &from, &to, &text));
                std::fprintf(out, "%s\t%.3f\t%.3f\t%s\n", path.c_str(), from, to, text);
            }
        }
        std::fflush(out);
        std::fprintf(stderr, "%s: %.2f s of audio in %.3f s, RTF %.3f\n", path.c_str(), audio, took, took / audio);
        if (speech_result_stop(result) == SPEECH_STOP_MODEL_LIMIT) {
            std::fprintf(stderr, "%s: the recognition reached the most the model writes and was stopped there; its text is what was written\n",
                         path.c_str());
            model_limit = true;
        }
        audio_total += audio;
        busy += took;
    }
    if (line.args.size() > 2) {
        std::fprintf(stderr, "%zu files, %.2f s of audio in %.3f s, RTF %.3f\n", line.args.size() - 1, audio_total, busy, busy / audio_total);
    }
    return model_limit ? 3 : 0;
}

}  // namespace

Command asr_command() {
    Command c;
    c.name = "asr";
    c.usage = "asr MODEL [options] AUDIO.wav...";
    c.summary = "write the text of WAVE files";
    c.description =
        "Recognizes each WAVE file (16-, 24- or 32-bit PCM or 32-bit float, any rate, channels averaged) and writes\n"
        "its text to stdout in the order given: with --format text one line per file, or with --timestamps one line\n"
        "per segment, FILE<TAB>START<TAB>END<TAB>TEXT, times in seconds; with --format json one object per file,\n"
        "{\"file\", \"text\", \"stop\"}, with \"languages\", the tags of the languages the model heard, where it names\n"
        "any, and \"segments\" and \"tokens\" when --timestamps is given. It exits with 3 when a recognition stopped at\n"
        "the most the model writes, after writing every file's text.";
    c.flags = {
        {"--format", "text|json", false, "text (the default) or one JSON object per file and line"},
        device_flag("auto (the first GPU, or the CPU without one), gpu, cpu or a name `speech devices` lists"),
        threads_flag(),
        verbose_flag("also report the release, the sample rate and the model's languages"),
    };
    c.request_options = true;
    c.model = ModelKind::Recognition;
    c.min_args = 2;
    c.max_args = SIZE_MAX;
    c.run = run_asr;
    return c;
}
