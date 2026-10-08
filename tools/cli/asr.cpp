#include <chrono>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "commands.h"
#include "fetch.h"
#include "json.h"
#include "regions.h"
#include "wave-input.h"

// speech asr: recognizes the speech in WAVE files and writes each file's text to stdout in the order the files are
// given, as text or as one JSON object per file, and reports on stderr the load and, for each file, its seconds of
// audio, the time to its text and the real-time factor. With --vad it transcribes each file by the regions where a
// detection model finds that someone speaks.

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

/** The flags and defaults of transcription by regions, for the help: "--threshold 0.5, ... and --max-speech-duration-s 10". */
std::string region_defaults() {
    const std::vector<RequestOption> defaults = region_options({});
    std::string out;
    for (size_t i = 0; i < defaults.size(); i++) {
        const OptionValue & v = defaults[i].value;
        const std::string value = v.index() == 1 ? std::to_string(std::get<int64_t>(v)) : json_number(std::get<double>(v));
        out += (i == 0 ? "" : i + 1 == defaults.size() ? " and " : ", ") + option_flag(defaults[i].option) + " " + value;
    }
    return out;
}

/** The detection model of --vad, and the request options that are its own: those it takes and the recognizer does not. */
struct Detection {
    Model model{nullptr, speech_model_free};
    std::vector<RequestOption> options;
};

std::optional<Detection> detection_of(const CommandLine & line, const speech_model_info * recognizer, std::vector<RequestOption> & options) {
    const std::optional<std::string> argument = line.value("--vad");
    if (!argument) return std::nullopt;
    const std::string path = model_file(*argument);
    const speech_task task = speech_model_info_task(file_info(path).get());
    if (task != SPEECH_TASK_DETECTION) {
        throw UsageError("--vad takes a model of speech detection, such as silero-vad, and " + *argument + " is a speech " + task_name(task) +
                         " model");
    }
    Detection d;
    d.model = load_model(path, line.loading(false));
    const ModelInfo info = model_info(d.model.get());
    std::vector<RequestOption> recognition;
    for (const RequestOption & o : options) {
        const bool own = speech_model_info_takes(info.get(), o.option) && !speech_model_info_takes(recognizer, o.option);
        (own ? d.options : recognition).push_back(o);
    }
    d.options = region_options(d.options);
    options = recognition;
    return d;
}

int run_asr(const CommandLine & line, FILE * out) {
    const std::string format = line.value("--format").value_or("text");
    if (format != "text" && format != "json") throw UsageError("--format takes text or json, not \"" + format + "\"");

    const auto t0 = Clock::now();
    const Model model = load_model(model_file(line.args[0]), line.loading(false));
    const ModelInfo info = model_info(model.get());
    const speech_model_info * m = info.get();
    std::vector<RequestOption> options = line.options;
    const std::optional<Detection> detection = detection_of(line, m, options);
    const std::string by = detection ? ", by the regions of " + std::string(speech_model_info_name(model_info(detection->model.get()).get())) : "";
    std::fprintf(stderr, "load %.2f s: %s on %s%s\n", seconds_since(t0), speech_model_info_name(m), speech_model_info_device(m), by.c_str());
    if (line.has("-v")) {
        std::string languages;
        for (size_t i = 0; i < speech_model_info_language_count(m); i++) languages += (i ? ", " : "") + std::string(speech_model_info_language(m, i));
        std::fprintf(stderr, "speech.cpp %s, %d Hz, languages: %s\n", speech_version(), speech_model_info_sample_rate(m), languages.c_str());
    }
    const bool timestamps = timestamps_in_effect(m, options);

    double audio_total = 0, busy = 0;
    bool model_limit = false;
    for (size_t i = 1; i < line.args.size(); i++) {
        const std::string & path = line.args[i];
        Transcript transcript;
        size_t regions = 0;
        double audio = 0;
        const auto start = Clock::now();
        try {
            const Wav wav = read_audio(path);
            const std::vector<float> samples = wav.mono();
            audio = (double) samples.size() / wav.sample_rate;
            Cancellation never;
            if (detection) {
                const Request request = detection_request(detection->model.get(), samples, wav.sample_rate, detection->options);
                const std::vector<Region> found = *detect_regions(request.get(), never);
                regions = found.size();
                transcript = *transcribe_regions(model.get(), samples, wav.sample_rate, found, options, timestamps, never);
            } else {
                const Request request = new_request(model.get());
                check(speech_request_set_audio(request.get(), samples.data(), samples.size(), wav.sample_rate));
                apply_options(request.get(), options);
                check(speech_transcribe(request.get()));
                transcript = transcript_of(speech_request_result(request.get()), timestamps);
            }
        } catch (const Failure & e) {
            throw Failure(e.code(), e.option(), path + ": " + e.what());
        }
        const double took = seconds_since(start);
        if (format == "json") {
            std::fprintf(out, "{\"file\":%s%s}\n", json_string(path).c_str(), recognition_members(transcript, timestamps).c_str());
        } else if (!timestamps) {
            std::fprintf(out, "%s\n", transcript.text.c_str());
        } else {
            for (const Timed & s : transcript.segments) std::fprintf(out, "%s\t%.3f\t%.3f\t%s\n", path.c_str(), s.start, s.end, s.text.c_str());
        }
        std::fflush(out);
        std::fprintf(stderr, "%s: %.2f s of audio in %.3f s, RTF %.3f", path.c_str(), audio, took, took / audio);
        if (detection) std::fprintf(stderr, ", %zu region%s", regions, regions == 1 ? "" : "s");
        std::fprintf(stderr, "\n");
        if (transcript.stop == SPEECH_STOP_MODEL_LIMIT) {
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
        "the most the model writes, after writing every file's text.\n"
        "With --vad, a detection model finds where someone speaks in each file, each region is recognized alone and the\n"
        "texts are joined, the times of the whole file. The detection options are those of speech vad, with OpenAI's\n"
        "server_vad defaults and a longest region:\n" + region_defaults() + ".\n"
        "A file in which no one speaks gives an empty text.";
    c.flags = {
        {"--format", "text|json", false, "text (the default) or one JSON object per file and line"},
        {"--vad", "MODEL", false, "transcribe by the regions where this detection model finds speech"},
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
