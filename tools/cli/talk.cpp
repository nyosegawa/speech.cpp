#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "commands.h"
#include "fetch.h"
#include "microphone.h"
#include "speaker.h"
#include "terminal.h"
#include "wave-input.h"
#include "wave-output.h"

// speech talk: holds a conversation with a dialogue model, which listens and speaks at once. With --live it hears the
// default microphone and plays the model's frames on the default audio output as they come, until Ctrl-C; with a WAVE
// file it hears the file and writes the model's frames end to end, its reply, as a 16-bit mono WAVE at the model's rate
// (wave-output.h). Either way it writes the model's text on stdout as the model says it, or on stderr where the WAVE
// goes to stdout. It reports on stderr the load time, and the seconds the model gave with the time it took to compute
// them and the RTFx.

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

/**
 * How far a live conversation may fall behind, in seconds of the microphone's audio not yet given to the model and of
 * the model's audio not yet played, before it stops: a frame computed in longer than it lasts is absorbed by the audio
 * queued to play, and a device that keeps computing slower than the frames last stops the conversation rather than
 * drop audio the model would never hear.
 */
constexpr double kBehind = 1.0;

std::atomic<bool> interrupted{false};

/** The first Ctrl-C ends the conversation; a second ends the program. */
void on_interrupt(int) {
    interrupted = true;
    std::signal(SIGINT, SIG_DFL);
}

/** Where the model's frames go, the speaker or a WAVE file, and its text; what was given; and the failure that stopped it. */
struct Talking {
    Speaker * speaker = nullptr;
    WavWriter * wav = nullptr;
    FILE * text = nullptr;
    uint64_t frames = 0, samples = 0;
    bool said = false;
    std::optional<Failure> failed;

    /** Ends the line of text, once anything was said. */
    void end_text() {
        if (said) std::fputc('\n', text);
        std::fflush(text);
        said = false;
    }
};

/** Plays or writes a frame and writes its text; a failure to write stops the conversation, since no exception may cross the C API. */
int on_frame(const float * s, size_t n, const char * text, void * user_data) {
    Talking & t = *static_cast<Talking *>(user_data);
    try {
        if (t.speaker) t.speaker->play(s, n);
        if (t.wav) t.wav->write(s, n);
    } catch (const Failure & f) {
        t.failed = f;
        return 1;
    }
    if (*text) {
        std::fputs(text, t.text);
        std::fflush(t.text);
        t.said = true;
    }
    t.frames++;
    t.samples += n;
    return 0;
}

using Dialogue = std::unique_ptr<speech_dialogue, decltype(&speech_dialogue_free)>;

/** Starts the conversation, which passes the model's first frame, at `rate`; the library computes for `busy` seconds. */
Dialogue start(const speech_request * request, int rate, Talking & talking, double & busy) {
    speech_dialogue * raw = nullptr;
    const auto t0 = Clock::now();
    check(speech_dialogue_start(request, rate, on_frame, &talking, &raw));
    busy += seconds_since(t0);
    Dialogue dialogue(raw, speech_dialogue_free);
    if (talking.failed) throw *talking.failed;
    return dialogue;
}

/** Gives the conversation `samples` it heard, which passes a frame of the model's for each whole frame of them. */
void push(speech_dialogue * dialogue, const std::vector<float> & samples, Talking & talking, double & busy) {
    const auto t0 = Clock::now();
    check(speech_dialogue_push(dialogue, samples.data(), samples.size()));
    busy += seconds_since(t0);
    if (talking.failed) throw *talking.failed;
}

/** Talks through the microphone and the audio output until Ctrl-C; the model computes its frames for `busy` seconds. */
void talk_live(const speech_request * request, int rate, FILE * out, Talking & talking, double & busy) {
    Speaker speaker(rate);
    Microphone microphone;
    talking.speaker = &speaker;
    talking.text = out;
    std::fprintf(stderr, "talking through %s at %d Hz and %s; use headphones, since the model hears what the speakers play; Ctrl-C ends\n",
                 microphone.name().c_str(), microphone.rate(), speaker.name().c_str());
    const Dialogue dialogue = start(request, microphone.rate(), talking, busy);
    // The conversation begins with the model's first frame: what the microphone heard while the prompt ran came before it.
    microphone.take(std::chrono::milliseconds(0));
    interrupted = false;
    std::signal(SIGINT, on_interrupt);
    while (!interrupted) {
        const std::vector<float> heard = microphone.take(std::chrono::milliseconds(50));
        const double behind = (double) heard.size() / microphone.rate() + (double) speaker.queued() / rate;
        if (behind > kBehind) {
            char message[320];
            std::snprintf(message, sizeof message,
                          "the conversation has fallen %.1f s behind the microphone, as the model computes its frames slower than they last on "
                          "this device; use a smaller type of the model or a faster device",
                          behind);
            throw Failure(speech_status_name(SPEECH_ERROR_DEVICE), "", message);
        }
        if (!heard.empty()) push(dialogue.get(), heard, talking, busy);
    }
    std::signal(SIGINT, SIG_DFL);
    check(speech_dialogue_end(dialogue.get()));
}

/** Talks with the WAVE file at `input` as what the model hears, writing its reply to `output`. */
void talk_file(const speech_request * request, int rate, const std::string & input, const std::string & output, FILE * out, Talking & talking,
               double & busy) {
    const Wav wav = read_audio(input);
    const std::vector<float> samples = wav.mono();
    Output file(output, out);
    WavWriter writer(file.file(), rate);
    talking.wav = &writer;
    talking.text = output == "-" ? stderr : out;
    const Dialogue dialogue = start(request, wav.sample_rate, talking, busy);
    push(dialogue.get(), samples, talking, busy);
    check(speech_dialogue_end(dialogue.get()));
    writer.finish();
    file.keep();
}

int run_talk(const CommandLine & line, FILE * out) {
    const bool live = line.has("--live");
    const std::optional<std::string> output = line.value("-o");
    if (live && line.args.size() > 1) throw UsageError("--live hears the microphone and takes no AUDIO; give one or the other");
    if (live && output) throw UsageError("--live plays the model's audio and writes no WAVE; give -o with an AUDIO file");
    if (!live && line.args.size() < 2) throw UsageError("give the WAVE file the model hears, or --live to talk through the microphone");
    if (!live && !output) throw UsageError("name the WAVE file to write the model's reply to with -o, or give -o - for stdout");

    // A model named for the first time is fetched here, before the load is timed.
    const std::string path = model_file(line.args[0]);
    require_options(path, line);
    const auto t0 = Clock::now();
    terminal::Status loading("Loading " + argument_label(line.args[0]));
    // A live conversation warms a GPU's kernels up first, so that its first frames keep up.
    const Model model = load_model(path, line.loading(live));
    const ModelInfo info = model_info(model.get());
    const speech_model_info * m = info.get();
    loading.done(loaded_line({m}, seconds_since(t0)));
    if (line.has("-v")) std::fprintf(stderr, "speech.cpp %s\n", speech_version());
    if (speech_model_info_task(m) != SPEECH_TASK_DIALOGUE) {
        throw Failure(speech_status_name(SPEECH_ERROR_UNSUPPORTED), "",
                      std::string(speech_model_info_name(m)) + " is a model of speech " + task_name(speech_model_info_task(m)) +
                          ", and speech talk holds conversations with a dialogue model");
    }
    const Request request = new_request(model.get());
    apply_options(request.get(), line.options);
    const int rate = speech_model_info_sample_rate(m);

    Talking talking;
    double busy = 0;
    try {
        if (live) talk_live(request.get(), rate, out, talking, busy);
        else talk_file(request.get(), rate, line.args[1], *output, out, talking, busy);
    } catch (...) {
        talking.end_text();
        throw;
    }
    talking.end_text();
    const double given = (double) talking.samples / rate;
    char said[200];
    std::snprintf(said, sizeof said, "%.2f s of audio in %llu frames, computed in %.3f s, RTFx %.2f", given, (unsigned long long) talking.frames, busy,
                  given / busy);
    const std::string where = live ? "" : *output == "-" ? line.args[1] + ": " : *output + ": ";
    std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Audio, where + terminal::paint(terminal::Color::Gray, said)).c_str());
    return 0;
}

}  // namespace

Command talk_command() {
    Command c;
    c.name = "talk";
    c.usage = "talk MODEL [options] --live|-o FILE|- AUDIO.wav";
    c.summary = "hold a conversation with a model that listens and speaks at once";
    c.description =
        "Holds a conversation with a dialogue model, which hears a frame of audio and gives a frame of its own, with the\n"
        "text it says in it, one frame ahead. With --live it hears the default microphone and plays its frames on the\n"
        "default audio output as they come, until Ctrl-C; use headphones, since the model hears what speakers play. A\n"
        "conversation that falls 1 s behind the microphone, as on a device that computes the frames slower than they last,\n"
        "stops with an error rather than drop audio the model would not hear. With AUDIO.wav (16-, 24- or 32-bit PCM or\n"
        "32-bit float, any rate, channels averaged) it hears the file and writes the model's frames end to end as a\n"
        "16-bit mono WAVE at the model's rate to -o, or to stdout with -o -: the frame of the start and one for each whole\n"
        "frame of the file, so that the reply lines up with the file in time. Either way it writes the model's text on\n"
        "stdout as it comes, or on stderr with -o -. A conversation lasts --max-seconds at most and holds the memory of\n"
        "that length from its start; --no-warmup applies to --live, which warms a GPU up first.";
    c.flags = {
        {"--live", "", false, "talk through the default microphone and audio output until Ctrl-C"},
        {"-o", "FILE", false, "the WAVE file to write the model's reply to, or - for stdout"},
        add_voice_flag(),
        device_flag("auto (the first GPU, or the CPU without one), gpu, cpu or a name `speech devices` lists"),
        threads_flag(),
        no_warmup_flag(),
        verbose_flag("also report the release"),
    };
    c.request_options = true;
    c.model = ModelKind::Dialogue;
    c.min_args = 1;
    c.max_args = 2;
    c.run = run_talk;
    c.release_notice = ReleaseNotice::AtEnd;
    return c;
}
