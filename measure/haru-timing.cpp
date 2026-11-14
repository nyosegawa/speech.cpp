// Times Haru's conversation on a recording of the user, from seed 1 with the model file's settings and no character
// card, in the voice of a reference WAVE file or voice file, or of none: the start, which runs the prompt and gives the
// first frame, and one step for each whole frame of the recording, each step's time and the mean of its parts (the
// codec's encoding of the user's frame, the graphs of the language model with its logits and the head, the sampling on
// the host and the codec's decoding of the model's frame), in milliseconds. A conversation keeps up when its steps take
// less than a frame's length; `late` counts the steps that took longer. Each conversation holds the keys and values of
// `max_seconds`, the file's 60 minutes unless given, as a request does, which the peak memory shows. A conversation of
// a few frames goes first, untimed, so that the times leave out the compilation of a GPU's kernels. It prints one JSON
// object per run.
//
// usage: haru-timing <model.gguf> <gpu|cpu|device name> <reference.wav|voice file|none> <user.wav> <runs> [max_seconds]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "haru/dialogue.h"
#include "haru/layout.h"
#include "haru/voice-file.h"
#include "json.h"
#include "resample.h"
#include "wav.h"

using namespace haru;

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

/** The value below which a share `q` of `v` lies, the nearest rank's. */
double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t) (q * (double) v.size()))];
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 6) {
        std::fprintf(stderr, "usage: %s <model.gguf> <gpu|cpu|device name> <reference.wav|voice file|none> <user.wav> <runs> [max_seconds]\n",
                     args[0].c_str());
        return 2;
    }
    try {
        const int runs = std::stoi(args[5]);
        if (runs < 1) throw std::runtime_error("runs must be positive");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(init_backend(args[2]), ggml_backend_free);
        const ModelFile model(args[1], backend.get(), model_layout);
        Dialogue dialogue(model, backend.get());
        Codec codec(model, backend.get());
        const Voice voice = args[3] == "none" ? Voice{} : load_voice(model, codec, backend.get(), args[3]);

        const int rate = (int) model.size("speech.sample_rate");
        const Wav wav = read_wav(args[4]);
        const std::vector<float> user = Resampler(wav.sample_rate, rate)(wav.mono());
        const int64_t frame = dialogue.frame_samples(), frames = (int64_t) user.size() / frame;
        if (frames < 1) throw std::runtime_error(args[4] + " is shorter than a frame");
        const double frame_seconds = (double) frame / rate;
        const auto user_frame = [&](int64_t i) { return std::vector<float>(user.begin() + i * frame, user.begin() + (i + 1) * frame); };

        DialogueSettings settings = file_dialogue_settings(model);
        if (args.size() > 6) settings.max_frames = (int64_t) (std::stod(args[6]) / frame_seconds);
        if (settings.max_frames < frames) throw std::runtime_error("max_seconds holds fewer frames than " + args[4] + " has");
        SeededRandomness warm(1);
        dialogue.start({}, voice.means, voice.quietest, settings, warm);
        for (int64_t i = 0; i < std::min<int64_t>(2, frames); i++) dialogue.step(user_frame(i), warm);
        for (int run = 0; run < runs; run++) {
            SeededRandomness r(1);
            const Clock::time_point starting = Clock::now();
            dialogue.start({}, voice.means, voice.quietest, settings, r);
            const double start = seconds_since(starting);
            const DialogueTimes before = dialogue.times();
            std::vector<double> steps;
            for (int64_t i = 0; i < frames; i++) {
                const Clock::time_point stepping = Clock::now();
                dialogue.step(user_frame(i), r);
                steps.push_back(seconds_since(stepping));
            }
            const DialogueTimes & after = dialogue.times();
            const double total = std::accumulate(steps.begin(), steps.end(), 0.0), ms = 1000.0 / (double) frames;
            const long late = std::count_if(steps.begin(), steps.end(), [&](double s) { return s > frame_seconds; });
            std::printf("{\"run\":%d,\"device\":%s,\"reference_frames\":%zu,\"max_frames\":%lld,\"start\":%.1f,\"frames\":%lld,\"step_p50\":%.1f,"
                        "\"step_p90\":%.1f,\"step_p99\":%.1f,\"late\":%ld,\"encode\":%.1f,\"model\":%.1f,\"sampling\":%.1f,\"decode\":%.1f,"
                        "\"rtfx\":%.2f}\n",
                        run, json_string(ggml_backend_name(backend.get())).c_str(), voice.means.size() / (size_t) codec.latent_dim(),
                        (long long) settings.max_frames, start * 1000.0, (long long) frames, quantile(steps, 0.5) * 1000.0, quantile(steps, 0.9) * 1000.0,
                        quantile(steps, 0.99) * 1000.0, late, (after.encode - before.encode) * ms, (after.model - before.model) * ms,
                        (after.sampling - before.sampling) * ms, (after.decode - before.decode) * ms, (double) frames * frame_seconds / total);
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
