// Checks the speaker encoder and the duration predictor of Irodori-TTS against the official implementation
// on every dump of reference/irodori-tts/dump.py, each stage from the dump's own inputs: the reference latent
// for the speaker encoder, and the text and speaker conditions for the duration predictor, or the null speaker
// for a dump without a reference. The length is checked against the frames the official runtime synthesized,
// with the dump's seconds, duration scale and speed: a dump with fixed seconds has no prediction, and only its
// length is checked.
//
// usage: irodori-condition-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "irodori-dumps.h"
#include "irodori-tts/duration.h"
#include "irodori-tts/layout.h"
#include "irodori-tts/speaker-encoder.h"
#include "npy.h"

using namespace irodori;

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        const ModelFile model(args[1], backend, model_layout);
        const SpeakerEncoder speaker(model);
        const DurationPredictor duration(model);
        const bool null_speaker = model.boolean("irodori-tts.duration.null_speaker");
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        // The speaker encoder and the duration predictor are the same in v4.1-Small-MF and v4.1-Small, so every
        // dump checks them, whichever model made it.
        std::vector<std::filesystem::path> dirs;
        for (const auto & e : std::filesystem::directory_iterator(std::filesystem::u8path(args[2]))) {
            if (std::filesystem::exists(e.path() / "meta.json") && std::filesystem::exists(e.path() / "dit_t.npy")) dirs.push_back(e.path());
        }
        std::sort(dirs.begin(), dirs.end());
        bool ok = !dirs.empty();
        for (const auto & dir : dirs) {
            const IrodoriDump d(dir);
            std::printf("%s\n", dir.filename().u8string().c_str());
            if (!d.reference() && !null_speaker) {
                std::printf("  skipped: the file lacks the null speaker that a request without a reference speaks with\n");
                continue;
            }
            std::vector<float> summary;
            if (d.reference()) {
                const Npy latent = read_npy(d.file("ref_latent.npy").u8string());
                const Npy encoded_ref = read_npy(d.file("speaker_encoded.npy").u8string());
                const Npy state_ref = read_npy(d.file("speaker_state.npy").u8string());
                Graph g;
                ggml_tensor * encoded = nullptr;
                ggml_tensor * state = speaker.build(g, latent.f32, &encoded);
                g.output(encoded);
                g.output(state);
                g.compute(backend, allocr);
                print_diff("  speaker encoder", compare(Graph::read(encoded), encoded_ref.f32));
                const Diff ds = compare(Graph::read(state), state_ref.f32);
                print_diff("  speaker condition", ds);
                // Measured on an Apple M5: 111 dB on the CPU in F32, 47 dB on Metal, 50 dB with F16 weights
                // and 25 dB with Q8_0. A wrong operation falls far below this bound.
                ok = ok && ds.snr_db > 20;
                summary.assign(state_ref.f32.begin(), state_ref.f32.begin() + state_ref.shape[1]);
            }
            const LengthOptions options = d.length();
            const int want = (int) d.number("latent_frames", -1);
            if (options.fixed()) {
                const int frames = duration.length(options, 0).frames;
                std::printf("  length: %g s at speed %g, %d frames (official %d)\n", options.seconds, options.speed, frames, want);
                ok = ok && frames == want;
                continue;
            }
            const Npy text = read_npy(d.file("text_state.npy").u8string());
            const Npy log_frames = read_npy(d.file("duration_log_frames.npy").u8string());
            Graph g;
            ggml_tensor * text_state = g.input(text.f32, text.shape[1], text.shape[0]);
            ggml_tensor * speaker_summary = d.reference() ? g.input(summary, (int64_t) summary.size()) : model.tensor("duration.null_speaker");
            ggml_tensor * sum = duration.build(g, text_state, speaker_summary);
            g.output(sum);
            g.compute(backend, allocr);
            const float predicted = Graph::read(sum)[0];
            const int frames = duration.length(options, predicted).frames;
            std::printf("  duration%s: log(1 + frames) %.6f (official %.6f), scale %g at speed %g, %d frames (official %d)\n",
                        d.reference() ? "" : " with the null speaker", std::log1p(predicted), log_frames.f32[0], options.duration_scale,
                        options.speed, frames, want);
            // F32 and F16 give the official frames; Q8_0 weights move a 27 s text by one frame (40 ms).
            ok = ok && std::abs(frames - want) <= 1;
        }
        ggml_gallocr_free(allocr);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
