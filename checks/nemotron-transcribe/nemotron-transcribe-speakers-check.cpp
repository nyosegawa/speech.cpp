// Checks Nemotron 3.5 Transcribe's speaker branch against NeMo on each dump of reference/nemotron-transcribe/dump.py:
// the streaming steps of Nemotron 3 Diarization, one per window of the encoder, from the dump's log-mel features, each
// step's speaker probabilities of the window's own 80 ms frames against forward_streaming_step()'s. The steps carry the
// speaker cache and the FIFO queue from window to window, so a step's error includes the earlier steps'. The fusion
// takes the probabilities above the encoder's activity threshold as 1 and the rest as 0, so the check also counts the
// frames whose decision differs from the official one. The dumps are those in <reference out dir>/<the file's
// general.name>/.
//
// usage: nemotron-transcribe-speakers-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <cstdio>
#include <string>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "json-reader.h"
#include "nemotron-transcribe/layout.h"
#include "nemotron-transcribe/speakers.h"
#include "npy.h"
#include "reference-dumps.h"
#include "weight-types.h"

using namespace nemotron_transcribe;

namespace {

int64_t integer(const JsonValue & v, const char * name) {
    return std::stoll(v.member(name)->text);
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        if (!takes_weights(args[1], layout, {"F32"}, "the bounds are set from F32's arithmetic")) return 2;
        start_ggml();
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "auto");
        const bool cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        std::printf("device: %s\n", ggml_backend_name(backend));
        bool ok = true;
        {
            const ModelFile model(args[1], backend, layout, [](const std::string & name) { return name.rfind("spk.", 0) == 0; });
            SpeakerBranch branch(model, backend);
            const int factor = branch.factor(), speakers = branch.speakers();
            const float threshold = model.f32("nemotron-transcribe.speaker.activity_threshold");
            const int mels = model.size("nemotron-transcribe.frontend.n_mels");
            Diff worst;
            int64_t flipped = 0, decided = 0;
            for (const auto & dump : reference_dumps(args[2], model)) {
                const JsonValue meta = parse_json(dump_text(dump / "meta.json"));
                const Npy features = read_npy((dump / "features.npy").u8string());
                const int64_t valid = integer(meta, "mel_frames");
                const std::vector<JsonValue> & windows = meta.member("windows")->items;
                branch.start();
                for (size_t k = 0; k < windows.size(); k++) {
                    const JsonValue & w = windows[k];
                    const int64_t start = integer(w, "start"), end = integer(w, "end"), cs = integer(w, "context_start"), ce = integer(w, "context_end");
                    if (start >= valid) continue;
                    const int64_t to = std::min(ce, valid), left = start - cs, right = to - std::min(end, valid);
                    const std::vector<float> got = branch.step(&features.f32[(size_t) (cs * mels)], to - cs, left, right, start / factor,
                                                               k + 1 == windows.size());
                    const Npy want_all = read_npy((dump / "diar_steps" / (std::to_string(k) + ".npy")).u8string());
                    const int64_t n = std::min<int64_t>((int64_t) got.size() / speakers, want_all.shape[0]);
                    const std::vector<float> want(want_all.f32.begin(), want_all.f32.begin() + n * speakers);
                    const std::vector<float> mine(got.begin(), got.begin() + n * speakers);
                    const Diff d = compare(mine, want);
                    int64_t step_flipped = 0;
                    for (size_t i = 0; i < want.size(); i++) step_flipped += (mine[i] > threshold) != (want[i] > threshold);
                    std::printf("%s window %zu (%lld frames, cache %lld, fifo %lld rows): %lld of %lld decisions differ\n",
                                dump.filename().u8string().c_str(), k, (long long) n, (long long) branch.state().cache_rows(),
                                (long long) branch.state().fifo_rows(), (long long) step_flipped, (long long) want.size());
                    print_diff("  probabilities", d);
                    keep_worst(worst, d);
                    flipped += step_flipped;
                    decided += (int64_t) want.size();
                }
            }
            // The bounds of float32 on the CPU, and of a GPU whose matrix kernels round their inputs to half precision.
            const double bound = cpu ? 60.0 : 25.0;
            print_diff("worst probabilities", worst);
            ok = worst.snr_db >= bound;
            std::printf("%s: probabilities SNR %.1f dB against a bound of %.0f dB, %lld of %lld decisions differ\n", ok ? "OK" : "FAIL", worst.snr_db,
                        bound, (long long) flipped, (long long) decided);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
