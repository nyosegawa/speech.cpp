// Checks Nemotron 3.5 Transcribe's perception module against NeMo on each dump of reference/nemotron-transcribe/dump.py,
// from the dump's audio to the audio's rows of the language model's input: the ASR encoder's kept frames of every window,
// the speaker branch's probabilities of them, the fusion's output and the projection's, each over the 80 ms frames the
// language model takes. It runs every stage from the stage before, so an error includes those of the stages before; the
// encoder and speakers checks run each from the dump's own inputs. The dumps are those in <reference out dir>/<the
// file's general.name>/.
//
// usage: nemotron-transcribe-perception-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <cstdio>
#include <string>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "json-reader.h"
#include "nemotron-transcribe/layout.h"
#include "nemotron-transcribe/perception.h"
#include "npy.h"
#include "reference-dumps.h"
#include "weight-types.h"

using namespace nemotron_transcribe;

namespace {

/** The first `n` rows of a dump of rows of `width` values. */
std::vector<float> first_rows(const Npy & a, int64_t n, int64_t width) {
    return std::vector<float>(a.f32.begin(), a.f32.begin() + n * width);
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
            const ModelFile model(args[1], backend, layout, [](const std::string & name) {
                for (const char * prefix : {"frontend.", "asr.", "spk.", "fuse.", "proj."}) {
                    if (name.rfind(prefix, 0) == 0) return true;
                }
                return false;
            });
            Perception perception(model, backend);
            const int64_t d = model.size("nemotron-transcribe.asr.d_model"), speakers = model.size("nemotron-transcribe.speaker.speakers");
            Diff worst;
            for (const auto & dump : reference_dumps(args[2], model)) {
                const JsonValue meta = parse_json(dump_text(dump / "meta.json"));
                const Npy audio = read_npy((dump / "audio.npy").u8string());
                const int64_t frames = std::stoll(meta.member("audio_frames")->text);
                PerceptionStages stages;
                const std::vector<float> rows = perception.encode(audio.f32, {}, &stages);
                if ((int64_t) rows.size() != frames * perception.hidden()) {
                    std::printf("FAIL: %s gives %lld rows, not the official %lld\n", dump.filename().u8string().c_str(),
                                (long long) (rows.size() / (size_t) perception.hidden()), (long long) frames);
                    ok = false;
                    continue;
                }
                std::printf("%s (%lld frames)\n", dump.filename().u8string().c_str(), (long long) frames);
                print_diff("  asr", compare(stages.asr, first_rows(read_npy((dump / "asr.npy").u8string()), frames, d)));
                print_diff("  speakers", compare(stages.speakers, first_rows(read_npy((dump / "diar.npy").u8string()), frames, speakers)));
                print_diff("  fused", compare(stages.fused, first_rows(read_npy((dump / "fused.npy").u8string()), frames, d)));
                const Diff out = compare(rows, first_rows(read_npy((dump / "audio_embeds.npy").u8string()), frames, perception.hidden()));
                print_diff("  audio rows", out);
                keep_worst(worst, out);
            }
            // The bounds of float32 on the CPU, and of a GPU whose matrix kernels round their inputs to half precision.
            const double bound = cpu ? 60.0 : 25.0;
            print_diff("worst audio rows", worst);
            ok = ok && worst.snr_db >= bound;
            std::printf("%s: audio rows SNR %.1f dB against a bound of %.0f dB\n", ok ? "OK" : "FAIL", worst.snr_db, bound);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
