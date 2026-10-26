// Checks Sortformer's network against NeMo on the steps of each configuration of reference/sortformer/dump.py that it
// dumped the stages of, each from the step's own input: the speaker cache's and the FIFO queue's rows as the official
// state held them before the step, and the chunk's and its right context's official features. It compares the chunk's
// pre-encoded frames, the embedding norm's and each layer's output where the dump has them, the final norm's,
// encoder_proj's, the upsampling's and the speakers' probabilities every 10 ms. The dumps are those in
// <reference out dir>/<the file's general.name>/.
//
// usage: sortformer-network-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <cstdio>
#include <string>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "sortformer-dumps.h"
#include "sortformer/diarizer.h"
#include "sortformer/layout.h"
#include "weight-types.h"

using namespace sortformer;
using namespace sortformer_dumps;

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
            Diarizer diarizer(args[1], backend);
            Network & network = diarizer.network();
            const int factor = network.factor(), mels = network.mels(), d = network.d_model(), speakers = network.speakers();
            Diff worst_layers, worst_probs;
            for (const auto & input : reference_dumps(args[2], diarizer.model())) {
                const Npy features = load(input, "features");
                for (const Config & c : configs(input)) {
                    const States states(c, speakers);
                    for (const int64_t k : c.encoder_steps) {
                        const Step & st = c.steps[(size_t) k];
                        const State before = states.after(c, k - 1, diarizer.silence(), speakers);
                        std::vector<float> rows = before.cache;
                        rows.insert(rows.end(), before.fifo.begin(), before.fifo.end());
                        const int64_t n_rows = before.cache_rows() + before.fifo_rows();
                        std::vector<float> stacked((size_t) (st.frames * factor * mels), 0.0f);
                        const float * from = &features.f32[(size_t) (st.first_frame * factor * mels)];
                        std::copy(from, from + st.features * mels, stacked.begin());

                        Graph g;
                        ggml_tensor * pre = nullptr;
                        StepStages stages;
                        ggml_tensor * probs = network.build(g, rows, n_rows, stacked, st.frames, &pre, &stages);
                        g.output(pre);
                        g.output(probs);
                        for (ggml_tensor * t : stages.layers) g.output(t);
                        g.output(stages.hidden);
                        g.output(stages.proj);
                        g.output(stages.upsampled);
                        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                        g.compute(backend, allocr);

                        const std::filesystem::path step = c.dir / ("step" + std::to_string(k));
                        std::printf("%s %s step %lld (%lld rows, %lld frames)\n", input.filename().u8string().c_str(), c.name.c_str(), (long long) k,
                                    (long long) n_rows, (long long) st.frames);
                        // The chunk's frames, then its right context's, as the official pre-encoded them.
                        std::vector<float> want_pre(&states.table.f32[(size_t) (st.first_frame * d)],
                                                    &states.table.f32[(size_t) ((st.first_frame + st.chunk_frames) * d)]);
                        want_pre.insert(want_pre.end(), &states.rc.f32[(size_t) (states.rc_offset[(size_t) k] * d)],
                                        &states.rc.f32[(size_t) ((states.rc_offset[(size_t) k] + st.rc_frames) * d)]);
                        const Diff pre_diff = compare(Graph::read(pre), want_pre);
                        print_diff("  pre-encoded", pre_diff);
                        if (k == c.layer_step) {
                            const Npy layers = load(step, "layers");
                            const size_t per = (size_t) ((n_rows + st.frames) * d);
                            for (size_t l = 0; l < stages.layers.size(); l++) {
                                const std::vector<float> want(layers.f32.begin() + (std::ptrdiff_t) (l * per), layers.f32.begin() + (std::ptrdiff_t) ((l + 1) * per));
                                const Diff diff = compare(Graph::read(stages.layers[l]), want);
                                if (l == 0 || l + 1 == stages.layers.size() || l % 10 == 0) print_diff(l == 0 ? "  embedding norm" : "  layer " + std::to_string(l - 1), diff);
                                keep_worst(worst_layers, diff);
                            }
                        }
                        const Diff hidden = compare(Graph::read(stages.hidden), load(step, "hidden").f32);
                        const Diff proj = compare(Graph::read(stages.proj), load(step, "proj").f32);
                        const Diff up = compare(Graph::read(stages.upsampled), load(step, "upsampled").f32);
                        const Diff p = compare(Graph::read(probs), load(step, "preds_hr").f32);
                        print_diff("  final norm", hidden);
                        print_diff("  encoder_proj", proj);
                        print_diff("  upsampled", up);
                        print_diff("  probabilities", p);
                        keep_worst(worst_probs, p);
                        // Measured on an Apple M5 (2026-10-26): on the CPU in F32 the pre-encoded frames lie at 115 dB and
                        // more, the final norm at 107 dB and more and the probabilities within 1.5e-5; on Metal, whose matrix
                        // kernel rounds its inputs to half precision, at 61 dB, 57 dB and within 2.9e-3. A wrong rotation,
                        // norm or kernel tap falls far below.
                        const double min_pre = cpu ? 100 : 50, min_hidden = cpu ? 95 : 45, max_prob = cpu ? 1e-4 : 2e-2;
                        ok = ok && pre_diff.snr_db > min_pre && hidden.snr_db > min_hidden && p.max_abs < max_prob && Graph::read(probs).size() == load(step, "preds_hr").f32.size();
                        ggml_gallocr_free(allocr);
                    }
                }
            }
            print_diff("worst layer", worst_layers);
            print_diff("worst probabilities", worst_probs);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
