// Checks SEMamba's network against RE-USE's official code on the dumps of reference/semamba/dump.py, each stage from its
// own dumped input: the dense encoder from the first chunk's magnitude and phase; where the dump has every block's output,
// each TFMamba block from the output before it and the first block's bidirectional Mamba over time and over frequency
// from its inputs; the two decoders from the last block's output; and the whole network from each chunk's magnitude and
// phase. The dumps are those in <reference out dir>/<the file's general.name>/.
//
// usage: semamba-network-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <cstdio>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "semamba-dumps.h"
#include "semamba/enhancer.h"
#include "semamba/layout.h"
#include "weight-types.h"

using namespace semamba_dumps;

namespace {

// float32 against CUDA's float32, with sums in other orders: on an Apple M5 (2026-11-01), the CPU and Metal gave the
// encoder 98.9 to 130 dB, each block and block 0's Mamba 124 to 138 dB, the decoders 88.0 to 138 dB and the whole
// network of a chunk 78.7 to 116 dB, where 30 blocks carry a stage's rounding to the next. A wrong kernel, norm, flip or
// layout falls far below, as products staged through half did at -21 dB.
constexpr double kStage = 80, kWhole = 60;

/** Computes `g` on `backend` and reads `t`. */
std::vector<float> run(Graph & g, ggml_tensor * t, ggml_backend_t backend, ggml_gallocr_t allocr) {
    g.output(t);
    g.compute(backend, allocr);
    return Graph::read(t);
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        if (!takes_weights(args[1], semamba::layout, {"F32"}, "the bounds are set from F32's arithmetic")) return 2;
        start_ggml();
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "auto");
        std::printf("device: %s\n", ggml_backend_name(backend));
        bool ok = true;
        {
            semamba::Enhancer enhancer(args[1], backend);
            const semamba::Network & net = enhancer.net();
            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            for (const auto & dump : dumps(args[2], enhancer.model())) {
                const Meta meta = read_meta(dump);
                const auto first = chunk(dump, 0);
                const Npy mag = read(first, "mag"), pha = read(first, "pha"), encoded = read(first, "encoded"), last = read(first, "mamba_out");
                const int64_t frames = mag.shape[0], bins = mag.shape[1];
                const int64_t f = encoded.shape[2], t = encoded.shape[1], h = encoded.shape[0];
                std::printf("%s (%d Hz, %lld frames of %lld bins, %lld by %lld after the encoder)\n", dump.filename().u8string().c_str(), meta.rate,
                            (long long) frames, (long long) bins, (long long) t, (long long) f);
                {
                    Graph g;
                    std::vector<float> input(mag.f32);
                    input.insert(input.end(), pha.f32.begin(), pha.f32.end());
                    const Diff d = compare(run(g, net.encode(g.ctx(), g.input(input, bins, frames, 2)), backend, allocr), encoded.f32);
                    print_diff("  encoder", d);
                    ok = ok && d.snr_db > kStage;
                }
                if (std::filesystem::exists(first / "blocks.npy")) {
                    const Npy blocks = read(first, "blocks");
                    const size_t size = (size_t) (f * t * h);
                    Diff worst{};
                    for (int b = 0; b < net.blocks(); b++) {
                        Graph g;
                        const std::vector<float> in = b == 0 ? encoded.f32 : std::vector<float>(blocks.f32.begin() + (b - 1) * size, blocks.f32.begin() + b * size);
                        ggml_context * ctx = g.ctx();
                        const auto over_frames = net.sequences(g, t, f), over_bins = net.sequences(g, f, t);
                        ggml_tensor * x = ggml_cont(ctx, ggml_permute(ctx, g.input(in, f, t, h), 2, 1, 0, 3));
                        x = ggml_cont(ctx, ggml_permute(ctx, net.block(ctx, b, x, over_frames, over_bins), 2, 1, 0, 3));
                        keep_worst(worst, compare(run(g, x, backend, allocr), std::vector<float>(blocks.f32.begin() + b * size, blocks.f32.begin() + (b + 1) * size)));
                    }
                    print_diff("  each block, the worst", worst);
                    ok = ok && worst.snr_db > kStage;
                    for (const char * part : {"time", "freq"}) {
                        const Npy in = read(first, (std::string(part) + "_in").c_str()), out = read(first, (std::string(part) + "_out").c_str());
                        Graph g;
                        const int64_t length = in.shape[1], count = in.shape[0];
                        const auto s = net.sequences(g, length, count);
                        const Diff d = compare(run(g, net.bidirectional(g.ctx(), std::string("blk.0.") + part, g.input(in.f32, h, length, count), s), backend, allocr), out.f32);
                        print_diff(std::string("  block 0's Mamba over ") + (std::string(part) == "time" ? "time" : "frequency"), d);
                        ok = ok && d.snr_db > kStage;
                    }
                }
                {
                    const Npy raw = read(first, "mag_raw"), real = read(first, "phase_r"), imag = read(first, "phase_i");
                    Graph g;
                    ggml_context * ctx = g.ctx();
                    ggml_tensor * x = g.input(last.f32, f, t, h);
                    ggml_tensor * mag_out = net.project(ctx, "mag.out", net.decode(ctx, "mag", x));
                    ggml_tensor * phase_x = net.decode(ctx, "pha", x);
                    ggml_tensor * r = net.project(ctx, "pha.r", phase_x), * i = net.project(ctx, "pha.i", phase_x);
                    g.output(r);
                    g.output(i);
                    const std::vector<float> got = run(g, mag_out, backend, allocr);
                    const Diff dm = compare(got, raw.f32), dr = compare(Graph::read(r), real.f32), di = compare(Graph::read(i), imag.f32);
                    print_diff("  magnitude decoder", dm);
                    print_diff("  phase decoder, real", dr);
                    print_diff("  phase decoder, imaginary", di);
                    ok = ok && dm.snr_db > kStage && dr.snr_db > kStage && di.snr_db > kStage;
                }
                Diff whole{};
                for (size_t k = 0; k < meta.starts.size(); k++) {
                    const auto folder = chunk(dump, k);
                    const Npy m = read(folder, "mag"), p = read(folder, "pha"), raw = read(folder, "mag_raw"), real = read(folder, "phase_r"),
                              imag = read(folder, "phase_i");
                    const semamba::Enhancer::Raw got = enhancer.network(m.f32, p.f32, m.shape[0], m.shape[1]);
                    keep_worst(whole, compare(got.magnitude, raw.f32));
                    keep_worst(whole, compare(got.real, real.f32));
                    keep_worst(whole, compare(got.imag, imag.f32));
                }
                print_diff("  network from each chunk, the worst", whole);
                ok = ok && whole.snr_db > kWhole;
            }
            ggml_gallocr_free(allocr);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
