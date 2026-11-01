// Times SEMamba's stages on one chunk of seeded noise: the STFT on the host, the dense encoder, one TFMamba block, all of
// them, the two decoders, the whole network, and the decoding and inverse STFT on the host. Each stage is its own graph,
// computed once untimed and then `runs` times; each line gives the median in milliseconds.
//
// usage: semamba-timing <model.gguf> <gpu|cpu|device name> <rate> <seconds> <runs>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "semamba/enhancer.h"

using namespace semamba;

namespace {

using Clock = std::chrono::steady_clock;

double median_ms(int runs, const std::function<void()> & run) {
    run();
    std::vector<double> times;
    for (int i = 0; i < runs; i++) {
        const auto start = Clock::now();
        run();
        times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

}  // namespace

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    if (args.size() < 6) {
        std::fprintf(stderr, "usage: %s <model.gguf> <gpu|cpu|device name> <rate> <seconds> <runs>\n", args[0].c_str());
        return 2;
    }
    try {
        start_ggml();
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(init_backend(args[2]), ggml_backend_free);
        const int rate = std::stoi(args[3]), runs = std::stoi(args[5]);
        const double seconds = std::stod(args[4]);
        Enhancer enhancer(args[1], backend.get());
        const Network & net = enhancer.net();
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocr(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())),
                                                                          ggml_gallocr_free);

        std::mt19937 rng(1);
        std::normal_distribution<float> normal(0.0f, 0.1f);
        std::vector<float> audio((size_t) (seconds * rate));
        for (float & s : audio) s = normal(rng);
        const Stft stft = enhancer.stft(rate);
        const int64_t frames = stft.frames(audio.size()), bins = stft.bins();
        std::vector<float> magnitude, phase;
        const double t_stft = median_ms(runs, [&] { stft.forward(audio.data(), audio.size(), magnitude, phase); });
        const int64_t f = net.encoded_bins(bins), t = net.encoded_frames(frames), h = net.hidden();
        std::vector<float> hidden((size_t) (f * t * h));
        for (float & v : hidden) v = normal(rng);

        const auto graph = [&](const std::function<ggml_tensor *(Graph &)> & build) {
            return median_ms(runs, [&] {
                Graph g;
                g.output(build(g));
                g.compute(backend.get(), allocr.get());
            });
        };
        const double t_encoder = graph([&](Graph & g) {
            std::vector<float> input(magnitude);
            input.insert(input.end(), phase.begin(), phase.end());
            return net.encode(g.ctx(), g.input(input, bins, frames, 2));
        });
        const auto blocks = [&](int count) {
            return graph([&](Graph & g) {
                ggml_context * ctx = g.ctx();
                ggml_tensor * x = ggml_cont(ctx, ggml_permute(ctx, g.input(hidden, f, t, h), 2, 1, 0, 3));
                for (int b = 0; b < count; b++) x = net.block(ctx, b, x);
                return x;
            });
        };
        const double t_block = blocks(1), t_blocks = blocks(net.blocks());
        const double t_decoders = graph([&](Graph & g) {
            ggml_context * ctx = g.ctx();
            ggml_tensor * x = g.input(hidden, f, t, h);
            ggml_tensor * m = net.project(ctx, "mag.out", net.decode(ctx, "mag", x));
            ggml_tensor * p = net.decode(ctx, "pha", x);
            ggml_tensor * r = net.project(ctx, "pha.r", p), * i = net.project(ctx, "pha.i", p);
            g.output(r);
            g.output(i);
            return m;
        });
        Enhancer::Raw raw;
        const double t_network = median_ms(runs, [&] { raw = enhancer.network(magnitude, phase, frames, bins); });
        const double t_inverse = median_ms(runs, [&] {
            const Decoded d = decode(raw.magnitude, raw.real, raw.imag, raw.bins, frames, bins, 0.5f);
            stft.inverse(d.magnitude, d.phase, (int) frames);
        });

        std::printf("device %s, %d Hz, %.2f s: %lld frames of %lld bins, %lld by %lld after the encoder\n", ggml_backend_name(backend.get()), rate,
                    seconds, (long long) frames, (long long) bins, (long long) t, (long long) f);
        std::printf("stft %.1f ms\nencoder %.1f ms\none block %.1f ms\n%d blocks %.1f ms\ndecoders %.1f ms\nnetwork %.1f ms\ndecode and inverse %.1f ms\n", t_stft,
                    t_encoder, t_block, net.blocks(), t_blocks, t_decoders, t_network, t_inverse);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
