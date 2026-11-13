// Checks Haru's output and MeanFlow head against the official implementation on a dump of reference/haru/dump.py, from
// the hidden state of each frame the dump draws: the text logits, the float32 logits of PAD and EOP and whether they
// keep the dump's choice between the two, then the head's condition, each of its layers, its normalized latents with the
// dump's noise, and the latents the codec decodes.
//
// usage: haru-head-check <model.gguf> <dump dir> [gpu|cpu|device name]

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "haru/head.h"
#include "haru/layout.h"
#include "json-reader.h"
#include "npy.h"
#include "weight-types.h"

using namespace haru;

namespace {

bool check(const std::string & what, const std::vector<float> & got, const std::vector<float> & want, double min_snr) {
    const Diff d = compare(got, want);
    print_diff(what, d);
    if (got.size() != want.size()) std::printf("  %zu values where the dump has %zu\n", got.size(), want.size());
    return got.size() == want.size() && d.snr_db > min_snr;
}

/** The index of the largest of `n` values from `v`. */
size_t argmax(const float * v, size_t n) { return (size_t) (std::max_element(v, v + n) - v); }

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <dump dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    const std::filesystem::path dir = std::filesystem::u8path(args[2]);
    if (!takes_weights(args[1], model_layout, {"F32"}, "its thresholds are those measured with them")) return 2;
    ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
    std::printf("backend: %s\n", ggml_backend_name(backend));
    const ModelFile model(args[1], backend, model_layout, [](const std::string & name) {
        return name == "token_embd" || name == "output_silent" || name.rfind("head.", 0) == 0;
    });
    std::ifstream meta_file(dir / "meta.json");
    const JsonValue meta = parse_json(std::string(std::istreambuf_iterator<char>(meta_file), std::istreambuf_iterator<char>()));
    const float guidance = std::stof(meta.member("guidance_scale")->text);
    const int64_t prompt_rows = std::stoll(meta.member("prompt_rows")->text);

    const Npy hidden_rows = read_npy((dir / "lm_hidden.npy").u8string());
    const Npy noise = read_npy((dir / "head_noise.npy").u8string());
    const int64_t h = hidden_rows.shape[1], n = noise.shape[0], latent = noise.shape[1];
    // Frame 0 comes from the prompt's last row and each frame after it from a step's row.
    const std::vector<float> hidden(hidden_rows.f32.begin() + (std::ptrdiff_t) ((prompt_rows - 1) * h),
                                    hidden_rows.f32.begin() + (std::ptrdiff_t) ((prompt_rows - 1 + n) * h));
    // Measured on an Apple M5 on 2026-11-14 for the ja-listen dump, the lowest SNR: on the CPU 135.5 dB for the logits
    // and 110.4 dB for the head (its third layer); on Metal, whose matrix kernel rounds its inputs to half precision,
    // 79.2 and 52.0 dB, with every frame's largest logit and choice between PAD and EOP the dump's on both. The official
    // head and logits in bfloat16, as the official code runs them, give 33.1 and 54.5 dB against the float32 dump.
    const bool cpu = ggml_backend_is_cpu(backend);
    const double logits_db = cpu ? 100 : 60, head_db = cpu ? 100 : 40;
    bool ok = true;

    Head head(model, backend);
    head.set_guidance(guidance);
    Graph g;
    ggml_tensor * x = g.input(hidden, h, n);
    ggml_tensor * logits = build_logits(g, model, x);
    ggml_tensor * silent = build_silent_logits(g, model, x);
    std::vector<ggml_tensor *> layers;
    ggml_tensor * cond = nullptr;
    ggml_tensor * latents = head.build(g, x, g.input(noise.f32, latent, n), &layers, &cond);
    for (ggml_tensor * t : {logits, silent, latents, cond}) g.output(t);
    for (ggml_tensor * t : layers) g.output(t);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    g.compute(backend, allocr);

    const Npy want_logits = read_npy((dir / "logits.npy").u8string());
    const std::vector<float> got_logits = Graph::read(logits);
    ok = check("text logits", got_logits, want_logits.f32, logits_db) && ok;
    const size_t vocab = (size_t) want_logits.shape[1];
    size_t same_argmax = 0;
    for (int64_t f = 0; f < n; f++) same_argmax += argmax(&got_logits[f * vocab], vocab) == argmax(&want_logits.f32[f * vocab], vocab);
    std::printf("%-34s %zu of %lld frames\n", "largest logit the dump's", same_argmax, (long long) n);
    const Npy want_silent = read_npy((dir / "silent_logits.npy").u8string());
    const std::vector<float> got_silent = Graph::read(silent);
    ok = check("PAD and EOP logits", got_silent, want_silent.f32, logits_db) && ok;
    // Which of PAD and EOP the logits favour decides whether the model stays silent or starts to speak.
    size_t same_choice = 0;
    for (int64_t f = 0; f < n; f++) same_choice += (got_silent[f * 2] > got_silent[f * 2 + 1]) == (want_silent.f32[f * 2] > want_silent.f32[f * 2 + 1]);
    std::printf("%-34s %zu of %lld frames\n", "PAD against EOP as the dump's", same_choice, (long long) n);
    ok = ok && same_argmax == (size_t) n && same_choice == (size_t) n;

    ok = check("head condition", Graph::read(cond), read_npy((dir / "head_cond.npy").u8string()).f32, head_db) && ok;
    const Npy want_layers = read_npy((dir / "head_layers.npy").u8string());
    for (size_t l = 0; l < layers.size(); l++) {
        const std::vector<float> got = Graph::read(layers[l]);
        std::vector<float> want((size_t) (n * h));
        for (int64_t f = 0; f < n; f++)
            std::copy_n(&want_layers.f32[(size_t) ((f * (int64_t) layers.size() + (int64_t) l) * h)], h, &want[(size_t) (f * h)]);
        ok = check("head layer " + std::to_string(l), got, want, head_db) && ok;
    }
    std::vector<float> normalized = Graph::read(latents);
    ok = check("head latents", normalized, read_npy((dir / "head_latents.npy").u8string()).f32, head_db) && ok;
    for (float & v : normalized) v = head.unscale(v);
    ok = check("latents", normalized, read_npy((dir / "latents.npy").u8string()).f32, head_db) && ok;

    ggml_gallocr_free(allocr);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
