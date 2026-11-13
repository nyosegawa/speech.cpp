// Checks Haru's codec against the official implementation on a dump of reference/haru/dump.py, each stage in the order
// data flows. Encoder: its stages and the latent means on the dump's voice reference encoded whole, the means of the
// user's frames encoded a frame at a time as a conversation encodes them, and the latent of digital silence. Decoder:
// the model's latents decoded a frame at a time as a conversation decodes them, and its stages on the latents decoded
// whole. Encoding and decoding a few frames at a time, in windows of 1 to 7 frames, against at once.
//
// usage: haru-codec-check <model.gguf> <dump dir> [gpu|cpu|device name]

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "haru/codec.h"
#include "haru/layout.h"
#include "npy.h"
#include "weight-types.h"

using namespace haru;

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/** Compares `got` with the dump `name`, prints the error and says whether its SNR is above `min_snr`. */
bool check(const std::string & what, const std::vector<float> & got, const std::string & dir, const std::string & name, double min_snr) {
    const Npy want = read_npy(dir + "/" + name + ".npy");
    const Diff d = compare(got, want.f32);
    print_diff(what, d);
    if (got.size() != want.f32.size()) std::printf("  %zu values where the dump has %zu\n", got.size(), want.f32.size());
    return got.size() == want.f32.size() && d.snr_db > min_snr;
}

/** Each stage against the dump's codec_<stage>, [time, channels], the channel fastest as ggml keeps it. */
bool check_stages(const CodecStages & stages, const std::string & side, int count, const std::string & dir, double min_snr) {
    bool ok = true;
    for (int i = 0; i <= count; i++) {
        const std::string name = side + "_" + (i < count ? std::to_string(i) : std::string("head"));
        ok = check(name, stages.at(name), dir, "codec_" + name, min_snr) && ok;
    }
    return ok;
}

/** The values of frames [first, first + n) of `values`, `width` values a frame. */
std::vector<float> frames_of(const std::vector<float> & values, int64_t width, int64_t first, int64_t n) {
    return std::vector<float>(values.begin() + (std::ptrdiff_t) (first * width), values.begin() + (std::ptrdiff_t) ((first + n) * width));
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <dump dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    const std::string dir = args[2];
    if (!takes_weights(args[1], model_layout, {"F32"}, "its thresholds are those measured with them")) return 2;
    ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
    std::printf("backend: %s\n", ggml_backend_name(backend));
    const ModelFile model(args[1], backend, model_layout, [](const std::string & name) { return name.rfind("codec.", 0) == 0; });
    Codec codec(model, backend);
    const int stages = (int) model.i32_array("haru.codec.ratios").size() + 1;
    const int64_t fs = codec.frame_samples(), latent = codec.latent_dim();
    const bool cpu = ggml_backend_is_cpu(backend);
    const double encoder_db = cpu ? 60 : 30, decoder_db = cpu ? 60 : 30, window_db = cpu ? 90 : 40;
    bool ok = true;

    const Npy reference = read_npy(dir + "/reference_audio.npy");
    CodecStages encoded;
    auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> means = codec.encode(reference.f32, &encoded);
    std::printf("encoding %.2f s of audio: %.3f s\n", (double) reference.f32.size() / 24000.0, seconds_since(t0));
    ok = check_stages(encoded, "encoder", stages, dir, encoder_db) && ok;
    ok = check("reference means", means, dir, "reference_means", encoder_db) && ok;

    const Npy user = read_npy(dir + "/user_audio.npy");
    const int64_t user_frames = (int64_t) user.f32.size() / fs;
    codec.reset_encoder();
    std::vector<float> user_means;
    t0 = std::chrono::steady_clock::now();
    for (int64_t f = 0; f < user_frames; f++) {
        const std::vector<float> m = codec.encode(frames_of(user.f32, fs, f, 1));
        user_means.insert(user_means.end(), m.begin(), m.end());
    }
    std::printf("encoding %lld user frames one at a time: %.3f s\n", (long long) user_frames, seconds_since(t0));
    ok = check("user means, a frame at a time", user_means, dir, "user_means", encoder_db) && ok;

    const int64_t silence_frames = model.size("haru.dialogue.silence_frames");
    codec.reset_encoder();
    const std::vector<float> zeros = codec.encode(std::vector<float>((size_t) (silence_frames * fs), 0.0f));
    ok = check("silence latent", frames_of(zeros, latent, silence_frames - 1, 1), dir, "silence_latent", encoder_db) && ok;

    const Npy latents = read_npy(dir + "/latents.npy");
    const int64_t n = latents.shape[0];
    codec.reset_decoder();
    std::vector<float> streamed;
    t0 = std::chrono::steady_clock::now();
    for (int64_t f = 0; f < n; f++) {
        const std::vector<float> a = codec.decode(frames_of(latents.f32, latent, f, 1), 1);
        streamed.insert(streamed.end(), a.begin(), a.end());
    }
    std::printf("decoding %lld frames one at a time: %.3f s\n", (long long) n, seconds_since(t0));
    ok = check("audio, a frame at a time", streamed, dir, "codec_audio", decoder_db) && ok;

    codec.reset_decoder();
    CodecStages decoded;
    const std::vector<float> whole = codec.decode(latents.f32, n, &decoded);
    ok = check_stages(decoded, "decoder", stages, dir, decoder_db) && ok;

    // Windows of 1 to 7 frames, which cross every stride's frame and every convolution's reach.
    codec.reset_decoder();
    std::vector<float> windowed;
    for (int64_t first = 0, size = 1; first < n; first += size, size = size % 7 + 1) {
        const int64_t count = std::min(size, n - first);
        const std::vector<float> a = codec.decode(frames_of(latents.f32, latent, first, count), count);
        windowed.insert(windowed.end(), a.begin(), a.end());
    }
    const Diff dw = compare(windowed, whole);
    print_diff("decoding in windows against at once", dw);
    ok = ok && windowed.size() == whole.size() && dw.snr_db > window_db;

    codec.reset_encoder();
    const std::vector<float> user_whole = codec.encode(std::vector<float>(user.f32.begin(), user.f32.begin() + (std::ptrdiff_t) (user_frames * fs)));
    codec.reset_encoder();
    std::vector<float> user_windowed;
    for (int64_t first = 0, size = 1; first < user_frames; first += size, size = size % 7 + 1) {
        const int64_t count = std::min(size, user_frames - first);
        const std::vector<float> m = codec.encode(frames_of(user.f32, fs, first, count));
        user_windowed.insert(user_windowed.end(), m.begin(), m.end());
    }
    const Diff de = compare(user_windowed, user_whole);
    print_diff("encoding in windows against at once", de);
    ok = ok && user_windowed.size() == user_whole.size() && de.snr_db > window_db;

    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
