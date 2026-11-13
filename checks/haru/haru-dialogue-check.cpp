// Checks a whole Haru conversation against the official one on a dump of reference/haru/dump.py: the official session's
// card, the reference's latent means, the user's audio a frame at a time, and the official draws, its noise and the
// tokens it drew, replayed, so that the conversation takes the dump's course. Each stage is compared as it comes, the
// prompt's rows and their input, each step's user latent and input row, each frame's hidden state, logits, head latent,
// latent and audio, and the text; each drawn token is checked to be one the conversation's own draw could take.
//
// Each frame's latent goes back into the model as the next row's input, so that a difference of rounding grows from
// frame to frame, most where the model starts to speak, where the head's latent turns on a small change of the hidden
// state: on a GPU, whose matrix kernel rounds its inputs to half precision, the conversation leaves the dump's course
// within some tens of frames, as the official code in bfloat16 does. On a GPU the prompt and the first frame, before any
// latent has come back, are held to bounds, and the whole conversation is printed; on the CPU in F32 the whole is held.
// The time of each step is printed, with the reading of the stages, on the F32 file this check takes.
//
// usage: haru-dialogue-check <model.gguf> <dump dir> [gpu|cpu|device name]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "haru/dialogue.h"
#include "haru/layout.h"
#include "json-reader.h"
#include "npy.h"
#include "weight-types.h"

using namespace haru;

namespace {

/** The official draws of a dump, given in the order a conversation takes them, and the tokens the official session drew. */
class Replay : public Randomness {
public:
    std::deque<float> normals;
    std::deque<int32_t> tokens;
    /** The drawn tokens the conversation's candidates gave no chance. */
    int unlikely = 0;

    void normal(float * out, size_t n) override {
        if (normals.size() < n) throw std::runtime_error("the dump has fewer draws than the conversation takes");
        for (size_t i = 0; i < n; i++, normals.pop_front()) out[i] = normals.front();
    }
    int32_t token(const Candidates & c) override {
        const int32_t t = tokens.front();
        tokens.pop_front();
        const auto at = std::find(c.tokens.begin(), c.tokens.end(), t);
        if (at == c.tokens.end() || c.weights[(size_t) (at - c.tokens.begin())] <= 0.0) unlikely++;
        return t;
    }
};

bool check(const std::string & what, const std::vector<float> & got, const std::vector<float> & want, double min_snr) {
    const Diff d = compare(got, want);
    print_diff(what, d);
    if (got.size() != want.size()) std::printf("  %zu values where the dump has %zu\n", got.size(), want.size());
    return got.size() == want.size() && d.snr_db > min_snr;
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
    const ModelFile model(args[1], backend, model_layout);
    Dialogue dialogue(model, backend);
    std::ifstream meta_file(dir + "/meta.json");
    const JsonValue meta = parse_json(std::string(std::istreambuf_iterator<char>(meta_file), std::istreambuf_iterator<char>()));
    const int64_t frames = std::stoll(meta.member("frames")->text);
    const int64_t fs = dialogue.frame_samples();

    Replay replay;
    const auto take = [&](const std::string & name) { return read_npy(dir + "/" + name + ".npy").f32; };
    const std::vector<float> ref_scale = take("reference_scale"), ref_noise = take("reference_noise"), head_noise = take("head_noise");
    const std::vector<float> user_scale = take("user_scale"), user_noise = take("user_noise");
    const int64_t latent = (int64_t) head_noise.size() / frames;
    replay.normals.insert(replay.normals.end(), ref_scale.begin(), ref_scale.end());
    replay.normals.insert(replay.normals.end(), ref_noise.begin(), ref_noise.end());
    replay.normals.insert(replay.normals.end(), head_noise.begin(), head_noise.begin() + latent);
    for (int64_t s = 0; s + 1 < frames; s++) {
        replay.normals.push_back(user_scale[(size_t) s]);
        replay.normals.insert(replay.normals.end(), user_noise.begin() + s * latent, user_noise.begin() + (s + 1) * latent);
        replay.normals.insert(replay.normals.end(), head_noise.begin() + (s + 1) * latent, head_noise.begin() + (s + 2) * latent);
    }
    for (int32_t t : read_npy(dir + "/tokens.npy").i32) replay.tokens.push_back(t);

    DialogueSettings s = file_dialogue_settings(model);
    s.guidance = std::stof(meta.member("guidance_scale")->text);
    s.speak_first = meta.member("speak_first")->boolean;
    s.max_frames = frames;
    const std::vector<int32_t> card = card_ids(model, dialogue.tokenizer(), dialogue.added(), meta.member("card")->text);
    const std::vector<float> reference_audio = take("reference_audio");
    DialogueStages stages;
    std::vector<std::string> text;
    auto t0 = std::chrono::steady_clock::now();
    // The reference's means are the dump's, the codec's stage of its own check.
    DialogueFrame f = dialogue.start(card, take("reference_means"), quietest_frame(reference_audio, fs), s, replay, &stages);
    std::printf("prompt of %zu rows and the first frame: %.3f s\n", stages.prompt_tokens.size(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::vector<float> audio = f.audio;
    text.push_back(f.text);
    const std::vector<float> user = take("user_audio");
    std::vector<double> times;
    for (int64_t step = 0; step + 1 < frames; step++) {
        const std::vector<float> frame(user.begin() + step * fs, user.begin() + (step + 1) * fs);
        t0 = std::chrono::steady_clock::now();
        f = dialogue.step(frame, replay, &stages);
        times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1000);
        audio.insert(audio.end(), f.audio.begin(), f.audio.end());
        text.push_back(f.text);
    }
    std::sort(times.begin(), times.end());
    if (!times.empty()) {
        std::printf("a step, with the stages' reading: p50 %.1f ms, p90 %.1f ms, max %.1f ms\n", times[times.size() / 2],
                    times[times.size() * 9 / 10], times.back());
    }

    // Measured on an Apple M5 on 2026-11-14 for the ja-listen and ja-speak dumps, the lowest SNR: on the CPU in F32 over
    // the whole conversation, 97.6 dB for the language model's stages, 73.2 dB for the latents (the last frames of
    // ja-listen, where the model starts to speak again, 55 to 68 dB; the rest 85 dB and up) and 60.7 dB for the audio;
    // on Metal, for the prompt and the first frame, 50.4 dB (ja-speak's first audio; its latent 57.4 dB). The official
    // session in bfloat16 on MPS, replayed with the same draws and tokens, gives 29.8 dB for ja-speak's first latent
    // against its float32 dump, and 5.2 dB for its latents and 3.3 dB for its audio over the conversation, where Metal
    // gives 10.9 and 6.0 dB.
    const bool cpu = ggml_backend_is_cpu(backend);
    bool ok = true;
    const std::vector<int32_t> tokens = read_npy(dir + "/prompt_tokens.npy").i32;
    const bool prompt_ok = stages.prompt_tokens == tokens;
    std::printf("%-34s %s\n", "prompt tokens", prompt_ok ? "equal" : "differ");
    ok = ok && prompt_ok;
    const Npy hidden = read_npy(dir + "/lm_hidden.npy");
    const int64_t h = hidden.shape[1], prompt_rows = (int64_t) tokens.size();
    const std::vector<float> hidden_frames(hidden.f32.begin() + (prompt_rows - 1) * h, hidden.f32.begin() + (prompt_rows - 1 + frames) * h);
    struct Stage {
        const char * what;
        const std::vector<float> & got;
        std::vector<float> want;
        int64_t first;
        double cpu_db;
    };
    const std::vector<float> embeddings = take("embeddings");
    const Stage all[] = {
        {"input rows", stages.embeddings, embeddings, prompt_rows * h, 90},
        {"user means", stages.user_means, take("user_means"), 0, 90},
        {"user latents", stages.user_latents, take("user_latents"), 0, 90},
        {"hidden states", stages.hidden, hidden_frames, h, 90},
        {"text logits", stages.logits, take("logits"), (int64_t) stages.logits.size() / frames, 90},
        {"PAD and EOP logits", stages.silent_logits, take("silent_logits"), 2, 90},
        {"head latents", stages.head_latents, take("head_latents"), latent, 60},
        {"latents", stages.latents, take("latents"), latent, 60},
        {"codec audio", stages.codec_audio, take("codec_audio"), fs, 50},
        {"audio", audio, take("audio"), fs, 50},
    };
    if (!cpu) {
        // The prompt's rows and the first frame, before any of the model's latents has come back.
        for (const Stage & st : all) {
            if (st.first == 0) continue;
            const std::vector<float> got(st.got.begin(), st.got.begin() + std::min<int64_t>(st.first, (int64_t) st.got.size()));
            const std::vector<float> want(st.want.begin(), st.want.begin() + std::min<int64_t>(st.first, (int64_t) st.want.size()));
            ok = check(std::string(st.what) + ", first", got, want, 40) && ok;
        }
    }
    for (const Stage & st : all) {
        const bool held = check(st.what, st.got, st.want, cpu ? st.cpu_db : -INFINITY);
        ok = (cpu ? held : true) && ok;
    }
    std::printf("%-34s %d of %lld outside the conversation's own candidates\n", "drawn tokens", replay.unlikely, (long long) frames);
    ok = ok && replay.unlikely == 0;
    int same_text = 0;
    for (size_t i = 0; i < text.size(); i++) same_text += text[i] == meta.member("text")->items[i].text;
    std::printf("%-34s %d of %zu frames equal\n", "text", same_text, text.size());
    ok = ok && same_text == (int) text.size();
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
