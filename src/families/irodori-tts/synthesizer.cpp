#include "synthesizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "error.h"
#include "layout.h"
#include "text-normalizer.h"

namespace irodori {

namespace {

/** Adds the seconds from its construction to its destruction to `total`. */
struct Timer {
    double & total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~Timer() { total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
};

bool starts_with_riff(const std::string & path) {
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) throw Error(Fault::Io, "cannot open " + path + "; check the path and that the file can be read");
    char magic[4] = {};
    f.read(magic, 4);
    return std::memcmp(magic, "RIFF", 4) == 0;
}

}  // namespace

TailCut::TailCut(const ModelFile & m)
    : window((int) m.u32("irodori-tts.tail.window")),
      std_threshold(m.f32("irodori-tts.tail.std_threshold")),
      mean_threshold(m.f32("irodori-tts.tail.mean_threshold")) {}

int TailCut::flattening_point(const std::vector<float> & latent, int frames, int latent_dim) const {
    for (int i = 0; i < frames; i++) {
        double sum = 0, squares = 0;
        for (int f = i; f < i + window; f++) {
            for (int d = 0; d < latent_dim; d++) {
                const double v = f < frames ? latent[(size_t) f * latent_dim + d] : 0.0;
                sum += v;
                squares += v * v;
            }
        }
        const double n = (double) window * latent_dim, mean = sum / n;
        const double deviation = std::sqrt(std::max(0.0, squares / n - mean * mean));
        if (deviation < std_threshold && std::fabs(mean) < mean_threshold) return i;
    }
    return frames;
}

Synthesizer::Synthesizer(const std::string & model_path, ggml_backend_t backend)
    : backend_(backend),
      model_(std::make_unique<ModelFile>(model_path, backend, model_layout)),
      codec_(*model_, backend),
      tokenizer_(*model_),
      text_(*model_),
      speaker_(*model_),
      duration_(*model_),
      dit_(*model_),
      sampler_(dit_, *model_, backend),
      reference_(*model_),
      tail_(*model_) {
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

Synthesizer::~Synthesizer() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

Voice Synthesizer::voice_from_latent(std::vector<float> latent) {
    Voice v;
    v.frames = (int) (latent.size() / codec_.latent_dim());
    v.latent = std::move(latent);
    Graph g;
    ggml_tensor * state = speaker_.build(g, v.latent);
    g.output(state);
    g.compute(backend_, allocr_);
    v.speaker = Graph::read(state);
    v.speaker_tokens = (int) state->ne[1];
    return v;
}

Voice Synthesizer::load_voice(const std::string & path) {
    if (starts_with_riff(path)) {
        return voice_from_latent(encode_reference(codec_, path, reference_).latent);
    }
    ModelFile file(path, backend_, voice_layout);
    const std::string codec = file.str("irodori-tts-voice.codec_sha256");
    if (codec != codec_.sha256()) {
        throw Error(Fault::InvalidArgument, path + " was made with the codec of SHA-256 " + codec + ", and " + model_->str("general.name") +
                                                " has the codec " + codec_.sha256() + "; make the voice again from its WAVE file with this model");
    }
    ggml_tensor * t = file.tensor("latent");
    if (t->type != GGML_TYPE_F32 || t->ne[0] != codec_.latent_dim() || ggml_n_dims(t) > 2) {
        throw Error(Fault::File, path + " holds no latent of " + std::to_string(codec_.latent_dim()) + " channels in float32; " + file.remedy());
    }
    std::vector<float> latent(ggml_nelements(t));
    ggml_backend_tensor_get(t, latent.data(), 0, ggml_nbytes(t));
    return voice_from_latent(std::move(latent));
}

size_t Synthesizer::synthesize(const Request & r, const Voice & voice, const AudioSink & sink, Stats * stats) {
    duration_.check(r.length);
    Stats local;
    Stats & st = stats ? *stats : local;
    const auto start = std::chrono::steady_clock::now();
    std::vector<float> text_state;
    float predicted = 0;
    int tokens = 0;
    {
        Timer t{st.text};
        const std::string text = normalize_text(r.text);
        if (text.empty()) throw Error(Fault::InvalidArgument, "the text is empty after normalization; give a text to speak", "text");
        const std::vector<int32_t> ids = tokenizer_.encode(text);
        tokens = (int) ids.size();
        if (tokens > text_.max_tokens()) {
            throw Error(Fault::OutOfRange, "the text is " + std::to_string(tokens) + " tokens long and Irodori-TTS takes at most " +
                                               std::to_string(text_.max_tokens()) + "; split it into sentences",
                        "text");
        }
        Graph g;
        ggml_tensor * state = text_.build(g, ids);
        g.output(state);
        // The runtime runs the duration predictor only when no length is fixed.
        ggml_tensor * sum = nullptr;
        if (!r.length.fixed()) {
            const std::vector<float> summary(voice.speaker.begin(), voice.speaker.begin() + speaker_.dim());
            sum = duration_.build(g, state, g.input(summary, speaker_.dim()));
            g.output(sum);
        }
        g.compute(backend_, allocr_);
        text_state = Graph::read(state);
        if (sum) predicted = Graph::read(sum)[0];
    }
    const Length length = duration_.length(r.length, predicted);
    const int frames = length.frames;
    std::vector<float> x;
    {
        Timer t{st.sampling};
        const Conditions c{text_state, tokens, voice.speaker, voice.speaker_tokens};
        const size_t n = (size_t) frames * codec_.latent_dim();
        if (!r.noise.empty() && r.noise.size() != n) {
            throw std::runtime_error("the given noise has " + std::to_string(r.noise.size() / codec_.latent_dim()) + " frames and the speech " +
                                     std::to_string(frames));
        }
        x = sampler_.sample(c, r.noise.empty() ? gaussian_noise(r.seed, n) : r.noise, frames, r.steps > 0 ? r.steps : sampler_.default_steps(),
                            r.progress);
    }
    if (x.empty()) return 0;
    const int flat = tail_.flattening_point(x, frames, codec_.latent_dim());
    int64_t samples = length.samples;
    if (flat > 0) samples = std::min(samples, (int64_t) flat * codec_.hop());

    size_t emitted = 0;
    {
        Timer t{st.codec};
        codec_.decode(x, samples, first_window, window, [&](const float * s, size_t n) {
            if (emitted == 0) st.first_audio = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            emitted += n;
            return sink(s, n);
        });
    }
    st.tokens = tokens;
    st.frames = frames;
    st.samples = emitted;
    return emitted;
}

}  // namespace irodori
