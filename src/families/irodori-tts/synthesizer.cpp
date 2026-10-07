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

void TailCut::check(const TailCut & asked) const {
    if (!asked.keep) return;
    const char * idle = asked.window != window                 ? "tail_window_size"
                        : asked.std_threshold != std_threshold   ? "tail_std_threshold"
                        : asked.mean_threshold != mean_threshold ? "tail_mean_threshold"
                                                                 : nullptr;
    if (idle) throw Error(Fault::InvalidArgument, std::string("keep_tail leaves the tail uncut, so ") + idle + " has no effect; leave out one of them", idle);
}

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
    const ModelFile file(path, backend_, voice_layout(*model_));
    ggml_tensor * t = file.tensor("latent");
    std::vector<float> latent(ggml_nelements(t));
    ggml_backend_tensor_get(t, latent.data(), 0, ggml_nbytes(t));
    return voice_from_latent(std::move(latent));
}

size_t Synthesizer::synthesize(const Request & r, const Voice & voice, const AudioSink & sink, Stats * stats) {
    if (r.guidance && dit_.meanflow()) throw std::logic_error("a MeanFlow model takes no guidance, and the options of its file offer none");
    const Guidance & guidance = r.guidance ? *r.guidance : sampler_.guidance();
    const TailCut & tail = r.tail ? *r.tail : tail_;
    const int steps = r.steps > 0 ? r.steps : sampler_.default_steps();
    duration_.check(r.length);
    sampler_.check(guidance, steps);
    tail_.check(tail);
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
        Conditions c{text_state, tokens, voice.speaker, voice.speaker_tokens, {}};
        const size_t n = (size_t) frames * codec_.latent_dim(), m = guidance.speaker_noise ? voice.speaker.size() : 0;
        std::vector<float> noise = r.noise, speaker_draw = r.speaker_noise;
        if (noise.empty()) {
            // The runtime draws the speaker's noise after the latent's from the same generator.
            noise = gaussian_noise(r.seed, n + m);
            speaker_draw.assign(noise.begin() + (std::ptrdiff_t) n, noise.end());
            noise.resize(n);
        }
        if (noise.size() != n || speaker_draw.size() != m) {
            throw std::runtime_error("the given noise has " + std::to_string(noise.size() / codec_.latent_dim()) + " frames and the speech " +
                                     std::to_string(frames) + ", or its speaker noise " + std::to_string(speaker_draw.size()) + " values where " +
                                     std::to_string(m) + " are wanted");
        }
        if (guidance.speaker_noise) c.speaker_noise = speaker_noise(speaker_draw, voice.speaker);
        x = sampler_.sample(c, std::move(noise), frames, steps, guidance, r.progress);
    }
    if (x.empty()) return 0;
    int64_t samples = length.samples;
    if (!tail.keep) {
        const int flat = tail.flattening_point(x, frames, codec_.latent_dim());
        if (flat > 0) samples = std::min(samples, (int64_t) flat * codec_.hop());
    }

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
