#include "dialogue.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "error.h"

namespace haru {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

/** π, which MSVC's <cmath> does not define as M_PI without _USE_MATH_DEFINES. */
constexpr double kPi = 3.14159265358979323846;

std::unique_ptr<Qwen2Tokenizer> file_tokenizer(const ModelFile & m) {
    Qwen2Pattern pattern;
    if (!qwen2_pattern(m.str("haru.tokenizer.pre_tokenizer"), &pattern)) {
        throw Error(Fault::File, m.path() + ": haru.tokenizer.pre_tokenizer is a pattern this release does not split by; " + m.remedy());
    }
    return std::make_unique<Qwen2Tokenizer>(m, "haru.tokenizer", pattern);
}

}  // namespace

void SeededRandomness::normal(float * out, size_t n) {
    for (size_t i = 0; i < n; i += 2) {
        double u1 = 0.0;
        while (u1 <= 0.0) u1 = 1.0 - uniform(rng_);
        const double u2 = uniform(rng_), radius = std::sqrt(-2.0 * std::log(u1)), angle = 2.0 * kPi * u2;
        out[i] = (float) (radius * std::cos(angle));
        if (i + 1 < n) out[i + 1] = (float) (radius * std::sin(angle));
    }
}

int32_t SeededRandomness::token(const Candidates & c) { return c.tokens[(size_t) pick(c.weights, uniform(rng_))]; }

DialogueSettings file_dialogue_settings(const ModelFile & m) {
    DialogueSettings s;
    s.sampling = file_sampling(m);
    s.guidance = m.f32("haru.head.guidance_scale");
    s.eop_bias = m.f32("haru.sampling.eop_bias");
    s.max_frames = m.size("haru.dialogue.max_frames");
    return s;
}

Dialogue::Dialogue(const ModelFile & m, ggml_backend_t backend)
    : m_(m), backend_(backend), tokenizer_(file_tokenizer(m)), added_(m, "haru.tokenizer"), codec_(m, backend), head_(m, backend), decoder_(m, "haru.text.", backend) {
    ids_ = file_text_ids(m);
    dry_config_ = file_dry(m);
    settings_ = file_dialogue_settings(m);
    latent_ = m.size("haru.latent_dim");
    hidden_ = m.size("haru.text.hidden_size");
    noise_std_ = m.f32("haru.codec.noise_std");
    norm_eps_ = m.f32("haru.audio_in.norm_eps");
}

Dialogue::~Dialogue() = default;

std::vector<float> Dialogue::encode_reference(const std::vector<float> & reference) {
    codec_.reset_encoder();
    std::vector<float> means = codec_.encode(reference);
    codec_.reset_encoder();
    return means;
}

const std::vector<float> & Dialogue::silence() {
    if (silence_.empty()) {
        const int64_t frames = m_.size("haru.dialogue.silence_frames");
        codec_.reset_encoder();
        const std::vector<float> zeros = codec_.encode(std::vector<float>((size_t) (frames * codec_.frame_samples()), 0.0f));
        silence_.assign(zeros.end() - latent_, zeros.end());
    }
    return silence_;
}

std::vector<float> Dialogue::sample(const std::vector<float> & mean, Randomness & r) {
    float scale = 0.0f;
    r.normal(&scale, 1);
    std::vector<float> noise(mean.size());
    r.normal(noise.data(), noise.size());
    std::vector<float> out(mean.size());
    // mean + scale x (latent_std / 0.8) x noise, multiplied left to right in float32 as the official code does.
    for (size_t i = 0; i < mean.size(); i++) out[i] = mean[i] + scale * noise_std_ * noise[i];
    return out;
}

std::vector<float> Dialogue::scaled(const std::vector<float> & latents) const {
    std::vector<float> out(latents.size());
    for (size_t i = 0; i < latents.size(); i++) out[i] = head_.scale(latents[i]);
    return out;
}

ggml_tensor * Dialogue::project(Graph & g, const std::string & side, ggml_tensor * latents) {
    ggml_context * c = g.ctx();
    const std::string p = "audio_in." + side + ".";
    ggml_tensor * x = ggml_add(c, mul_mat(c, m_.tensor(p + "fc1.weight"), latents), m_.tensor(p + "fc1.bias"));
    x = ggml_mul(c, ggml_rms_norm(c, x, norm_eps_), m_.tensor(p + "norm"));
    return ggml_add(c, mul_mat(c, m_.tensor(p + "fc2.weight"), x), m_.tensor(p + "fc2.bias"));
}

ggml_tensor * Dialogue::build_embed(Graph & g, const std::vector<int32_t> & tokens, const std::vector<float> & model, const std::vector<float> & user,
                                    const std::vector<float> * model_mask, const std::vector<float> * user_mask) {
    ggml_context * c = g.ctx();
    const int64_t n = (int64_t) tokens.size();
    ggml_tensor * x = ggml_get_rows(c, m_.tensor("token_embd"), g.input(tokens, n));
    const std::pair<const char *, std::pair<const std::vector<float> *, const std::vector<float> *>> sides[] = {
        {"model", {&model, model_mask}}, {"user", {&user, user_mask}}};
    for (const auto & [side, data] : sides) {
        ggml_tensor * y = project(g, side, g.input(scaled(*data.first), latent_, n));
        if (data.second) y = ggml_mul(c, y, g.input(*data.second, 1, n));
        x = ggml_add(c, x, y);
    }
    return x;
}

DialogueFrame Dialogue::start(const std::vector<int32_t> & card, const std::vector<float> & reference, int64_t quietest, const DialogueSettings & s,
                              Randomness & r, DialogueStages * stages) {
    if (s.max_frames < 1) throw Error(Fault::OutOfRange, "a conversation runs one frame at least", "max_frames");
    settings_ = s;
    head_.set_guidance(s.guidance);
    frames_ = 0;
    times_ = DialogueTimes();
    dry_ = std::make_unique<Dry>(dry_config_, std::vector<int32_t>{ids_.pad, ids_.eop, ids_.eos});
    detokenizer_ = std::make_unique<StreamDetokenizer>(*tokenizer_, std::vector<int32_t>{ids_.pad, ids_.eop});
    const int64_t n_ref = (int64_t) reference.size() / latent_;
    const std::vector<float> latents = n_ref > 0 ? sample(reference, r) : std::vector<float>();
    // Rows before any of the model's audio carry the speaker's room tone and the user's encoded silence.
    const bool grid = s.speak_first || (card.empty() && n_ref == 0);
    std::vector<float> room, quiet;
    if (grid) {
        quiet = silence();
        room = n_ref > 0 && quietest >= 0 ? std::vector<float>(latents.begin() + quietest * latent_, latents.begin() + (quietest + 1) * latent_) : quiet;
    }
    const Prompt p = build_prompt(card, latents, s.speak_first, room, quiet, ids_, latent_);
    codec_.reset_encoder();
    codec_.reset_decoder();
    must_speak_ = s.speak_first;
    decoder_.start(p.rows() + s.max_frames);

    Graph g;
    ggml_tensor * x = build_embed(g, p.tokens, p.model_latents, p.user_latents, &p.model_mask, &p.user_mask);
    g.output(x);
    g.compute(backend_, decoder_.allocator());
    const std::vector<float> embeddings = Graph::read(x);
    const std::vector<float> rows = decoder_.run(embeddings, p.rows());
    if (stages) {
        stages->prompt_tokens = p.tokens;
        stages->prompt_model = p.model_latents;
        stages->prompt_user = p.user_latents;
        stages->prompt_model_mask = p.model_mask;
        stages->prompt_user_mask = p.user_mask;
        stages->embeddings.insert(stages->embeddings.end(), embeddings.begin(), embeddings.end());
    }
    Graph frame;
    return emit(frame, frame.input(std::vector<float>(rows.end() - hidden_, rows.end()), hidden_, 1), r, stages);
}

DialogueFrame Dialogue::step(const std::vector<float> & user, Randomness & r, DialogueStages * stages) {
    if ((int64_t) user.size() != codec_.frame_samples()) {
        throw Error(Fault::InvalidArgument, "a frame of the user's audio is " + std::to_string(codec_.frame_samples()) + " samples", "audio");
    }
    if (done()) throw Error(Fault::OutOfRange, "the conversation has given its " + std::to_string(settings_.max_frames) + " frames", "max_frames");
    const Clock::time_point encoding = Clock::now();
    const std::vector<float> mean = codec_.encode(user);
    times_.encode += seconds_since(encoding);
    const std::vector<float> latent = sample(mean, r);
    if (stages) {
        stages->user_means.insert(stages->user_means.end(), mean.begin(), mean.end());
        stages->user_latents.insert(stages->user_latents.end(), latent.begin(), latent.end());
    }
    const Clock::time_point building = Clock::now();
    Graph g;
    ggml_tensor * x = build_embed(g, {token_}, latent_last_, latent, nullptr, nullptr);
    if (stages) g.output(x);
    ggml_tensor * hidden = decoder_.build_rows(g, x);
    times_.model += seconds_since(building);
    DialogueFrame f = emit(g, hidden, r, stages);
    decoder_.advance(1);
    if (stages) {
        const std::vector<float> e = Graph::read(x);
        stages->embeddings.insert(stages->embeddings.end(), e.begin(), e.end());
    }
    return f;
}

DialogueFrame Dialogue::emit(Graph & g, ggml_tensor * hidden, Randomness & r, DialogueStages * stages) {
    const Clock::time_point computing = Clock::now();
    std::vector<float> noise((size_t) latent_);
    r.normal(noise.data(), noise.size());
    ggml_tensor * logits = build_logits(g, m_, hidden);
    ggml_tensor * silent = build_silent_logits(g, m_, hidden);
    ggml_tensor * latent = head_.build(g, hidden, g.input(noise, latent_, 1));
    for (ggml_tensor * t : {hidden, logits, silent, latent}) g.output(t);
    g.compute(backend_, decoder_.allocator());
    std::vector<float> l = Graph::read(logits);
    const std::vector<float> two = Graph::read(silent), z = Graph::read(latent);
    times_.model += seconds_since(computing);
    if (stages) {
        const std::vector<float> h = Graph::read(hidden);
        stages->hidden.insert(stages->hidden.end(), h.begin(), h.end());
        stages->logits.insert(stages->logits.end(), l.begin(), l.end());
        stages->silent_logits.insert(stages->silent_logits.end(), two.begin(), two.end());
        stages->head_latents.insert(stages->head_latents.end(), z.begin(), z.end());
    }
    const Clock::time_point sampling = Clock::now();
    shape_logits(l, *dry_, two.data(), ids_, settings_.eop_bias, must_speak_);
    must_speak_ = false;
    DialogueFrame f;
    f.token = r.token(candidates(l, settings_.sampling));
    times_.sampling += seconds_since(sampling);
    latent_last_.resize(z.size());
    for (size_t i = 0; i < z.size(); i++) latent_last_[i] = head_.unscale(z[i]);
    const Clock::time_point decoding = Clock::now();
    f.audio = codec_.decode(latent_last_, 1);
    times_.decode += seconds_since(decoding);
    if (stages) {
        stages->latents.insert(stages->latents.end(), latent_last_.begin(), latent_last_.end());
        stages->codec_audio.insert(stages->codec_audio.end(), f.audio.begin(), f.audio.end());
    }
    for (float & a : f.audio) a = std::clamp(a, -1.0f, 1.0f);
    token_ = f.token;
    dry_->update(f.token);
    f.text = detokenizer_->push(f.token);
    frames_++;
    return f;
}

}  // namespace haru
