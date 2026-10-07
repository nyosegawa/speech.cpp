#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

#include "error.h"

namespace irodori {

namespace {

/**
 * torch.linspace() on the CPU in float32: the first half from the start, the second from the end. (Its
 * vectorized path can round the last bit of a few points differently; dit_t in the dumps shows the times
 * the official run used.)
 */
std::vector<float> linspace(float start, float end, int64_t n) {
    std::vector<float> out((size_t) n);
    const float step = (end - start) / (float) (n - 1);
    for (int64_t i = 0; i < n; i++) out[(size_t) i] = i < n / 2 ? start + step * (float) i : end - step * (float) (n - i - 1);
    return out;
}

std::string number(double v) {
    char s[32];
    std::snprintf(s, sizeof s, "%g", v);
    return s;
}

}  // namespace

Sampler::Sampler(const Dit & dit, const ModelFile & m, ggml_backend_t backend) : dit_(dit), backend_(backend) {
    default_steps_ = (int) m.u32("irodori-tts.sampler.default_steps");
    if (!dit.meanflow()) {
        guidance_.text = m.f32("irodori-tts.sampler.cfg_text");
        guidance_.speaker = m.f32("irodori-tts.sampler.cfg_speaker");
        guidance_.min_t = m.f32("irodori-tts.sampler.cfg_min_t");
        guidance_.max_t = m.f32("irodori-tts.sampler.cfg_max_t");
    }
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

Sampler::~Sampler() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

void Sampler::check(const Guidance & g, int steps) const {
    if (g.rescale_k.has_value() != g.rescale_sigma.has_value()) {
        const char * missing = g.rescale_k ? "rescale_sigma" : "rescale_k";
        throw Error(Fault::InvalidArgument, std::string("the rescaling takes rescale_k and rescale_sigma together; give ") + missing + " as well",
                    missing);
    }
    if (g.min_t > g.max_t) {
        throw Error(Fault::InvalidArgument,
                    "cfg_min_t is " + number(g.min_t) + " and cfg_max_t " + number(g.max_t) +
                        ", so the guidance runs at no step; give a cfg_min_t no greater than cfg_max_t",
                    "cfg_min_t");
    }
    const bool text = g.text > 0, speaker = g.speaker > 0;
    if (!text && !speaker) {
        // Without a scale above 0 the runtime runs no guidance and ignores how it would have run.
        const char * idle = g.mode != guidance_.mode       ? "cfg_guidance_mode"
                            : g.min_t != guidance_.min_t   ? "cfg_min_t"
                            : g.max_t != guidance_.max_t   ? "cfg_max_t"
                            : g.speaker_noise              ? "speaker_uncond_mode"
                                                           : nullptr;
        if (idle) {
            throw Error(Fault::InvalidArgument,
                        std::string("with cfg_scale_text and cfg_scale_speaker at 0 no guidance runs, so ") + idle + " has no effect; leave it out",
                        idle);
        }
    }
    if (g.mode == GuidanceMode::Joint && text && speaker && g.text != g.speaker) {
        throw Error(Fault::InvalidArgument,
                    "the joint guidance leaves out the text and the speaker together and takes one scale, and cfg_scale_text is " +
                        number(g.text) + " and cfg_scale_speaker " + number(g.speaker) +
                        "; give them the same value, or another cfg_guidance_mode",
                    "cfg_guidance_mode");
    }
    // Only the joint guidance leaves the speaker out without a speaker scale above 0, since it leaves out everything.
    if (g.speaker_noise && !speaker && g.mode != GuidanceMode::Joint) {
        throw Error(Fault::InvalidArgument,
                    "speaker_uncond_mode noise fills the branch without the speaker, which the guidance runs only with cfg_scale_speaker above "
                    "0 or the joint guidance; leave it out or give a cfg_scale_speaker",
                    "speaker_uncond_mode");
    }
    const std::vector<float> t = schedule(steps, g.sway);
    for (size_t i = 0; i + 1 < t.size(); i++) {
        if (!(t[i] > t[i + 1])) {
            throw Error(Fault::OutOfRange,
                        "a sway_coeff of " + number(g.sway) + " leaves steps " + std::to_string(i) + " and " + std::to_string(i + 1) + " of " +
                            std::to_string(steps) + " at the same time; give a coefficient nearer to 0",
                        "sway_coeff");
        }
    }
}

std::vector<float> Sampler::schedule(int steps, float sway) const {
    if (dit_.meanflow()) return linspace(1.0f, 0.0f, (int64_t) steps + 1);
    // Sway Sampling (F5-TTS) as the runtime bends the linear schedule, in float32: a coefficient of 0 leaves each time
    // as it is. RF then starts just short of pure noise.
    std::vector<float> t = linspace(0.0f, 1.0f, (int64_t) steps + 1);
    // Python's 0.5 * math.pi, which torch rounds to float32 to multiply a float32 tensor.
    const float half_pi = (float) 1.5707963267948966;
    for (float & u : t) {
        const float bend = std::cos(half_pi * u) + u - 1.0f;
        u = std::min(1.0f, std::max(0.0f, u + sway * bend));
        u = (1.0f - u) * 0.999f;
    }
    return t;
}

std::vector<Branch> Sampler::branches(int step, float t, const Guidance & g, std::vector<float> * scales) const {
    std::vector<Branch> out(1);
    if (scales) scales->clear();
    struct Guide {
        bool text;
        float scale;
    };
    std::vector<Guide> enabled;
    if (g.text > 0) enabled.push_back({true, g.text});
    if (g.speaker > 0) enabled.push_back({false, g.speaker});
    if (dit_.meanflow() || enabled.empty() || !((double) t >= g.min_t && (double) t <= g.max_t)) return out;
    const Branch::Speaker left = g.speaker_noise ? Branch::Speaker::Noise : Branch::Speaker::Left;
    auto without = [&](const Guide & e) {
        Branch b;
        if (e.text) b.text = false;
        else b.speaker = left;
        return b;
    };
    // The runtime's order: the text's guidance added first, then the speaker's.
    switch (g.mode) {
        case GuidanceMode::Independent:
            for (const Guide & e : enabled) {
                out.push_back(without(e));
                if (scales) scales->push_back(e.scale);
            }
            break;
        case GuidanceMode::Joint:
            out.push_back({false, left});
            if (scales) scales->push_back(enabled[0].scale);
            break;
        case GuidanceMode::Alternating: {
            // The runtime counts the steps without guidance too.
            const Guide & e = enabled[(size_t) step % enabled.size()];
            out.push_back(without(e));
            if (scales) scales->push_back(e.scale);
            break;
        }
    }
    return out;
}

std::vector<float> Sampler::velocity(const Conditions & c, const std::vector<float> & x, int frames, int step, float t, float t_next,
                                     const Guidance & g, std::vector<float> * out) {
    std::vector<float> scales;
    const std::vector<Branch> b = branches(step, t, g, &scales);
    Graph graph;
    ggml_tensor * result = dit_.build(graph, x, frames, t, t - t_next, c, b);
    graph.output(result);
    graph.compute(backend_, allocr_);
    std::vector<float> v = Graph::read(result);
    if (out) *out = v;
    const size_t n = x.size();
    if (b.size() > 1) {
        std::vector<float> guided(v.begin(), v.begin() + n);
        for (size_t k = 1; k < b.size(); k++) {
            const float scale = scales[k - 1];
            for (size_t i = 0; i < n; i++) guided[i] = guided[i] + scale * (v[i] - v[k * n + i]);
        }
        v = std::move(guided);
    }
    if (g.rescale_k && (double) t < 1.0) {
        // The temporal score rescaling of the runtime (temporal_score_rescale()): the ratio in double precision, as
        // Python computes it, and the velocity in float32 with the ratio and 1 - t rounded to it.
        const double one_minus_t = 1.0 - (double) t, snr = one_minus_t * one_minus_t / ((double) t * (double) t);
        const double sigma_sq = *g.rescale_sigma * *g.rescale_sigma;
        const float ratio = (float) ((snr * sigma_sq + 1.0) / (snr * sigma_sq / *g.rescale_k + 1.0)), omt = (float) one_minus_t;
        for (size_t i = 0; i < n; i++) v[i] = (ratio * (omt * v[i] + x[i]) - x[i]) / omt;
    }
    return v;
}

std::vector<float> Sampler::sample(const Conditions & c, std::vector<float> x, int frames, int steps, const Guidance & g,
                                   const std::function<bool(double done)> & progress) {
    const std::vector<float> times = schedule(steps, g.sway);
    if (g.truncation) {
        for (float & v : x) v = v * *g.truncation;
    }
    for (int i = 0; i < steps; i++) {
        if (progress && !progress((double) i / steps)) return {};
        const std::vector<float> v = velocity(c, x, frames, i, times[i], times[i + 1], g);
        const float dt = times[i + 1] - times[i];
        for (size_t j = 0; j < x.size(); j++) x[j] = x[j] + v[j] * dt;
    }
    if (progress && !progress(1)) return {};
    return x;
}

std::vector<float> gaussian_noise(uint64_t seed, size_t n) {
    std::mt19937_64 rng(seed);
    std::vector<float> out(n);
    constexpr double kTwoPi = 6.283185307179586;
    for (size_t i = 0; i < n; i += 2) {
        // Uniform on (0, 1]: 53 random bits, offset so that the logarithm never sees zero.
        const double u1 = ((rng() >> 11) + 1.0) * 0x1.0p-53, u2 = (rng() >> 11) * 0x1.0p-53;
        const double r = std::sqrt(-2.0 * std::log(u1));
        out[i] = (float) (r * std::cos(kTwoPi * u2));
        if (i + 1 < n) out[i + 1] = (float) (r * std::sin(kTwoPi * u2));
    }
    return out;
}

std::vector<float> speaker_noise(const std::vector<float> & draw, const std::vector<float> & speaker) {
    // torch's std() of a float32 tensor accumulates in double precision.
    double sum = 0;
    for (float v : speaker) sum += v;
    const double mean = sum / (double) speaker.size();
    double squares = 0;
    for (float v : speaker) squares += ((double) v - mean) * ((double) v - mean);
    const float deviation = std::max((float) std::sqrt(squares / (double) (speaker.size() - 1)), 1e-6f);
    std::vector<float> out(draw.size());
    for (size_t i = 0; i < draw.size(); i++) out[i] = draw[i] * deviation;
    return out;
}

}  // namespace irodori
