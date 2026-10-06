#include "duration.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "error.h"
#include "layers.h"

namespace irodori {

DurationPredictor::DurationPredictor(const ModelFile & m) : m_(m) {
    layers_ = (int) m.u32("irodori-tts.duration.num_layers");
    eps_ = m.f32("irodori-tts.norm_eps");
    sample_rate_ = (int) m.u32("speech.sample_rate");
    hop_ = (int) m.u32("irodori-tts.codec.hop_length");
    min_seconds_ = m.f32("irodori-tts.length.min_seconds");
    max_seconds_ = m.f32("irodori-tts.length.max_seconds");
    min_speed_ = m.f32("irodori-tts.length.min_speed");
    max_speed_ = m.f32("irodori-tts.length.max_speed");
    min_frames_ = std::max(1, (int) std::ceil(min_seconds_ * sample_rate_ / (double) hop_));
    max_frames_ = std::max(1, (int) std::floor(max_seconds_ * sample_rate_ / (double) hop_));
}

ggml_tensor * DurationPredictor::build(Graph & g, ggml_tensor * text_state, ggml_tensor * speaker_summary) const {
    ggml_context * ctx = g.ctx();
    const Layers l{ctx, m_, eps_};
    ggml_tensor * speaker = ggml_silu(ctx, speaker_summary);
    ggml_tensor * caption = ggml_silu(ctx, m_.tensor("duration.null_caption"));
    ggml_tensor * h = l.linear(text_state, "duration.in_proj");
    const int64_t dim = h->ne[0];
    for (int i = 0; i < layers_; i++) {
        const std::string b = "duration.blk." + std::to_string(i) + ".";
        ggml_tensor * mod = ggml_add(ctx, l.linear(speaker, b + "mod"), l.linear(caption, b + "caption_mod"));
        auto part = [&](int k) { return ggml_view_1d(ctx, mod, dim, k * dim * ggml_element_size(mod)); };
        ggml_tensor * x = l.modulate(l.rms(h, b + "norm"), part(0), part(1));
        h = ggml_add(ctx, h, ggml_mul(ctx, l.swiglu(x, b), ggml_tanh(ctx, part(2))));
    }
    ggml_tensor * per_token = ggml_softplus(ctx, l.linear(l.rms(h, "duration.out_norm"), "duration.out_proj"));
    return ggml_sum(ctx, per_token);
}

namespace {

std::string number(double v) {
    char s[32];
    std::snprintf(s, sizeof s, "%g", v);
    return s;
}

}  // namespace

void DurationPredictor::check(const LengthOptions & o) const {
    if (!(o.speed >= min_speed_ && o.speed <= max_speed_)) {
        throw Error(Fault::OutOfRange, "the speed is " + number(o.speed) + "; Irodori-TTS takes a speed from " + number(min_speed_) + " to " +
                                           number(max_speed_) + ", 1 being its own rate",
                    "speed");
    }
    if (!(o.duration_scale > 0 && std::isfinite(o.duration_scale))) {
        throw Error(Fault::OutOfRange, "the duration scale is " + number(o.duration_scale) + "; give a factor above 0, 1 for the predicted length",
                    "duration_scale");
    }
    if (o.seconds == 0) return;
    if (!(o.seconds >= min_seconds_ && o.seconds <= max_seconds_)) {
        throw Error(Fault::OutOfRange, "the length is " + number(o.seconds) + " s; Irodori-TTS speaks from " + number(min_seconds_) + " to " +
                                           number(max_seconds_) + " s",
                    "seconds");
    }
    if (o.duration_scale != 1) {
        throw Error(Fault::InvalidArgument,
                    "a request fixes the length in seconds or scales the predicted one, not both; leave out seconds or the duration scale",
                    "seconds");
    }
    const double seconds = o.seconds / o.speed;
    if (!(seconds >= min_seconds_ && seconds <= max_seconds_)) {
        throw Error(Fault::OutOfRange, "a length of " + number(seconds) + " s (seconds divided by speed) is outside the " + number(min_seconds_) +
                                           " to " + number(max_seconds_) + " s that Irodori-TTS speaks; ask for a length within them",
                    "seconds");
    }
}

Length DurationPredictor::length(const LengthOptions & o, float predicted_sum) const {
    check(o);
    // Irodori-TTS-Server divides both the fixed seconds and the duration scale by OpenAI's speed
    // (Aratako/Irodori-TTS-Server@61012c760f22f7b4a6c21c5c5f8f9e148120b6f9, src/irodori_openai_tts/app.py).
    if (o.fixed()) {
        const int64_t samples = std::max<int64_t>(1, (int64_t) (o.seconds / o.speed * sample_rate_));
        return {(int) ((samples + hop_ - 1) / hop_), samples};
    }
    const float predicted = std::expm1(std::log1p(std::max(predicted_sum, 0.0f)));
    if (o.duration_scale == 1 && o.speed == 1) {
        // The bounds are whole frames, so bounding before the rounding gives the frames of the runtime's bounding after it.
        const int frames = (int) std::nearbyint(std::max((double) min_frames_, std::min((double) max_frames_, (double) predicted)));
        return {frames, (int64_t) frames * hop_};
    }
    const double frames = std::nearbyint((double) predicted * (o.duration_scale / o.speed));
    if (!(frames >= min_frames_ && frames <= max_frames_)) {
        const double seconds_per_frame = (double) hop_ / sample_rate_;
        throw Error(Fault::OutOfRange, "the predicted length of " + number(predicted * seconds_per_frame) + " s, times the duration scale and over " +
                                           "the speed, is " + number(frames * seconds_per_frame) + " s, outside the " + number(min_seconds_) +
                                           " to " + number(max_seconds_) + " s that Irodori-TTS speaks; ask for a length within them",
                    o.duration_scale != 1 ? "duration_scale" : "speed");
    }
    return {(int) frames, (int64_t) frames * hop_};
}

}  // namespace irodori
