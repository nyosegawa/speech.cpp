#include "perception.h"

#include <algorithm>

#include "graph.h"

namespace nemotron_transcribe {

namespace {

const std::string kKeys = "nemotron-transcribe.";

ggml_tensor * layer_norm(ggml_context * ctx, const ModelFile & m, ggml_tensor * x, const std::string & name, float eps) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), m.tensor(name + ".weight")), m.tensor(name + ".bias"));
}

}  // namespace

Perception::Perception(const ModelFile & m, ggml_backend_t backend)
    : m_(m),
      backend_(backend),
      frontend_(m, kKeys, true),
      asr_(m, backend),
      speakers_(m, backend),
      window_(m.size(kKeys + "asr.window_frames")),
      left_((int) m.u32(kKeys + "asr.window_left")),
      right_((int) m.u32(kKeys + "asr.window_right")),
      hidden_((int) m.tensor("proj.weight")->ne[1]),
      threshold_(m.f32(kKeys + "speaker.activity_threshold")),
      scale_(m.f32(kKeys + "speaker.kernel_scale")),
      eps_(m.f32(kKeys + "asr.norm_eps")),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {}

Perception::~Perception() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

int64_t Perception::frames(size_t samples) const {
    const int64_t factor = asr_.factor();
    return (frontend_.frames(samples) + factor - 1) / factor;
}

std::vector<float> Perception::encode(const std::vector<float> & samples, const std::function<bool(double)> & progress, PerceptionStages * stages) {
    const std::vector<double> mel = frontend_.log_mels(samples);
    const std::vector<float> normalized = frontend_.normalized(mel);
    const std::vector<float> raw(mel.begin(), mel.end());
    const int mels = frontend_.mels(), factor = asr_.factor(), d = asr_.d_model(), speakers = speakers_.speakers();
    const int64_t valid = (int64_t) mel.size() / mels, frames = (valid + factor - 1) / factor;

    // The windows of the official encoder, over the audio's frames and the one more the preprocessor gives it: cores of
    // chunk frames from the start, and as much context on either side as the frames reach. A window whose core holds
    // only that extra frame gives no frame of the audio and is left out.
    const int64_t chunk = (int64_t) window_ * factor, left = (int64_t) left_ * factor, right = (int64_t) right_ * factor, total = valid + 1;
    const int64_t windows = std::max<int64_t>(1, (total + chunk - 1) / chunk);
    std::vector<float> rows;
    rows.reserve((size_t) (frames * hidden_));
    speakers_.start();
    for (int64_t k = 0; k < windows; k++) {
        const int64_t start = k * chunk, end = std::min(start + chunk, total);
        if (start >= valid) break;
        const int64_t cs = std::max<int64_t>(start - left, 0), ce = std::min(std::min(end + right, total), valid);
        // The core's frames in the encoder's output: the boundaries of start and end rounded up, as the Transformer
        // encoder's output grid gives them, cut to the frames the window's features give.
        const int64_t drop = (start - cs) / factor, core = (std::min(end, valid) + factor - 1) / factor - start / factor;
        const std::vector<float> out = asr_.encode(&normalized[(size_t) (cs * mels)], ce - cs);
        const int64_t kept = std::min<int64_t>(core, (int64_t) out.size() / d - drop);
        const std::vector<float> asr(out.begin() + drop * d, out.begin() + (drop + kept) * d);
        std::vector<float> p = speakers_.step(&raw[(size_t) (cs * mels)], ce - cs, start - cs, ce - std::min(end, valid), start / factor,
                                              k + 1 == windows || (k + 1) * chunk >= valid);
        // _align_diar_frames(): the window's probabilities cut to its kept frames, or the last repeated up to them.
        p.resize((size_t) (std::min<int64_t>((int64_t) p.size() / speakers, kept) * speakers));
        while ((int64_t) p.size() < kept * speakers) p.insert(p.end(), p.end() - speakers, p.end());
        const std::vector<float> fused = fuse(asr, p, kept, stages);
        rows.insert(rows.end(), fused.begin(), fused.end());
        if (stages) {
            stages->asr.insert(stages->asr.end(), asr.begin(), asr.end());
            stages->speakers.insert(stages->speakers.end(), p.begin(), p.end());
        }
        if (progress && !progress((double) (k + 1) / (double) windows)) return {};
    }
    return rows;
}

std::vector<float> Perception::fuse(const std::vector<float> & asr, const std::vector<float> & probs, int64_t frames, PerceptionStages * stages) {
    const int speakers = speakers_.speakers();
    std::vector<float> active(probs.size());
    for (size_t i = 0; i < probs.size(); i++) active[i] = probs[i] > threshold_ ? 1.0f : 0.0f;
    Graph g;
    ggml_context * ctx = g.ctx();
    ggml_tensor * states = layer_norm(ctx, m_, g.input(asr, asr_.d_model(), frames), "fuse.asr_norm", eps_);
    ggml_tensor * targets = layer_norm(ctx, m_, g.input(active, speakers, frames), "fuse.speaker_norm", eps_);
    ggml_tensor * kernel = ggml_cont(ctx, ggml_transpose(ctx, m_.tensor("fuse.speaker_kernel")));
    ggml_tensor * fused = ggml_add(ctx, states, ggml_scale(ctx, mul_mat(ctx, kernel, targets), scale_));
    ggml_tensor * out = ggml_add(ctx, mul_mat(ctx, m_.tensor("proj.weight"), fused), m_.tensor("proj.bias"));
    g.output(out);
    if (stages) g.output(fused);
    g.compute(backend_, allocr_);
    if (stages) {
        const std::vector<float> f = Graph::read(fused);
        stages->fused.insert(stages->fused.end(), f.begin(), f.end());
    }
    return Graph::read(out);
}

}  // namespace nemotron_transcribe
