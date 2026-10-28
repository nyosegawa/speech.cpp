#include "speakers.h"

#include <algorithm>
#include <cmath>

namespace nemotron_transcribe {

namespace {

const std::string kKeys = "nemotron-transcribe.speaker.";

sortformer::NetworkShape network_shape(const ModelFile & m) {
    sortformer::NetworkShape s;
    s.d_model = m.size(kKeys + "d_model");
    s.heads = m.size(kKeys + "num_heads");
    s.layers = m.count(kKeys + "num_layers");
    s.factor = m.size("nemotron-transcribe.asr.subsampling_factor");
    s.speakers = m.size(kKeys + "speakers");
    s.mels = m.size("nemotron-transcribe.frontend.n_mels");
    s.norm_eps = m.f32("nemotron-transcribe.asr.norm_eps");
    s.rope_base = m.f32(kKeys + "rope_base");
    return s;
}

}  // namespace

SpeakerBranch::SpeakerBranch(const ModelFile & m, ggml_backend_t backend)
    : network_(m, backend, network_shape(m), "spk."), compression_(sortformer::compression_of(m, kKeys)) {
    // The windows are the chunks: ParallelExpertEncoder._init_streaming_diar() gives the Sortformer the encoder's window
    // as its chunk_len, and its FIFO and speaker cache settings.
    streaming_.chunk_len = m.size("nemotron-transcribe.asr.window_frames");
    streaming_.spkcache_len = m.size(kKeys + "streaming.spkcache_len");
    streaming_.fifo_len = (int) m.u32(kKeys + "streaming.fifo_len");
    streaming_.spkcache_update_period = m.size(kKeys + "streaming.spkcache_update_period");
    ggml_tensor * t = m.tensor("spk.silence.embedding");
    silence_.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, silence_.data(), 0, ggml_nbytes(t));
}

void SpeakerBranch::start() {
    state_ = sortformer::State();
}

std::vector<float> SpeakerBranch::step(const float * features, int64_t frames, int64_t left, int64_t right, int64_t first, bool last) {
    const int factor = network_.factor(), mels = network_.mels(), speakers = network_.speakers();
    // forward_streaming_step(): the left context's 80 ms frames rounded, the right context's rounded up.
    const int64_t lc = (int64_t) std::lround((double) left / factor), rc = (right + factor - 1) / factor;
    const int64_t n = (frames + factor - 1) / factor, chunk = n - lc - rc;
    std::vector<float> stacked((size_t) (n * factor * mels), 0.0f);
    std::copy(features, features + frames * mels, stacked.begin());
    std::vector<float> rows = state_.cache;
    rows.insert(rows.end(), state_.fifo.begin(), state_.fifo.end());
    const int64_t cached = state_.cache_rows() + state_.fifo_rows();
    const sortformer::StepResult result = network_.step(rows, cached, stacked, n);

    // Each 80 ms row the average of its factor rows of 10 ms (SortformerModules.downsample_preds()), for the update
    // and for the window's own frames.
    std::vector<float> coarse((size_t) ((cached + n) * speakers));
    for (int64_t r = 0; r < cached + n; r++) {
        for (int s = 0; s < speakers; s++) {
            float sum = 0;
            for (int j = 0; j < factor; j++) sum += result.probs[(size_t) ((r * factor + j) * speakers + s)];
            coarse[(size_t) (r * speakers + s)] = sum / (float) factor;
        }
    }
    std::vector<float> out(&coarse[(size_t) ((cached + lc) * speakers)], &coarse[(size_t) ((cached + lc + chunk) * speakers)]);
    if (!last) {
        sortformer::update(state_, streaming_, compression_, silence_, &result.pre_encoded[(size_t) (lc * network_.d_model())], chunk, first,
                           coarse.data(), lc);
    }
    return out;
}

}  // namespace nemotron_transcribe
