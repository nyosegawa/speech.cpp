#include "diarizer.h"

#include <algorithm>
#include <stdexcept>

#include "layout.h"

namespace sortformer {

Diarizer::Diarizer(const std::string & path, ggml_backend_t backend)
    : model_(path, backend, layout),
      frontend_(model_, "sortformer.", false),
      network_(model_, backend),
      compression_(compression_of(model_)),
      modes_(model_.str_array("sortformer.streaming.modes")) {
    ggml_tensor * t = model_.tensor("silence.embedding");
    silence_.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, silence_.data(), 0, ggml_nbytes(t));
}

Streaming Diarizer::mode(const std::string & name) const {
    const auto it = std::find(modes_.begin(), modes_.end(), name);
    if (it == modes_.end()) throw std::logic_error("the model file has no streaming configuration " + name);
    const size_t i = (size_t) (it - modes_.begin());
    const auto at = [&](const char * key) { return (int) model_.i32_array(std::string("sortformer.streaming.") + key)[i]; };
    return {at("spkcache_len"), at("fifo_len"), at("chunk_len"), at("chunk_right_context"), at("spkcache_update_period")};
}

Postprocessing Diarizer::postprocessing() const {
    const auto at = [&](const char * key) { return model_.f64(std::string("sortformer.postprocessing.") + key); };
    return {at("onset"), at("offset"), at("pad_onset"), at("pad_offset"), at("min_duration_on"), at("min_duration_off")};
}

ChunkStream::ChunkStream(Diarizer & diarizer, const Streaming & streaming, std::function<void(int64_t step)> on_step)
    : diarizer_(diarizer), streaming_(streaming), on_step_(std::move(on_step)) {}

void ChunkStream::push(const float * samples, size_t n, std::vector<float> & probs) {
    if (n == 0) return;
    const size_t at = x_.size();
    x_.resize(at + n);
    diarizer_.frontend().preemphasize(samples, n, received_ == 0, last_, x_.data() + at);
    last_ = samples[n - 1];
    received_ += (int64_t) n;
    compute_frames(false);
    run_steps(false, probs);
}

void ChunkStream::end(std::vector<float> & probs) {
    compute_frames(true);
    run_steps(true, probs);
}

void ChunkStream::compute_frames(bool ended) {
    const NemoFrontend & frontend = diarizer_.frontend();
    const int hop = frontend.hop_length(), mels = frontend.mels();
    // Before the audio ends, a frame waits for the last sample of its window.
    const int64_t reach = frontend.window_end(0);
    const int64_t target = ended ? frontend.frames((size_t) received_) : received_ >= reach ? (received_ - reach) / hop + 1 : 0;
    if (target > computed_) {
        std::vector<double> mel((size_t) ((target - computed_) * mels));
        frontend.log_mel(x_.data(), x_begin_, received_, computed_, target - computed_, mel.data());
        features_.insert(features_.end(), mel.begin(), mel.end());
        computed_ = target;
    }
    // The next frame's window starts n_fft / 2 before its centre.
    const int64_t keep = std::max<int64_t>(computed_ * hop - reach, 0);
    if (keep > x_begin_) {
        x_.erase(x_.begin(), x_.begin() + (keep - x_begin_));
        x_begin_ = keep;
    }
}

void ChunkStream::run_steps(bool ended, std::vector<float> & probs) {
    Network & network = diarizer_.network();
    const int factor = network.factor(), mels = network.mels(), speakers = network.speakers();
    const int64_t chunk = (int64_t) streaming_.chunk_len * factor, right = (int64_t) streaming_.chunk_right_context * factor;
    const int64_t total = diarizer_.frontend().frames((size_t) received_);
    for (;;) {
        const int64_t first = step_ * chunk;
        int64_t last = first + chunk, context = right;
        if (ended) {
            if (first >= total) break;
            last = std::min(last, total);
            context = std::min(right, total - last);
        } else if (computed_ < last + right) {
            break;
        }
        // FeatureStacking: factor frames of features to each 80 ms frame, the last padded with zeros.
        const int64_t features = last + context - first, frames = (features + factor - 1) / factor;
        const int64_t chunk_frames = (last - first + factor - 1) / factor;
        std::vector<float> stacked((size_t) (frames * factor * mels), 0.0f);
        const float * from = &features_[(size_t) ((first - features_begin_) * mels)];
        std::copy(from, from + features * mels, stacked.begin());

        std::vector<float> input = state_.cache;
        input.insert(input.end(), state_.fifo.begin(), state_.fifo.end());
        const int64_t rows = state_.cache_rows() + state_.fifo_rows();
        const StepResult result = network.step(input, rows, stacked, frames);

        // The 80 ms probabilities the update reads: each row the average of its factor rows of 10 ms.
        std::vector<float> coarse((size_t) ((rows + frames) * speakers));
        for (int64_t r = 0; r < rows + frames; r++) {
            for (int s = 0; s < speakers; s++) {
                float sum = 0;
                for (int j = 0; j < factor; j++) sum += result.probs[(size_t) (((r * factor) + j) * speakers + s)];
                coarse[(size_t) (r * speakers + s)] = sum / (float) factor;
            }
        }
        update(state_, streaming_, diarizer_.compression(), diarizer_.silence(), result.pre_encoded.data(), chunk_frames,
               step_ * streaming_.chunk_len, coarse.data());
        // The chunk's rows of 10 ms, cut to the audio's frames.
        const float * out = &result.probs[(size_t) (rows * factor * speakers)];
        probs.insert(probs.end(), out, out + (last - first) * speakers);
        if (on_step_) on_step_(step_);
        step_++;

        const int64_t next = step_ * chunk;
        if (next > features_begin_) {
            const int64_t drop = std::min<int64_t>(next, computed_) - features_begin_;
            features_.erase(features_.begin(), features_.begin() + drop * mels);
            features_begin_ += drop;
        }
    }
}

}  // namespace sortformer
