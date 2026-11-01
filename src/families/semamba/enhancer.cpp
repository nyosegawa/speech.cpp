#include "enhancer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

#include "error.h"
#include "layout.h"

namespace semamba {

namespace {

constexpr double kPi = 3.14159265358979323846;

}  // namespace

Decoded decode(const std::vector<float> & magnitude, const std::vector<float> & real, const std::vector<float> & imag, int64_t out_bins,
               int64_t frames, int64_t bins, float zero_portion) {
    Decoded d;
    d.magnitude.resize((size_t) (frames * bins));
    d.phase.resize((size_t) (frames * bins));
    d.zeroed.assign((size_t) frames, 0);
    for (int64_t t = 0; t < frames; t++) {
        int64_t zeros = 0;
        for (int64_t k = 0; k < bins; k++) {
            const size_t from = (size_t) (t * out_bins + k), to = (size_t) (t * bins + k);
            d.magnitude[to] = magnitude[from];
            d.phase[to] = std::atan2(imag[from], real[from]);
            // expm1(relu(m)) is zero exactly where m is not above zero.
            if (!(magnitude[from] > 0.0f)) zeros++;
        }
        if ((float) zeros / (float) bins > zero_portion) {
            d.zeroed[(size_t) t] = 1;
            std::fill(d.magnitude.begin() + t * bins, d.magnitude.begin() + (t + 1) * bins, 0.0f);
        }
    }
    return d;
}

Enhancer::Enhancer(const std::string & path, ggml_backend_t backend)
    : backend_(backend),
      model_(path, backend, layout),
      network_(model_),
      reference_rate_(model_.size("semamba.stft.sampling_rate")),
      n_fft_(model_.size("semamba.stft.n_fft")),
      hop_(model_.size("semamba.stft.hop_size")),
      lowest_(model_.size("semamba.sample_rate_min")),
      highest_(model_.size("semamba.sample_rate_max")),
      window_floor_(model_.f32("semamba.window_floor")),
      zero_portion_(model_.f32("semamba.zero_portion")),
      pad_value_(model_.f32("semamba.pad_value")),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {
    if (!allocr_) throw Error(Fault::OutOfMemory, "cannot create a graph allocator");
    // A chunk of a few frames has every operation of the network, which nothing here moves to the CPU.
    Graph g;
    const Stft s = stft(lowest_);
    const int64_t frames = 6, bins = s.bins();
    const Network::Outputs probe =
        network_.forward(g, std::vector<float>((size_t) (frames * bins), 0.0f), std::vector<float>((size_t) (frames * bins), 0.0f), frames, bins);
    for (ggml_tensor * t : {probe.magnitude, probe.real, probe.imag}) g.output(t);
    for (int i = 0; i < ggml_graph_n_nodes(g.cgraph()); i++) {
        ggml_tensor * node = ggml_graph_node(g.cgraph(), i);
        if (!ggml_backend_supports_op(backend, node)) {
            ggml_gallocr_free(allocr_);
            throw Error(Fault::Device, std::string(ggml_backend_name(backend)) + " does not compute " + ggml_op_desc(node) + " as " +
                                           model_.str("general.name") + " needs it; load it on the CPU (device \"cpu\")",
                        "device");
        }
    }
}

Enhancer::~Enhancer() {
    ggml_gallocr_free(allocr_);
}

Stft Enhancer::stft(int rate) const {
    return Stft(scaled(n_fft_, rate, reference_rate_), scaled(hop_, rate, reference_rate_));
}

size_t Enhancer::fewest_samples(int rate) const {
    return (size_t) (scaled(n_fft_, rate, reference_rate_) / 2 + 1);
}

Enhancer::Raw Enhancer::network(const std::vector<float> & magnitude, const std::vector<float> & phase, int64_t frames, int64_t bins) {
    Graph g;
    const Network::Outputs out = network_.forward(g, magnitude, phase, frames, bins);
    g.output(out.magnitude);
    g.output(out.real);
    g.output(out.imag);
    g.compute(backend_, allocr_);
    return {Graph::read(out.magnitude), Graph::read(out.real), Graph::read(out.imag), out.magnitude->ne[0], out.magnitude->ne[1]};
}

std::vector<float> Enhancer::chunk(const float * samples, size_t n, int rate) {
    const Stft s = stft(rate);
    std::vector<float> magnitude, phase;
    s.forward(samples, n, magnitude, phase);
    const int64_t frames = s.frames(n), bins = s.bins();
    const Raw raw = network(magnitude, phase, frames, bins);
    const Decoded d = decode(raw.magnitude, raw.real, raw.imag, raw.bins, frames, bins, zero_portion_);
    std::vector<float> out = s.inverse(d.magnitude, d.phase, (int) frames);
    // pad_or_trim_to_match() pads a short inverse with pad_value and cuts a long one.
    out.resize(n, pad_value_);
    return out;
}

bool Enhancer::enhance(const std::vector<float> & samples, int rate, double chunk_size_in_seconds, double hop_length_portion,
                       const std::function<bool(const float * samples, size_t n)> & emit, const std::function<bool(double done)> & progress) {
    Stream stream(*this, rate, chunk_size_in_seconds, hop_length_portion, emit, progress, (int64_t) samples.size());
    return stream.push(samples.data(), samples.size()) && stream.end();
}

Stream::Stream(Enhancer & enhancer, int rate, double chunk_size_in_seconds, double hop_length_portion,
               std::function<bool(const float * samples, size_t n)> emit, std::function<bool(double done)> progress, int64_t total)
    : Stream([&enhancer, rate](const float * samples, size_t n) { return enhancer.chunk(samples, n, rate); }, enhancer.window_floor_, rate,
             chunk_size_in_seconds, hop_length_portion, std::move(emit), std::move(progress), total) {}

Stream::Stream(Chunk chunk, float window_floor, int rate, double chunk_size_in_seconds, double hop_length_portion,
               std::function<bool(const float * samples, size_t n)> emit, std::function<bool(double done)> progress, int64_t total)
    : chunk_(std::move(chunk)),
      size_((int64_t) (chunk_size_in_seconds * rate)),
      hop_((int64_t) (hop_length_portion * (double) size_)),
      window_((size_t) size_),
      emit_(std::move(emit)),
      progress_(std::move(progress)) {
    if (size_ <= 0 || hop_ <= 0) throw std::logic_error("a stream's chunks or hop have no samples");
    // inference_chunk.py's count of chunks, max(1, ceil((N - size) / hop) + 1), where the length is known.
    if (total > 0) total_chunks_ = std::max<int64_t>(1, (int64_t) std::ceil((double) (total - size_) / (double) hop_) + 1);
    // torch.hann_window(size), periodic, in float, kept above window_floor.
    for (int64_t i = 0; i < size_; i++) {
        window_[(size_t) i] = std::max((float) (0.5 - 0.5 * std::cos(2.0 * kPi * (double) i / (double) size_)), window_floor);
    }
}

bool Stream::compute(int64_t start, int64_t length) {
    const std::vector<float> enhanced = chunk_(audio_.data() + (start - first_), (size_t) length);
    const int64_t end = start + length;
    if ((int64_t) sum_.size() < end - passed_) {
        sum_.resize((size_t) (end - passed_), 0.0f);
        weight_.resize((size_t) (end - passed_), 0.0f);
    }
    for (int64_t j = 0; j < length; j++) {
        sum_[(size_t) (start + j - passed_)] += enhanced[(size_t) j] * window_[(size_t) j];
        weight_[(size_t) (start + j - passed_)] += window_[(size_t) j];
    }
    computed_++;
    return !progress_ || total_chunks_ == 0 || progress_((double) computed_ / (double) total_chunks_);
}

bool Stream::pass(int64_t settled) {
    if (settled <= passed_) return true;
    const size_t n = (size_t) (settled - passed_);
    out_.resize(n);
    for (size_t j = 0; j < n; j++) out_[j] = sum_[j] / weight_[j];
    sum_.erase(sum_.begin(), sum_.begin() + (ptrdiff_t) n);
    weight_.erase(weight_.begin(), weight_.begin() + (ptrdiff_t) n);
    passed_ = settled;
    return emit_(out_.data(), n);
}

bool Stream::push(const float * samples, size_t n) {
    if (stopped_) return false;
    audio_.insert(audio_.end(), samples, samples + n);
    received_ += (int64_t) n;
    // A chunk that starts a whole number of hops in sits where it sits whether or not it turns out to be the last, so it
    // is computed once its samples are there. The last chunk starts after the start of the one before it, so every
    // sample up to that start is settled.
    while (received_ >= computed_ * hop_ + size_) {
        const int64_t start = computed_ * hop_;
        if (!compute(start, size_) || !pass(start)) return !(stopped_ = true);
        const int64_t keep = start - first_;
        audio_.erase(audio_.begin(), audio_.begin() + (ptrdiff_t) keep);
        first_ = start;
    }
    return true;
}

bool Stream::end() {
    if (stopped_) return false;
    const int64_t count = std::max<int64_t>(1, (int64_t) std::ceil((double) (received_ - size_) / (double) hop_) + 1);
    // Every chunk but the last has its samples and was computed; the last, ending with the audio, starts on a hop only
    // when the audio ends a whole chunk after one, and then it was computed too.
    if (computed_ < count) {
        const int64_t start = std::max<int64_t>(0, received_ - size_);
        if (!compute(start, std::min(size_, received_ - start))) return !(stopped_ = true);
    }
    stopped_ = true;
    return pass(received_);
}

}  // namespace semamba
