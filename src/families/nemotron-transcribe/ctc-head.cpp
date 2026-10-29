#include "ctc-head.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "graph.h"

namespace nemotron_transcribe {

namespace {

const std::string kKeys = "nemotron-transcribe.";

/**
 * The CTC frames one graph classifies at a time: the log-softmax holds a value of every class for each, 23,863 of them,
 * and the frames of a window hold several times that.
 */
constexpr int64_t kClassifiedFrames = 256;

NemoTransformer::Shape shape_of(const ModelFile & m) {
    const std::string c = kKeys + "ctc.";
    NemoTransformer::Shape s;
    s.d_model = m.size(kKeys + "asr.d_model");
    s.heads = m.size(c + "num_heads");
    s.layers = m.count(c + "num_layers");
    s.rotary_dim = m.size(c + "rotary_dim");
    s.norm_eps = m.f32(c + "norm_eps");
    s.rope_base = m.f32(c + "rope_base");
    s.qk_norm = true;
    return s;
}

}  // namespace

CtcHead::CtcHead(const ModelFile & m, ggml_backend_t backend)
    : m_(m),
      backend_(backend),
      transformer_(m, "ctc.", shape_of(m)),
      d_(m.size(kKeys + "asr.d_model")),
      classes_((int) m.str_array(kKeys + "ctc.vocabulary").size() + 1),
      upsampling_(m.size(kKeys + "ctc.upsampling_factor")),
      window_(m.size(kKeys + "asr.window_frames")),
      left_((int) m.u32(kKeys + "asr.window_left")),
      right_((int) m.u32(kKeys + "asr.window_right")),
      residual_scale_(m.f32(kKeys + "ctc.residual_scale")),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {}

CtcHead::~CtcHead() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

std::vector<float> CtcHead::log_probs(const std::vector<uint16_t> & states, int64_t frames, const std::vector<int32_t> & classes,
                                      std::vector<CtcHeadStages> * stages, const std::function<bool()> & proceed) {
    const int64_t kept = (int64_t) classes.size(), up = upsampling_;
    std::vector<float> out((size_t) (frames * up * kept));
    if (stages) stages->clear();
    // The windows of _iter_ctc_windows(), as the encoder's windowed inference cuts them.
    for (int64_t start = 0; start < frames; start += window_) {
        if (proceed && !proceed()) return {};
        const int64_t end = std::min<int64_t>(start + window_, frames);
        const int64_t cs = std::max<int64_t>(start - left_, 0), ce = std::min<int64_t>(end + right_, frames), n = ce - cs;
        std::vector<float> x((size_t) (n * d_));
        for (size_t i = 0; i < x.size(); i++) {
            const uint32_t bits = (uint32_t) states[(size_t) (cs * d_) + i] << 16;
            std::memcpy(&x[i], &bits, sizeof bits);
        }
        std::vector<int32_t> positions((size_t) (n * up));
        for (size_t i = 0; i < positions.size(); i++) positions[i] = (int32_t) i;

        Graph g;
        ggml_context * ctx = g.ctx();
        // The transposed convolution of kernel and stride `up`: the weight [d_in, d_out, up] as one matrix of d_out * up
        // outputs, whose output [d_out, up, n] is the upsampled frames in order, frame t's j-th at t * up + j.
        ggml_tensor * w = m_.tensor("ctc.upsample.weight");
        ggml_tensor * u = mul_mat(ctx, ggml_reshape_2d(ctx, w, w->ne[0], w->ne[1] * w->ne[2]), g.input(x, d_, n));
        u = ggml_add(ctx, ggml_reshape_2d(ctx, u, d_, n * up), m_.tensor("ctc.upsample.bias"));
        std::vector<ggml_tensor *> layers;
        ggml_tensor * t = transformer_.build(ctx, u, g.input(positions, n * up), stages ? &layers : nullptr);
        ggml_tensor * trunk = ggml_add(ctx, u, ggml_scale(ctx, t, residual_scale_));
        if (stages) {
            g.output(u);
            for (ggml_tensor * l : layers) g.output(l);
            g.output(t);
            g.output(trunk);
        }

        // Each slice of the window's own frames classified, shifted by its largest logit as log_softmax() shifts it; the
        // kept classes' shifted logits and the sums of the exponentials, of which the log is taken here.
        const int64_t first = (start - cs) * up, count = (end - start) * up;
        struct Slice {
            int64_t first, count;
            ggml_tensor * kept, * sums;
        };
        std::vector<Slice> slices;
        for (int64_t s = 0; s < count; s += kClassifiedFrames) {
            const int64_t m = std::min(kClassifiedFrames, count - s);
            ggml_tensor * frames_in = ggml_view_2d(ctx, trunk, d_, m, trunk->nb[1], (size_t) (first + s) * trunk->nb[1]);
            ggml_tensor * logits = ggml_add(ctx, mul_mat(ctx, m_.tensor("ctc.output.weight"), frames_in), m_.tensor("ctc.output.bias"));
            ggml_tensor * best = ggml_get_rows(ctx, ggml_reshape_3d(ctx, logits, 1, classes_, m), ggml_reshape_2d(ctx, ggml_argmax(ctx, logits), 1, m));
            ggml_tensor * shifted = ggml_sub(ctx, logits, ggml_reshape_2d(ctx, best, 1, m));
            std::vector<int32_t> index((size_t) (kept * m));
            for (int64_t f = 0; f < m; f++) std::copy(classes.begin(), classes.end(), index.begin() + f * kept);
            ggml_tensor * picked = ggml_get_rows(ctx, ggml_reshape_3d(ctx, shifted, 1, classes_, m),
                                                 ggml_reshape_2d(ctx, g.input(index, kept * m), kept, m));
            ggml_tensor * sums = ggml_sum_rows(ctx, ggml_exp(ctx, shifted));
            g.output(picked);
            g.output(sums);
            slices.push_back({s, m, picked, sums});
        }
        g.compute(backend_, allocr_);

        for (const Slice & s : slices) {
            const std::vector<float> picked = Graph::read(s.kept), sums = Graph::read(s.sums);
            for (int64_t f = 0; f < s.count; f++) {
                const float log_sum = std::log(sums[(size_t) f]);
                float * row = &out[(size_t) ((start * up + s.first + f) * kept)];
                for (int64_t k = 0; k < kept; k++) row[k] = picked[(size_t) (f * kept + k)] - log_sum;
            }
        }
        if (stages) {
            CtcHeadStages st;
            st.upsampled = Graph::read(u);
            for (ggml_tensor * l : layers) st.layers.push_back(Graph::read(l));
            st.output = Graph::read(t);
            st.trunk = Graph::read(trunk);
            stages->push_back(std::move(st));
        }
    }
    return out;
}

}  // namespace nemotron_transcribe
