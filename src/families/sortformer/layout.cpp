#include "layout.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "error.h"

namespace sortformer {

namespace {

/**
 * How the layout stores a tensor in a file of each weight type: a matrix of the encoder, encoder_proj or the upsampling,
 * which ggml_mul_mat() alone reads, in the file's type; the speaker head, whose 37K parameters give the probabilities,
 * and the frontend, the norms, the biases and the silence embedding in F32.
 */
const Storage kMatrix = quantized_storage({{GGML_TYPE_F32, "0.9.0"},
                                           {GGML_TYPE_F16, "0.9.0"},
                                           {GGML_TYPE_Q8_0, "0.9.0"},
                                           {GGML_TYPE_Q6_K, "0.9.0"},
                                           {GGML_TYPE_Q5_K, "0.9.0"},
                                           {GGML_TYPE_Q4_K, "0.9.0"}});
const Storage kF32 = float32_storage();

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

/** Reads every key of the layout, checks the ones that must agree, and names the tensors they call for. */
std::vector<TensorSpec> tensors(const ModelFile & m) {
    check_model_keys(m, "diarization", nullptr);
    const std::string p = "sortformer.";
    const int n_fft = m.size(p + "frontend.n_fft"), mels = m.size(p + "frontend.n_mels");
    m.size(p + "frontend.hop_length");
    for (const char * key : {"preemphasis", "log_guard"}) m.f32(p + "frontend." + key);
    // The frontend's real FFT packs the frame's values in pairs.
    require(n_fft % 2 == 0, m, "sortformer.frontend.n_fft is odd");
    const int d = m.size(p + "encoder.d_model"), heads = m.size(p + "encoder.num_heads"), factor = m.size(p + "encoder.subsampling_factor");
    // RoPE rotates each head's channels in two halves.
    require(d % heads == 0 && (d / heads) % 2 == 0, m, "sortformer.encoder.d_model is not a multiple of sortformer.encoder.num_heads with an even quotient");
    const float eps = m.f32(p + "encoder.norm_eps"), base = m.f32(p + "encoder.rope_base");
    require(eps > 0 && std::isfinite(eps) && base > 1 && std::isfinite(base), m, "sortformer.encoder.norm_eps or sortformer.encoder.rope_base is out of range");
    const int layers = m.count(p + "encoder.num_layers");
    const int speakers = m.size(p + "speakers");

    const uint32_t silence = m.u32(p + "cache.silence_frames");
    const float threshold = m.f32(p + "cache.score_threshold");
    require(threshold > 0 && threshold < 1, m, "sortformer.cache.score_threshold is not a probability between 0 and 1");
    for (const char * key : {"boost_latest", "strong_boost_rate", "weak_boost_rate", "min_pos_scores_rate"}) {
        const float v = m.f32(p + "cache." + key);
        require(v >= 0 && std::isfinite(v), m, p + "cache." + key + " is negative or not finite");
    }

    // The streaming configurations: one value of each setting for each name, the cache long enough for a silence frame
    // and one more of each speaker (SortformerModules._check_streaming_parameters()), chunks of a frame at least, and
    // the offline configuration among them, which a whole recording takes by default.
    const std::vector<std::string> modes = m.str_array(p + "streaming.modes");
    require(!modes.empty() && std::find(modes.begin(), modes.end(), "offline") != modes.end(), m, "sortformer.streaming.modes names no offline configuration");
    const std::string stream_default = m.str(p + "streaming.stream_default");
    require(std::find(modes.begin(), modes.end(), stream_default) != modes.end(), m, "sortformer.streaming.stream_default is none of sortformer.streaming.modes");
    for (const char * key : {"spkcache_len", "fifo_len", "chunk_len", "chunk_right_context", "spkcache_update_period"}) {
        const std::vector<int32_t> values = m.i32_array(p + "streaming." + key);
        require(values.size() == modes.size(), m, p + "streaming." + key + " does not give each of sortformer.streaming.modes one value");
        const std::string k = key;
        const int32_t least = k == "spkcache_len" ? (int32_t) ((1 + silence) * speakers) : k == "chunk_len" || k == "spkcache_update_period" ? 1 : 0;
        require(std::all_of(values.begin(), values.end(), [&](int32_t v) { return v >= least; }), m,
                p + "streaming." + key + " holds a value below " + std::to_string(least));
    }
    for (const char * key : {"onset", "offset"}) {
        const double v = m.f64(p + "postprocessing." + key);
        require(v >= 0 && v <= 1, m, p + "postprocessing." + key + " is not a probability");
    }
    for (const char * key : {"pad_onset", "pad_offset", "min_duration_on", "min_duration_off"}) {
        const double v = m.f64(p + "postprocessing." + key);
        require(v >= 0 && std::isfinite(v), m, p + "postprocessing." + key + " is negative or not finite");
    }

    // The widths of the feed-forward layers, the head and the window have no key: each is the width of one tensor,
    // which the others are checked against.
    const int64_t window = m.width("frontend.window", 0), ff = m.width("blk.0.ffn_up.weight", 1), hidden = m.width("encoder_proj.weight", 1);
    require(window <= n_fft, m, "frontend.window is longer than sortformer.frontend.n_fft");
    std::vector<TensorSpec> t = {
        {"frontend.window", {window}, kF32},
        {"frontend.filterbank", {n_fft / 2 + 1, mels}, kF32},
        {"pre_encode.weight", {(int64_t) mels * factor, d}, kMatrix},
        {"embed_norm.weight", {d}, kF32},
        {"embed_norm.bias", {d}, kF32},
    };
    add_numbered(t, "blk.", layers,
                 {{"attn_norm.weight", {d}, kF32},
                  {"attn_norm.bias", {d}, kF32},
                  {"attn_q.weight", {d, d}, kMatrix},
                  {"attn_k.weight", {d, d}, kMatrix},
                  {"attn_v.weight", {d, d}, kMatrix},
                  {"attn_out.weight", {d, d}, kMatrix},
                  {"attn_out.bias", {d}, kF32},
                  {"ffn_norm.weight", {d}, kF32},
                  {"ffn_norm.bias", {d}, kF32},
                  {"ffn_up.weight", {d, ff}, kMatrix},
                  {"ffn_up.bias", {ff}, kF32},
                  {"ffn_down.weight", {ff, d}, kMatrix},
                  {"ffn_down.bias", {d}, kF32}});
    add_block(t, "",
              {{"final_norm.weight", {d}, kF32},
               {"final_norm.bias", {d}, kF32},
               {"encoder_proj.weight", {d, hidden}, kMatrix},
               {"encoder_proj.bias", {hidden}, kF32},
               {"upsample.weight", {3 * hidden, hidden * factor}, kMatrix},
               {"upsample.bias", {hidden * factor}, kF32},
               {"head.hidden.weight", {hidden, hidden}, kF32},
               {"head.hidden.bias", {hidden}, kF32},
               {"head.out.weight", {hidden, speakers}, kF32},
               {"head.out.bias", {speakers}, kF32},
               {"silence.embedding", {d}, kF32}});
    return t;
}

}  // namespace

const Layout layout = {"sortformer", 1,
                       "convert it again with reference/sortformer/convert.py, or download it again from its Hugging Face repository", tensors};

}  // namespace sortformer
