#include "layout.h"

#include <algorithm>
#include <climits>
#include <set>
#include <string>
#include <vector>

#include "error.h"
#include "qwen2-tokenizer.h"

namespace haru {

namespace {

/**
 * How the layout stores a tensor in a file of each weight type. A matrix or the embedding of the language model, which
 * ggml_mul_mat() and ggml_get_rows() alone read, takes the file's type. The MeanFlow head, the audio projections and the
 * codec take F16, or F32 in an F32 file: the official 8-bit model quantizes the language model alone. The norms, the
 * biases, the codec's scales and depthwise kernels, the rows of the two silent tokens, whose logits the official code
 * computes in float32, and the weights a quantized type would bend where the model is sensitive take F32: the Gated
 * DeltaNet's convolution and the projections that give its decay and its write strength.
 */
const Storage kMatrix = quantized_storage({{GGML_TYPE_F32, "0.11.0"},
                                           {GGML_TYPE_F16, "0.11.0"},
                                           {GGML_TYPE_Q8_0, "0.11.0"},
                                           {GGML_TYPE_Q6_K, "0.11.0"},
                                           {GGML_TYPE_Q5_K, "0.11.0"},
                                           {GGML_TYPE_Q4_K, "0.11.0"}});
const Storage kHalf = half_storage();
const Storage kF32 = float32_storage();

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

/** An [i32] key of one positive value per stage of the codec, `stages` of them. */
std::vector<int32_t> per_stage(const ModelFile & m, const std::string & key, size_t stages) {
    const std::vector<int32_t> v = m.i32_array(key);
    require(v.size() == stages && std::all_of(v.begin(), v.end(), [&](int32_t r) { return r >= 1 && r <= m.tensor_count(); }), m,
            key + " does not hold one positive value for each of the codec's " + std::to_string(stages) + " stages");
    return v;
}

/** A block of the codec of `ch` channels: a depthwise convolution and an MLP, each after an RMS norm and scaled. */
void add_codec_blocks(std::vector<TensorSpec> & t, const std::string & prefix, int count, int64_t ch, int64_t width, int64_t wide) {
    add_numbered(t, prefix, count,
                 {{"norm", {ch}, kF32},           {"conv.weight", {ch, width}, kF32}, {"conv.bias", {ch}, kF32},
                  {"gamma", {ch}, kF32},          {"ffn_norm", {ch}, kF32},           {"fc1.weight", {ch, wide * ch}, kHalf},
                  {"fc1.bias", {wide * ch}, kF32}, {"fc2.weight", {wide * ch, ch}, kHalf}, {"fc2.bias", {ch}, kF32},
                  {"ffn_gamma", {ch}, kF32}});
}

/** The codec's tensors, from the keys of haru.codec. */
void add_codec(std::vector<TensorSpec> & t, const ModelFile & m, int64_t latent, int64_t frame_samples) {
    const std::string c = "haru.codec.";
    require(m.size(c + "latent_dim") == latent, m, "haru.codec.latent_dim is not haru.latent_dim");
    const int64_t base = m.size(c + "base_channels"), width = m.size(c + "kernel"), wide = m.size(c + "ffn_ratio");
    const std::vector<int32_t> ratios = m.i32_array(c + "ratios");
    require(!ratios.empty() && ratios.size() < 16 && std::all_of(ratios.begin(), ratios.end(), [](int32_t r) { return r >= 1; }), m,
            "haru.codec.ratios is empty, holds a ratio below 1 or more stages than a channel count can double through");
    int64_t product = 1;
    for (int32_t r : ratios) product = std::min<int64_t>(product * r, (int64_t) INT_MAX + 1);
    require(product == frame_samples, m, "haru.codec.ratios do not multiply to haru.frame_samples");
    const size_t stages = ratios.size() + 1;
    const std::vector<int32_t> enc = per_stage(m, c + "encoder_depths", stages), dec = per_stage(m, c + "decoder_depths", stages);
    m.f32(c + "norm_eps");
    m.f32(c + "noise_std");
    const int n = (int) ratios.size();

    add_block(t, "codec.enc.stem.", {{"weight", {1, base, width}, kHalf}, {"bias", {base}, kF32}});
    for (int i = 0; i <= n; i++) {
        const int64_t ch = base << i;
        if (i > 0) {
            // The encoder downsamples by the ratios in reverse order.
            const int64_t in = base << (i - 1), s = ratios[(size_t) (n - i)];
            add_block(t, "codec.enc.down." + std::to_string(i - 1) + ".",
                      {{"first", {s * in, ch}, kHalf}, {"second", {s * in, ch}, kHalf}, {"bias", {ch}, kF32}});
        }
        add_codec_blocks(t, "codec.enc.blk." + std::to_string(i) + ".", enc[(size_t) i], ch, width, wide);
    }
    add_block(t, "codec.enc.head.", {{"weight", {base << n, latent, width}, kHalf}, {"bias", {latent}, kF32}});
    add_block(t, "codec.dec.stem.", {{"weight", {latent, base << n, width}, kHalf}, {"bias", {base << n}, kF32}});
    for (int i = 0; i <= n; i++) {
        const int64_t ch = base << (n - i);
        if (i > 0) {
            const int64_t in = base << (n - i + 1), s = ratios[(size_t) (i - 1)];
            add_block(t, "codec.dec.up." + std::to_string(i - 1) + ".", {{"weight", {in, ch, 2 * s}, kHalf}, {"bias", {ch}, kF32}});
        }
        add_codec_blocks(t, "codec.dec.blk." + std::to_string(i) + ".", dec[(size_t) i], ch, width, wide);
    }
    add_block(t, "codec.dec.head.", {{"weight", {base, 1, width}, kHalf}, {"bias", {1}, kF32}});
}

/** The card's keys: its fields, its languages, which are the model's, and each language's wording. */
void check_card(const ModelFile & m) {
    const std::string k = "haru.card.";
    const std::vector<std::string> fields = m.str_array(k + "fields");
    require(!fields.empty() && std::set<std::string>(fields.begin(), fields.end()).size() == fields.size(), m,
            "haru.card.fields is empty or names a field twice");
    std::vector<std::string> languages = m.str_array(k + "languages");
    std::vector<std::string> sorted = languages;
    std::sort(sorted.begin(), sorted.end());
    require(sorted == m.str_array("general.languages"), m, "haru.card.languages are not the model's general.languages");
    const std::vector<std::string> genders = m.str_array(k + "gender_values"), backchannels = m.str_array(k + "backchannel_values");
    require(!genders.empty() && !backchannels.empty(), m, "haru.card.gender_values or backchannel_values is empty");
    for (const std::string & l : languages) {
        m.str(k + l + ".lead");
        require(m.str_array(k + l + ".headers").size() == fields.size(), m, "haru.card." + l + ".headers does not hold a header for each field");
        require(m.str_array(k + l + ".gender").size() == genders.size(), m, "haru.card." + l + ".gender does not label each gender value");
        require(m.str_array(k + l + ".backchannel").size() == backchannels.size(), m,
                "haru.card." + l + ".backchannel does not label each backchannel value");
    }
}

/** Reads every key of the layout, checks the ones that must agree, and names the tensors they call for. */
std::vector<TensorSpec> tensors(const ModelFile & m) {
    check_model_keys(m, "dialogue", "steers");
    const std::string p = "haru.", x = p + "text.", h = p + "head.", s = p + "sampling.", d = p + "dialogue.";

    const int64_t frame_samples = m.size(p + "frame_samples"), latent = m.size(p + "latent_dim");
    m.f32(p + "latent_scale");
    m.f32(p + "latent_bias");

    const int64_t hidden = m.size(x + "hidden_size"), vocab = m.size(x + "vocab_size"), ffn = m.size(x + "intermediate_size");
    const int layers = m.count(x + "num_hidden_layers");
    const std::vector<std::string> types = m.str_array(x + "layer_types");
    require((int) types.size() == layers, m, "haru.text.layer_types does not name a kind for each of the language model's layers");
    const int64_t heads = m.size(x + "num_attention_heads"), kv = m.size(x + "num_key_value_heads"), hd = m.size(x + "head_dim");
    require(heads % kv == 0, m, "the language model's attention heads do not share its key/value heads evenly");
    const int rotary = m.size(x + "rotary_dim");
    require(rotary <= hd && rotary % 2 == 0, m, "haru.text.rotary_dim is odd or wider than a head");
    const int64_t nk = m.size(x + "linear_num_key_heads"), nv = m.size(x + "linear_num_value_heads");
    const int64_t dk = m.size(x + "linear_key_head_dim"), dv = m.size(x + "linear_value_head_dim");
    require(nv % nk == 0, m, "the Gated DeltaNet's value heads do not share its key heads evenly");
    require(dk == dv, m, "the Gated DeltaNet's key and value heads differ in width, which ggml_gated_delta_net() does not take");
    const int64_t conv_width = m.size(x + "linear_conv_kernel_dim"), conv = 2 * nk * dk + nv * dv;
    m.size(x + "max_position_embeddings");
    m.f32(x + "rms_norm_eps");
    m.f32(x + "rope_theta");

    const uint32_t pad = m.u32(p + "pad_token_id"), eop = m.u32(p + "eop_token_id"), eos = m.u32(p + "eos_token_id");
    require(pad < (uint64_t) vocab && eop < (uint64_t) vocab && eos < (uint64_t) vocab && pad != eop, m,
            "haru.pad_token_id, eop_token_id or eos_token_id is past the vocabulary, or PAD is EOP");
    m.f32(p + "audio_in.norm_eps");

    const int head_layers = m.count(h + "num_layers");
    const int64_t head_ffn = m.size(h + "ffn_size"), frequencies = m.size(h + "frequency_dim");
    require(frequencies % 2 == 0, m, "haru.head.frequency_dim is odd, where its features are a cosine and a sine of each frequency");
    m.f32(h + "norm_eps");
    m.f32(h + "interval_embed");
    m.f32(h + "frequency_base");
    const float g = m.f32(h + "guidance_scale"), g_min = m.f32(h + "guidance_min"), g_max = m.f32(h + "guidance_max");
    require(g_min > 0.0f && g_min <= g && g <= g_max, m, "haru.head.guidance_scale is not within guidance_min and guidance_max above 0");

    m.f32(s + "temperature");
    m.u32(s + "top_k");
    const float top_p = m.f32(s + "top_p");
    require(top_p > 0.0f && top_p <= 1.0f, m, "haru.sampling.top_p is not above 0 and at most 1");
    m.f32(s + "eop_bias");
    m.f64(s + "dry_multiplier");
    m.f64(s + "dry_base");
    m.u32(s + "dry_allowed_length");
    m.u32(s + "dry_lookback");
    m.size(s + "dry_max_match_length");
    m.f64(s + "dry_max_penalty");

    m.size(d + "max_frames");
    const float ref_min = m.f32(d + "reference_min_seconds"), ref_max = m.f32(d + "reference_max_seconds");
    require(ref_min > 0.0f && ref_min <= ref_max, m, "haru.dialogue.reference_min_seconds is not above 0 and at most reference_max_seconds");
    m.size(d + "card_max_chars");
    m.f32(d + "silence_dbfs");
    m.f32(d + "silence_margin_seconds");
    m.size(d + "silence_window_divisor");
    m.size(d + "silence_frames");
    m.str(d + "system_open");
    m.str(d + "system_close");
    check_card(m);

    const std::vector<std::string> tokens = m.str_array(p + "tokenizer.tokens");
    require(tokens.size() <= (size_t) vocab, m, "haru.tokenizer.tokens has more tokens than the language model's vocabulary");
    m.str_array(p + "tokenizer.merges");
    const std::vector<int32_t> added = m.i32_array(p + "tokenizer.added_ids");
    require(std::all_of(added.begin(), added.end(), [&](int32_t id) { return id >= 0 && (size_t) id < tokens.size() && !tokens[(size_t) id].empty(); }),
            m, "haru.tokenizer.added_ids holds an id that is not one of the tokens");
    Qwen2Pattern pattern;
    require(qwen2_pattern(m.str(p + "tokenizer.pre_tokenizer"), &pattern), m, "haru.tokenizer.pre_tokenizer is a pattern this release does not split by");

    std::vector<TensorSpec> t = {{"token_embd", {hidden, vocab}, kMatrix}, {"output_silent", {hidden, 2}, kF32}, {"output_norm", {hidden}, kF32}};
    for (int i = 0; i < layers; i++) {
        const std::string o = "blk." + std::to_string(i) + ".";
        add_block(t, o, {{"attn_norm", {hidden}, kF32}, {"post_attn_norm", {hidden}, kF32}});
        if (types[(size_t) i] == "full_attention") {
            add_block(t, o, {{"attn_q", {hidden, heads * hd}, kMatrix},      {"attn_gate", {hidden, heads * hd}, kMatrix},
                             {"attn_k", {hidden, kv * hd}, kMatrix},         {"attn_v", {hidden, kv * hd}, kMatrix},
                             {"attn_output", {heads * hd, hidden}, kMatrix}, {"attn_q_norm", {hd}, kF32},
                             {"attn_k_norm", {hd}, kF32}});
        } else {
            require(types[(size_t) i] == "linear_attention", m,
                    "haru.text.layer_types names " + types[(size_t) i] + ", which is neither full_attention nor linear_attention");
            add_block(t, o, {{"ssm_in_qkv", {hidden, conv}, kMatrix},      {"ssm_in_z", {hidden, nv * dv}, kMatrix},
                             {"ssm_in_b", {hidden, nv}, kF32},             {"ssm_in_a", {hidden, nv}, kF32},
                             {"ssm_conv1d", {conv_width, conv}, kF32},     {"ssm_dt_bias", {nv}, kF32},
                             {"ssm_a_log", {nv}, kF32},                    {"ssm_norm", {dv}, kF32},
                             {"ssm_out", {nv * dv, hidden}, kMatrix}});
        }
        add_block(t, o, {{"ffn_gate", {hidden, ffn}, kMatrix}, {"ffn_up", {hidden, ffn}, kMatrix}, {"ffn_down", {ffn, hidden}, kMatrix}});
    }

    for (const char * side : {"model", "user"}) {
        add_block(t, std::string("audio_in.") + side + ".",
                  {{"fc1.weight", {latent, hidden}, kHalf}, {"fc1.bias", {hidden}, kF32}, {"norm", {hidden}, kF32},
                   {"fc2.weight", {hidden, hidden}, kHalf}, {"fc2.bias", {hidden}, kF32}});
    }

    add_block(t, "head.", {{"cond_proj", {hidden, hidden}, kHalf}, {"latent_proj", {latent, hidden}, kHalf}});
    for (const char * embed : {"t_embed", "dt_embed", "h_embed", "omega_embed"}) {
        add_block(t, std::string("head.") + embed + ".", {{"fc1", {frequencies, hidden}, kHalf}, {"fc2", {hidden, hidden}, kHalf}});
    }
    add_numbered(t, "head.blk.", head_layers,
                 {{"norm", {hidden}, kF32},                {"ada", {hidden, 3 * hidden}, kHalf},  {"ffn_gate", {hidden, head_ffn}, kHalf},
                  {"ffn_up", {hidden, head_ffn}, kHalf},   {"ffn_down", {head_ffn, hidden}, kHalf}});
    add_block(t, "head.final.", {{"ada", {hidden, 2 * hidden}, kHalf}, {"proj", {hidden, latent}, kHalf}});

    add_codec(t, m, latent, frame_samples);
    return t;
}

}  // namespace

const Layout model_layout = {"haru", 1, "convert it again with reference/haru/convert.py, or download it again from its Hugging Face repository",
                             tensors};

}  // namespace haru
