#include "layout.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <vector>

#include "error.h"
#include "qwen2-tokenizer.h"

namespace nemotron_transcribe {

namespace {

/**
 * How the layout stores a tensor in a file of each weight type: a matrix or an embedding that ggml_mul_mat(),
 * ggml_mul_mat_id() or ggml_get_rows() alone reads, in the file's type; the frontend, the norms, the biases, the Mamba
 * layers' convolutions and decays, the routers, the speaker head and the fusion's kernel in F32.
 */
const Storage kMatrix = quantized_storage({{GGML_TYPE_F32, "0.9.1"},
                                           {GGML_TYPE_F16, "0.9.1"},
                                           {GGML_TYPE_Q8_0, "0.9.1"},
                                           {GGML_TYPE_Q6_K, "0.9.1"},
                                           {GGML_TYPE_Q5_K, "0.9.1"},
                                           {GGML_TYPE_Q4_K, "0.9.1"}});
const Storage kF32 = float32_storage();

const std::string p = "nemotron-transcribe.";

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

/** A positive, finite float key. */
float positive(const ModelFile & m, const std::string & key) {
    const float v = m.f32(key);
    require(v > 0 && std::isfinite(v), m, key + " is not a positive finite number");
    return v;
}

/** The tensors of `layers` of NeMo's Transformer encoder under `prefix`, d_model wide with heads of head_dim and a
 * feed-forward ff wide, and their queries' and keys' norms where `qk_norm`. */
void transformer(std::vector<TensorSpec> & t, const std::string & prefix, int layers, int64_t d, int64_t head_dim, int64_t ff, bool qk_norm) {
    add_block(t, prefix, {{"embed_norm.weight", {d}, kF32}, {"embed_norm.bias", {d}, kF32}});
    add_numbered(t, prefix + "blk.", layers,
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
    if (qk_norm) {
        add_numbered(t, prefix + "blk.", layers,
                     {{"attn_q_norm.weight", {head_dim}, kF32},
                      {"attn_q_norm.bias", {head_dim}, kF32},
                      {"attn_k_norm.weight", {head_dim}, kF32},
                      {"attn_k_norm.bias", {head_dim}, kF32}});
    }
    add_block(t, prefix, {{"final_norm.weight", {d}, kF32}, {"final_norm.bias", {d}, kF32}});
}

/** The tensors of a MoE layer of the language model under `prefix`. */
void moe(std::vector<TensorSpec> & t, const std::string & prefix, int64_t h, int64_t experts, int64_t ff, int64_t shared_ff) {
    add_block(t, prefix,
              {{"ffn_gate_inp.weight", {h, experts}, kF32},
               {"exp_probs_b.bias", {experts}, kF32},
               {"ffn_up_exps.weight", {h, ff, experts}, kMatrix},
               {"ffn_down_exps.weight", {ff, h, experts}, kMatrix},
               {"ffn_up_shexp.weight", {h, shared_ff}, kMatrix},
               {"ffn_down_shexp.weight", {shared_ff, h}, kMatrix}});
}

/** The tensors of an attention layer of the language model under `prefix`. */
void attention(std::vector<TensorSpec> & t, const std::string & prefix, int64_t h, int64_t q, int64_t kv) {
    add_block(t, prefix,
              {{"attn_q.weight", {h, q}, kMatrix},
               {"attn_k.weight", {h, kv}, kMatrix},
               {"attn_v.weight", {h, kv}, kMatrix},
               {"attn_output.weight", {q, h}, kMatrix}});
}

/** Reads every key of the layout, checks the ones that must agree, and names the tensors they call for. */
std::vector<TensorSpec> tensors(const ModelFile & m) {
    check_model_keys(m, "recognition", "checked");
    const int n_fft = m.size(p + "frontend.n_fft"), mels = m.size(p + "frontend.n_mels");
    m.size(p + "frontend.hop_length");
    for (const char * key : {"preemphasis", "log_guard", "std_guard"}) m.f32(p + "frontend." + key);
    // The frontend's real FFT packs the frame's values in pairs.
    require(n_fft % 2 == 0, m, p + "frontend.n_fft is odd");
    m.size(p + "audio.max_samples");

    // The ASR encoder: half of each head rotated at most, in pairs.
    const std::string e = p + "asr.";
    const int d = m.size(e + "d_model"), heads = m.size(e + "num_heads"), factor = m.size(e + "subsampling_factor");
    require(d % heads == 0, m, e + "d_model is not a multiple of " + e + "num_heads");
    const int head_dim = d / heads, rotary = m.size(e + "rotary_dim");
    require(rotary % 2 == 0 && rotary <= head_dim, m, e + "rotary_dim is odd or wider than a head");
    positive(m, e + "rope_base");
    positive(m, e + "norm_eps");
    const int layers = m.count(e + "num_layers");
    m.size(e + "window_frames");
    m.u32(e + "window_left");
    m.u32(e + "window_right");

    // The speaker branch: every channel of a head rotated.
    const std::string s = p + "speaker.";
    const int sd = m.size(s + "d_model"), sheads = m.size(s + "num_heads");
    require(sd % sheads == 0 && (sd / sheads) % 2 == 0, m, s + "d_model is not a multiple of " + s + "num_heads with an even quotient");
    positive(m, s + "rope_base");
    const int slayers = m.count(s + "num_layers"), speakers = m.size(s + "speakers");
    const uint32_t silence = m.u32(s + "cache.silence_frames");
    const float threshold = m.f32(s + "cache.score_threshold");
    require(threshold > 0 && threshold < 1, m, s + "cache.score_threshold is not a probability between 0 and 1");
    for (const char * key : {"boost_latest", "strong_boost_rate", "weak_boost_rate", "min_pos_scores_rate"}) {
        const float v = m.f32(s + "cache." + key);
        require(v >= 0 && std::isfinite(v), m, s + "cache." + key + " is negative or not finite");
    }
    // The speaker cache holds a silence frame and one more of each speaker (SortformerModules._check_streaming_parameters()).
    require(m.size(s + "streaming.spkcache_len") >= (int) ((1 + silence) * speakers), m, s + "streaming.spkcache_len is shorter than a silence frame and a frame of each speaker");
    m.u32(s + "streaming.fifo_len");
    m.size(s + "streaming.spkcache_update_period");
    const float activity = m.f32(s + "activity_threshold");
    require(activity >= 0 && activity < 1, m, s + "activity_threshold is not a probability below 1");
    require(std::isfinite(m.f32(s + "kernel_scale")), m, s + "kernel_scale is not finite");

    // The language model.
    const std::string l = p + "llm.";
    const int64_t h = m.size(l + "hidden_size"), vocab = m.size(l + "vocab_size");
    const int64_t mamba_heads = m.size(l + "mamba_num_heads"), mamba_head_dim = m.size(l + "mamba_head_dim");
    const int64_t groups = m.size(l + "n_groups"), state = m.size(l + "ssm_state_size"), conv = m.size(l + "conv_kernel");
    const int64_t attn_heads = m.size(l + "num_attention_heads"), kv_heads = m.size(l + "num_key_value_heads"), attn_dim = m.size(l + "head_dim");
    const int64_t experts = m.size(l + "n_routed_experts"), used = m.size(l + "num_experts_per_tok");
    const int64_t expert_ff = m.size(l + "moe_intermediate_size"), shared_ff = m.size(l + "moe_shared_expert_intermediate_size");
    const int64_t expert_groups = m.size(l + "n_group"), topk_groups = m.size(l + "topk_group");
    require(mamba_heads % groups == 0, m, l + "mamba_num_heads is not a multiple of " + l + "n_groups");
    require(attn_heads % kv_heads == 0, m, l + "num_attention_heads is not a multiple of " + l + "num_key_value_heads");
    require(used <= experts && experts % expert_groups == 0 && topk_groups <= expert_groups, m,
            l + "num_experts_per_tok, n_group or topk_group does not fit " + l + "n_routed_experts");
    // Groups of experts, which a router of n_group > 1 takes the best of, are a routing this release does not compute.
    require(expert_groups == 1 && topk_groups == 1, m, l + "n_group or topk_group is not 1, a routing this release does not compute");
    positive(m, l + "norm_eps");
    positive(m, l + "layer_norm_epsilon");
    m.boolean(l + "norm_topk_prob");
    positive(m, l + "routed_scaling_factor");
    m.size(l + "context_length");
    const std::vector<std::string> types = m.str_array(l + "layer_types");
    const std::vector<std::string> mtp_types = m.str_array(l + "mtp.layer_types");
    for (const std::vector<std::string> * list : {&types, &mtp_types}) {
        for (const std::string & type : *list) {
            require(type == "mamba" || type == "attention" || type == "moe", m, l + "layer_types names a layer of type " + type + ", which this release does not read");
        }
    }
    require(!types.empty() && !mtp_types.empty(), m, l + "layer_types or " + l + "mtp.layer_types is empty");
    m.size(l + "mtp.depth");

    // The tokenizer, a byte-level BPE with Nemotron's pre-tokenizer, and the prompt around the user's text and the audio.
    const std::string k = p + "tokenizer.";
    // The tokens of the language model's vocabulary, and after them the audio locator's, which no embedding has.
    const int64_t token_count = (int64_t) m.str_array(k + "tokens").size();
    require(token_count > vocab, m, k + "tokens does not hold a token of each id of " + l + "vocab_size and the audio locator's");
    m.str_array(k + "merges");
    for (const char * key : {"added_ids", "special_ids"}) {
        for (int32_t id : m.i32_array(k + key)) require(id >= 0 && id < token_count, m, k + key + " holds an id out of " + k + "tokens");
    }
    Qwen2Pattern pattern;
    require(qwen2_pattern(m.str(k + "pre_tokenizer"), &pattern) && pattern == Qwen2Pattern::Nemotron, m,
            k + "pre_tokenizer is a pattern this release does not split by");
    require(m.boolean(k + "ignore_merges"), m, k + "ignore_merges is false, which this release does not tokenize by");
    const std::string q = p + "prompt.";
    for (const char * key : {"before_text", "before_audio", "after_audio", "instruction", "context_before", "context_after"}) m.str(q + key);
    const uint32_t audio_id = m.u32(q + "audio_id");
    require(audio_id >= (uint32_t) vocab && audio_id < (uint32_t) token_count, m, q + "audio_id is not a token after the language model's vocabulary");
    const std::string g = p + "generation.";
    for (int32_t id : m.i32_array(g + "eos_ids")) require(id >= 0 && id < vocab, m, g + "eos_ids holds an id out of the vocabulary");
    for (const char * key : {"char_run", "block_run", "block_max_chars"}) m.size(g + key);

    // The CTC timestamp head: a Transformer encoder of the ASR encoder's width after the upsampling, and an output of a
    // class for each piece of its vocabulary and the blank.
    const std::string c = p + "ctc.";
    const int ctc_layers = m.count(c + "num_layers"), ctc_heads = m.size(c + "num_heads");
    require(d % ctc_heads == 0, m, e + "d_model is not a multiple of " + c + "num_heads");
    require(m.size(c + "rotary_dim") % 2 == 0 && m.size(c + "rotary_dim") <= d / ctc_heads, m, c + "rotary_dim is odd or wider than a head");
    positive(m, c + "rope_base");
    const int upsampling = m.size(c + "upsampling_factor");
    require(std::isfinite(m.f32(c + "residual_scale")), m, c + "residual_scale is not finite");
    const int64_t classes = (int64_t) m.str_array(c + "vocabulary").size() + 1;

    // The widths of the feed-forward layers and of the frontend's window have no key: each is the width of one tensor,
    // which the others are checked against.
    const int64_t window = m.width("frontend.window", 0);
    const int64_t ff = m.width("asr.blk.0.ffn_up.weight", 1), sff = m.width("spk.blk.0.ffn_up.weight", 1);
    const int64_t hidden = m.width("spk.encoder_proj.weight", 1), ctc_ff = m.width("ctc.blk.0.ffn_up.weight", 1);
    require(window <= n_fft, m, "frontend.window is longer than " + p + "frontend.n_fft");
    const int64_t mamba_inner = mamba_heads * mamba_head_dim, conv_channels = mamba_inner + 2 * groups * state;
    std::vector<TensorSpec> t = {
        {"frontend.window", {window}, kF32},
        {"frontend.filterbank", {n_fft / 2 + 1, mels}, kF32},
        {"asr.pre_encode.weight", {(int64_t) mels * factor, d}, kMatrix},
        {"spk.pre_encode.weight", {(int64_t) mels * factor, sd}, kMatrix},
    };
    transformer(t, "asr.", layers, d, head_dim, ff, true);
    transformer(t, "spk.", slayers, sd, sd / sheads, sff, false);
    add_block(t, "spk.",
              {{"encoder_proj.weight", {sd, hidden}, kMatrix},
               {"encoder_proj.bias", {hidden}, kF32},
               {"upsample.weight", {3 * hidden, hidden * factor}, kMatrix},
               {"upsample.bias", {hidden * factor}, kF32},
               {"head.hidden.weight", {hidden, hidden}, kF32},
               {"head.hidden.bias", {hidden}, kF32},
               {"head.out.weight", {hidden, speakers}, kF32},
               {"head.out.bias", {speakers}, kF32},
               {"silence.embedding", {sd}, kF32}});
    add_block(t, "",
              {{"fuse.asr_norm.weight", {d}, kF32},
               {"fuse.asr_norm.bias", {d}, kF32},
               {"fuse.speaker_norm.weight", {speakers}, kF32},
               {"fuse.speaker_norm.bias", {speakers}, kF32},
               {"fuse.speaker_kernel", {d, speakers}, kF32},
               {"proj.weight", {d, h}, kMatrix},
               {"proj.bias", {h}, kF32},
               {"token_embd.weight", {h, vocab}, kMatrix},
               {"output_norm.weight", {h}, kF32},
               {"output.weight", {h, vocab}, kMatrix}});
    for (size_t i = 0; i < types.size(); i++) {
        const std::string b = "blk." + std::to_string(i) + ".";
        add_block(t, b, {{"norm.weight", {h}, kF32}});
        if (types[i] == "mamba") {
            add_block(t, b,
                      {{"ssm_in.weight", {h, mamba_inner + conv_channels + mamba_heads}, kMatrix},
                       {"ssm_conv1d.weight", {conv, conv_channels}, kF32},
                       {"ssm_conv1d.bias", {conv_channels}, kF32},
                       {"ssm_a", {1, mamba_heads}, kF32},
                       {"ssm_d", {1, mamba_heads}, kF32},
                       {"ssm_dt.bias", {mamba_heads}, kF32},
                       {"ssm_norm.weight", {mamba_inner}, kF32},
                       {"ssm_out.weight", {mamba_inner, h}, kMatrix}});
        } else if (types[i] == "attention") {
            attention(t, b, h, attn_heads * attn_dim, kv_heads * attn_dim);
        } else {
            moe(t, b, h, experts, expert_ff, shared_ff);
        }
    }
    for (size_t i = 0; i < mtp_types.size(); i++) {
        const std::string b = "mtp.blk." + std::to_string(i) + ".";
        add_block(t, b, {{"norm.weight", {h}, kF32}});
        if (i == 0) {
            add_block(t, b, {{"eh_proj.weight", {2 * h, h}, kMatrix}, {"enorm.weight", {h}, kF32}, {"hnorm.weight", {h}, kF32}});
        }
        if (i + 1 == mtp_types.size()) add_block(t, b, {{"final_norm.weight", {h}, kF32}});
        if (mtp_types[i] == "attention") attention(t, b, h, attn_heads * attn_dim, kv_heads * attn_dim);
        else if (mtp_types[i] == "moe") moe(t, b, h, experts, expert_ff, shared_ff);
        else require(false, m, l + "mtp.layer_types names a Mamba layer, which this release does not read");
    }
    add_block(t, "ctc.",
              {{"upsample.weight", {d, d, upsampling}, kMatrix},
               {"upsample.bias", {d}, kF32},
               {"output.weight", {d, classes}, kMatrix},
               {"output.bias", {classes}, kF32}});
    transformer(t, "ctc.", ctc_layers, d, d / ctc_heads, ctc_ff, true);
    return t;
}

}  // namespace

const Layout layout = {"nemotron-transcribe", 1,
                       "convert it again with reference/nemotron-transcribe/convert.py, or download it again from its Hugging Face repository", tensors};

}  // namespace nemotron_transcribe
