#include "layout.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "codec.h"

namespace {

/** The layers of the talker's and the code predictor's decoder stacks. */
const std::initializer_list<const char *> kDecoderLayer = {"attn_norm", "ffn_norm", "attn_q", "attn_k", "attn_v", "attn_o",
                                                           "attn_q_norm", "attn_k_norm", "ffn_gate", "ffn_up", "ffn_down"};

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw std::runtime_error(m.path() + ": " + what + "; " + m.remedy());
}

void read_stack(const ModelFile & m, const std::string & prefix) {
    for (const char * key : {"hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                             "head_dim", "vocab_size"}) {
        m.u32(prefix + key);
    }
    m.f32(prefix + "rms_norm_eps");
    m.f32(prefix + "rope_theta");
}

void read_sampling(const ModelFile & m, const std::string & prefix) {
    m.boolean(prefix + "do_sample");
    m.f32(prefix + "temperature");
    m.u32(prefix + "top_k");
    m.f32(prefix + "top_p");
    m.f32(prefix + "repetition_penalty");
}

/** Reads every key of the layout, checks the ones that must agree, and names the tensors they call for. */
std::vector<std::string> tensors(const ModelFile & m) {
    check_model_keys(m, "synthesis", "steers");
    const std::vector<std::string> languages = m.str_array("speech.languages");
    const std::vector<std::string> voices = m.str_array("speech.voices");
    const size_t n = voices.size();
    require(n > 0 && std::is_sorted(voices.begin(), voices.end()), m, "speech.voices is empty or not sorted");
    const std::vector<std::string> voice_languages = m.str_array("speech.voice_languages");
    const std::vector<std::string> genders = m.str_array("speech.voice_genders");
    const std::vector<std::string> descriptions = m.str_array("speech.voice_descriptions");
    require(voice_languages.size() == n && genders.size() == n && descriptions.size() == n, m,
            "speech.voice_languages, speech.voice_genders and speech.voice_descriptions do not each have one entry per voice");
    for (size_t i = 0; i < n; i++) {
        require(std::count(languages.begin(), languages.end(), voice_languages[i]) == 1, m,
                "the voice " + voices[i] + " speaks " + voice_languages[i] + ", which is not one of speech.languages");
        require(genders[i] == "female" || genders[i] == "male", m, "the gender of the voice " + voices[i] + " is neither female nor male");
    }

    const std::string p = "qwen3-tts.";
    read_stack(m, p + "talker.");
    read_stack(m, p + "code_predictor.");
    for (const char * key : {"num_code_groups", "max_position_embeddings", "codec_bos_id", "codec_eos_token_id", "codec_pad_id",
                             "codec_think_id", "codec_nothink_id", "codec_think_bos_id", "codec_think_eos_id", "suppressed_tokens"}) {
        m.u32(p + "talker." + key);
    }
    for (const char * key : {"tts_bos_token_id", "tts_eos_token_id", "tts_pad_token_id", "im_start_token_id", "im_end_token_id",
                             "assistant_token_id", "newline_token_id"}) {
        m.u32(p + "text." + key);
    }
    require(m.i32_array(p + "language_ids").size() == languages.size(), m, "qwen3-tts.language_ids does not have one id per language");
    require(m.i32_array(p + "speaker_ids").size() == n && m.i32_array(p + "dialect_ids").size() == n, m,
            "qwen3-tts.speaker_ids and qwen3-tts.dialect_ids do not each have one id per voice");
    const std::string dialect_language = m.str(p + "dialect_language");
    require(std::count(languages.begin(), languages.end(), dialect_language) == 1, m,
            "qwen3-tts.dialect_language is " + dialect_language + ", which is not one of speech.languages");
    const uint32_t vocab = m.u32(p + "talker.vocab_size");
    require(m.u32(p + "talker.suppressed_tokens") <= vocab, m, "qwen3-tts.talker.suppressed_tokens exceeds the talker's vocabulary");
    const uint32_t min_frames = m.u32(p + "generation.min_frames"), max_frames = m.u32(p + "generation.max_frames");
    require(min_frames <= max_frames && max_frames < m.u32(p + "talker.max_position_embeddings"), m,
            "qwen3-tts.generation.max_frames is below min_frames or leaves the talker no position for a prompt");
    read_sampling(m, p + "generation.talker.");
    read_sampling(m, p + "generation.code_predictor.");
    m.str_array(p + "tokenizer.tokens");
    m.str_array(p + "tokenizer.merges");

    for (const char * key : {"latent_dim", "codebook_dim", "hidden_size", "head_dim", "sliding_window"}) m.u32(p + "codec." + key);
    m.f32(p + "codec.rms_norm_eps");
    m.f32(p + "codec.rope_theta");
    const uint32_t groups = m.u32(p + "talker.num_code_groups"), quantizers = m.u32(p + "codec.num_quantizers");
    require(quantizers == groups, m, "qwen3-tts.codec.num_quantizers differs from qwen3-tts.talker.num_code_groups");
    require(m.u32(p + "codec.num_key_value_heads") == m.u32(p + "codec.num_attention_heads"), m,
            "the codec's transformer has fewer key/value heads than query heads, which this reader does not run");
    const uint32_t upsampling = (uint32_t) m.i32_array(p + "codec.upsampling_ratios").size();
    const uint32_t blocks = (uint32_t) m.i32_array(p + "codec.upsample_rates").size();

    std::vector<std::string> names = {"talker.text_embd",     "talker.text_proj.fc1.weight", "talker.text_proj.fc1.bias",
                                      "talker.text_proj.fc2.weight", "talker.text_proj.fc2.bias", "talker.codec_embd",
                                      "talker.codec_head",    "talker.norm",                 "cp.norm"};
    add_numbered(names, "talker.blk.", m.u32(p + "talker.num_hidden_layers"), kDecoderLayer);
    add_numbered(names, "cp.blk.", m.u32(p + "code_predictor.num_hidden_layers"), kDecoderLayer);
    add_numbered(names, "cp.codec_embd.", groups - 1, {""});
    add_numbered(names, "cp.head.", groups - 1, {""});
    // The official model makes small_to_mtp_projection a Linear exactly when the two widths differ (the 1.7B model).
    if (m.u32(p + "code_predictor.hidden_size") != m.u32(p + "talker.hidden_size")) {
        names.insert(names.end(), {"cp.in_proj.weight", "cp.in_proj.bias"});
    }

    names.insert(names.end(), {"codec.vq.first.codebook.0", "codec.vq.first.out_proj", "codec.vq.rest.out_proj", "codec.pre_conv.weight",
                               "codec.pre_conv.bias", "codec.tf.in_proj.weight", "codec.tf.in_proj.bias", "codec.tf.out_proj.weight",
                               "codec.tf.out_proj.bias", "codec.tf.norm", "codec.dec.in_conv.weight", "codec.dec.in_conv.bias",
                               "codec.dec.out_snake.alpha", "codec.dec.out_snake.inv_beta", "codec.dec.out_conv.weight",
                               "codec.dec.out_conv.bias"});
    add_numbered(names, "codec.vq.rest.codebook.", quantizers - 1, {""});
    add_numbered(names, "codec.tf.blk.", m.u32(p + "codec.num_hidden_layers"),
                 {"attn_norm", "ffn_norm", "attn_q", "attn_k", "attn_v", "attn_o", "attn_scale", "ffn_gate", "ffn_up", "ffn_down",
                  "ffn_scale"});
    add_numbered(names, "codec.up.", upsampling,
                 {"tconv.weight", "tconv.bias", "dwconv.weight", "dwconv.bias", "norm.weight", "norm.bias", "pw1.weight", "pw1.bias",
                  "pw2.weight", "pw2.bias", "gamma"});
    add_numbered(names, "codec.dec.blk.", blocks, {"snake.alpha", "snake.inv_beta", "tconv.weight", "tconv.bias"});
    for (uint32_t b = 0; b < blocks; b++) {
        add_numbered(names, "codec.dec.blk." + std::to_string(b) + ".res.", kCodecResidualUnits,
                     {"snake1.alpha", "snake1.inv_beta", "conv1.weight", "conv1.bias", "snake2.alpha", "snake2.inv_beta", "conv2.weight",
                      "conv2.bias"});
    }
    return names;
}

}  // namespace

const Layout qwen3_tts_layout = {"qwen3-tts", 1,
                                 "convert it again with reference/qwen3-tts/convert.py, or download it again from its Hugging Face "
                                 "repository",
                                 tensors};
