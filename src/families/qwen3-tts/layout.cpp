#include "layout.h"

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>

#include "codec.h"
#include "error.h"
#include "prompt.h"

namespace {

/**
 * The types reference/qwen3-tts/convert.py stores a tensor in by its --type: a matrix of the talker or the code
 * predictor in Q8_0, F16 or F32, F16 standing for Q8_0 where a matrix's rows are no multiple of 32; a convolution or
 * matrix of the codec in F16 or F32; and the rest in F32.
 */
const std::vector<ggml_type> kMatrix = {GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32};
const std::vector<ggml_type> kCodecWeight = {GGML_TYPE_F16, GGML_TYPE_F32};
const std::vector<ggml_type> kF32 = {GGML_TYPE_F32};

/**
 * What the official codec decoder fixes in its code rather than in its configuration: the width of pre_conv, the
 * width of every other convolution but a residual unit's second, which is 1, and the fourfold widening of a ConvNeXt
 * block.
 */
constexpr int64_t kPreConvWidth = 3;
constexpr int64_t kConvWidth = 7;
constexpr int64_t kConvNextWidening = 4;

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

/** The talker's or the code predictor's decoder stack. */
struct Stack {
    int hidden, ffn, layers, heads, kv_heads, head_dim, vocab;
};

Stack read_stack(const ModelFile & m, const std::string & prefix) {
    const Stack s = {m.size(prefix + "hidden_size"),         m.size(prefix + "intermediate_size"), m.count(prefix + "num_hidden_layers"),
                     m.size(prefix + "num_attention_heads"), m.size(prefix + "num_key_value_heads"), m.size(prefix + "head_dim"),
                     m.size(prefix + "vocab_size")};
    m.f32(prefix + "rms_norm_eps");
    m.f32(prefix + "rope_theta");
    require(s.heads % s.kv_heads == 0, m, prefix + "num_attention_heads is not a multiple of " + prefix + "num_key_value_heads");
    require(s.head_dim % 2 == 0, m, prefix + "head_dim is odd, where RoPE turns pairs of channels");
    return s;
}

void add_stack(std::vector<TensorSpec> & t, const std::string & prefix, const Stack & s) {
    const int64_t h = s.hidden, q = (int64_t) s.heads * s.head_dim, kv = (int64_t) s.kv_heads * s.head_dim;
    add_numbered(t, prefix, s.layers,
                 {{"attn_norm", {h}, kF32},         {"ffn_norm", {h}, kF32},           {"attn_q", {h, q}, kMatrix},
                  {"attn_k", {h, kv}, kMatrix},     {"attn_v", {h, kv}, kMatrix},      {"attn_o", {q, h}, kMatrix},
                  {"attn_q_norm", {s.head_dim}, kF32}, {"attn_k_norm", {s.head_dim}, kF32}, {"ffn_gate", {h, s.ffn}, kMatrix},
                  {"ffn_up", {h, s.ffn}, kMatrix},  {"ffn_down", {s.ffn, h}, kMatrix}});
}

void read_sampling(const ModelFile & m, const std::string & prefix) {
    m.boolean(prefix + "do_sample");
    m.f32(prefix + "temperature");
    m.u32(prefix + "top_k");
    m.f32(prefix + "top_p");
    m.f32(prefix + "repetition_penalty");
}

/** An [i32] key of the strides of the codec's stages, one per stage: a stride below 1 throws, and so do more stages than the file holds tensors. */
std::vector<int32_t> strides(const ModelFile & m, const std::string & key) {
    const std::vector<int32_t> s = m.i32_array(key);
    require((int64_t) s.size() <= m.tensor_count() && std::all_of(s.begin(), s.end(), [](int32_t r) { return r >= 1; }), m,
            key + " holds a stride below 1 or names more stages than the file holds tensors");
    return s;
}

/** Reads every key of the layout, checks the ones that must agree, and names the tensors they call for. */
std::vector<TensorSpec> tensors(const ModelFile & m) {
    check_model_keys(m, "synthesis", "steers");
    const std::vector<std::string> languages = m.str_array("general.languages");
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
                "the voice " + voices[i] + " speaks " + voice_languages[i] + ", which is not one of general.languages");
        require(genders[i] == "female" || genders[i] == "male", m, "the gender of the voice " + voices[i] + " is neither female nor male");
    }

    const std::string p = "qwen3-tts.";
    const Stack talker = read_stack(m, p + "talker."), cp = read_stack(m, p + "code_predictor.");
    const int vocab = talker.vocab;
    const int groups = m.count(p + "talker.num_code_groups");
    require(groups >= 2, m, "qwen3-tts.talker.num_code_groups is 1, where the code predictor predicts the codes after a frame's first");
    for (const char * key : {"codec_bos_id", "codec_eos_token_id", "codec_pad_id", "codec_think_id", "codec_nothink_id", "codec_think_bos_id",
                             "codec_think_eos_id"}) {
        require(m.u32(p + "talker." + key) < (uint32_t) vocab, m, p + "talker." + key + " is not in the talker's vocabulary");
    }
    const uint32_t suppressed = m.u32(p + "talker.suppressed_tokens");
    require(suppressed < (uint32_t) vocab, m, "qwen3-tts.talker.suppressed_tokens leaves the talker no code to sample");
    const size_t tokens = m.str_array(p + "tokenizer.tokens").size();
    m.str_array(p + "tokenizer.merges");
    for (const char * key : {"tts_bos_token_id", "tts_eos_token_id", "tts_pad_token_id", "im_start_token_id", "im_end_token_id",
                             "assistant_token_id", "newline_token_id"}) {
        require(m.u32(p + "text." + key) < tokens, m, p + "text." + key + " is not one of qwen3-tts.tokenizer.tokens");
    }
    const auto codec_id = [&](int32_t id) { return id >= 0 && id < vocab; };
    const std::vector<int32_t> language_ids = m.i32_array(p + "language_ids");
    const std::vector<int32_t> speaker_ids = m.i32_array(p + "speaker_ids"), dialect_ids = m.i32_array(p + "dialect_ids");
    require(language_ids.size() == languages.size(), m, "qwen3-tts.language_ids does not have one id per language");
    require(speaker_ids.size() == n && dialect_ids.size() == n, m, "qwen3-tts.speaker_ids and qwen3-tts.dialect_ids do not each have one id per voice");
    require(std::all_of(language_ids.begin(), language_ids.end(), codec_id) && std::all_of(speaker_ids.begin(), speaker_ids.end(), codec_id), m,
            "qwen3-tts.language_ids or qwen3-tts.speaker_ids holds an id outside the talker's vocabulary");
    require(std::all_of(dialect_ids.begin(), dialect_ids.end(), [&](int32_t id) { return id == -1 || codec_id(id); }), m,
            "qwen3-tts.dialect_ids holds an id that is neither -1 nor in the talker's vocabulary");
    const std::string dialect_language = m.str(p + "dialect_language");
    require(std::count(languages.begin(), languages.end(), dialect_language) == 1, m,
            "qwen3-tts.dialect_language is " + dialect_language + ", which is not one of general.languages");
    const uint32_t min_frames = m.u32(p + "generation.min_frames");
    const int max_frames = m.size(p + "generation.max_frames");
    require(min_frames <= (uint32_t) max_frames && (int64_t) max_frames + kPromptRows < m.size(p + "talker.max_position_embeddings"), m,
            "qwen3-tts.generation.max_frames is below min_frames or leaves the talker no position for a text");
    read_sampling(m, p + "generation.talker.");
    read_sampling(m, p + "generation.code_predictor.");

    const int latent = m.size(p + "codec.latent_dim"), codebook_dim = m.size(p + "codec.codebook_dim");
    const int codec_hidden = m.size(p + "codec.hidden_size"), codec_heads = m.size(p + "codec.num_attention_heads");
    const int codec_head_dim = m.size(p + "codec.head_dim"), codec_layers = m.count(p + "codec.num_hidden_layers");
    m.size(p + "codec.sliding_window");
    m.f32(p + "codec.rms_norm_eps");
    m.f32(p + "codec.rope_theta");
    const int quantizers = m.size(p + "codec.num_quantizers");
    require(quantizers == groups, m, "qwen3-tts.codec.num_quantizers differs from qwen3-tts.talker.num_code_groups");
    require(m.u32(p + "codec.num_key_value_heads") == (uint32_t) codec_heads, m,
            "the codec's transformer has fewer key/value heads than query heads, which this reader does not run");
    require(codec_head_dim % 2 == 0, m, "qwen3-tts.codec.head_dim is odd, where RoPE turns pairs of channels");
    require(codebook_dim % 2 == 0, m, "qwen3-tts.codec.codebook_dim is odd, where the codebooks are half its width");
    const std::vector<int32_t> ratios = strides(m, p + "codec.upsampling_ratios"), rates = strides(m, p + "codec.upsample_rates");
    int64_t samples = 1;
    for (const std::vector<int32_t> * stages : {&ratios, &rates}) {
        for (int32_t r : *stages) samples = std::min<int64_t>(samples * r, (int64_t) INT_MAX + 1);
    }
    require(samples <= INT_MAX, m, "the codec's strides give a frame more samples than speech.cpp counts");

    // The checkpoint's text_hidden_size, text_vocab_size, the codec's codebook_size, decoder_dim and intermediate_size
    // have no key: each is the width of one tensor, which the others are checked against.
    const int64_t text_width = m.width("talker.text_embd", 0), text_rows = m.width("talker.text_embd", 1);
    require(text_rows >= (int64_t) tokens, m, "talker.text_embd has fewer rows than qwen3-tts.tokenizer.tokens has tokens");
    const int64_t codebook_size = m.width("codec.vq.first.codebook.0", 1);
    require(codebook_size >= vocab - (int64_t) suppressed && codebook_size >= cp.vocab, m,
            "the codec's codebooks have fewer entries than the talker or the code predictor has codes");
    const int64_t decoder = m.width("codec.dec.in_conv.weight", 1), codec_ffn = m.width("codec.tf.blk.0.ffn_gate", 1);
    std::vector<int64_t> channels = {decoder};
    for (size_t b = 0; b < rates.size(); b++) {
        require(channels.back() % 2 == 0, m,
                "codec.dec.in_conv.weight has " + std::to_string(decoder) + " outputs, which the codec's decoder blocks cannot each halve");
        channels.push_back(channels.back() / 2);
    }

    const int64_t h = talker.hidden;
    std::vector<TensorSpec> t = {{"talker.text_embd", {text_width, text_rows}, kMatrix},
                                 {"talker.text_proj.fc1.weight", {text_width, text_width}, kMatrix},
                                 {"talker.text_proj.fc1.bias", {text_width}, kF32},
                                 {"talker.text_proj.fc2.weight", {text_width, h}, kMatrix},
                                 {"talker.text_proj.fc2.bias", {h}, kF32},
                                 {"talker.codec_embd", {h, vocab}, kMatrix},
                                 {"talker.codec_head", {h, vocab}, kMatrix},
                                 {"talker.norm", {h}, kF32},
                                 {"cp.norm", {cp.hidden}, kF32}};
    add_stack(t, "talker.blk.", talker);
    add_stack(t, "cp.blk.", cp);
    // The code predictor embeds the codes after a frame's first in the talker's width, since the talker reads them too.
    add_numbered(t, "cp.codec_embd.", groups - 1, {{"", {h, cp.vocab}, kMatrix}});
    add_numbered(t, "cp.head.", groups - 1, {{"", {cp.hidden, cp.vocab}, kMatrix}});
    // The official model makes small_to_mtp_projection a Linear exactly when the two widths differ (the 1.7B model).
    if (cp.hidden != talker.hidden) add_block(t, "cp.in_proj.", {{"weight", {h, cp.hidden}, kMatrix}, {"bias", {cp.hidden}, kF32}});

    const int64_t half = codebook_dim / 2, c = codec_hidden, attention = (int64_t) codec_heads * codec_head_dim, out = channels.back();
    add_block(t, "codec.", {{"vq.first.codebook.0", {half, codebook_size}, kF32},
                            {"vq.first.out_proj", {half, codebook_dim}, kF32},
                            {"vq.rest.out_proj", {half, codebook_dim}, kF32},
                            {"pre_conv.weight", {codebook_dim, latent, kPreConvWidth}, kCodecWeight},
                            {"pre_conv.bias", {latent}, kF32},
                            {"tf.in_proj.weight", {latent, c}, kCodecWeight},
                            {"tf.in_proj.bias", {c}, kF32},
                            {"tf.out_proj.weight", {c, latent}, kCodecWeight},
                            {"tf.out_proj.bias", {latent}, kF32},
                            {"tf.norm", {c}, kF32},
                            {"dec.in_conv.weight", {latent, decoder, kConvWidth}, kCodecWeight},
                            {"dec.in_conv.bias", {decoder}, kF32},
                            {"dec.out_snake.alpha", {out}, kF32},
                            {"dec.out_snake.inv_beta", {out}, kF32},
                            {"dec.out_conv.weight", {out, 1, kConvWidth}, kCodecWeight},
                            {"dec.out_conv.bias", {1}, kF32}});
    add_numbered(t, "codec.vq.rest.codebook.", quantizers - 1, {{"", {half, codebook_size}, kF32}});
    add_numbered(t, "codec.tf.blk.", codec_layers,
                 {{"attn_norm", {c}, kF32},                 {"ffn_norm", {c}, kF32},
                  {"attn_q", {c, attention}, kCodecWeight}, {"attn_k", {c, attention}, kCodecWeight},
                  {"attn_v", {c, attention}, kCodecWeight}, {"attn_o", {attention, c}, kCodecWeight},
                  {"attn_scale", {c}, kF32},                {"ffn_gate", {c, codec_ffn}, kCodecWeight},
                  {"ffn_up", {c, codec_ffn}, kCodecWeight}, {"ffn_down", {codec_ffn, c}, kCodecWeight},
                  {"ffn_scale", {c}, kF32}});
    const int64_t l = latent, wide = kConvNextWidening * latent;
    for (size_t i = 0; i < ratios.size(); i++) {
        add_block(t, "codec.up." + std::to_string(i) + ".",
                  {{"tconv.weight", {l, l, ratios[i]}, kCodecWeight}, {"tconv.bias", {l}, kF32},
                   {"dwconv.weight", {l, kConvWidth}, kF32},          {"dwconv.bias", {l}, kF32},
                   {"norm.weight", {l}, kF32},                        {"norm.bias", {l}, kF32},
                   {"pw1.weight", {l, wide}, kCodecWeight},           {"pw1.bias", {wide}, kF32},
                   {"pw2.weight", {wide, l}, kCodecWeight},           {"pw2.bias", {l}, kF32},
                   {"gamma", {l}, kF32}});
    }
    for (size_t b = 0; b < rates.size(); b++) {
        const int64_t in = channels[b], o = channels[b + 1];
        const std::string prefix = "codec.dec.blk." + std::to_string(b) + ".";
        add_block(t, prefix, {{"snake.alpha", {in}, kF32}, {"snake.inv_beta", {in}, kF32},
                              {"tconv.weight", {in, o, 2 * (int64_t) rates[b]}, kCodecWeight}, {"tconv.bias", {o}, kF32}});
        add_numbered(t, prefix + "res.", kCodecResidualUnits,
                     {{"snake1.alpha", {o}, kF32},                   {"snake1.inv_beta", {o}, kF32},
                      {"conv1.weight", {o, o, kConvWidth}, kCodecWeight}, {"conv1.bias", {o}, kF32},
                      {"snake2.alpha", {o}, kF32},                   {"snake2.inv_beta", {o}, kF32},
                      {"conv2.weight", {o, o, 1}, kCodecWeight},     {"conv2.bias", {o}, kF32}});
    }
    return t;
}

}  // namespace

const Layout qwen3_tts_layout = {"qwen3-tts", 1,
                                 "convert it again with reference/qwen3-tts/convert.py, or download it again from its Hugging Face "
                                 "repository",
                                 tensors};
