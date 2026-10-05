#include "layout.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "codec.h"

namespace irodori {

namespace {

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw std::runtime_error(m.path() + ": " + what + "; " + m.remedy());
}

/** The residual units of a codec block, "<prefix>res.<j>." each. */
void add_residual_units(std::vector<std::string> & names, const std::string & prefix) {
    add_numbered(names, prefix + "res.", kCodecResidualUnits,
                 {"snake1.alpha", "snake1.inv_alpha", "conv1.weight", "conv1.bias", "snake2.alpha", "snake2.inv_alpha", "conv2.weight",
                  "conv2.bias"});
}

/** Reads every key of the model layout, checks the ones that must agree, and names the tensors they call for. */
std::vector<std::string> model_tensors(const ModelFile & m) {
    check_model_keys(m, "synthesis", "checked");
    const std::string p = "irodori-tts.";
    const bool meanflow = m.one_of(p + "flow", {"meanflow", "rf_velocity"}) == "meanflow";
    m.u32(p + "latent_dim");
    m.f32(p + "norm_eps");
    m.f32(p + "rope_theta");
    for (const char * key : {"hidden_size", "num_heads", "window", "dim", "max_tokens"}) m.u32(p + "text." + key);
    for (const char * key : {"norm_eps", "rope_theta_global", "rope_theta_local"}) m.f32(p + "text." + key);
    const uint32_t text_layers = m.u32(p + "text.num_layers");
    require(m.i32_array(p + "text.layer_global").size() == text_layers, m, "irodori-tts.text.layer_global does not have one entry per layer");
    for (const char * key : {"dim", "num_heads", "patch_size"}) m.u32(p + "speaker." + key);
    for (const char * key : {"dim", "num_heads", "timestep_dim"}) m.u32(p + "dit." + key);
    m.u32(p + "sampler.default_steps");
    if (!meanflow) {
        for (const char * key : {"cfg_text", "cfg_speaker", "cfg_min_t", "cfg_max_t"}) m.f32(p + "sampler." + key);
    }
    require(m.f32(p + "length.min_seconds") <= m.f32(p + "length.max_seconds") && m.f32(p + "length.min_speed") <= m.f32(p + "length.max_speed"),
            m, "the bounds of irodori-tts.length are reversed");
    m.f32(p + "reference.max_seconds");
    m.f32(p + "reference.lufs");
    m.u32(p + "tail.window");
    m.f32(p + "tail.std_threshold");
    m.f32(p + "tail.mean_threshold");
    const size_t pieces = m.str_array(p + "tokenizer.tokens").size();
    require(m.f64_array(p + "tokenizer.scores").size() == pieces, m, "irodori-tts.tokenizer.scores does not have one score per piece");
    for (int32_t id : m.i32_array(p + "tokenizer.added_ids")) require(id >= 0 && (size_t) id < pieces, m, "an added token is not a piece");
    require(m.u32(p + "tokenizer.bos_id") < pieces && m.u32(p + "tokenizer.unknown_id") < pieces, m,
            "the tokenizer's <s> or unknown piece is not a piece");
    m.u32(p + "codec.hop_length");
    require(m.str(p + "codec.sha256").size() == 64, m, "irodori-tts.codec.sha256 is not a SHA-256 in hexadecimal");

    std::vector<std::string> names = {"text.embd",        "text.embd_norm",          "text.final_norm",          "text.proj.weight",
                                      "text.proj.bias",   "text.proj.res_norm",      "text.proj.res_up.weight",  "text.proj.res_up.bias",
                                      "text.proj.res_down.weight", "text.proj.res_down.bias", "text.norm",       "speaker.in_proj.weight",
                                      "speaker.in_proj.bias", "speaker.norm",        "duration.in_proj.weight",  "duration.in_proj.bias",
                                      "duration.out_norm", "duration.out_proj.weight", "duration.out_proj.bias", "duration.null_caption",
                                      "dit.in_proj.weight", "dit.in_proj.bias",      "dit.out_norm",             "dit.out_proj.weight",
                                      "dit.out_proj.bias"};
    add_numbered(names, "text.blk.", text_layers, {"attn_q", "attn_k", "attn_v", "attn_out", "ffn_norm", "ffn_act", "ffn_gate", "ffn_down"});
    // ModernBERT normalizes the attention's input of every layer but the first.
    for (uint32_t l = 1; l < text_layers; l++) names.push_back("text.blk." + std::to_string(l) + ".attn_norm");
    add_numbered(names, "speaker.blk.", m.u32(p + "speaker.num_layers"),
                 {"attn_norm", "attn_q", "attn_k", "attn_v", "attn_o", "attn_gate", "q_norm", "k_norm", "ffn_norm", "ffn_gate", "ffn_up",
                  "ffn_down"});
    add_numbered(names, "duration.blk.", m.u32(p + "duration.num_layers"),
                 {"norm", "mod.weight", "mod.bias", "caption_mod.weight", "caption_mod.bias", "ffn_gate", "ffn_up", "ffn_down"});
    add_numbered(names, "dit.cond.", 3, {""});
    if (meanflow) add_numbered(names, "dit.delta_cond.", 3, {""});
    add_numbered(names, "dit.blk.", m.u32(p + "dit.num_layers"),
                 {"attn_q", "attn_k", "attn_v", "attn_o", "attn_gate", "attn_k_text", "attn_v_text", "attn_k_speaker", "attn_v_speaker",
                  "q_norm", "k_norm", "ffn_gate", "ffn_up", "ffn_down", "attn_ada.shift.down", "attn_ada.shift.up.weight",
                  "attn_ada.shift.up.bias", "attn_ada.scale.down", "attn_ada.scale.up.weight", "attn_ada.scale.up.bias",
                  "attn_ada.gate.down", "attn_ada.gate.up.weight", "attn_ada.gate.up.bias", "ffn_ada.shift.down", "ffn_ada.shift.up.weight",
                  "ffn_ada.shift.up.bias", "ffn_ada.scale.down", "ffn_ada.scale.up.weight", "ffn_ada.scale.up.bias", "ffn_ada.gate.down",
                  "ffn_ada.gate.up.weight", "ffn_ada.gate.up.bias"});

    const uint32_t encoder_blocks = (uint32_t) m.i32_array(p + "codec.encoder_rates").size();
    const uint32_t decoder_blocks = (uint32_t) m.i32_array(p + "codec.decoder_rates").size();
    names.insert(names.end(), {"codec.enc.conv_in.weight", "codec.enc.conv_in.bias", "codec.enc.snake.alpha", "codec.enc.snake.inv_alpha",
                               "codec.enc.conv_out.weight", "codec.enc.conv_out.bias", "codec.bottleneck.mean.weight",
                               "codec.bottleneck.mean.bias", "codec.dec.in_proj.weight", "codec.dec.in_proj.bias", "codec.dec.conv_in.weight",
                               "codec.dec.conv_in.bias", "codec.dec.out_snake.alpha", "codec.dec.out_snake.inv_alpha",
                               "codec.dec.conv_out.weight", "codec.dec.conv_out.bias"});
    add_numbered(names, "codec.enc.blk.", encoder_blocks, {"snake.alpha", "snake.inv_alpha", "down.first", "down.second", "down.bias"});
    for (uint32_t b = 0; b < encoder_blocks; b++) add_residual_units(names, "codec.enc.blk." + std::to_string(b) + ".");
    add_numbered(names, "codec.dec.blk.", decoder_blocks, {"snake.alpha", "snake.inv_alpha", "up.weight", "up.bias"});
    for (uint32_t b = 0; b < decoder_blocks; b++) add_residual_units(names, "codec.dec.blk." + std::to_string(b) + ".");
    return names;
}

std::vector<std::string> voice_tensors(const ModelFile & m) {
    const std::string p = "irodori-tts-voice.";
    require(m.str(p + "codec_sha256").size() == 64, m, "irodori-tts-voice.codec_sha256 is not a SHA-256 in hexadecimal");
    require(m.f32(p + "reference_seconds") > 0 && m.u32(p + "reference_sample_rate") > 0, m,
            "the voice file gives its recording no length or no rate");
    m.one_of(p + "device_kind", {"cpu", "gpu", "igpu"});
    return {"latent"};
}

}  // namespace

const Layout model_layout = {"irodori-tts", 1,
                             "convert it again with reference/irodori-tts/convert.py, or download it again from its Hugging Face "
                             "repository",
                             model_tensors};

const Layout voice_layout = {"irodori-tts-voice", 1, "make it again from its WAVE file with speech-tts make-voice", voice_tensors};

}  // namespace irodori
