#include "layout.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "error.h"
namespace fastconformer {

namespace {

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

/** Reads every key of the layout, checks the ones that must agree, and names the tensors they call for. */
std::vector<std::string> tensors(const ModelFile & m) {
    check_model_keys(m, "recognition", "checked");
    const std::string p = "fastconformer.";
    for (const char * key : {"n_fft", "hop_length", "n_mels"}) m.u32(p + "frontend." + key);
    for (const char * key : {"preemphasis", "log_guard", "std_guard"}) m.f32(p + "frontend." + key);
    for (const char * key : {"d_model", "num_heads", "conv_kernel"}) m.u32(p + "encoder." + key);
    for (const char * key : {"norm_eps", "pos_base", "xscale", "ff_factor"}) m.f32(p + "encoder." + key);
    if (m.one_of(p + "encoder.attention", {"rel_pos", "rel_pos_local_attn"}) == "rel_pos_local_attn") {
        m.u32(p + "encoder.attention_context");
        m.u32(p + "encoder.global_tokens");
    }
    const uint32_t factor = m.u32(p + "encoder.subsampling_factor");
    uint32_t halvings = 0;
    for (uint32_t f = factor; f > 1; f /= 2) halvings++;
    require(factor >= 2 && (1u << halvings) == factor, m, "fastconformer.encoder.subsampling_factor is not a power of two");
    const bool use_bias = m.boolean(p + "encoder.use_bias");

    const bool tdt = m.one_of(p + "decoder.kind", {"tdt", "rnnt"}) == "tdt";
    const uint32_t blank = m.u32(p + "decoder.blank_id");
    if (tdt) {
        m.i32_array(p + "decoder.tdt.durations");
        m.u32(p + "decoder.tdt.max_symbols");
    } else {
        m.u32(p + "decoder.rnnt.beam_size");
        m.boolean(p + "decoder.rnnt.score_norm");
        m.f32(p + "decoder.rnnt.max_target_ratio");
    }
    m.str_array(p + "segment.separators");
    m.str_array(p + "segment.breaks");
    require(m.str_array(p + "tokenizer.tokens").size() == blank, m, "the blank does not follow the last of fastconformer.tokenizer.tokens");
    require(m.u32(p + "tokenizer.unknown_id") < blank, m, "fastconformer.tokenizer.unknown_id is not a piece");
    m.str(p + "tokenizer.unknown_surface");
    m.boolean(p + "tokenizer.strip_leading_space");
    m.str_array(p + "tokenizer.punctuation");

    std::vector<std::string> names = {"frontend.window", "frontend.filterbank", "sub.conv.0.weight", "sub.conv.0.bias", "sub.out.weight",
                                      "sub.out.bias",    "pred.embed.weight",   "joint.enc.weight",  "joint.enc.bias",  "joint.pred.weight",
                                      "joint.pred.bias", "joint.out.weight",    "joint.out.bias"};
    for (uint32_t i = 1; i < halvings; i++) {
        const std::string c = "sub.conv." + std::to_string(i) + ".";
        names.insert(names.end(), {c + "dw.weight", c + "dw.bias", c + "pw.weight", c + "pw.bias"});
    }
    add_numbered(names, "blk.", m.u32(p + "encoder.num_layers"),
                 {"ff1_norm.weight", "ff1_norm.bias", "ff2_norm.weight", "ff2_norm.bias", "attn_norm.weight", "attn_norm.bias",
                  "conv_norm.weight", "conv_norm.bias", "out_norm.weight", "out_norm.bias", "attn_pos.weight", "attn_pos_bias_u",
                  "attn_pos_bias_v", "conv_dw.weight", "conv_dw.bias", "ff1_up.weight", "ff1_down.weight", "ff2_up.weight",
                  "ff2_down.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_out.weight", "conv_pw1_a.weight",
                  "conv_pw1_gate.weight", "conv_pw2.weight"});
    if (use_bias) {
        add_numbered(names, "blk.", m.u32(p + "encoder.num_layers"),
                     {"ff1_up.bias", "ff1_down.bias", "ff2_up.bias", "ff2_down.bias", "attn_q.bias", "attn_k.bias", "attn_v.bias",
                      "attn_out.bias", "conv_pw1_a.bias", "conv_pw1_gate.bias", "conv_pw2.bias"});
    }
    add_numbered(names, "pred.lstm.", m.u32(p + "decoder.prediction_layers"), {"ih.weight", "hh.weight", "bias"});
    return names;
}

}  // namespace

const Layout layout = {"fastconformer", 1,
                       "convert it again with reference/fastconformer/convert.py, or download it again from its Hugging Face "
                       "repository",
                       tensors};

}  // namespace fastconformer
