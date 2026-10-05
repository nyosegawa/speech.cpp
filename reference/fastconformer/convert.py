"""Converts a pinned NeMo FastConformer checkpoint with a TDT decoder to the GGUF the C++ port reads.

usage: uv run python convert.py <model> <out dir> [--type f32|f16]

Writes <model>-<type>.gguf: the frontend's window and mel filterbank, the subsampling, the conformer layers,
the prediction network and the joint, with the SentencePiece pieces the token ids name and the TDT greedy
decoding's durations and limit. A hybrid checkpoint's CTC head is left out, since NeMo decodes with TDT.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis: a Linear weight [out, in] is stored as is
(ne = [in, out]). --type f16 applies to the matrices of the linear layers; convolution kernels, norms, biases,
the frontend and the rest stay float32. The batch norm of each convolution module is folded into its
depthwise convolution, which it follows in evaluation, and each LSTM layer's two biases are summed. A checkpoint
whose conformer layers have no biases (ConformerEncoder's use_bias false) is written without them, and
fastconformer.use_bias says so.
"""

import argparse
import os

import numpy as np
from gguf import GGUFWriter
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel, EncDecRNNTBPEModel
from sentencepiece import sentencepiece_model_pb2

from pins import MODELS, NEMO, restore

ARCH = "fastconformer"
# The BCP 47 tags of the languages each checkpoint transcribes, from its model card. Neither takes a language:
# parakeet-tdt-0.6b-v3 finds the language of the audio itself.
LANGUAGES = {
    "parakeet-tdt_ctc-0.6b-ja": ["ja"],
    "parakeet-tdt-0.6b-v3": ["en", "es", "fr", "de", "bg", "hr", "cs", "da", "nl", "et", "fi", "el", "hu", "it", "lv",
                             "lt", "mt", "pl", "pt", "ro", "sk", "sl", "sv", "ru", "uk"],
}
LICENSES = {"parakeet-tdt_ctc-0.6b-ja": "CC-BY-4.0", "parakeet-tdt-0.6b-v3": "CC-BY-4.0"}

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
parser.add_argument("--type", choices=["f32", "f16"], default="f16")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

pin = MODELS[args.model]
model = restore(pin)
cfg = model.cfg
enc = cfg.encoder
featurizer = model.preprocessor.featurizer
assert enc.subsampling == "dw_striding" and enc.self_attention_model == "rel_pos"
assert list(enc.att_context_size) == [-1, -1] and enc.conv_norm_type == "batch_norm"
assert featurizer.exact_pad is False and featurizer.frame_splicing == 1 and featurizer.mag_power == 2.0
assert featurizer.normalize == "per_feature" and featurizer.log and featurizer.log_zero_guard_type == "add"
assert featurizer.pad_value == 0
# The decoding transcribe() runs: greedy TDT with the blank as the prediction network's padding (its embedding
# zero, so the blank fed first is the start of the sequence) and a ReLU joint. The joint's log-softmax, which
# NeMo applies on the CPU alone, changes no argmax and is left out.
# A model with a prompt (EncDecRNNTBPEModelWithPrompt and its hybrid) is told its language through an input the C++
# does not have; a hybrid's CTC head is left aside while cur_decoder is "rnnt".
dec, joint, decoding = model.decoder, model.joint, model.decoding
assert type(model) in (EncDecRNNTBPEModel, EncDecHybridRNNTCTCBPEModel)
assert type(model) is EncDecRNNTBPEModel or model.cur_decoder == "rnnt"
assert decoding.cfg.model_type == "tdt" and decoding.cfg.strategy == "greedy_batch"
assert not decoding.cfg.get("big_blank_durations")
assert dec.blank_as_pad and dec.blank_idx == decoding.blank_id and not dec.random_state_sampling
# LSTMDropout is the plain LSTM rnn() makes without a normalization; its dropout is off in evaluation.
assert type(dec.prediction.dec_rnn).__name__ == "LSTMDropout" and dec.prediction.dec_rnn.lstm.proj_size == 0
assert dec.prediction.dec_rnn.lstm.bias and not dec.prediction.dec_rnn.lstm.bidirectional
assert not dec.is_adapter_available() and not joint.is_adapter_available()
assert float(dec.prediction.embed.weight[dec.blank_idx].abs().max()) == 0
assert joint.activation == "relu" and joint.temperature == 1.0
durations = [int(d) for d in decoding.cfg.durations]
assert joint.num_extra_outputs == len(durations) == len(decoding.durations)
assert joint.joint_net[-1].out_features == decoding.blank_id + 1 + len(durations)

path = os.path.join(args.out_dir, f"{args.model}-{args.type}.gguf")
w = GGUFWriter(path, ARCH)
w.add_name(args.model)
w.add_license(LICENSES[args.model])
w.add_source_url(f"https://huggingface.co/{pin['repository']}/tree/{pin['revision']}")
w.add_string("fastconformer.nemo_version", NEMO)
w.add_languages(LANGUAGES[args.model])
w.add_array("speech.languages", LANGUAGES[args.model])
w.add_bool("speech.language_selectable", False)

# The frontend: FilterbankFeatures' parameters, and the normalization guard, CONSTANT in features.py.
w.add_uint32("fastconformer.sample_rate", int(featurizer.sample_rate))
w.add_uint32("fastconformer.n_fft", int(featurizer.n_fft))
w.add_uint32("fastconformer.hop_length", int(featurizer.hop_length))
w.add_uint32("fastconformer.n_mels", int(featurizer.nfilt))
w.add_float32("fastconformer.preemphasis", float(featurizer.preemph))
w.add_float32("fastconformer.log_guard", float(featurizer.log_zero_guard_value))
w.add_float32("fastconformer.std_guard", 1e-5)

# The encoder. pos_base is INF_VAL of multi_head_attention.py, the base of RelPositionalEncoding's
# wavelengths; ff_factor is ConformerLayer's fc_factor, the weight of each half-step feed-forward.
w.add_uint32("fastconformer.d_model", int(enc.d_model))
w.add_uint32("fastconformer.num_layers", int(enc.n_layers))
w.add_uint32("fastconformer.num_heads", int(enc.n_heads))
w.add_uint32("fastconformer.conv_kernel", int(enc.conv_kernel_size))
w.add_uint32("fastconformer.subsampling_factor", int(enc.subsampling_factor))
w.add_float32("fastconformer.norm_eps", float(model.encoder.layers[0].norm_out.eps))
w.add_float32("fastconformer.pos_base", 10000.0)
w.add_float32("fastconformer.xscale", float(model.encoder.xscale or 1.0))
w.add_float32("fastconformer.ff_factor", float(model.encoder.layers[0].fc_factor))
# 1 when the linear layers of the feed-forward modules and the attention and the pointwise convolutions have biases;
# the depthwise convolution has one either way once the batch norm is folded into it.
use_bias = bool(model.encoder.layers[0].feed_forward1.use_bias)
assert all(bool(m.use_bias) == use_bias for layer in model.encoder.layers
           for m in (layer.feed_forward1, layer.feed_forward2, layer.self_attn, layer.conv))
w.add_uint32("fastconformer.use_bias", int(use_bias))

# The joint's classes are the tokenizer's ids, a blank after them, then one per duration.
tokenizer = model.tokenizer
assert not tokenizer.legacy
proto = sentencepiece_model_pb2.ModelProto()
proto.ParseFromString(tokenizer.tokenizer.serialized_model_proto())
# Normal, user-defined and unknown pieces only: SentencePiece writes a user-defined piece (parakeet-tdt-0.6b-v3's
# <|en|> and the other tags) as it writes a normal one, and the C++ detokenizer handles no control or byte pieces.
allowed = (proto.SentencePiece.NORMAL, proto.SentencePiece.USER_DEFINED, proto.SentencePiece.UNKNOWN)
assert all(p.type in allowed for p in proto.pieces)
assert not proto.trainer_spec.treat_whitespace_as_suffix
blank = int(decoding.blank_id)
assert blank == len(proto.pieces) == tokenizer.vocab_size
w.add_uint32("fastconformer.blank_id", blank)
w.add_uint32("fastconformer.prediction.num_layers", int(dec.pred_rnn_layers))
w.add_array("fastconformer.tdt.durations", durations)
# The most tokens emitted on one frame before the decoding moves to the next.
w.add_uint32("fastconformer.tdt.max_symbols", int(decoding.decoding.max_symbols))
w.add_string("tokenizer.model", "sentencepiece")
w.add_array("tokenizer.tokens", [p.piece for p in proto.pieces])
w.add_uint32("tokenizer.unknown_id", int(tokenizer.tokenizer.unk_id()))
w.add_string("tokenizer.unknown_surface", proto.trainer_spec.unk_surface)
# 1 when SentencePiece's decoder drops the leading "▁" of each piece until the text is no longer empty, which
# it does when either normalizer option is set.
w.add_uint32("tokenizer.strip_leading_space",
             int(proto.normalizer_spec.add_dummy_prefix or proto.normalizer_spec.remove_extra_whitespaces))
# The marks before which the decoding removes one whitespace character. The C++ looks for a space only,
# the one whitespace character a decoded text can hold when no piece holds another and the unknown surface
# holds only spaces.
assert not any(c.isspace() for p in proto.pieces for c in p.piece)
assert all(c == " " or not c.isspace() for c in proto.trainer_spec.unk_surface)
w.add_array("tokenizer.punctuation", sorted(decoding.supported_punctuation or []))
# The decoding strips no language tags from the text (strip_lang_tags, for models that write them), which the C++
# does not do.
assert not decoding.strip_lang_tags

sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}


def add(name, data, matrix=False):
    data = np.ascontiguousarray(data, dtype=np.float32)
    w.add_tensor(name, data.astype(np.float16) if matrix and args.type == "f16" else data)


add("frontend.window", sd["preprocessor.featurizer.window"])
add("frontend.filterbank", sd["preprocessor.featurizer.fb"][0])

# The subsampling's nn.Sequential: Conv2d, ReLU, then (depthwise Conv2d, pointwise Conv2d, ReLU) per halving.
s = "encoder.pre_encode."
add("sub.conv.0.weight", sd[s + "conv.0.weight"])
add("sub.conv.0.bias", sd[s + "conv.0.bias"])
for i in range(1, int(np.log2(enc.subsampling_factor))):
    dw, pw = 3 * i - 1, 3 * i
    add(f"sub.conv.{i}.dw.weight", sd[s + f"conv.{dw}.weight"])
    add(f"sub.conv.{i}.dw.bias", sd[s + f"conv.{dw}.bias"])
    add(f"sub.conv.{i}.pw.weight", sd[s + f"conv.{pw}.weight"][:, :, 0, 0], True)
    add(f"sub.conv.{i}.pw.bias", sd[s + f"conv.{pw}.bias"])
add("sub.out.weight", sd[s + "out.weight"], True)
add("sub.out.bias", sd[s + "out.bias"])

d = int(enc.d_model)
for l in range(int(enc.n_layers)):
    a, o = f"encoder.layers.{l}.", f"blk.{l}."

    def linear(src, dst, bias=use_bias):
        add(dst + ".weight", sd[a + src + ".weight"], True)
        if bias:
            add(dst + ".bias", sd[a + src + ".bias"])

    def norm(src, dst):
        add(dst + ".weight", sd[a + src + ".weight"])
        add(dst + ".bias", sd[a + src + ".bias"])

    for ff in ("1", "2"):
        norm(f"norm_feed_forward{ff}", o + f"ff{ff}_norm")
        linear(f"feed_forward{ff}.linear1", o + f"ff{ff}_up")
        linear(f"feed_forward{ff}.linear2", o + f"ff{ff}_down")
    norm("norm_self_att", o + "attn_norm")
    for x in ("q", "k", "v", "out"):
        linear(f"self_attn.linear_{x}", o + f"attn_{x}")
    linear("self_attn.linear_pos", o + "attn_pos", bias=False)
    add(o + "attn_pos_bias_u", sd[a + "self_attn.pos_bias_u"])
    add(o + "attn_pos_bias_v", sd[a + "self_attn.pos_bias_v"])

    norm("norm_conv", o + "conv_norm")
    # GLU keeps the first half of the pointwise convolution's channels and gates them with the second.
    pw1 = sd[a + "conv.pointwise_conv1.weight"][:, :, 0]
    add(o + "conv_pw1_a.weight", pw1[:d], True)
    add(o + "conv_pw1_gate.weight", pw1[d:], True)
    if use_bias:
        add(o + "conv_pw1_a.bias", sd[a + "conv.pointwise_conv1.bias"][:d])
        add(o + "conv_pw1_gate.bias", sd[a + "conv.pointwise_conv1.bias"][d:])
    bn = a + "conv.batch_norm."
    scale = sd[bn + "weight"] / np.sqrt(sd[bn + "running_var"] + model.encoder.layers[l].conv.batch_norm.eps)
    dw_bias = sd[a + "conv.depthwise_conv.bias"] if use_bias else 0
    add(o + "conv_dw.weight", sd[a + "conv.depthwise_conv.weight"][:, 0, :] * scale[:, None])
    add(o + "conv_dw.bias", (dw_bias - sd[bn + "running_mean"]) * scale + sd[bn + "bias"])
    add(o + "conv_pw2.weight", sd[a + "conv.pointwise_conv2.weight"][:, :, 0], True)
    if use_bias:
        add(o + "conv_pw2.bias", sd[a + "conv.pointwise_conv2.bias"])
    norm("norm_out", o + "out_norm")

# The prediction network: an embedding of the tokens and the blank, then LSTM layers whose gates are stacked as
# PyTorch stacks them, input, forget, cell and output.
add("pred.embed.weight", sd["decoder.prediction.embed.weight"], True)
for l in range(int(dec.pred_rnn_layers)):
    r = "decoder.prediction.dec_rnn.lstm."
    add(f"pred.lstm.{l}.ih.weight", sd[r + f"weight_ih_l{l}"], True)
    add(f"pred.lstm.{l}.hh.weight", sd[r + f"weight_hh_l{l}"], True)
    add(f"pred.lstm.{l}.bias", sd[r + f"bias_ih_l{l}"] + sd[r + f"bias_hh_l{l}"])
for x in ("enc", "pred"):
    add(f"joint.{x}.weight", sd[f"joint.{x}.weight"], True)
    add(f"joint.{x}.bias", sd[f"joint.{x}.bias"])
add("joint.out.weight", sd["joint.joint_net.2.weight"], True)
add("joint.out.bias", sd["joint.joint_net.2.bias"])

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", path)
