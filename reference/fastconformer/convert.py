"""Converts a pinned NeMo FastConformer checkpoint with a CTC head to the GGUF the C++ port reads.

usage: uv run python convert.py <model> <out dir> [--type f32|f16]

Writes <model>-<type>.gguf: the frontend's window and mel filterbank, the subsampling, the conformer layers
and the CTC head, with the SentencePiece pieces the CTC head's ids name.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis: a Linear weight [out, in] is stored as is
(ne = [in, out]). --type f16 applies to the matrices of the linear layers; convolution kernels, norms, biases,
the frontend and the rest stay float32. The batch norm of each convolution module is folded into its
depthwise convolution, which it follows in evaluation.
"""

import argparse
import os

import numpy as np
from gguf import GGUFWriter
from sentencepiece import sentencepiece_model_pb2

from pins import MODELS, NEMO, restore

ARCH = "fastconformer"
# The BCP 47 tags of the languages each checkpoint transcribes, from its model card.
LANGUAGES = {"parakeet-tdt_ctc-0.6b-ja": ["ja"]}
LICENSES = {"parakeet-tdt_ctc-0.6b-ja": "CC-BY-4.0"}

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
assert featurizer.pad_value == 0 and model.ctc_decoder.temperature == 1.0

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

# The CTC head's classes are the tokenizer's ids and a blank after them.
tokenizer = model.tokenizer
assert not tokenizer.legacy
proto = sentencepiece_model_pb2.ModelProto()
proto.ParseFromString(tokenizer.tokenizer.serialized_model_proto())
# Normal pieces and the unknown piece only: the C++ detokenizer handles no control, user-defined or byte pieces.
assert all(p.type in (proto.SentencePiece.NORMAL, proto.SentencePiece.UNKNOWN) for p in proto.pieces)
assert not proto.trainer_spec.treat_whitespace_as_suffix
blank = int(model.ctc_decoder.num_classes_with_blank) - 1
assert blank == len(proto.pieces) == tokenizer.vocab_size
w.add_uint32("fastconformer.ctc.blank_id", blank)
w.add_string("tokenizer.model", "sentencepiece")
w.add_array("tokenizer.tokens", [p.piece for p in proto.pieces])
w.add_uint32("tokenizer.unknown_id", int(tokenizer.tokenizer.unk_id()))
w.add_string("tokenizer.unknown_surface", proto.trainer_spec.unk_surface)
# 1 when SentencePiece's decoder drops the leading "▁" of each piece until the text is no longer empty, which
# it does when either normalizer option is set.
w.add_uint32("tokenizer.strip_leading_space",
             int(proto.normalizer_spec.add_dummy_prefix or proto.normalizer_spec.remove_extra_whitespaces))
# The marks before which the CTC decoding removes one whitespace character. The C++ looks for a space only,
# the one whitespace character a decoded text can hold when no piece holds another and the unknown surface
# holds only spaces.
assert not any(c.isspace() for p in proto.pieces for c in p.piece)
assert all(c == " " or not c.isspace() for c in proto.trainer_spec.unk_surface)
w.add_array("tokenizer.punctuation", sorted(model.ctc_decoding.supported_punctuation or []))

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

    def linear(src, dst, bias=True):
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
    pw1, pw1_bias = sd[a + "conv.pointwise_conv1.weight"][:, :, 0], sd[a + "conv.pointwise_conv1.bias"]
    add(o + "conv_pw1_a.weight", pw1[:d], True)
    add(o + "conv_pw1_a.bias", pw1_bias[:d])
    add(o + "conv_pw1_gate.weight", pw1[d:], True)
    add(o + "conv_pw1_gate.bias", pw1_bias[d:])
    bn = a + "conv.batch_norm."
    scale = sd[bn + "weight"] / np.sqrt(sd[bn + "running_var"] + model.encoder.layers[l].conv.batch_norm.eps)
    add(o + "conv_dw.weight", sd[a + "conv.depthwise_conv.weight"][:, 0, :] * scale[:, None])
    add(o + "conv_dw.bias", (sd[a + "conv.depthwise_conv.bias"] - sd[bn + "running_mean"]) * scale + sd[bn + "bias"])
    add(o + "conv_pw2.weight", sd[a + "conv.pointwise_conv2.weight"][:, :, 0], True)
    add(o + "conv_pw2.bias", sd[a + "conv.pointwise_conv2.bias"])
    norm("norm_out", o + "out_norm")

add("ctc.weight", sd["ctc_decoder.decoder_layers.0.weight"][:, :, 0], True)
add("ctc.bias", sd["ctc_decoder.decoder_layers.0.bias"])

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", path)
