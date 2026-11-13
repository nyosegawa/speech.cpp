"""Converts the pinned Haru checkpoint, its codec included, to the one GGUF file the C++ port reads.

usage: uv run python convert.py <out dir>

Writes the model's file, named under GGUF's naming convention, in layout 1: the language model (a Qwen3.5 text model of
Gated DeltaNet and gated attention layers), the projections of the model's and the user's audio latents into it, the
MeanFlow head that draws a frame's audio latent from its hidden state, the VibeVoice acoustic codec, the tokenizer, the
character card's wording in each language, and every constant the C++ reads, with the model's identity in the GGUF
specification's general keys. Each key says where its value comes from: the checkpoint's config.json, a buffer of the
checkpoint, or a constant of the official code at the commit pyproject.toml pins.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis: a Linear weight [out, in] is stored as is (ne = [in,
out]). Every tensor is written in float32, about 21 GB, a tensor at a time; `speech quantize` makes the other weight
types, each tensor in the type src/families/haru/layout.cpp gives it.

- A zero-centred RMS norm of the language model, which scales by 1 + weight, is stored as 1 + weight, the float32 sum
  the official code computes on every call. The Gated DeltaNet's gated norm and every other norm scale by the weight.
- An attention's query projection, whose rows give each head its query and then its output gate, is stored as two
  matrices.
- The rows of the token embedding of the two tokens a frame is silent with, PAD and EOP, are stored again as
  `output_silent`, which every weight type keeps in float32: the official code computes their logits in float32, since
  the choice between staying silent and starting to speak is often closer than bfloat16 resolves.
- The MeanFlow head's adaptive norms keep their one projection [3 x hidden, hidden] (shift, scale and gate) or
  [2 x hidden, hidden] (shift and scale), whose output the C++ cuts into its parts.
- The codec is channel-first, as ggml's matrix products want it. A convolution's weight [out, in, k] is stored as numpy
  [k, out, in] (ne = [in, out, k]), so that tap k is a plain [in, out] matrix; a depthwise one [channels, 1, k] as
  [k, channels]; a transposed convolution's [in, out, k] as [k, out, in]. A strided convolution of width 2s is stored as
  its two halves, each [out, s * in] with the input channel fastest.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from gguf import GGUFValueType, GGUFWriter, LlamaFileType, naming_convention
from safetensors import safe_open

import haru.audio as haru_audio
import haru.server as haru_server
from haru.models import meanflow_head
from haru.prompt import CARD_FIELDS, card_templates
from haru.sampling import DryConfig
from haru.session import DuplexConfig

from pins import CODE, MODEL, snapshot

ARCH = "haru"
# Each layout this converter has written, with the first release of speech.cpp whose reader takes it; it writes the
# last.
RELEASES = {1: "0.11.0"}
LAYOUT = max(RELEASES)
BASENAME, SIZE_LABEL = "Haru", "4B"
# The values of the card's two fields that are chosen rather than written, in the order the official template lists them.
CARD_CHOICES = {"gender": ["female", "male"], "backchannel": ["low", "mid", "high"]}

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

folder = snapshot()
config = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
text, codec, head = config["lm"], config["codec"], config["head"]
assert config["model_type"] == "haru" and text["model_type"] == "qwen3_5_text"
assert text["attn_output_gate"] and text["hidden_act"] == "silu" and not text["attention_bias"] and text["tie_word_embeddings"]
assert text["rope_parameters"]["rope_type"] == "default" and text["rope_parameters"]["mrope_interleaved"]
assert text["rope_parameters"]["partial_rotary_factor"] == text["partial_rotary_factor"]
assert text["linear_key_head_dim"] == text["linear_value_head_dim"], "ggml_gated_delta_net() takes keys and values of one width"
assert head["num_steps"] == 1, "the head's sampling is one step of the mean flow"
card_lines = open(os.path.join(folder, "README.md"), encoding="utf-8").read().split("---\n")[1].split("\n")
languages = []
for line in card_lines[card_lines.index("language:") + 1:]:
    if not line.startswith("- "):
        break
    languages.append(line[2:].strip())
assert languages and all(2 <= len(code) <= 3 and code.isalpha() and code.islower() for code in languages)
languages = sorted(languages)
tokenizer = json.load(open(os.path.join(folder, "tokenizer.json"), encoding="utf-8"))
assert tokenizer["model"]["type"] == "BPE" and tokenizer["normalizer"] == {"type": "NFC"}
pre = tokenizer["pre_tokenizer"]["pretokenizers"]
assert pre[0]["type"] == "Split" and pre[0]["behavior"] == "Isolated" and pre[1]["type"] == "ByteLevel" and not pre[1]["use_regex"]
templates = card_templates()
assert [entry["code"] for entry in templates["languages"]] == list(templates["templates"])
weights = safe_open(os.path.join(folder, "model.safetensors"), "pt")
read = set()


def checkpoint(name):
    read.add(name)
    return weights.get_tensor(name)


def f32(name):
    return checkpoint(name).to(torch.float32).numpy()


def zero_centred(name):
    """A zero-centred RMS norm's weight w, by which it scales as 1 + w; stored as 1 + w, the float32 sum it computes."""
    return np.float32(1.0) + f32(name)


path = os.path.join(args.out_dir, naming_convention(None, BASENAME, None, None, SIZE_LABEL, "f32") + ".gguf")
w = GGUFWriter(path, ARCH)
plan = []


def add_array(key, values, element):
    """An array with its element type given, not inferred from its first value."""
    w.add_key_value(key, list(values), GGUFValueType.ARRAY, sub_type=element)


def add(name, shape, make):
    """A tensor of numpy shape `shape` whose float32 values `make` gives when the tensor is written."""
    plan.append((name, tuple(int(d) for d in shape), make))


repository = f"https://huggingface.co/{MODEL['repository']}"
w.add_name("haru-4b")
w.add_organization(MODEL["repository"].split("/")[0])
w.add_basename(BASENAME)
w.add_size_label(SIZE_LABEL)
w.add_license("apache-2.0")
w.add_source_url(f"{repository}/tree/{MODEL['revision']}")
w.add_source_repo_url(repository)
w.add_file_type(LlamaFileType.ALL_F32)
w.add_languages(languages)
w.add_uint32("speech.layout", LAYOUT)
w.add_string("speech.requires", RELEASES[LAYOUT])
w.add_string("speech.task", "dialogue")
w.add_uint32("speech.sample_rate", int(config["sample_rate"]))
# A request's language chooses the wording of the character card, which steers what the model says.
w.add_string("speech.language_use", "steers")

p = f"{ARCH}."
# config.json.
w.add_uint32(p + "frame_samples", int(config["frame_samples"]))
w.add_uint32(p + "latent_dim", int(config["latent_dim"]))
w.add_uint32(p + "pad_token_id", int(config["pad_token_id"]))
w.add_uint32(p + "eop_token_id", int(config["eop_token_id"]))
w.add_uint32(p + "eos_token_id", int(text["eos_token_id"]))
# The checkpoint's buffers, by which the codec's latents are normalized for the head and the projections:
# (latent + bias) x scale.
w.add_float32(p + "latent_scale", float(checkpoint("latent_scale").float()))
w.add_float32(p + "latent_bias", float(checkpoint("latent_bias").float()))

t = p + "text."
for key in ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "intermediate_size", "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
            "linear_value_head_dim", "linear_conv_kernel_dim", "vocab_size", "max_position_embeddings"]:
    w.add_uint32(t + key, int(text[key]))
add_array(t + "layer_types", text["layer_types"], GGUFValueType.STRING)
assert set(text["layer_types"]) == {"linear_attention", "full_attention"}
w.add_float32(t + "rms_norm_eps", float(text["rms_norm_eps"]))
w.add_float32(t + "rope_theta", float(text["rope_parameters"]["rope_theta"]))
w.add_uint32(t + "rotary_dim", int(text["head_dim"] * text["rope_parameters"]["partial_rotary_factor"]))

# haru.models.haru.AudioProjector, whose RMSNorm the official code builds with this epsilon.
w.add_float32(p + "audio_in.norm_eps", 1e-6)

h = p + "head."
H = int(text["hidden_size"])
assert int(head["hidden_size"]) == H and int(head["latent_dim"]) == int(config["latent_dim"])
w.add_uint32(h + "num_layers", int(head["num_layers"]))
w.add_uint32(h + "ffn_size", int(H * float(head["ffn_ratio"])))
w.add_float32(h + "norm_eps", float(head["norm_eps"]))
w.add_float32(h + "interval_embed", float(head["interval_embed"]))
w.add_float32(h + "guidance_scale", float(head["guidance_scale"]))
w.add_float32(h + "guidance_min", float(head["guidance_range"][0]))
w.add_float32(h + "guidance_max", float(head["guidance_range"][1]))
# haru.models.meanflow_head: the sinusoidal features of a scalar, cos then sin of value x base^(-i / half) for i < half.
w.add_uint32(h + "frequency_dim", meanflow_head._FREQUENCY_DIM)
w.add_float32(h + "frequency_base", 10000.0)

c = p + "codec."
for key in ["latent_dim", "base_channels"]:
    w.add_uint32(c + key, int(codec[key]))
for key in ["ratios", "encoder_depths", "decoder_depths"]:
    add_array(c + key, [int(v) for v in codec[key]], GGUFValueType.INT32)
assert math.prod(codec["ratios"]) == int(config["frame_samples"])
w.add_float32(c + "norm_eps", float(codec["norm_eps"]))
# VibeVoiceCodec.sample(): a latent drawn about its mean as mean + s x latent_std / 0.8 x noise, s one normal draw per
# clip and noise one per value.
w.add_float32(c + "noise_std", float(codec["latent_std"]) / 0.8)
# haru.models.vibevoice_codec: the width of the stems, the heads and the blocks' depthwise convolutions, and the
# blocks' MLP, 4 x the channels.
w.add_uint32(c + "kernel", 7)
w.add_uint32(c + "ffn_ratio", 4)

s = p + "sampling."
defaults, dry = DuplexConfig(), DryConfig()
w.add_float32(s + "temperature", float(defaults.temperature))
w.add_uint32(s + "top_k", int(defaults.top_k))
w.add_float32(s + "top_p", float(defaults.top_p))
w.add_float32(s + "eop_bias", float(defaults.eop_bias))
# DRY's penalty, min(max_penalty, multiplier x base^(length - allowed_length)), is computed in double by the official
# code and rounded to float32 where it is subtracted; its constants are kept in double to give the same float.
w.add_float64(s + "dry_multiplier", float(dry.multiplier))
w.add_float64(s + "dry_base", float(dry.base))
w.add_uint32(s + "dry_allowed_length", int(dry.allowed_length))
w.add_uint32(s + "dry_lookback", int(dry.lookback))
w.add_uint32(s + "dry_max_match_length", int(dry.max_match_length))
w.add_float64(s + "dry_max_penalty", float(dry.max_penalty))

d = p + "dialogue."
# DuplexConfig: the most frames a conversation runs (60 minutes) and the most seconds of a voice reference it keeps.
w.add_uint32(d + "max_frames", int(defaults.max_frames))
w.add_float32(d + "reference_max_seconds", float(defaults.reference_seconds))
# haru.server: the shortest voice reference it takes, and the longest character card in characters.
w.add_float32(d + "reference_min_seconds", float(haru_server.REFERENCE_MIN_SECONDS))
assert haru_server.REFERENCE_MAX_SECONDS == defaults.reference_seconds
w.add_uint32(d + "card_max_chars", int(haru_server.SYSTEM_PROMPT_MAX_CHARS))
# haru.audio.trim_silence(), which prepare_reference() cuts a voice reference's leading and trailing silence with: windows
# of 1/50 s quieter than this level in dBFS, keeping this many seconds of margin.
w.add_float32(d + "silence_dbfs", float(haru_audio._SILENCE_DBFS))
w.add_float32(d + "silence_margin_seconds", float(haru_audio._SILENCE_MARGIN_SECONDS))
w.add_uint32(d + "silence_window_divisor", 50)
# haru.prompt.silence_latent(): the latent of digital silence is the last of this many frames of zeros.
w.add_uint32(d + "silence_frames", 16)
# haru.prompt.system_prompt_ids(): the card is a ChatML system turn.
w.add_string(d + "system_open", "<|im_start|>system\n")
w.add_string(d + "system_close", "<|im_end|>\n")

# haru.prompt.render_card(): the lead sentence, then each field that has a value, its header and its value on lines of
# their own; the gender and the backchannels are chosen from labels.
k = p + "card."
add_array(k + "fields", CARD_FIELDS, GGUFValueType.STRING)
add_array(k + "languages", list(templates["templates"]), GGUFValueType.STRING)
for field, values in CARD_CHOICES.items():
    add_array(k + field + "_values", values, GGUFValueType.STRING)
for code, template in templates["templates"].items():
    assert sorted(template["headers"]) == sorted(CARD_FIELDS)
    w.add_string(f"{k}{code}.lead", template["lead"])
    add_array(f"{k}{code}.headers", [template["headers"][f] for f in CARD_FIELDS], GGUFValueType.STRING)
    for field, values in CARD_CHOICES.items():
        assert sorted(template[field]) == sorted(values)
        add_array(f"{k}{code}.{field}", [template[field][v] for v in values], GGUFValueType.STRING)

z = p + "tokenizer."
vocab = tokenizer["model"]["vocab"]
added = {entry["id"]: entry for entry in tokenizer["added_tokens"]}
size = max([*vocab.values(), *added]) + 1
tokens = [""] * size
for token, id_ in vocab.items():
    tokens[id_] = token
for id_, entry in added.items():
    assert not tokens[id_], f"the added token {entry['content']} takes the id of a token of the vocabulary"
    tokens[id_] = entry["content"]
assert all(tokens) and size <= int(text["vocab_size"])
add_array(z + "tokens", tokens, GGUFValueType.STRING)
add_array(z + "merges", [m if isinstance(m, str) else " ".join(m) for m in tokenizer["model"]["merges"]], GGUFValueType.STRING)
# Every added token, special or not, which the tokenizers library finds in a text before its pre-tokenizer runs.
assert all(not (e["lstrip"] or e["rstrip"] or e["single_word"] or e["normalized"]) for e in added.values()), \
    "an added token strips space, matches whole words or matches the normalized text, which the C++ does not"
add_array(z + "added_ids", sorted(added), GGUFValueType.INT32)
w.add_string(z + "pre_tokenizer", pre[0]["pattern"]["Regex"])

# The language model.
m = "lm."
V = int(text["vocab_size"])
heads, kv, hd = int(text["num_attention_heads"]), int(text["num_key_value_heads"]), int(text["head_dim"])
I = int(text["intermediate_size"])
silent = [int(config["pad_token_id"]), int(config["eop_token_id"])]
add("token_embd", (V, H), lambda: f32(m + "embed_tokens.weight"))
add("output_silent", (len(silent), H), lambda: f32(m + "embed_tokens.weight")[silent])
add("output_norm", (H,), lambda: zero_centred(m + "norm.weight"))
for i, kind in enumerate(text["layer_types"]):
    s_, o = f"{m}layers.{i}.", f"blk.{i}."
    add(o + "attn_norm", (H,), lambda s_=s_: zero_centred(s_ + "input_layernorm.weight"))
    add(o + "post_attn_norm", (H,), lambda s_=s_: zero_centred(s_ + "post_attention_layernorm.weight"))
    if kind == "full_attention":
        a_ = s_ + "self_attn."
        add(o + "attn_q", (heads * hd, H), lambda a_=a_: f32(a_ + "q_proj.weight").reshape(heads, 2, hd, H)[:, 0].reshape(-1, H))
        add(o + "attn_gate", (heads * hd, H), lambda a_=a_: f32(a_ + "q_proj.weight").reshape(heads, 2, hd, H)[:, 1].reshape(-1, H))
        for part, name in (("attn_k", "k_proj"), ("attn_v", "v_proj")):
            add(o + part, (kv * hd, H), lambda a_=a_, name=name: f32(a_ + name + ".weight"))
        add(o + "attn_output", (H, heads * hd), lambda a_=a_: f32(a_ + "o_proj.weight"))
        add(o + "attn_q_norm", (hd,), lambda a_=a_: zero_centred(a_ + "q_norm.weight"))
        add(o + "attn_k_norm", (hd,), lambda a_=a_: zero_centred(a_ + "k_norm.weight"))
    else:
        l_ = s_ + "linear_attn."
        nk, nv = int(text["linear_num_key_heads"]), int(text["linear_num_value_heads"])
        dk, dv = int(text["linear_key_head_dim"]), int(text["linear_value_head_dim"])
        conv = 2 * nk * dk + nv * dv
        add(o + "ssm_in_qkv", (conv, H), lambda l_=l_: f32(l_ + "in_proj_qkv.weight"))
        add(o + "ssm_in_z", (nv * dv, H), lambda l_=l_: f32(l_ + "in_proj_z.weight"))
        add(o + "ssm_in_b", (nv, H), lambda l_=l_: f32(l_ + "in_proj_b.weight"))
        add(o + "ssm_in_a", (nv, H), lambda l_=l_: f32(l_ + "in_proj_a.weight"))
        add(o + "ssm_conv1d", (conv, int(text["linear_conv_kernel_dim"])), lambda l_=l_: f32(l_ + "conv1d.weight")[:, 0, :])
        add(o + "ssm_dt_bias", (nv,), lambda l_=l_: f32(l_ + "dt_bias"))
        add(o + "ssm_a_log", (nv,), lambda l_=l_: f32(l_ + "A_log"))
        add(o + "ssm_norm", (dv,), lambda l_=l_: f32(l_ + "norm.weight"))
        add(o + "ssm_out", (H, nv * dv), lambda l_=l_: f32(l_ + "out_proj.weight"))
    x = s_ + "mlp."
    add(o + "ffn_gate", (I, H), lambda x=x: f32(x + "gate_proj.weight"))
    add(o + "ffn_up", (I, H), lambda x=x: f32(x + "up_proj.weight"))
    add(o + "ffn_down", (H, I), lambda x=x: f32(x + "down_proj.weight"))

# The projections of the audio latents into the language model.
L = int(config["latent_dim"])
for side in ("model", "user"):
    src, dst = f"{side}_audio_proj.", f"audio_in.{side}."
    add(dst + "fc1.weight", (H, L), lambda src=src: f32(src + "fc1.weight"))
    add(dst + "fc1.bias", (H,), lambda src=src: f32(src + "fc1.bias"))
    add(dst + "norm", (H,), lambda src=src: f32(src + "norm.weight"))
    add(dst + "fc2.weight", (H, H), lambda src=src: f32(src + "fc2.weight"))
    add(dst + "fc2.bias", (H,), lambda src=src: f32(src + "fc2.bias"))

# The MeanFlow head.
F = int(H * float(head["ffn_ratio"]))
add("head.cond_proj", (H, H), lambda: f32("head.cond_proj.weight"))
add("head.latent_proj", (H, L), lambda: f32("head.latent_proj.weight"))
for embed in ("t_embed", "dt_embed", "h_embed", "omega_embed"):
    add(f"head.{embed}.fc1", (H, meanflow_head._FREQUENCY_DIM), lambda e=embed: f32(f"head.{e}.fc1.weight"))
    add(f"head.{embed}.fc2", (H, H), lambda e=embed: f32(f"head.{e}.fc2.weight"))
for i in range(int(head["num_layers"])):
    s_, o = f"head.layers.{i}.", f"head.blk.{i}."
    add(o + "norm", (H,), lambda s_=s_: f32(s_ + "norm.weight"))
    add(o + "ada", (3 * H, H), lambda s_=s_: f32(s_ + "ada.weight"))
    add(o + "ffn_gate", (F, H), lambda s_=s_: f32(s_ + "ffn.gate_proj.weight"))
    add(o + "ffn_up", (F, H), lambda s_=s_: f32(s_ + "ffn.up_proj.weight"))
    add(o + "ffn_down", (H, F), lambda s_=s_: f32(s_ + "ffn.down_proj.weight"))
add("head.final.ada", (2 * H, H), lambda: f32("head.final.ada.weight"))
add("head.final.proj", (L, H), lambda: f32("head.final.proj.weight"))


# The codec.
def conv_weight(name):
    """A convolution's weight [out, in, k] as numpy [k, out, in]."""
    return np.ascontiguousarray(f32(name).transpose(2, 0, 1))


def depthwise_weight(name):
    """A depthwise convolution's weight [channels, 1, k] as numpy [k, channels]."""
    return np.ascontiguousarray(f32(name)[:, 0, :].T)


def transposed_weight(name):
    """A transposed convolution's weight [in, out, k] as numpy [k, out, in]."""
    return np.ascontiguousarray(f32(name).transpose(2, 1, 0))


def halves(name, stride, half):
    """Half `half` of a strided convolution of width 2s, [out, in, 2s], as [out, s * in] with the input channel fastest."""
    weight = f32(name)
    out, inp, width = weight.shape
    assert width == 2 * stride
    return np.ascontiguousarray(weight.transpose(0, 2, 1).reshape(out, 2, stride * inp)[:, half])


def conv(src, dst, inp, out, kernel):
    add(dst + ".weight", (kernel, out, inp), lambda: conv_weight(src + ".weight"))
    add(dst + ".bias", (out,), lambda: f32(src + ".bias"))


def block(src, dst, ch):
    add(dst + ".norm", (ch,), lambda: f32(src + ".norm.weight"))
    add(dst + ".conv.weight", (7, ch), lambda: depthwise_weight(src + ".conv.weight"))
    add(dst + ".conv.bias", (ch,), lambda: f32(src + ".conv.bias"))
    add(dst + ".gamma", (ch,), lambda: f32(src + ".gamma"))
    add(dst + ".ffn_norm", (ch,), lambda: f32(src + ".ffn_norm.weight"))
    add(dst + ".fc1.weight", (4 * ch, ch), lambda: f32(src + ".fc1.weight"))
    add(dst + ".fc1.bias", (4 * ch,), lambda: f32(src + ".fc1.bias"))
    add(dst + ".fc2.weight", (ch, 4 * ch), lambda: f32(src + ".fc2.weight"))
    add(dst + ".fc2.bias", (ch,), lambda: f32(src + ".fc2.bias"))
    add(dst + ".ffn_gamma", (ch,), lambda: f32(src + ".ffn_gamma"))


base, ratios = int(codec["base_channels"]), [int(r) for r in codec["ratios"]]
n = len(ratios)
strides = ratios[::-1]
conv("codec.encoder.stem", "codec.enc.stem", 1, base, 7)
for i, depth in enumerate(codec["encoder_depths"]):
    if i > 0:
        src, dst, st = f"codec.encoder.down.{i - 1}", f"codec.enc.down.{i - 1}", strides[i - 1]
        inp, out = base << (i - 1), base << i
        add(dst + ".first", (out, st * inp), lambda src=src, st=st: halves(src + ".weight", st, 0))
        add(dst + ".second", (out, st * inp), lambda src=src, st=st: halves(src + ".weight", st, 1))
        add(dst + ".bias", (out,), lambda src=src: f32(src + ".bias"))
    for j in range(int(depth)):
        block(f"codec.encoder.stages.{i}.{j}", f"codec.enc.blk.{i}.{j}", base << i)
conv("codec.encoder.head", "codec.enc.head", base << n, L, 7)
conv("codec.decoder.stem", "codec.dec.stem", L, base << n, 7)
for i, depth in enumerate(codec["decoder_depths"]):
    if i > 0:
        src, dst = f"codec.decoder.up.{i - 1}", f"codec.dec.up.{i - 1}"
        inp, out, st = base << (n - i + 1), base << (n - i), ratios[i - 1]
        add(dst + ".weight", (2 * st, out, inp), lambda src=src: transposed_weight(src + ".weight"))
        add(dst + ".bias", (out,), lambda src=src: f32(src + ".bias"))
    for j in range(int(depth)):
        block(f"codec.decoder.stages.{i}.{j}", f"codec.dec.blk.{i}.{j}", base << (n - i))
conv("codec.decoder.head", "codec.dec.head", base, 1, 7)
names = [name for name, _, _ in plan]
assert len(names) == len(set(names)), "a tensor is written twice"

# Every tensor's info first, so that the values are written a tensor at a time.
for name, shape, _ in plan:
    w.add_tensor_info(name, shape, np.dtype(np.float32), int(np.prod(shape)) * 4)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for name, shape, make in plan:
    values = np.ascontiguousarray(make(), dtype=np.float32)
    assert values.shape == shape, (name, values.shape, shape)
    w.write_tensor_data(values)
w.close()

unread = set(weights.keys()) - read
assert not unread, f"the conversion leaves out {sorted(unread)[:5]}"
print(f"{path}: layout {LAYOUT}, {len(plan)} tensors, code {CODE['repository']}@{CODE['commit'][:7]}")
