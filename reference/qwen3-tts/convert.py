"""Converts a pinned official Qwen3-TTS 12Hz CustomVoice checkpoint to the one GGUF file the C++ port reads.

usage: uv run python convert.py <0.6b|1.7b> <out dir>

Writes Qwen3-TTS-12Hz-<0.6B|1.7B>-CustomVoice-F32.gguf, named under GGUF's naming convention, in layout 1: the talker,
the code predictor, the text embedding, the tokenizer and the 12Hz codec's decoder, every tensor in float32, with every
constant the C++ reads and the model's identity in the GGUF specification's general keys. `speech quantize` makes the
other weight types of it, each tensor in the type src/families/qwen3-tts/layout.cpp gives it.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis. A Linear weight [out, in] is stored as is
(ne = [in, out]). Every convolution weight is stored as numpy [k, out, in] (ne = [in, out, k]), so that tap k is
a plain [in, out] matrix: the C++ side runs a convolution as one matrix product per tap on channel-first
activations. A depthwise weight [c, 1, k] is stored as [k, c].
"""

import argparse
import inspect
import json
import os
import re

import numpy as np
import yaml
from gguf import GGUFValueType, GGUFWriter, LlamaFileType, naming_convention
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from safetensors import safe_open

from pins import MODELS, snapshot

ARCH = "qwen3-tts"
# Each layout this converter has written, with the first release of speech.cpp whose reader takes it; it writes the
# last.
RELEASES = {1: "0.7.0"}
LAYOUT = max(RELEASES)

# The ISO 639 two-letter code of each language the checkpoint names, which requests give as a BCP 47 tag. Its dialects
# have none: a dialect is spoken only by its speakers, when the language is Chinese or left to the model.
LANGUAGE_TAGS = {"chinese": "zh", "english": "en", "french": "fr", "german": "de", "italian": "it",
                 "japanese": "ja", "korean": "ko", "portuguese": "pt", "russian": "ru", "spanish": "es"}
# The SPDX identifier of each license a model card names.
LICENSES = {"apache-2.0": "Apache-2.0"}
# The parts of the models' names under GGUF's naming convention (ggml's docs/gguf.md) besides the size: 12Hz, the
# codec's frame rate, belongs to the line, and CustomVoice is what the line was fine-tuned toward.
BASENAME, FINETUNE = "Qwen3-TTS-12Hz", "CustomVoice"

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

pin = MODELS[args.model]
model_dir = snapshot(pin)
config = json.load(open(os.path.join(model_dir, "config.json")))
generation = json.load(open(os.path.join(model_dir, "generation_config.json")))
talker_cfg = config["talker_config"]
cp_cfg = talker_cfg["code_predictor_config"]
codec_dir = os.path.join(model_dir, "speech_tokenizer")
codec_top = json.load(open(os.path.join(codec_dir, "config.json")))
codec_cfg = codec_top["decoder_config"]
size_label = {"0b6": "0.6B", "1b7": "1.7B"}[config["tts_model_size"]]
assert size_label.lower() == args.model
model_name = pin["repository"].split("/")[1]
assert model_name == f"{BASENAME}-{size_label}-{FINETUNE}", f"{model_name} is not named {BASENAME}, its size and {FINETUNE}"
assert config["tts_model_type"] == "custom_voice", "only CustomVoice checkpoints are supported"
assert config["tokenizer_type"] == "qwen3_tts_tokenizer_12hz"
assert codec_cfg["num_key_value_heads"] == codec_cfg["num_attention_heads"], "the C++ runs a codec with as many key/value heads as heads"
assert codec_cfg["num_quantizers"] == talker_cfg["num_code_groups"]

# The constants the official generate() (Qwen3TTSForConditionalGeneration.generate() in
# qwen_tts/core/models/modeling_qwen3_tts.py, at the commit pyproject.toml pins) writes in its code rather than in the
# checkpoint: at least two frames before the end of speech, the last 1024 ids of the talker's vocabulary never sampled
# but the end of speech, and a voice's dialect taken when the language is Chinese or auto.
official_generate = inspect.getsource(Qwen3TTSForConditionalGeneration.generate)
MIN_FRAMES = 2
SUPPRESSED_TOKENS = 1024
DIALECT_LANGUAGE = "chinese"
assert f'"min_new_tokens": {MIN_FRAMES}' in official_generate
assert f"vocab_size - {SUPPRESSED_TOKENS}, " in official_generate
assert f'language.lower() in ["{DIALECT_LANGUAGE}", "auto"]' in official_generate

# The model card: its license in the front matter, and the table of the speakers with the description and the
# native language of each.
card = open(os.path.join(model_dir, "README.md"), encoding="utf-8").read()
license_id = yaml.safe_load(card.split("---\n")[1])["license"]
voice_rows = {}
lines = card.splitlines()
header = next(i for i, line in enumerate(lines) if re.fullmatch(r"\|\s*Speaker\s*\|\s*Voice Description\s*\|\s*Native language\s*\|",
                                                                 line.strip(), re.IGNORECASE))
for line in lines[header + 2:]:
    if not line.startswith("|"):
        break
    speaker, description, native = [cell.strip() for cell in line.strip().strip("|").split("|")]
    genders = set(re.findall(r"\b(female|male)\b", description.lower()))
    assert len(genders) == 1, f"the description of {speaker} names no single gender: {description}"
    # A dialect's speaker is listed as "Chinese (Beijing Dialect)"; the voice's language is the language of its dialect.
    voice_rows[speaker.lower()] = (LANGUAGE_TAGS[native.split("(")[0].strip().lower()], genders.pop(), description)


def load(path):
    f = safe_open(path, "pt")
    return {k: f.get_tensor(k).float().numpy() for k in f.keys()}


path = os.path.join(args.out_dir, naming_convention(None, BASENAME, FINETUNE, None, size_label, "f32") + ".gguf")
w = GGUFWriter(path, ARCH)


def add_array(key, values, element):
    """An array with its element type given, not inferred from its first value."""
    w.add_key_value(key, list(values), GGUFValueType.ARRAY, sub_type=element)


languages = talker_cfg["codec_language_id"]
dialects = {d for d in talker_cfg["spk_is_dialect"].values() if d}
unknown = [language for language in languages if language not in LANGUAGE_TAGS and language not in dialects]
assert not unknown, f"no ISO 639 code for the languages {unknown}"
tags = sorted((LANGUAGE_TAGS[language], language) for language in languages if language in LANGUAGE_TAGS)

repository = f"https://huggingface.co/{pin['repository']}"
w.add_name(model_name)
w.add_organization(pin["repository"].split("/")[0])
w.add_basename(BASENAME)
w.add_size_label(size_label)
w.add_finetune(FINETUNE)
w.add_license(LICENSES[license_id])
w.add_source_url(f"{repository}/tree/{pin['revision']}")
w.add_source_repo_url(repository)
w.add_file_type(LlamaFileType.ALL_F32)
w.add_languages([tag for tag, _ in tags])
w.add_uint32("speech.layout", LAYOUT)
w.add_string("speech.requires", RELEASES[LAYOUT])
w.add_string("speech.task", "synthesis")
w.add_uint32("speech.sample_rate", int(codec_top["output_sample_rate"]))
w.add_string("speech.language_use", "steers")
voices = sorted(talker_cfg["spk_id"])
assert sorted(voice_rows) == voices, f"the model card lists the speakers {sorted(voice_rows)}, the checkpoint {voices}"
add_array("speech.voices", voices, GGUFValueType.STRING)
add_array("speech.voice_languages", [voice_rows[v][0] for v in voices], GGUFValueType.STRING)
add_array("speech.voice_genders", [voice_rows[v][1] for v in voices], GGUFValueType.STRING)
add_array("speech.voice_descriptions", [voice_rows[v][2] for v in voices], GGUFValueType.STRING)

p = f"{ARCH}."
for prefix, cfg in (("talker", talker_cfg), ("code_predictor", cp_cfg)):
    for key in ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "head_dim", "vocab_size"]:
        w.add_uint32(f"{p}{prefix}.{key}", int(cfg[key]))
    w.add_float32(f"{p}{prefix}.rms_norm_eps", float(cfg["rms_norm_eps"]))
    w.add_float32(f"{p}{prefix}.rope_theta", float(cfg["rope_theta"]))
for key in ["num_code_groups", "max_position_embeddings", "codec_bos_id", "codec_eos_token_id", "codec_pad_id",
            "codec_think_id", "codec_nothink_id", "codec_think_bos_id", "codec_think_eos_id"]:
    w.add_uint32(f"{p}talker.{key}", int(talker_cfg[key]))
w.add_uint32(f"{p}talker.suppressed_tokens", SUPPRESSED_TOKENS)
for key in ["tts_bos_token_id", "tts_eos_token_id", "tts_pad_token_id", "im_start_token_id", "im_end_token_id",
            "assistant_token_id"]:
    w.add_uint32(f"{p}text.{key}", int(config[key]))
vocab = json.load(open(os.path.join(model_dir, "vocab.json"), encoding="utf-8"))
# The chat template's "\n", as the byte-level BPE spells it.
w.add_uint32(f"{p}text.newline_token_id", int(vocab["Ċ"]))

add_array(f"{p}language_ids", [int(languages[name]) for _, name in tags], GGUFValueType.INT32)
add_array(f"{p}speaker_ids", [int(talker_cfg["spk_id"][v]) for v in voices], GGUFValueType.INT32)
dialect_of = talker_cfg["spk_is_dialect"]
add_array(f"{p}dialect_ids", [int(languages[dialect_of[v]]) if dialect_of[v] else -1 for v in voices], GGUFValueType.INT32)
w.add_string(f"{p}dialect_language", LANGUAGE_TAGS[DIALECT_LANGUAGE])

w.add_uint32(f"{p}generation.min_frames", MIN_FRAMES)
w.add_uint32(f"{p}generation.max_frames", int(generation["max_new_tokens"]))
for stack, (do_sample, temperature, top_k, top_p) in (
        ("talker", ("do_sample", "temperature", "top_k", "top_p")),
        ("code_predictor", ("subtalker_dosample", "subtalker_temperature", "subtalker_top_k", "subtalker_top_p"))):
    g = f"{p}generation.{stack}."
    w.add_bool(g + "do_sample", bool(generation[do_sample]))
    w.add_float32(g + "temperature", float(generation[temperature]))
    w.add_uint32(g + "top_k", int(generation[top_k]))
    w.add_float32(g + "top_p", float(generation[top_p]))
w.add_float32(f"{p}generation.talker.repetition_penalty", float(generation["repetition_penalty"]))
# The official code passes the code predictor no repetition penalty, so its generate() takes its configuration's.
w.add_float32(f"{p}generation.code_predictor.repetition_penalty", float(cp_cfg["repetition_penalty"]))

# The Qwen2 byte-level BPE: tokens in id order, with the added tokens, and merges in rank order.
tok_cfg = json.load(open(os.path.join(model_dir, "tokenizer_config.json"), encoding="utf-8"))
added = {int(i): v["content"] for i, v in tok_cfg["added_tokens_decoder"].items()}
n_tokens = max(max(vocab.values()), max(added)) + 1
tokens = [""] * n_tokens
for t, i in vocab.items():
    tokens[i] = t
for i, t in added.items():
    tokens[i] = t
merges = [line.rstrip("\n") for line in open(os.path.join(model_dir, "merges.txt"), encoding="utf-8")
          if line.strip() and not line.startswith("#version")]
add_array(f"{p}tokenizer.tokens", tokens, GGUFValueType.STRING)
add_array(f"{p}tokenizer.merges", merges, GGUFValueType.STRING)

for key in ["num_quantizers", "latent_dim", "codebook_dim", "hidden_size", "num_attention_heads", "num_key_value_heads",
            "head_dim", "num_hidden_layers", "sliding_window"]:
    w.add_uint32(f"{p}codec.{key}", int(codec_cfg[key]))
w.add_float32(f"{p}codec.rms_norm_eps", float(codec_cfg["rms_norm_eps"]))
w.add_float32(f"{p}codec.rope_theta", float(codec_cfg["rope_theta"]))
add_array(f"{p}codec.upsample_rates", [int(r) for r in codec_cfg["upsample_rates"]], GGUFValueType.INT32)
add_array(f"{p}codec.upsampling_ratios", [int(r) for r in codec_cfg["upsampling_ratios"]], GGUFValueType.INT32)


def add(name, data):
    w.add_tensor(name, np.ascontiguousarray(data, dtype=np.float32))


# ---------------------------------------------------------------- talker
weights = load(os.path.join(model_dir, "model.safetensors"))
t = "talker."
add("talker.text_embd", weights[t + "model.text_embedding.weight"])
add("talker.text_proj.fc1.weight", weights[t + "text_projection.linear_fc1.weight"])
add("talker.text_proj.fc1.bias", weights[t + "text_projection.linear_fc1.bias"])
add("talker.text_proj.fc2.weight", weights[t + "text_projection.linear_fc2.weight"])
add("talker.text_proj.fc2.bias", weights[t + "text_projection.linear_fc2.bias"])
add("talker.codec_embd", weights[t + "model.codec_embedding.weight"])
add("talker.codec_head", weights[t + "codec_head.weight"])
add("talker.norm", weights[t + "model.norm.weight"])


def add_layers(prefix_in, prefix_out, n):
    for i in range(n):
        a = f"{prefix_in}.layers.{i}."
        b = f"{prefix_out}.blk.{i}."
        add(b + "attn_norm", weights[a + "input_layernorm.weight"])
        add(b + "ffn_norm", weights[a + "post_attention_layernorm.weight"])
        add(b + "attn_q", weights[a + "self_attn.q_proj.weight"])
        add(b + "attn_k", weights[a + "self_attn.k_proj.weight"])
        add(b + "attn_v", weights[a + "self_attn.v_proj.weight"])
        add(b + "attn_o", weights[a + "self_attn.o_proj.weight"])
        add(b + "attn_q_norm", weights[a + "self_attn.q_norm.weight"])
        add(b + "attn_k_norm", weights[a + "self_attn.k_norm.weight"])
        add(b + "ffn_gate", weights[a + "mlp.gate_proj.weight"])
        add(b + "ffn_up", weights[a + "mlp.up_proj.weight"])
        add(b + "ffn_down", weights[a + "mlp.down_proj.weight"])


add_layers(t + "model", "talker", talker_cfg["num_hidden_layers"])
c = t + "code_predictor."
add_layers(c + "model", "cp", cp_cfg["num_hidden_layers"])
add("cp.norm", weights[c + "model.norm.weight"])
for i in range(talker_cfg["num_code_groups"] - 1):
    add(f"cp.codec_embd.{i}", weights[c + f"model.codec_embedding.{i}.weight"])
    add(f"cp.head.{i}", weights[c + f"lm_head.{i}.weight"])
# The official model makes small_to_mtp_projection a Linear exactly when the two widths differ, and an identity
# otherwise.
projected = cp_cfg["hidden_size"] != talker_cfg["hidden_size"]
assert (c + "small_to_mtp_projection.weight" in weights) == projected
if projected:
    add("cp.in_proj.weight", weights[c + "small_to_mtp_projection.weight"])
    add("cp.in_proj.bias", weights[c + "small_to_mtp_projection.bias"])

# ---------------------------------------------------------------- codec decoder
cw = {k[len("decoder."):]: v for k, v in load(os.path.join(codec_dir, "model.safetensors")).items()
      if k.startswith("decoder.")}


def codebook(prefix):
    usage = cw[prefix + "._codebook.cluster_usage"]
    return cw[prefix + "._codebook.embedding_sum"] / np.maximum(usage, 1e-5)[:, None]


# The first quantizer and the other fifteen each project 256 -> 512 with their own 1x1 conv.
add("codec.vq.first.codebook.0", codebook("quantizer.rvq_first.vq.layers.0"))
add("codec.vq.first.out_proj", cw["quantizer.rvq_first.output_proj.weight"][:, :, 0])
for i in range(codec_cfg["num_quantizers"] - 1):
    add(f"codec.vq.rest.codebook.{i}", codebook(f"quantizer.rvq_rest.vq.layers.{i}"))
add("codec.vq.rest.out_proj", cw["quantizer.rvq_rest.output_proj.weight"][:, :, 0])


def conv(name, weight):
    """Conv1d weight [out, in, k] -> [k, out, in]."""
    add(name, np.transpose(weight, (2, 0, 1)))


def tconv(name, weight):
    """ConvTranspose1d weight [in, out, k] -> [k, out, in]."""
    add(name, np.transpose(weight, (2, 1, 0)))


conv("codec.pre_conv.weight", cw["pre_conv.conv.weight"])
add("codec.pre_conv.bias", cw["pre_conv.conv.bias"])

tf = "pre_transformer."
add("codec.tf.in_proj.weight", cw[tf + "input_proj.weight"])
add("codec.tf.in_proj.bias", cw[tf + "input_proj.bias"])
add("codec.tf.out_proj.weight", cw[tf + "output_proj.weight"])
add("codec.tf.out_proj.bias", cw[tf + "output_proj.bias"])
add("codec.tf.norm", cw[tf + "norm.weight"])
for i in range(codec_cfg["num_hidden_layers"]):
    a = f"{tf}layers.{i}."
    b = f"codec.tf.blk.{i}."
    add(b + "attn_norm", cw[a + "input_layernorm.weight"])
    add(b + "ffn_norm", cw[a + "post_attention_layernorm.weight"])
    add(b + "attn_q", cw[a + "self_attn.q_proj.weight"])
    add(b + "attn_k", cw[a + "self_attn.k_proj.weight"])
    add(b + "attn_v", cw[a + "self_attn.v_proj.weight"])
    add(b + "attn_o", cw[a + "self_attn.o_proj.weight"])
    add(b + "attn_scale", cw[a + "self_attn_layer_scale.scale"])
    add(b + "ffn_gate", cw[a + "mlp.gate_proj.weight"])
    add(b + "ffn_up", cw[a + "mlp.up_proj.weight"])
    add(b + "ffn_down", cw[a + "mlp.down_proj.weight"])
    add(b + "ffn_scale", cw[a + "mlp_layer_scale.scale"])

for i in range(len(codec_cfg["upsampling_ratios"])):
    a = f"upsample.{i}."
    b = f"codec.up.{i}."
    tconv(b + "tconv.weight", cw[a + "0.conv.weight"])
    add(b + "tconv.bias", cw[a + "0.conv.bias"])
    add(b + "dwconv.weight", np.transpose(cw[a + "1.dwconv.conv.weight"][:, 0, :]))
    add(b + "dwconv.bias", cw[a + "1.dwconv.conv.bias"])
    add(b + "norm.weight", cw[a + "1.norm.weight"])
    add(b + "norm.bias", cw[a + "1.norm.bias"])
    add(b + "pw1.weight", cw[a + "1.pwconv1.weight"])
    add(b + "pw1.bias", cw[a + "1.pwconv1.bias"])
    add(b + "pw2.weight", cw[a + "1.pwconv2.weight"])
    add(b + "pw2.bias", cw[a + "1.pwconv2.bias"])
    add(b + "gamma", cw[a + "1.gamma"])


def snake(prefix, name):
    """SnakeBeta: x + inv_beta * sin(alpha * x)^2, with alpha = exp(a) and inv_beta = 1 / (exp(b) + 1e-9)."""
    add(name + ".alpha", np.exp(cw[prefix + ".alpha"]))
    add(name + ".inv_beta", 1.0 / (np.exp(cw[prefix + ".beta"]) + 1e-9))


conv("codec.dec.in_conv.weight", cw["decoder.0.conv.weight"])
add("codec.dec.in_conv.bias", cw["decoder.0.conv.bias"])
n_blocks = len(codec_cfg["upsample_rates"])
for i in range(n_blocks):
    a = f"decoder.{i + 1}.block."
    b = f"codec.dec.blk.{i}."
    snake(a + "0", b + "snake")
    tconv(b + "tconv.weight", cw[a + "1.conv.weight"])
    add(b + "tconv.bias", cw[a + "1.conv.bias"])
    # The three residual units of dilations 1, 3 and 9, which the C++ runs as the architecture's.
    for j in range(3):
        r = f"{a}{j + 2}."
        s = f"{b}res.{j}."
        snake(r + "act1", s + "snake1")
        conv(s + "conv1.weight", cw[r + "conv1.conv.weight"])
        add(s + "conv1.bias", cw[r + "conv1.conv.bias"])
        snake(r + "act2", s + "snake2")
        conv(s + "conv2.weight", cw[r + "conv2.conv.weight"])
        add(s + "conv2.bias", cw[r + "conv2.conv.bias"])
snake(f"decoder.{n_blocks + 1}", "codec.dec.out_snake")
conv("codec.dec.out_conv.weight", cw[f"decoder.{n_blocks + 2}.conv.weight"])
add("codec.dec.out_conv.bias", cw[f"decoder.{n_blocks + 2}.conv.bias"])

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", path)
