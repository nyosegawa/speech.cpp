"""Converts a pinned Qwen3-ASR checkpoint to the one GGUF file the C++ port reads.

usage: uv run python convert.py <Qwen3-ASR-0.6B|Qwen3-ASR-1.7B> <out dir> [--type f32|f16|q8_0]

Writes Qwen3-ASR-<0.6B|1.7B>-<F32|F16|Q8_0>.gguf, named under GGUF's naming convention, in layout 1: the frontend's
window and mel filterbank, the encoder's convolutions and layers, the projector, the decoder and its tokenizer, with
every constant the C++ reads and the model's identity in the GGUF specification's general keys. The constants come
from the checkpoint's config.json, preprocessor_config.json, generation_config.json, chat_template.json and tokenizer
files, and where it has none from the official code docs/adr/0018 takes as the reference: transformers 5.18's
Qwen3-ASR (the feature extractor, the encoder and the language tags) and qwen-asr 0.0.6's inference/utils.py (the
limits of the audio, the prompt's forced language, and the parse of the output), both pinned by uv.lock.

--type applies to the matrices of the encoder's and the decoder's linear layers, the projector and the token
embeddings, which are also the decoder's output matrix; the convolution kernels are float16 in an F16 or Q8_0 file and
float32 in an F32 one, and the norms, the biases and the frontend stay float32. The encoder holds in Q8_0: with the
decoder's weights of the Q8_0 file, teacher-forced on the dumps' ids of every input of up to 30 s (requests auto and
auto-prompt), its output from a Q8_0 file moved the argmax of 1 or 2 of 605 steps with the 0.6B model and of none of 604
with the 1.7B, on the CPU and on Metal of an Apple M5, where the official encoder's output moved 2 and none
(2026-10-06); each step it moved had a margin of 0.13 or less in the official model.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis: a Linear weight [out, in] is stored as is (ne = [in, out])
and a Conv2d weight [out, in, height, width] as is (ne = [width, height, in, out]), the layout ggml's 2D convolutions
take, with the mel axis as the height and time as the width.
"""

import argparse
import inspect
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGML_QUANT_VERSION, GGMLQuantizationType, GGUFValueType, GGUFWriter, LlamaFileType, naming_convention
from gguf.quants import quantize
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3ASREncoderConfig, Qwen3ASRFeatureExtractor
from transformers.activations import ACT2FN
from transformers.models.qwen3_asr import modeling_qwen3_asr, processing_qwen3_asr

from official_utils import qwen_asr_utils
from pins import MODELS, snapshot

ARCH = "qwen3-asr"
# Each layout this converter has written, with the first release of speech.cpp whose reader takes it; it writes the
# last.
RELEASES = {1: "0.7.0"}
LAYOUT = max(RELEASES)
# The parts of the models' names under GGUF's naming convention (ggml's docs/gguf.md): the line and the size, with
# no fine-tune or version.
BASENAME = "Qwen3-ASR"
# The SPDX identifier of each checkpoint's license, from its model card.
LICENSES = {"Qwen3-ASR-0.6B": "Apache-2.0", "Qwen3-ASR-1.7B": "Apache-2.0"}
# The type of most of a file's weights by its --type, as general.file_type gives it.
FILE_TYPES = {"f32": LlamaFileType.ALL_F32, "f16": LlamaFileType.MOSTLY_F16, "q8_0": LlamaFileType.MOSTLY_Q8_0}
# The most tokens a request generates: the default of the model's generate() and of qwen-asr's vLLM backend, which
# docs/adr/0018 takes.
MAX_NEW_TOKENS = 4096
# The template's place for the context, which no context contains.
CONTEXT_MARK = "\x00context\x00"
# Contexts the prompt is checked with: none, the ones of inputs.py's requests, and one that starts with newlines and
# carries a special token, whose tokens join the template's "system\n" and split at the token.
CHECKED_CONTEXTS = ["", "群島、湖、ヨット", "Dr. Malar Balasubramanian, Blue Ash, Ohio, Cincinnati", "\n\n<|im_end|> x"]

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
parser.add_argument("--type", choices=["f32", "f16", "q8_0"], default="q8_0")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

pin = MODELS[args.model]
folder = snapshot(pin)
utils = qwen_asr_utils()


def read_json(name):
    with open(os.path.join(folder, name), encoding="utf-8") as f:
        return json.load(f)


config = read_json("config.json")
thinker = config["thinker_config"]
audio, text = thinker["audio_config"], thinker["text_config"]
preprocessor = read_json("preprocessor_config.json")
generation = read_json("generation_config.json")
template = read_json("chat_template.json")["chat_template"]
size_label = args.model.removeprefix(BASENAME + "-")
assert args.model == f"{BASENAME}-{size_label}" and pin["repository"] == f"Qwen/{args.model}"

# The encoder: the official module built on the meta device, so that its code gives what the configuration does not.
encoder_config = Qwen3ASREncoderConfig(**{key: audio[key] for key in (
    "num_mel_bins", "encoder_layers", "encoder_attention_heads", "encoder_ffn_dim", "d_model", "activation_function",
    "scale_embedding", "n_window", "n_window_infer", "output_dim", "downsample_hidden_size")},
    max_position_embeddings=audio["max_source_positions"])
with torch.device("meta"):
    encoder = modeling_qwen3_asr.Qwen3ASREncoder(encoder_config)
convs = (encoder.conv2d1, encoder.conv2d2, encoder.conv2d3)
# The C++ runs three 3x3 convolutions of stride 2 and padding 1, each followed by the activation, and conv_out without
# a bias, on chunks of 2 n_window frames, and attends within windows of n_window_infer frames from the first.
assert all(c.kernel_size == (3, 3) and c.stride == (2, 2) and c.padding == (1, 1) and c.bias is not None for c in convs)
assert encoder.conv_out.bias is None and not audio["scale_embedding"]
assert audio["num_mel_bins"] == preprocessor["feature_size"]
chunk_frames, window_frames = 2 * audio["n_window"], audio["n_window_infer"]
assert window_frames % chunk_frames == 0
# torch's erf GELU, which the C++ runs as ggml_gelu_erf.
probe = torch.linspace(-6, 6, 1001)
assert torch.equal(ACT2FN[audio["activation_function"]](probe), F.gelu(probe))
eps = {encoder.ln_post.eps} | {n.eps for layer in encoder.layers for n in (layer.self_attn_layer_norm, layer.final_layer_norm)}
assert len(eps) == 1
[norm_eps] = eps
# The positions of a chunk's tokens restart at 0 in every chunk, so a chunk's tokens must fit the table.
chunk_tokens = chunk_frames
for _ in convs:
    chunk_tokens = (chunk_tokens - 1) // 2 + 1
assert chunk_tokens <= audio["max_source_positions"]
max_timescale = encoder.positional_embedding.max_timescale
# The projector maps the encoder's width to the decoder's.
assert audio["output_dim"] == text["hidden_size"]

# The decoder: the Qwen3 block of src/common/qwen3-decoder.cpp with its output tied to the token embeddings.
# qwen-asr runs interleaved multimodal RoPE, but a prompt of audio and text gives each of its sections the same
# position, so it is the plain rotate-half RoPE the C++ runs.
assert not text["attention_bias"] and text["hidden_act"] == "silu" and text["tie_word_embeddings"]
assert text["rope_scaling"]["rope_type"] == "default" and sum(text["rope_scaling"]["mrope_section"]) * 2 == text["head_dim"]

# The frontend: transformers' feature extractor with the checkpoint's parameters, whose code the C++ ports: a centred
# STFT with reflected padding and a periodic Hann window, the power of every frame but the last through a Slaney mel
# filterbank, the log10 of it floored at a guard, floored again at the utterance's maximum less a range, and shifted and
# scaled. An utterance shorter than min_length samples is padded with zeros to it.
assert preprocessor["feature_extractor_type"] == "WhisperFeatureExtractor" and preprocessor["dither"] == 0
assert preprocessor["padding_value"] == 0
extractor = Qwen3ASRFeatureExtractor(
    feature_size=preprocessor["feature_size"], sampling_rate=utils.SAMPLE_RATE, hop_length=preprocessor["hop_length"],
    n_fft=preprocessor["n_fft"], padding_value=preprocessor["padding_value"], dither=preprocessor["dither"],
    n_window=audio["n_window"])
LOG_FLOOR, DYNAMIC_RANGE, LOG_OFFSET, LOG_DIVISOR = 1e-10, 8.0, 4.0, 4.0
extract = inspect.getsource(Qwen3ASRFeatureExtractor._torch_extract_fbank_features)
for code in ("torch.hann_window(self.n_fft", "stft[..., :-1].abs() ** 2", f"torch.clamp(mel_spec, min={LOG_FLOOR}).log10()",
             f"log_spec.max() - {DYNAMIC_RANGE}", f"(log_spec + {LOG_OFFSET}) / {LOG_DIVISOR}"):
    assert code in extract, f"the feature extractor no longer computes {code}"
window = torch.hann_window(extractor.n_fft).numpy()
filterbank = extractor.mel_filters.T.astype(np.float32)
assert filterbank.shape == (extractor.feature_size, extractor.n_fft // 2 + 1)
# qwen-asr pads a part of a split shorter than MIN_ASR_INPUT_SECONDS to it, as transformers pads any utterance.
min_samples = int(utils.MIN_ASR_INPUT_SECONDS * utils.SAMPLE_RATE)
assert extractor.min_length == min_samples

# The limits of the audio and the split of long audio (split_audio_into_chunks()), in samples as qwen-asr counts them.
split = inspect.signature(utils.split_audio_into_chunks).parameters
max_samples = int(utils.MAX_ASR_INPUT_SECONDS * utils.SAMPLE_RATE)
split_search_samples = int(split["search_expand_sec"].default * utils.SAMPLE_RATE)
split_window_samples = int((split["min_window_ms"].default / 1000.0) * utils.SAMPLE_RATE)
assert "win = max(4, int((min_window_ms / 1000.0) * sr))" in inspect.getsource(utils.split_audio_into_chunks)
assert split_window_samples >= 4

# The parse of the output: detect_and_fix_repetitions() keeps one of a run of more than `threshold` of one character
# and one of `threshold` or more repeats of a pattern of up to `max_len` characters.
repetitions = inspect.getsource(utils.detect_and_fix_repetitions)
repetition_threshold = inspect.signature(utils.detect_and_fix_repetitions).parameters["threshold"].default
[repetition_max_period] = [int(n) for n in re.findall(r"def fix_pattern_repeats\(s, thresh, max_len=(\d+)\)", repetitions)]
assert "text = fix_pattern_repeats(text, threshold)" in repetitions
# The text of the output is what follows <asr_text>, stripped, whether the model wrote a language or None for audio
# without speech, and the whole output when the language was forced; the C++ parses it so, and reports no language.
for raw, forced, parsed in (("language None<asr_text>", None, ""), ("language None<asr_text> x ", None, "x"),
                            (" language Japanese<asr_text>\u3000x\n", None, "x"), ("x<asr_text>y", "Japanese", "x<asr_text>y")):
    assert utils.parse_asr_output(raw, user_language=forced)[1] == parsed, raw

# Greedy decoding to MAX_NEW_TOKENS, stopping at any of the generation configuration's end tokens.
assert not generation["do_sample"] and generation.get("repetition_penalty", 1.0) == 1.0
with open(os.path.join(os.path.dirname(utils.__file__), "qwen3_asr.py"), encoding="utf-8") as f:
    assert f"max_new_tokens: Optional[int] = {MAX_NEW_TOKENS}," in f.read(), "qwen-asr's vLLM backend has another default"
eos_ids = list(generation["eos_token_id"])

# The languages: transformers' tags of the names the checkpoint supports, which qwen-asr's forced language writes.
# general.languages holds each language's shortest ISO 639 code, as BCP 47 names it (docs/adr/0015): the ISO 639-1 code
# of two letters, and for Cantonese and Filipino, which have none, the three letters of ISO 639-3 (yue) and 639-2
# (fil).
names = processing_qwen3_asr.LANGUAGE_CODE_TO_NAME
assert sorted(names.values()) == sorted(config["support_languages"]) == sorted(utils.SUPPORTED_LANGUAGES)
tags = sorted(names)
assert all(re.fullmatch("[a-z]{2,3}", tag) for tag in tags)
assert sorted(names[tag] for tag in tags if len(tag) == 3) == ["Cantonese", "Filipino"], "a language has a code longer than its shortest"

# The tokenizer: the Qwen2 byte-level BPE with the added tokens, which encoding splits text at and decoding skips when
# they are special.
tokenizer = AutoTokenizer.from_pretrained(folder)
tokenizer_config = read_json("tokenizer_config.json")
added = {int(i): token for i, token in tokenizer_config["added_tokens_decoder"].items()}
vocab = read_json("vocab.json")
tokens = [""] * (max(max(vocab.values()), max(added)) + 1)
for token, i in vocab.items():
    tokens[i] = token
for i, token in added.items():
    tokens[i] = token["content"]
assert all(tokens) and not any(token["lstrip"] or token["rstrip"] or token["single_word"] for token in added.values())
with open(os.path.join(folder, "merges.txt"), encoding="utf-8") as f:
    merges = [line.rstrip("\n") for line in f if line.strip() and not line.startswith("#version")]
added_ids = sorted(added)
special_ids = sorted(i for i, token in added.items() if token["special"])
assert tokenizer.decode(added_ids, skip_special_tokens=True) == "".join(tokens[i] for i in added_ids if i not in special_ids)
assert len(tokens) <= text["vocab_size"] and all(i < len(tokens) for i in eos_ids)

# The prompt: the chat template as qwen-asr fills it (_build_messages()), around the context and the audio, and the
# forced language after it. Its text splits into the template's pieces before the context, between the context and
# the audio tokens, and after them; the C++ tokenizes the text before the audio and the text after it, the forced
# language included, each splitting at the added tokens, which gives the official ids of the whole when the audio's
# tokens border on added tokens.
messages = [{"role": "system", "content": CONTEXT_MARK}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
filled = tokenizer.apply_chat_template(messages, chat_template=template, add_generation_prompt=True, tokenize=False)
audio_token = tokenizer.audio_token
before_context, rest = filled.split(CONTEXT_MARK)
before_audio, after_audio = rest.split(audio_token)
assert before_audio.endswith(tokenizer.audio_bos_token) and after_audio.startswith(tokenizer.audio_eos_token)
# The audio's tokens are the added token whose rows of the embeddings the projector's output replaces.
assert tokenizer.audio_token_id == thinker["audio_token_id"] and tokens[thinker["audio_token_id"]] == audio_token
asr_text = utils._ASR_TEXT_TAG
assert asr_text in [tokens[i] for i in added_ids]


def ids(piece):
    return tokenizer(piece)["input_ids"]


for context in CHECKED_CONTEXTS:
    for name in [None] + [names[tag] for tag in tags]:
        forced = f"{utils._LANG_PREFIX}{name}{asr_text}" if name else ""
        whole = ids(before_context + context + before_audio + audio_token * 3 + after_audio + forced)
        assert whole == ids(before_context + context + before_audio) + [tokenizer.audio_token_id] * 3 + ids(after_audio + forced)

weights = {}
for name in pin["weights"]:
    f = safe_open(os.path.join(folder, name), "pt")
    weights.update((key, f) for key in f.keys())


def weight(key):
    return weights.pop(key).get_tensor(key).float().numpy()


path = os.path.join(args.out_dir, naming_convention(None, BASENAME, None, None, size_label, args.type) + ".gguf")
# Tensors go to a temporary file as they are added, so that the 1.7B model in float32 is never held whole.
w = GGUFWriter(path, ARCH, use_temp_file=True)


def add_array(key, values, element):
    """An array with its element type given, not inferred from its first value."""
    w.add_key_value(key, list(values), GGUFValueType.ARRAY, sub_type=element)


repository = f"https://huggingface.co/{pin['repository']}"
w.add_name(args.model)
w.add_organization(pin["repository"].split("/")[0])
w.add_basename(BASENAME)
w.add_size_label(size_label)
w.add_license(LICENSES[args.model])
w.add_source_url(f"{repository}/tree/{pin['revision']}")
w.add_source_repo_url(repository)
w.add_file_type(FILE_TYPES[args.type])
if args.type == "q8_0":
    w.add_quantization_version(GGML_QUANT_VERSION)
w.add_languages(tags)
w.add_uint32("speech.layout", LAYOUT)
w.add_string("speech.requires", RELEASES[LAYOUT])
w.add_string("speech.task", "recognition")
w.add_uint32("speech.sample_rate", utils.SAMPLE_RATE)
w.add_string("speech.language_use", "steers")

p = f"{ARCH}."
add_array(p + "language_names", [names[tag] for tag in tags], GGUFValueType.STRING)
w.add_uint32(p + "frontend.n_fft", extractor.n_fft)
w.add_uint32(p + "frontend.hop_length", extractor.hop_length)
w.add_uint32(p + "frontend.n_mels", extractor.feature_size)
w.add_float32(p + "frontend.log_floor", LOG_FLOOR)
w.add_float32(p + "frontend.dynamic_range", DYNAMIC_RANGE)
w.add_float32(p + "frontend.log_offset", LOG_OFFSET)
w.add_float32(p + "frontend.log_divisor", LOG_DIVISOR)
w.add_uint32(p + "audio.min_samples", min_samples)
w.add_uint32(p + "audio.max_samples", max_samples)
w.add_uint32(p + "audio.split_search_samples", split_search_samples)
w.add_uint32(p + "audio.split_window_samples", split_window_samples)
w.add_uint32(p + "encoder.d_model", audio["d_model"])
w.add_uint32(p + "encoder.num_layers", audio["encoder_layers"])
w.add_uint32(p + "encoder.num_heads", audio["encoder_attention_heads"])
w.add_uint32(p + "encoder.ffn_dim", audio["encoder_ffn_dim"])
w.add_uint32(p + "encoder.chunk_frames", chunk_frames)
w.add_uint32(p + "encoder.window_frames", window_frames)
w.add_float32(p + "encoder.norm_eps", norm_eps)
w.add_float32(p + "encoder.max_timescale", max_timescale)
for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "vocab_size", "max_position_embeddings"):
    w.add_uint32(p + "decoder." + key, text[key])
w.add_float32(p + "decoder.rms_norm_eps", text["rms_norm_eps"])
w.add_float32(p + "decoder.rope_theta", text["rope_theta"])
w.add_string(p + "prompt.before_context", before_context)
w.add_string(p + "prompt.before_audio", before_audio)
w.add_string(p + "prompt.after_audio", after_audio)
w.add_string(p + "prompt.audio_token", audio_token)
w.add_string(p + "prompt.language_prefix", utils._LANG_PREFIX)
w.add_string(p + "prompt.asr_text", asr_text)
w.add_uint32(p + "output.repetition_threshold", repetition_threshold)
w.add_uint32(p + "output.repetition_max_period", repetition_max_period)
add_array(p + "generation.eos_ids", eos_ids, GGUFValueType.INT32)
w.add_uint32(p + "generation.max_new_tokens", MAX_NEW_TOKENS)
add_array(p + "tokenizer.tokens", tokens, GGUFValueType.STRING)
add_array(p + "tokenizer.merges", merges, GGUFValueType.STRING)
add_array(p + "tokenizer.added_ids", added_ids, GGUFValueType.INT32)
add_array(p + "tokenizer.special_ids", special_ids, GGUFValueType.INT32)

conv_type = np.float32 if args.type == "f32" else np.float16


def add(name, data, kind):
    """kind: 'matrix' is stored in the requested type, 'conv' as float16 or float32 by it, 'f32' as float32."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    if kind == "matrix" and args.type == "q8_0":
        assert data.ndim == 2 and data.shape[-1] % 32 == 0, f"{name}'s rows are no whole Q8_0 blocks"
        w.add_tensor(name, quantize(data, GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0)
    elif kind == "matrix" and args.type == "f16":
        w.add_tensor(name, data.astype(np.float16))
    elif kind == "conv":
        w.add_tensor(name, data.astype(conv_type))
    else:
        w.add_tensor(name, data)


add("frontend.window", window, "f32")
add("frontend.filterbank", filterbank, "f32")

a = "thinker.audio_tower."
for i in (1, 2, 3):
    add(f"enc.conv.{i}.weight", weight(f"{a}conv2d{i}.weight"), "conv")
    add(f"enc.conv.{i}.bias", weight(f"{a}conv2d{i}.bias"), "f32")
add("enc.conv_out.weight", weight(a + "conv_out.weight"), "matrix")
for i in range(audio["encoder_layers"]):
    s, d = f"{a}layers.{i}.", f"enc.blk.{i}."
    for source, target, kind in (("self_attn_layer_norm", "attn_norm", "f32"), ("self_attn.q_proj", "attn_q", "matrix"),
                                 ("self_attn.k_proj", "attn_k", "matrix"), ("self_attn.v_proj", "attn_v", "matrix"),
                                 ("self_attn.out_proj", "attn_out", "matrix"), ("final_layer_norm", "ffn_norm", "f32"),
                                 ("fc1", "ffn_up", "matrix"), ("fc2", "ffn_down", "matrix")):
        add(d + target + ".weight", weight(s + source + ".weight"), kind)
        add(d + target + ".bias", weight(s + source + ".bias"), "f32")
add("enc.norm.weight", weight(a + "ln_post.weight"), "f32")
add("enc.norm.bias", weight(a + "ln_post.bias"), "f32")
for i in (1, 2):
    add(f"proj.{i}.weight", weight(f"{a}proj{i}.weight"), "matrix")
    add(f"proj.{i}.bias", weight(f"{a}proj{i}.bias"), "f32")

t = "thinker.model."
# The checkpoint stores the tied output matrix a second time, as lm_head.
embeddings = weight(t + "embed_tokens.weight")
assert np.array_equal(embeddings, weight("thinker.lm_head.weight"))
add("dec.token_embd", embeddings, "matrix")
del embeddings
for i in range(text["num_hidden_layers"]):
    s, d = f"{t}layers.{i}.", f"dec.blk.{i}."
    for source, target, kind in (("input_layernorm", "attn_norm", "f32"), ("post_attention_layernorm", "ffn_norm", "f32"),
                                 ("self_attn.q_proj", "attn_q", "matrix"), ("self_attn.k_proj", "attn_k", "matrix"),
                                 ("self_attn.v_proj", "attn_v", "matrix"), ("self_attn.o_proj", "attn_o", "matrix"),
                                 ("self_attn.q_norm", "attn_q_norm", "f32"), ("self_attn.k_norm", "attn_k_norm", "f32"),
                                 ("mlp.gate_proj", "ffn_gate", "matrix"), ("mlp.up_proj", "ffn_up", "matrix"),
                                 ("mlp.down_proj", "ffn_down", "matrix")):
        add(d + target, weight(s + source + ".weight"), kind)
add("dec.norm", weight(t + "norm.weight"), "f32")
assert not weights, f"the checkpoint holds tensors the converter does not write: {sorted(weights)[:5]}"

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", path)
