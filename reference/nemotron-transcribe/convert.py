"""Converts the pinned Nemotron 3.5 Transcribe checkpoint to the one GGUF file the C++ port reads.

usage: uv run python convert.py <out dir>

Writes NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-F32.gguf, named under GGUF's naming convention, in layout 1, about 136 GB,
a tensor at a time, so that the conversion holds one tensor of the language model in memory at most:

- the perception module: the frontend's window and mel filterbank, the ASR encoder (the feature stacking's projection
  and its Transformer layers, which normalize each head's queries and keys and rotate half of each head's channels), the
  speaker branch (Nemotron 3 Diarization's encoder and head, its silence embedding and the settings of its speaker
  cache), the windows the parallel expert encoder takes long audio in, the fusion's norms with its sinusoidal speaker
  kernel, and the projection to the language model's width;
- the language model, Nemotron 3.5 Lightning: its embeddings, its Mamba-2, attention and MoE layers, the final norm, the
  head, and the multi-token prediction layers;
- the tokenizer, the prompt the official vLLM plugin renders, and the generation's settings;
- the CTC timestamp head: its decoder, the vocabulary and the SentencePiece model it aligns words with, and the settings
  of its alignment, as the checkpoint holds them.

The constants come from the checkpoint's config.json, llm_backbone/config.json, generation_config.json, tokenizer files
and the safetensors' metadata, read through the official modules official.py builds; where the checkpoint has none,
from the official code (NeMo Speech's ParallelExpertEncoder, its nemotron3p5 prompt format and its vLLM plugin's prompt)
or from the model card, which says how the model is asked to transcribe and how its output is cleaned.

Two parts of the checkpoint are left out, since no inference reads them: the speaker branch's own preprocessor, which is
asserted to equal the perception module's, whose features both branches take, and its hidden_to_spks head, which
SortformerModules builds and never applies.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis: a Linear weight [out, in] is stored as is (ne = [in,
out]). A layer's experts are stored as one tensor [experts, out, in], which ggml_mul_mat_id() reads; a Mamba layer's
convolution [channels, 1, width] as [channels, width] and its A as -exp(A_log), which ggml_ssm_scan() takes, both as
the official code computes them in float32. The upsampling Conv1d of the speaker head is stored as reference/sortformer
stores it; the timestamp head's ConvTranspose1d [in, out, k], whose stride is its width, as [k, out, in], so that each
tap is a plain matrix, and its output Conv1d of width 1 as a matrix. Every tensor is written in float32, the checkpoint's BF16 and F32 values exactly; `speech quantize` makes the
other weight types of the file, each tensor in the type src/families/nemotron-transcribe/layout.cpp gives it.
"""

import argparse
import json
import math
import os
import re

import numpy as np
import torch
from gguf import GGUFValueType, GGUFWriter, LlamaFileType, naming_convention
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

import official
from pins import MODELS, snapshot

ARCH = "nemotron-transcribe"
# Each layout this converter has written, with the first release of speech.cpp whose reader takes it; it writes the
# last.
RELEASES = {1: "0.9.1"}
LAYOUT = max(RELEASES)
MODEL = "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
BASENAME = "NVIDIA-Nemotron-3.5-Transcribe"
SIZE_LABEL = "30B-A4B"
# OpenMDW 1.1, the model card's license, which SPDX's list does not yet hold.
LICENSE = "LicenseRef-OpenMDW-1.1"
# The model card's languages (its metadata's language list), ISO 639-1 codes all.
LANGUAGES = ["ar", "bg", "bn", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "he", "hi", "hr", "hu", "id", "it",
             "ja", "ko", "lt", "lv", "mr", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "ta", "te", "th", "tl",
             "tr", "uk", "vi", "zh"]
# The model card's prompts for transcription, which it asks in the user's turn with no system turn: verbatim
# transcription, its recommended prompt and the one of its English, multilingual and long-form results, and contextual
# transcription, whose terms follow "Context terms: ".
INSTRUCTION = ("Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. "
               "Formatting: punctuation and capitalization. Output: transcript only.")
CONTEXT_BEFORE = "Transcribe with punctuation and capitalization. Context terms: "
CONTEXT_AFTER = "."
# The model card's context length, the --max-model-len of its vLLM server, and the longest recording its server takes,
# VLLM_MAX_AUDIO_DECODE_DURATION_S of the same command.
CONTEXT_LENGTH = 65536
MAX_AUDIO_SECONDS = 3600
# The model card's generation-loop guardrail, which its reported transcriptions apply to the decoded text: a character
# repeated 21 times or more is kept once (CHAR_RUN = (.)\1{20,}), then a block of 1 to 4096 characters repeated 20 times
# or more (BLOCK_RUN = (.{1,4096}?)\1{19,}).
CHAR_RUN = 21
BLOCK_RUN = 20
BLOCK_MAX_CHARS = 4096
# Instructions the prompt's tokens are checked with: both of the card's, and texts that end in spaces and newlines,
# carry a special token, or hold Japanese, whose tokens might join the text around them.
CHECKED_TEXTS = [INSTRUCTION, CONTEXT_BEFORE + "FLEURS, Nemotron" + CONTEXT_AFTER, "", "  x\n\n", "<|im_end|> x",
                 "日本語で書き起こしてください。", "Context terms: 群島、湖、ヨット."]

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
torch.set_grad_enabled(False)

pin = MODELS[MODEL]
folder = snapshot(pin)


def read_json(name):
    with open(os.path.join(folder, name), encoding="utf-8") as f:
        return json.load(f)


config = read_json("config.json")
generation = read_json("generation_config.json")
llm_config = AutoConfig.from_pretrained(os.path.join(folder, "llm_backbone"))
model = official.load(folder, with_language_model=False)
perception = model.perception
pee = perception.encoder
asr = pee.asr_encoder
diar = pee.diarization_model
sm = diar.sortformer_modules
featurizer = perception.preprocessor.featurizer
sd = {k: v.detach().float().numpy() for k, v in perception.state_dict().items()}

# The frontend: NeMo's mel spectrogram without its own normalization, which the parallel expert encoder applies itself,
# per feature for the ASR branch and none for the speaker branch.
assert featurizer.exact_pad is False and featurizer.frame_splicing == 1 and featurizer.mag_power == 2.0
assert featurizer.normalize is None and featurizer.log and featurizer.log_zero_guard_type == "add"
assert featurizer.pad_value == 0 and featurizer.preemph is not None and featurizer.pad_to == 0
assert pee.asr_normalize_type == "per_feature" and pee.diar_normalize_type in (None, "NA")
for name in ("featurizer.window", "featurizer.fb"):
    a, b = sd["preprocessor." + name], sd["encoder.diarization_model.preprocessor." + name]
    assert np.array_equal(a, b), f"the speaker branch's {name} differs from the perception module's"
diar_featurizer = diar.preprocessor.featurizer
for attribute in ("sample_rate", "n_fft", "hop_length", "win_length", "nfilt", "preemph", "log_zero_guard_value", "mag_power"):
    assert getattr(diar_featurizer, attribute) == getattr(featurizer, attribute), attribute

# The ASR encoder: NeMo's Transformer encoder with the feature stacking, queries and keys normalized per head, and the
# first half of each head's channels rotated.
assert type(asr).__name__ == "TransformerEncoder" and type(asr.pre_encode).__name__ == "FeatureStacking"
assert asr.self_attention_model == "rope" and asr.attn_mode == "full" and asr.xscale is None and asr.out_proj is None
assert type(asr.embed_norm).__name__ == "LayerNorm" and asr.pre_encode.proj.bias is None
layer = asr.layers[0]
assert layer.attn.qk_norm and layer.attn.w_qkv.bias is None and layer.attn.out_proj.bias is not None
assert type(layer.ffn.net[1]).__name__ == "GELU" and layer.ffn.net[1].approximate == "none"
asr_rope = asr.pos_enc
asr_rotary = int(asr_rope.d_k_rot)
assert asr_rotary % 2 == 0 and asr_rotary <= asr_rope.d_k == asr.d_model // asr.n_heads

# The speaker branch: Nemotron 3 Diarization as reference/sortformer converts it.
encoder = diar.encoder
assert type(encoder).__name__ == "TransformerEncoder" and type(encoder.pre_encode).__name__ == "FeatureStacking"
assert encoder.self_attention_model == "rope" and encoder.attn_mode == "full" and encoder.xscale is None
assert encoder.pos_enc.d_k_rot == encoder.pos_enc.d_k and encoder.out_proj is None and diar.transformer_encoder is None
assert not encoder.layers[0].attn.qk_norm and encoder.layers[0].attn.w_qkv.bias is None
assert diar.high_resolution and diar.upsample_factor == encoder.subsampling_factor == asr.subsampling_factor
assert sm.use_learnable_sil_emb and sm.encoder_proj is not None and sm.subpixel_upsample.kernel_size == (3,)
assert diar.streaming_mode and not diar.async_streaming
eps = {m.eps for m in perception.modules() if type(m).__name__ == "LayerNorm"}
assert len(eps) == 1, eps
norm_eps = eps.pop()

# The fusion: speaker probabilities over speaker_activity_threshold taken as 1 and the rest as 0, normalized over the
# speakers and turned into the ASR encoder's width by the sinusoidal kernel, added to the normalized ASR states.
assert pee.speaker_feature_mode == "thresholded" and pee.n_spk == sm.n_spk
kernel = pee.diar_kernel.float().numpy()
assert np.array_equal(kernel, pee._build_sinusoid_position_encoding(pee.n_spk, pee.asr_d_model).float().numpy())
assert type(perception.modality_adapter).__name__ == "IdentityConnector" and perception.rote is None

# The language model. Its layers' types are read as llm_backbone/config.json names them: Transformers' configuration
# renames them (mamba to linear_attention, attention to full_attention).
backbone = read_json("llm_backbone/config.json")
types = list(backbone["layers_block_type"])
assert set(types) == {"mamba", "moe", "attention"} and len(types) == llm_config.num_hidden_layers
assert llm_config.mamba_hidden_act == "silu" and llm_config.mlp_hidden_act == "relu2" and not llm_config.mlp_bias
assert not llm_config.attention_bias and not llm_config.use_bias and llm_config.use_conv_bias
assert llm_config.n_routed_experts % llm_config.n_group == 0 and llm_config.moe_latent_size is None
assert not llm_config.tie_word_embeddings and llm_config.n_shared_experts == 1
assert tuple(llm_config.time_step_limit) == (0.0, math.inf), "a time step limit the scan would clamp to"
assert config["mtp"]["use_repeated_layer"] and config["mtp"]["hybrid_override_pattern"] == "*E"
mtp_types = list(backbone["mtp_layers_block_type"])
assert mtp_types == ["attention", "moe"] and llm_config.num_nextn_predict_layers == 1

# The tokenizer: a byte-level BPE whose pre-tokenizer splits by one pattern, with no normalizer, and whose merges a word
# of the vocabulary skips (ignore_merges).
tokenizer = AutoTokenizer.from_pretrained(folder)
backend = json.loads(tokenizer.backend_tokenizer.to_str())
assert backend["normalizer"] is None and backend["model"]["type"] == "BPE" and backend["model"]["byte_fallback"] is False
pre = backend["pre_tokenizer"]
assert pre["type"] == "Sequence" and [p["type"] for p in pre["pretokenizers"]] == ["Split", "ByteLevel"]
assert pre["pretokenizers"][0]["behavior"] == "Isolated" and not pre["pretokenizers"][0]["invert"]
assert pre["pretokenizers"][1] == {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}
vocab = backend["model"]["vocab"]
added = {entry["id"]: entry for entry in backend["added_tokens"]}
tokens = [""] * (max([*vocab.values(), *added]) + 1)
for token, i in vocab.items():
    tokens[i] = token
for i, entry in added.items():
    assert tokens[i] in ("", entry["content"]), f"the added token {entry['content']} takes the id of another token"
    tokens[i] = entry["content"]
# The tokenizer's last added token, after the language model's vocabulary, is the audio locator, whose id the audio's
# rows replace and no embedding has.
assert all(tokens) and len(tokens) == llm_config.vocab_size + 1
assert all(not (e["lstrip"] or e["rstrip"] or e["single_word"] or e["normalized"]) for e in added.values()), \
    "an added token strips space, matches whole words or matches the normalized text, which the C++ does not"
special_ids = sorted(i for i, e in added.items() if e["special"])

# The prompt, as NVIDIA's vLLM plugin renders a request (vllm/salm/runtime_compat.py): the user's text, one space and the
# audio locator in the user's turn, with no system turn, formatted by the checkpoint's nemotron3p5 format with reasoning
# off. The locator's one row the audio's rows replace; the C++ tokenizes the text before it and the text after it, each
# splitting at the added tokens.
locator = model.audio_locator_tag
locator_id = model.audio_locator_tag_id
assert locator == config["audio_locator_tag"] and locator_id == llm_config.vocab_size and tokens[locator_id] == locator
assert model.text_eos_id == generation["eos_token_id"][0]
from nemo.collections.common.prompts import PromptFormatter

formatter = PromptFormatter.resolve(model.cfg.prompt_format)(model.tokenizer)
mark = "\x00text\x00"
rendered = model.tokenizer.tokenizer.apply_chat_template([{"role": "user", "content": f"{mark} {locator}"}],
                                                         add_generation_prompt=True, enable_thinking=False, tokenize=False)
before_text, rest = rendered.split(mark)
before_audio, after_audio = rest.split(locator)
assert before_audio == " "


def ids(piece):
    return model.tokenizer.tokenizer(piece, add_special_tokens=False)["input_ids"]


for text in CHECKED_TEXTS:
    turns = [{"role": "user", "content": f"{text} {locator}"}]
    # encode_dialog() rewrites the turns it is given, adding the system turn and renaming content to slots.
    whole = formatter.encode_dialog(turns=[dict(t) for t in turns], enable_thinking=False)["input_ids"].tolist()
    template = model.tokenizer.tokenizer.apply_chat_template(turns, add_generation_prompt=True, enable_thinking=False,
                                                             tokenize=True, return_dict=False)
    assert whole == list(template), f"the nemotron3p5 format and the chat template tokenize {text!r} apart"
    assert whole == ids(before_text + text + before_audio) + [locator_id] + ids(after_audio), \
        f"the prompt of {text!r} does not split at the locator"

eos_ids = list(generation["eos_token_id"])
assert all(i in special_ids for i in eos_ids)

# The CTC timestamp head, as the checkpoint's metadata configures it.
head, ctc_decoder, ctc_tokenizer = official.timestamp_head(folder)
with safe_open(os.path.join(folder, official.WEIGHTS), framework="pt") as f:
    keys = list(f.keys())
    model_proto = f.get_tensor("ctc_timestamp.tokenizer.model_proto").numpy()
assert model_proto.dtype == np.uint8 and model_proto.ndim == 1 and model_proto.size
ctc_alignment = config["ctc_timestamps"]["alignment_config"]
assert ctc_decoder == config["ctc_timestamps"]["decoder_config"] and ctc_tokenizer == config["ctc_timestamps"]["tokenizer_config"]
assert int(head.upsample_factor) == ctc_decoder["upsampling_factor"]
assert ctc_decoder["use_transformer"] and ctc_decoder["upsampling_mode"] == "conv_transpose" and ctc_decoder["residual"]
assert not ctc_decoder["learnable_residual_scale"] and ctc_decoder["self_attention_model"] == "rope"
assert ctc_decoder["attn_mode"] == "full" and ctc_decoder["qk_norm"] and not ctc_decoder["qkv_bias"]
assert len(ctc_decoder["vocabulary"]) == ctc_decoder["num_classes"]

path = os.path.join(args.out_dir, naming_convention(None, BASENAME, None, None, SIZE_LABEL, "f32") + ".gguf")
w = GGUFWriter(path, ARCH)
plan = []


def add_array(key, values, element):
    """An array with its element type given, not inferred from its first value."""
    w.add_key_value(key, list(values), GGUFValueType.ARRAY, sub_type=element)


def add(name, shape, make):
    """A tensor of numpy shape `shape` whose float32 values `make` gives when the tensor is written."""
    plan.append((name, tuple(int(d) for d in shape), make))


repository = f"https://huggingface.co/{pin['repository']}"
w.add_name(BASENAME + "-" + SIZE_LABEL)
w.add_organization(pin["repository"].split("/")[0])
w.add_basename(BASENAME)
w.add_size_label(SIZE_LABEL)
w.add_license(LICENSE)
w.add_source_url(f"{repository}/tree/{pin['revision']}")
w.add_source_repo_url(repository)
w.add_file_type(LlamaFileType.ALL_F32)
w.add_languages(LANGUAGES)
w.add_uint32("speech.layout", LAYOUT)
w.add_string("speech.requires", RELEASES[LAYOUT])
w.add_string("speech.task", "recognition")
w.add_uint32("speech.sample_rate", int(featurizer.sample_rate))
# The model is asked to transcribe the language as spoken; a request's language is only checked against the model's.
w.add_string("speech.language_use", "checked")

p = f"{ARCH}."
w.add_uint32(p + "frontend.n_fft", int(featurizer.n_fft))
w.add_uint32(p + "frontend.hop_length", int(featurizer.hop_length))
w.add_uint32(p + "frontend.n_mels", int(featurizer.nfilt))
w.add_float32(p + "frontend.preemphasis", float(featurizer.preemph))
w.add_float32(p + "frontend.log_guard", float(featurizer.log_zero_guard_value))
# normalize_batch()'s guard of the standard deviation of a per-feature normalization (CONSTANT in features.py).
from nemo.collections.asr.parts.preprocessing import features as nemo_features

w.add_float32(p + "frontend.std_guard", float(nemo_features.CONSTANT))
w.add_uint32(p + "audio.max_samples", MAX_AUDIO_SECONDS * int(featurizer.sample_rate))

e = p + "asr."
w.add_uint32(e + "d_model", int(asr.d_model))
w.add_uint32(e + "num_layers", len(asr.layers))
w.add_uint32(e + "num_heads", int(asr.n_heads))
w.add_uint32(e + "subsampling_factor", int(asr.subsampling_factor))
w.add_uint32(e + "rotary_dim", asr_rotary)
w.add_float32(e + "rope_base", float(asr_rope.rope_base))
w.add_float32(e + "norm_eps", float(norm_eps))
# The windows the parallel expert encoder takes audio in when it generates, in 80 ms frames: a core, with the context
# it hears on either side, which both branches take.
w.add_uint32(e + "window_frames", int(pee.online_inference_length))
w.add_uint32(e + "window_left", int(pee.chunk_left_context))
w.add_uint32(e + "window_right", int(pee.chunk_right_context))

s = p + "speaker."
w.add_uint32(s + "d_model", int(encoder.d_model))
w.add_uint32(s + "num_layers", int(encoder.n_layers))
w.add_uint32(s + "num_heads", int(encoder.n_heads))
w.add_float32(s + "rope_base", float(encoder.pos_enc.rope_base))
w.add_uint32(s + "speakers", int(sm.n_spk))
w.add_uint32(s + "cache.silence_frames", int(sm.spkcache_sil_frames_per_spk))
w.add_float32(s + "cache.score_threshold", float(sm.pred_score_threshold))
w.add_float32(s + "cache.boost_latest", float(sm.scores_boost_latest))
w.add_float32(s + "cache.strong_boost_rate", float(sm.strong_boost_rate))
w.add_float32(s + "cache.weak_boost_rate", float(sm.weak_boost_rate))
w.add_float32(s + "cache.min_pos_scores_rate", float(sm.min_pos_scores_rate))
# The streaming the encoder runs the speaker branch in, a step per window (ParallelExpertEncoder._init_streaming_diar()).
w.add_uint32(s + "streaming.spkcache_len", int(pee.diar_spkcache_len))
w.add_uint32(s + "streaming.fifo_len", int(pee.diar_fifo_len))
w.add_uint32(s + "streaming.spkcache_update_period", int(pee.diar_spkcache_update_period))
w.add_float32(s + "activity_threshold", float(pee.speaker_activity_threshold))
w.add_float32(s + "kernel_scale", float(pee.spk_kernel_scale))

m = p + "llm."
for key in ["hidden_size", "vocab_size", "mamba_num_heads", "mamba_head_dim", "n_groups", "ssm_state_size", "conv_kernel",
            "num_attention_heads", "num_key_value_heads", "head_dim", "n_routed_experts", "num_experts_per_tok",
            "moe_intermediate_size", "moe_shared_expert_intermediate_size", "n_group", "topk_group"]:
    w.add_uint32(m + key, int(getattr(llm_config, key)))
add_array(m + "layer_types", types, GGUFValueType.STRING)
w.add_float32(m + "norm_eps", float(llm_config.norm_eps))
w.add_float32(m + "layer_norm_epsilon", float(llm_config.layer_norm_epsilon))
w.add_bool(m + "norm_topk_prob", bool(llm_config.norm_topk_prob))
w.add_float32(m + "routed_scaling_factor", float(llm_config.routed_scaling_factor))
w.add_uint32(m + "context_length", CONTEXT_LENGTH)
# The multi-token prediction head: one layer of each type of mtp_layers_block_type, run mtp.num_nextn_predict_layers
# times, each time on its own last output.
add_array(m + "mtp.layer_types", mtp_types, GGUFValueType.STRING)
w.add_uint32(m + "mtp.depth", int(config["mtp"]["num_nextn_predict_layers"]))

k = p + "tokenizer."
add_array(k + "tokens", tokens, GGUFValueType.STRING)
add_array(k + "merges", [x if isinstance(x, str) else " ".join(x) for x in backend["model"]["merges"]], GGUFValueType.STRING)
add_array(k + "added_ids", sorted(added), GGUFValueType.INT32)
add_array(k + "special_ids", special_ids, GGUFValueType.INT32)
w.add_string(k + "pre_tokenizer", pre["pretokenizers"][0]["pattern"]["Regex"])
w.add_bool(k + "ignore_merges", bool(backend["model"]["ignore_merges"]))

q = p + "prompt."
w.add_string(q + "before_text", before_text)
w.add_string(q + "before_audio", before_audio)
w.add_string(q + "after_audio", after_audio)
w.add_uint32(q + "audio_id", locator_id)
w.add_string(q + "instruction", INSTRUCTION)
w.add_string(q + "context_before", CONTEXT_BEFORE)
w.add_string(q + "context_after", CONTEXT_AFTER)
g = p + "generation."
add_array(g + "eos_ids", eos_ids, GGUFValueType.INT32)
w.add_uint32(g + "char_run", CHAR_RUN)
w.add_uint32(g + "block_run", BLOCK_RUN)
w.add_uint32(g + "block_max_chars", BLOCK_MAX_CHARS)

c = p + "ctc."
w.add_uint32(c + "num_layers", int(ctc_decoder["n_layers"]))
w.add_uint32(c + "num_heads", int(ctc_decoder["n_heads"]))
w.add_float32(c + "rope_base", float(ctc_decoder["rope_base"]))
w.add_uint32(c + "rotary_dim", int(head.transformer.pos_enc.d_k_rot))
w.add_uint32(c + "upsampling_factor", int(ctc_decoder["upsampling_factor"]))
w.add_float32(c + "residual_scale", float(head.residual_scale))
add_array(c + "vocabulary", ctc_decoder["vocabulary"], GGUFValueType.STRING)
add_array(c + "tokenizer.model_proto", model_proto.tolist(), GGUFValueType.UINT8)
for key, value in ctc_tokenizer.items():
    (w.add_bool if isinstance(value, bool) else w.add_string)(c + "tokenizer." + key, value)
assert ctc_alignment["schema_version"] == 1 and ctc_alignment["basis_version"] == 1
assert ctc_alignment["upsampling_factor"] == ctc_decoder["upsampling_factor"]


def add_setting(key, value):
    if isinstance(value, bool):
        w.add_bool(key, value)
    elif isinstance(value, int):
        w.add_int32(key, value)
    elif isinstance(value, float):
        w.add_float32(key, value)
    elif isinstance(value, str):
        w.add_string(key, value)
    elif isinstance(value, dict):
        for name, inner in value.items():
            add_setting(key + "." + name, inner)
    elif isinstance(value, list) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        add_array(key, [float(v) for v in value], GGUFValueType.FLOAT32)
    elif isinstance(value, list) and value and all(isinstance(r, list) and len(r) == len(value[0]) for r in value):
        # A matrix, its rows one after another, and the number of its rows.
        add_setting(key, [v for row in value for v in row])
        w.add_uint32(key + ".rows", len(value))
    else:
        raise SystemExit(f"{key} is a {type(value).__name__}, which the converter does not write")


for key, value in ctc_alignment["parameters"].items():
    add_setting(c + "alignment." + key, value)

# The perception module's tensors, from its state dict in float32.
used = set()


def perception_tensor(name, key, data=None):
    used.add(key)
    values = sd[key] if data is None else data
    add(name, values.shape, lambda values=values: values)


perception_tensor("frontend.window", "preprocessor.featurizer.window")
perception_tensor("frontend.filterbank", "preprocessor.featurizer.fb", sd["preprocessor.featurizer.fb"][0])
used.update({"encoder.diarization_model.preprocessor.featurizer.window", "encoder.diarization_model.preprocessor.featurizer.fb"})


def transformer(prefix, source, d_model, layers, qk_norm):
    """NeMo's Transformer encoder layers: w_qkv, whose rows are the queries, the keys and the values, each head after
    head, held as three matrices."""
    for l in range(layers):
        a, o = f"{source}layers.{l}.", f"{prefix}blk.{l}."
        for src, dst in (("norm1", "attn_norm"), ("norm2", "ffn_norm")):
            perception_tensor(o + dst + ".weight", a + src + ".weight")
            perception_tensor(o + dst + ".bias", a + src + ".bias")
        qkv = sd[a + "attn.w_qkv.weight"]
        for i, part in enumerate(("attn_q", "attn_k", "attn_v")):
            perception_tensor(o + part + ".weight", a + "attn.w_qkv.weight", qkv[i * d_model:(i + 1) * d_model])
        if qk_norm:
            for src, dst in (("q_norm", "attn_q_norm"), ("k_norm", "attn_k_norm")):
                perception_tensor(o + dst + ".weight", a + "attn." + src + ".weight")
                perception_tensor(o + dst + ".bias", a + "attn." + src + ".bias")
        perception_tensor(o + "attn_out.weight", a + "attn.out_proj.weight")
        perception_tensor(o + "attn_out.bias", a + "attn.out_proj.bias")
        for src, dst in (("ffn.net.0", "ffn_up"), ("ffn.net.3", "ffn_down")):
            perception_tensor(o + dst + ".weight", a + src + ".weight")
            perception_tensor(o + dst + ".bias", a + src + ".bias")


a = "encoder.asr_encoder."
perception_tensor("asr.pre_encode.weight", a + "pre_encode.proj.weight")
for n in ("embed_norm", "final_norm"):
    perception_tensor(f"asr.{n}.weight", a + n + ".weight")
    perception_tensor(f"asr.{n}.bias", a + n + ".bias")
transformer("asr.", a, int(asr.d_model), len(asr.layers), True)

d = "encoder.diarization_model."
perception_tensor("spk.pre_encode.weight", d + "encoder.pre_encode.proj.weight")
for n in ("embed_norm", "final_norm"):
    perception_tensor(f"spk.{n}.weight", d + "encoder." + n + ".weight")
    perception_tensor(f"spk.{n}.bias", d + "encoder." + n + ".bias")
transformer("spk.", d + "encoder.", int(encoder.d_model), int(encoder.n_layers), False)
t = d + "sortformer_modules."
perception_tensor("spk.encoder_proj.weight", t + "encoder_proj.weight")
perception_tensor("spk.encoder_proj.bias", t + "encoder_proj.bias")
upsample = sd[t + "subpixel_upsample.weight"]
perception_tensor("spk.upsample.weight", t + "subpixel_upsample.weight", upsample.transpose(0, 2, 1).reshape(upsample.shape[0], -1))
perception_tensor("spk.upsample.bias", t + "subpixel_upsample.bias")
perception_tensor("spk.head.hidden.weight", t + "first_hidden_to_hidden.weight")
perception_tensor("spk.head.hidden.bias", t + "first_hidden_to_hidden.bias")
perception_tensor("spk.head.out.weight", t + "single_hidden_to_spks.weight")
perception_tensor("spk.head.out.bias", t + "single_hidden_to_spks.bias")
perception_tensor("spk.silence.embedding", t + "learnable_sil_emb")
used.update({t + "hidden_to_spks.weight", t + "hidden_to_spks.bias"})

for n, dst in (("asr_norm", "fuse.asr_norm"), ("diar_norm", "fuse.speaker_norm")):
    perception_tensor(dst + ".weight", f"encoder.{n}.weight")
    perception_tensor(dst + ".bias", f"encoder.{n}.bias")
add("fuse.speaker_kernel", kernel.shape, lambda: kernel)
perception_tensor("proj.weight", "proj.weight")
perception_tensor("proj.bias", "proj.bias")
assert set(sd) == used, sorted(set(sd) - used)[:8]

# The language model's and the timestamp head's tensors, read from the checkpoint as they are written.
checkpoint = safe_open(os.path.join(folder, official.WEIGHTS), framework="pt")
read = set()


def tensor(key):
    read.add(key)
    return checkpoint.get_tensor(key).float()


def llm_tensor(name, key, transform=lambda x: x):
    shape = transform(torch.empty(checkpoint.get_slice(key).get_shape(), device="meta")).shape
    add(name, shape, lambda: transform(tensor(key)).numpy())


def experts(o, source):
    """A layer's experts, [experts, out, in]: Automodel holds each projection's as [experts, in, out]."""
    llm_tensor(o + "ffn_up_exps.weight", source + "experts.gate_and_up_projs", lambda x: x.transpose(1, 2))
    llm_tensor(o + "ffn_down_exps.weight", source + "experts.down_projs", lambda x: x.transpose(1, 2))
    llm_tensor(o + "ffn_gate_inp.weight", source + "gate.weight")
    llm_tensor(o + "exp_probs_b.bias", source + "gate.e_score_correction_bias")
    llm_tensor(o + "ffn_up_shexp.weight", source + "shared_experts.up_proj.weight")
    llm_tensor(o + "ffn_down_shexp.weight", source + "shared_experts.down_proj.weight")


def attention(o, source):
    for src, dst in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
        llm_tensor(o + dst + ".weight", source + src + ".weight")


llm_tensor("token_embd.weight", "llm.model.embed_tokens.weight")
for l, kind in enumerate(types):
    a, o = f"llm.model.layers.{l}.", f"blk.{l}."
    llm_tensor(o + "norm.weight", a + "norm.weight")
    x = a + "mixer."
    if kind == "mamba":
        llm_tensor(o + "ssm_in.weight", x + "in_proj.weight")
        llm_tensor(o + "ssm_conv1d.weight", x + "conv1d.weight", lambda t: t[:, 0, :])
        llm_tensor(o + "ssm_conv1d.bias", x + "conv1d.bias")
        # The official code computes A = -exp(A_log) in float32 on every call (Transformers' and Automodel's mixers).
        llm_tensor(o + "ssm_a", x + "_fp32_params.A_log", lambda t: -torch.exp(t)[:, None])
        llm_tensor(o + "ssm_d", x + "_fp32_params.D", lambda t: t[:, None])
        llm_tensor(o + "ssm_dt.bias", x + "_fp32_params.dt_bias")
        llm_tensor(o + "ssm_norm.weight", x + "norm.weight")
        llm_tensor(o + "ssm_out.weight", x + "out_proj.weight")
    elif kind == "attention":
        attention(o, x)
    else:
        experts(o, x)
llm_tensor("output_norm.weight", "llm.model.norm.weight")
llm_tensor("output.weight", "llm.lm_head.weight")
for l, kind in enumerate(mtp_types):
    a, o = f"llm.mtp.layers.{l}.", f"mtp.blk.{l}."
    llm_tensor(o + "norm.weight", a + "norm.weight")
    if l == 0:
        llm_tensor(o + "eh_proj.weight", a + "eh_proj.weight")
        llm_tensor(o + "enorm.weight", a + "enorm.weight")
        llm_tensor(o + "hnorm.weight", a + "hnorm.weight")
    if l == len(mtp_types) - 1:
        llm_tensor(o + "final_norm.weight", a + "final_layernorm.weight")
    (attention if kind == "attention" else experts)(o, a + "mixer.")

h = "ctc_timestamp.decoder."
llm_tensor("ctc.upsample.weight", h + "upsample.weight", lambda t: t.permute(2, 1, 0))
llm_tensor("ctc.upsample.bias", h + "upsample.bias")
for n in ("embed_norm", "final_norm"):
    llm_tensor(f"ctc.{n}.weight", h + f"transformer.{n}.weight")
    llm_tensor(f"ctc.{n}.bias", h + f"transformer.{n}.bias")
ctc_d = int(ctc_decoder["feat_in"])
for l in range(int(ctc_decoder["n_layers"])):
    a, o = f"{h}transformer.layers.{l}.", f"ctc.blk.{l}."
    for src, dst in (("norm1", "attn_norm"), ("norm2", "ffn_norm"), ("attn.q_norm", "attn_q_norm"), ("attn.k_norm", "attn_k_norm"),
                     ("attn.out_proj", "attn_out"), ("ffn.net.0", "ffn_up"), ("ffn.net.3", "ffn_down")):
        llm_tensor(o + dst + ".weight", a + src + ".weight")
        llm_tensor(o + dst + ".bias", a + src + ".bias")
    for i, part in enumerate(("attn_q", "attn_k", "attn_v")):
        llm_tensor(o + part + ".weight", a + "attn.w_qkv.weight", lambda t, i=i: t[i * ctc_d:(i + 1) * ctc_d])
llm_tensor("ctc.output.weight", h + "decoder_layers.0.weight", lambda t: t[:, :, 0])
llm_tensor("ctc.output.bias", h + "decoder_layers.0.bias")
read.update({"ctc_timestamp.decoder.residual_scale", "ctc_timestamp.decoder.upsample_factor", "ctc_timestamp.tokenizer.model_proto"})

names = [name for name, _, _ in plan]
assert len(names) == len(set(names)), "a tensor is written twice"
expected = {key for key in keys if not key.startswith("perception.") and not key.endswith("._extra_state")}
# Every tensor info first, so that the values are written a tensor at a time.
for name, shape, _ in plan:
    w.add_tensor_info(name, shape, np.dtype(np.float32), int(np.prod(shape)) * 4)
w.kv_data[0] = dict(sorted(w.kv_data[0].items(), key=lambda item: not item[0].startswith("general.")))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for n, (name, shape, make) in enumerate(plan):
    values = np.ascontiguousarray(make(), dtype=np.float32)
    assert values.shape == shape, f"{name} is {values.shape}, not {shape}"
    w.write_tensor_data(values)
    if name.endswith("ffn_down_exps.weight"):
        print(f"{n + 1}/{len(plan)} {name}", flush=True)
w.close()
assert read == expected, f"the checkpoint's tensors {sorted(expected - read)[:5]} are not written"
print(path)
