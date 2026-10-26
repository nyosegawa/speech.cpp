"""Converts the pinned Nemotron 3 Diarization checkpoint to the one GGUF file the C++ port reads.

usage: uv run python convert.py <out dir>

Writes Nemotron-3-Diarization-99M-F32.gguf, named under GGUF's naming convention, in layout 1: the frontend's window and
mel filterbank, the feature stacking's projection, the Transformer encoder's layers, their fused projection of the
queries, keys and values held as three matrices, encoder_proj, the upsampling
convolution and the speaker head, the learnable silence embedding the speaker cache holds in place of a frame, the
settings of the cache's compression, the streaming configurations of the model card's table, the defaults of the
postprocessing diarize() runs, every other constant the C++ reads and the model's identity in the GGUF specification's
general keys. The file names no language: the model tells speakers apart in any.

Two heads of the checkpoint are left out, since no inference reads them: hidden_to_spks, which SortformerModules builds
and never applies, and the activity head, which only the auxiliary loss of training and forward_streaming_step()'s
return_logits read.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis: a Linear weight [out, in] is stored as is
(ne = [in, out]). The upsampling Conv1d's kernel [out, in, 3] is stored as one matrix [out, 3 * in] whose columns run
over the three taps, then the input channels, so that it multiplies the frame before, the frame and the frame after
stacked. Every tensor is written in float32, the checkpoint's BF16 values exactly; `speech quantize` makes the other
weight types of the file, each tensor in the type src/families/sortformer/layout.cpp gives it.
"""

import argparse
import os

import numpy as np
from gguf import GGUFValueType, LlamaFileType, GGUFWriter, naming_convention, size_label
from nemo.collections.asr.parts.utils.vad_utils import PostProcessingParams

from pins import MODELS, configurations, restore

ARCH = "sortformer"
# Each layout this converter has written, with the first release of speech.cpp whose reader takes it; it writes the
# last.
RELEASES = {1: "0.9.0"}
LAYOUT = max(RELEASES)
# OpenMDW 1.1, the model card's license, which SPDX's list does not yet hold (it holds 1.0).
LICENSE = "LicenseRef-OpenMDW-1.1"

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

pin = MODELS["Nemotron-3-Diarization"]
model = restore(pin)
cfg = model.cfg
featurizer = model.preprocessor.featurizer
encoder = model.encoder
sm = model.sortformer_modules
modes, stream_default = configurations(pin)

assert featurizer.exact_pad is False and featurizer.frame_splicing == 1 and featurizer.mag_power == 2.0
assert featurizer.normalize == "NA" and featurizer.log and featurizer.log_zero_guard_type == "add"
assert featurizer.pad_value == 0 and featurizer.preemph is not None
assert type(encoder).__name__ == "TransformerEncoder" and type(encoder.pre_encode).__name__ == "FeatureStacking"
assert encoder.self_attention_model == "rope" and encoder.attn_mode == "full" and encoder.xscale is None
assert encoder.pos_enc.d_k_rot == encoder.pos_enc.d_k and encoder.out_proj is None and model.transformer_encoder is None
assert type(encoder.embed_norm).__name__ == "LayerNorm"
layer = encoder.layers[0]
assert not layer.attn.qk_norm and layer.attn.w_qkv.bias is None and layer.attn.out_proj.bias is not None
assert type(layer.ffn.net[1]).__name__ == "GELU" and layer.ffn.net[1].approximate == "none"
eps = {m.eps for m in model.modules() if type(m).__name__ == "LayerNorm"}
assert len(eps) == 1, eps
# Predictions every 10 ms, upsampled from the encoder's 80 ms frames, and no transformer after the encoder; the speaker
# cache holds the learnable silence embedding where a frame is disabled, and chunks take no left context.
assert model.high_resolution and model.output_subsampling_factor == 1 and model.upsample_factor == encoder.subsampling_factor
assert sm.use_learnable_sil_emb and sm.chunk_left_context == 0 and sm.encoder_proj is not None
assert sm.subpixel_upsample.kernel_size == (3,) and sm.subpixel_upsample.padding == (1,)
assert model.streaming_mode and not model.async_streaming

w = GGUFWriter(None, ARCH)


def add_array(key, values, element):
    """An array with its element type given, not inferred from its first value."""
    w.add_key_value(key, list(values), GGUFValueType.ARRAY, sub_type=element)


w.add_uint32("speech.layout", LAYOUT)
w.add_string("speech.requires", RELEASES[LAYOUT])
w.add_string("speech.task", "diarization")
w.add_uint32("speech.sample_rate", int(featurizer.sample_rate))

# The frontend: FilterbankFeatures' parameters; its features are not normalized (normalize "NA").
w.add_uint32("sortformer.frontend.n_fft", int(featurizer.n_fft))
w.add_uint32("sortformer.frontend.hop_length", int(featurizer.hop_length))
w.add_uint32("sortformer.frontend.n_mels", int(featurizer.nfilt))
w.add_float32("sortformer.frontend.preemphasis", float(featurizer.preemph))
w.add_float32("sortformer.frontend.log_guard", float(featurizer.log_zero_guard_value))

# The encoder: FeatureStacking's factor, which is also the upsampling's, the Transformer's sizes, and RoPE's base,
# TransformerEncoder's rope_base, which RotaryPositionalEncoding rotates every channel of a head by.
w.add_uint32("sortformer.encoder.d_model", int(encoder.d_model))
w.add_uint32("sortformer.encoder.num_layers", int(encoder.n_layers))
w.add_uint32("sortformer.encoder.num_heads", int(encoder.n_heads))
w.add_uint32("sortformer.encoder.subsampling_factor", int(encoder.subsampling_factor))
w.add_float32("sortformer.encoder.norm_eps", float(eps.pop()))
w.add_float32("sortformer.encoder.rope_base", float(encoder.pos_enc.rope_base))
w.add_uint32("sortformer.speakers", int(sm.n_spk))

# The compression of the speaker cache (SortformerModules._compress_spkcache()): the silence frames it keeps for each
# speaker, the probabilities its scores are clamped to, the boost of the frames the FIFO queue has just given it, and
# the shares of each speaker's frames that the strong and the weak boosts and the disabling of non-positive scores
# reach.
w.add_uint32("sortformer.cache.silence_frames", int(sm.spkcache_sil_frames_per_spk))
w.add_float32("sortformer.cache.score_threshold", float(sm.pred_score_threshold))
w.add_float32("sortformer.cache.boost_latest", float(sm.scores_boost_latest))
w.add_float32("sortformer.cache.strong_boost_rate", float(sm.strong_boost_rate))
w.add_float32("sortformer.cache.weak_boost_rate", float(sm.weak_boost_rate))
w.add_float32("sortformer.cache.min_pos_scores_rate", float(sm.min_pos_scores_rate))

# The streaming configurations, the longest chunk first, and the one a stream takes by default.
order = sorted(modes, key=lambda name: -modes[name]["chunk_len"])
assert order[0] == "offline"
add_array("sortformer.streaming.modes", order, GGUFValueType.STRING)
for field in ("spkcache_len", "fifo_len", "chunk_len", "chunk_right_context", "spkcache_update_period"):
    add_array(f"sortformer.streaming.{field}", [modes[name][field] for name in order], GGUFValueType.INT32)
w.add_string("sortformer.streaming.stream_default", stream_default)

# The defaults of the postprocessing diarize() runs, PostProcessingParams', in seconds where they are times.
post = PostProcessingParams()
for field in ("onset", "offset", "pad_onset", "pad_offset", "min_duration_on", "min_duration_off"):
    w.add_float64(f"sortformer.postprocessing.{field}", float(getattr(post, field)))

sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
used = set()


def add(name, key, data=None):
    used.add(key)
    w.add_tensor(name, np.ascontiguousarray(sd[key] if data is None else data, dtype=np.float32))


add("frontend.window", "preprocessor.featurizer.window")
add("frontend.filterbank", "preprocessor.featurizer.fb", sd["preprocessor.featurizer.fb"][0])
add("pre_encode.weight", "encoder.pre_encode.proj.weight")
add("embed_norm.weight", "encoder.embed_norm.weight")
add("embed_norm.bias", "encoder.embed_norm.bias")
for l in range(int(encoder.n_layers)):
    a, o = f"encoder.layers.{l}.", f"blk.{l}."
    for src, dst in (("norm1", "attn_norm"), ("norm2", "ffn_norm")):
        add(o + dst + ".weight", a + src + ".weight")
        add(o + dst + ".bias", a + src + ".bias")
    # w_qkv's rows are the queries, the keys and the values, each head after head, which the file holds apart.
    qkv = sd[a + "attn.w_qkv.weight"]
    d = int(encoder.d_model)
    for i, part in enumerate(("attn_q", "attn_k", "attn_v")):
        add(o + part + ".weight", a + "attn.w_qkv.weight", qkv[i * d : (i + 1) * d])
    add(o + "attn_out.weight", a + "attn.out_proj.weight")
    add(o + "attn_out.bias", a + "attn.out_proj.bias")
    for src, dst in (("ffn.net.0", "ffn_up"), ("ffn.net.3", "ffn_down")):
        add(o + dst + ".weight", a + src + ".weight")
        add(o + dst + ".bias", a + src + ".bias")
add("final_norm.weight", "encoder.final_norm.weight")
add("final_norm.bias", "encoder.final_norm.bias")
s = "sortformer_modules."
add("encoder_proj.weight", s + "encoder_proj.weight")
add("encoder_proj.bias", s + "encoder_proj.bias")
kernel = sd[s + "subpixel_upsample.weight"]
add("upsample.weight", s + "subpixel_upsample.weight", kernel.transpose(0, 2, 1).reshape(kernel.shape[0], -1))
add("upsample.bias", s + "subpixel_upsample.bias")
add("head.hidden.weight", s + "first_hidden_to_hidden.weight")
add("head.hidden.bias", s + "first_hidden_to_hidden.bias")
add("head.out.weight", s + "single_hidden_to_spks.weight")
add("head.out.bias", s + "single_hidden_to_spks.bias")
add("silence.embedding", s + "learnable_sil_emb")
left_out = {k for k in sd if k.startswith(s + "hidden_to_spks.") or k.startswith(s + "activity_head.")}
assert set(sd) == used | left_out, sorted(set(sd) - used - left_out)

# The model's identity in the GGUF specification's general keys, which name the file. They follow the tensors, whose
# parameters give the size label, since the name gives none, and then go first after general.architecture, since
# GGUFWriter writes the keys in the order they were added.
name = pin["repository"].split("/")[1]
label = size_label(*w.get_total_parameter_count())
repository = f"https://huggingface.co/{pin['repository']}"
w.add_name(name)
w.add_organization(pin["repository"].split("/")[0])
w.add_basename(name)
w.add_size_label(label)
w.add_license(LICENSE)
w.add_source_url(f"{repository}/tree/{pin['revision']}")
w.add_source_repo_url(repository)
w.add_file_type(LlamaFileType.ALL_F32)
w.kv_data[0] = dict(sorted(w.kv_data[0].items(), key=lambda item: not item[0].startswith("general.")))

path = os.path.join(args.out_dir, naming_convention(None, name, None, None, label, "f32") + ".gguf")
w.write_header_to_file(path)
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", path)
