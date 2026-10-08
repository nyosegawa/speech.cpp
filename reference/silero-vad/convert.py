"""Converts the pinned Silero VAD 16 kHz model to the one GGUF file the C++ port reads.

usage: uv run python convert.py <out dir>

Writes silero-vad-309K-v6.2-F32.gguf, named under GGUF's naming convention, in layout 1: the 16 kHz model of the JIT
file the package loads (its _model; the 8 kHz _model_8k is left out, since speech.cpp resamples every input to the
model's 16 kHz), the sizes of its chunks, context and STFT, the strides and padding of its encoder, the defaults of the
options of get_speech_timestamps() a request may set and the constants of its rule for regions, and the model's
identity in the GGUF specification's general keys. The model hears speech in any language and has no general.languages.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis: the STFT's basis [258, 1, 256] is stored as [258, 256]
(ne = [256, 258]), its first 129 rows the real parts and the rest the imaginary parts; a Conv1d weight [out, in, width]
as is (ne = [width, in, out]); the LSTM cell's matrices as is, with its two biases summed. Every tensor is float32, the
only type of this model's files: its 309K parameters take 1.2 MB.
"""

import argparse
import inspect
import os
import re

import numpy as np
import torch
from gguf import GGUFValueType, GGUFWriter, LlamaFileType, naming_convention, size_label
from silero_vad import utils_vad

from pins import MODEL, load

ARCH = "silero-vad"
# Each layout this converter has written, with the first release of speech.cpp whose reader takes it; it writes the
# last.
RELEASES = {1: "0.8.0"}
LAYOUT = max(RELEASES)
SAMPLE_RATE = 16000
# The model's version: the release whose weights these are, which v6.2.0 of the package first shipped, unchanged
# since.
VERSION = "v6.2"

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

model = load()
vad = model._model
stft = vad.stft
blocks = [getattr(vad.encoder, str(i)) for i in range(4)]
assert SAMPLE_RATE in model.sample_rates
# The JIT model takes chunks of 512 samples at 16 kHz and refuses any other length, as get_speech_timestamps() cuts them.
chunk = 512
model.reset_states()
model(torch.zeros(chunk), SAMPLE_RATE)
try:
    model(torch.zeros(chunk + 1), SAMPLE_RATE)
    raise AssertionError("the model took a chunk of another length")
except Exception as e:
    assert "Provided number of samples" in str(e), e
model.reset_states()
context = int(vad.context_size_samples)
# The STFT reflects the end of its input by a width its code fixes, measured here on a ramp: a ramp reflected at its end
# runs back down from its last sample but one.
ramp = torch.arange(context + chunk, dtype=torch.float32).unsqueeze(0)
padded = stft.padding(ramp)[0]
reflect = padded.shape[0] - ramp.shape[1]
assert torch.equal(padded[:ramp.shape[1]], ramp[0]) and torch.equal(padded[ramp.shape[1]:], ramp[0, -1 - reflect:-1].flip(0))
n_fft, hop = int(stft.filter_length), int(stft.hop_length)
assert stft.forward_basis_buffer.shape == (n_fft + 2, 1, n_fft)
# Each encoder block is a convolution, an identity in place of its squeeze-and-excitation, and a ReLU.
for block in blocks:
    assert block.se.original_name == "Identity" and block.activation.original_name == "ReLU"
    conv = block.reparam_conv
    assert conv.dilation == (1,) and conv.groups == 1 and conv.padding_mode == "zeros"
# The decoder: a dropout, inactive in evaluation, a ReLU, a convolution of width 1 and a sigmoid, on the LSTM cell's h.
names = [m.original_name for _, m in vad.decoder.decoder.named_children()]
assert names == ["Dropout", "ReLU", "Conv1d", "Sigmoid"] and vad.decoder.rnn.original_name == "LSTMCell", names

# The defaults of the options a request sets, from get_speech_timestamps()'s signature, and the constants of the rule
# that turns probabilities into regions, from get_speech_timestamps_from_probs(): the silence, in ms, that a region
# longer than max_speech_duration_s may be cut at, and the threshold below which a chunk ends a region, the threshold
# less 0.15 but at least 0.01, which the code writes as literals. The rule runs with use_max_poss_sil_at_max_speech at
# its default, true, the one the C++ ports: it cuts at the longest of the silences.
defaults = {k: v.default for k, v in inspect.signature(utils_vad.get_speech_timestamps).parameters.items()}
rule = {k: v.default for k, v in inspect.signature(utils_vad.get_speech_timestamps_from_probs).parameters.items()}
for k in ("threshold", "min_speech_duration_ms", "min_silence_duration_ms", "speech_pad_ms", "max_speech_duration_s",
          "min_silence_at_max_speech", "use_max_poss_sil_at_max_speech"):
    assert defaults[k] == rule[k], k
assert defaults["max_speech_duration_s"] == float("inf") and defaults["use_max_poss_sil_at_max_speech"] is True
source = inspect.getsource(utils_vad.get_speech_timestamps_from_probs)
negative = re.search(r"neg_threshold = max\(threshold - ([0-9.]+), ([0-9.]+)\)", source)
assert negative, "get_speech_timestamps_from_probs() no longer computes neg_threshold as this converter reads it"

w = GGUFWriter(None, ARCH)
w.add_uint32("speech.layout", LAYOUT)
w.add_string("speech.requires", RELEASES[LAYOUT])
w.add_string("speech.task", "detection")
w.add_uint32("speech.sample_rate", SAMPLE_RATE)

w.add_uint32("silero-vad.chunk_size", chunk)
w.add_uint32("silero-vad.context_size", context)
w.add_uint32("silero-vad.stft.n_fft", n_fft)
w.add_uint32("silero-vad.stft.hop_length", hop)
w.add_uint32("silero-vad.stft.reflect", reflect)
w.add_key_value("silero-vad.encoder.strides", [int(b.reparam_conv.stride[0]) for b in blocks], GGUFValueType.ARRAY,
                sub_type=GGUFValueType.INT32)
w.add_key_value("silero-vad.encoder.padding", [int(b.reparam_conv.padding[0]) for b in blocks], GGUFValueType.ARRAY,
                sub_type=GGUFValueType.INT32)
w.add_float64("silero-vad.threshold", float(defaults["threshold"]))
w.add_uint32("silero-vad.min_speech_duration_ms", int(defaults["min_speech_duration_ms"]))
w.add_uint32("silero-vad.min_silence_duration_ms", int(defaults["min_silence_duration_ms"]))
w.add_uint32("silero-vad.speech_pad_ms", int(defaults["speech_pad_ms"]))
w.add_uint32("silero-vad.min_silence_at_max_speech_ms", int(defaults["min_silence_at_max_speech"]))
w.add_float64("silero-vad.neg_threshold_offset", float(negative.group(1)))
w.add_float64("silero-vad.neg_threshold_floor", float(negative.group(2)))

sd = {k.removeprefix("_model."): v.detach().float().numpy() for k, v in model.state_dict().items() if k.startswith("_model.")}


def add(name, data):
    w.add_tensor(name, np.ascontiguousarray(data, dtype=np.float32))


add("stft.basis", sd["stft.forward_basis_buffer"][:, 0, :])
for i in range(len(blocks)):
    add(f"encoder.{i}.weight", sd[f"encoder.{i}.reparam_conv.weight"])
    add(f"encoder.{i}.bias", sd[f"encoder.{i}.reparam_conv.bias"])
# The cell's four gates, input, forget, cell and output, are stacked in its matrices and its summed biases.
add("lstm.ih.weight", sd["decoder.rnn.weight_ih"])
add("lstm.hh.weight", sd["decoder.rnn.weight_hh"])
add("lstm.bias", sd["decoder.rnn.bias_ih"] + sd["decoder.rnn.bias_hh"])
add("decoder.weight", sd["decoder.decoder.2.weight"][:, :, 0])
add("decoder.bias", sd["decoder.decoder.2.bias"])

# The model's identity in the GGUF specification's general keys, which name the file. They follow the tensors, whose
# parameters give the size label, and then go first after general.architecture, since GGUFWriter writes the keys in
# the order they were added.
label = size_label(*w.get_total_parameter_count())
repository = f"https://github.com/{MODEL['repository']}"
w.add_name(MODEL["repository"].split("/")[1])
w.add_organization(MODEL["repository"].split("/")[0])
w.add_basename(ARCH)
w.add_size_label(label)
w.add_version(VERSION)
# The license of the repository, which the package's metadata names too.
w.add_license("MIT")
w.add_source_url(f"{repository}/tree/{MODEL['revision']}")
w.add_source_repo_url(repository)
w.add_file_type(LlamaFileType.ALL_F32)
w.kv_data[0] = dict(sorted(w.kv_data[0].items(), key=lambda item: not item[0].startswith("general.")))

path = os.path.join(args.out_dir, naming_convention(None, ARCH, None, VERSION, label, "f32") + ".gguf")
w.write_header_to_file(path)
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", path)
