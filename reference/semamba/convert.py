"""Converts a pinned RE-USE checkpoint to the one GGUF file the C++ port reads.

usage: uv run python convert.py <model> <out dir>

<model> is a key of pins.MODELS. Writes NVIDIA-NemotronLabs-RE-USE-9.6M-F32.gguf, or the laughter checkpoint's with its
fine-tune in the name, under GGUF's naming convention, in layout 1: every tensor of the checkpoint, the STFT's sizes at
the reference rate, the rates the model card lists, the defaults of inference_chunk.sh, the constants of the official
code the C++ reads, and the model's identity in the GGUF specification's general keys. The file names no language: the
model card calls the model language-agnostic.

The sizes of the network have no key where a tensor gives them: the hidden channels are the encoder's first
convolution's outputs, Mamba's inner width, state and convolution are those of its in_proj, A_log and conv1d, and its
dt_rank is dt_proj's input. The counts, strides and factors the official modules build in code have keys, each with
the module that sets it.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis: a Conv2d's kernel [out, in, kh, kw] is stored as is
(ne = [kw, kh, in, out]), the kernel along the frequency bins first; a Linear weight [out, in] as is (ne = [in, out]);
Mamba's depthwise Conv1d kernel [d_inner, 1, d_conv] as [d_inner, d_conv], the matrix ggml_ssm_conv() takes. A_log is
stored as the checkpoint holds it, from which the C++ computes A = -exp(A_log) as Mamba does. Every tensor is written in
float32, the checkpoint's own type.
"""

import argparse
import os

import numpy as np
from gguf import GGUFValueType, GGUFWriter, LlamaFileType, naming_convention, size_label

from pins import CHUNK_SIZE_IN_SECONDS, HOP_LENGTH_PORTION, MODELS, SAMPLE_RATES, load

ARCH = "semamba"
# Each layout this converter has written, with the first release of speech.cpp whose reader takes it; it writes the
# last.
RELEASES = {1: "0.10.1"}
LAYOUT = max(RELEASES)
# OpenMDW 1.1, the model card's license, which SPDX's list does not yet hold (it holds 1.0).
LICENSE = "LicenseRef-OpenMDW-1.1"

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

pin = MODELS[args.model]
model, cfg = load(args.model, "cuda")
mc, sc = cfg["model_cfg"], cfg["stft_cfg"]

# The configuration the modules below are read under: a magnitude mapped directly (no mask, no nonlinearity), the
# magnitude compressed by log1p after a ReLU on the way back, an STFT whose window is its FFT, and one Mamba layer to
# each direction.
assert mc["mapping"] is True and mc["nonlinear"] == "None" and mc["compress_factor"] == "relu_log1p"
assert mc["input_channel"] == 2 and mc["output_channel"] == 1 and mc["inner_mamba_nlayer"] == 1
assert sc["n_fft"] == sc["win_size"] and sc["sfi"] is True
encoder, mag, pha = model.dense_encoder, model.mask_decoder, model.phase_decoder
conv_2 = encoder.dense_conv_2[0]
assert conv_2.kernel_size == (1, 3) and conv_2.padding == (0, 0)
up = [mag.up_conv1[0], mag.up_conv2[0], pha.up_conv1[0], pha.up_conv2[0]]
# The first up-sampling multiplies the bins by what the encoder divided them by, the second the frames.
assert up[0].r == up[2].r == conv_2.stride[1] and up[1].r == up[3].r == conv_2.stride[0]
assert all(u.conv.kernel_size == (1, 3) and u.pad1.padding == (1, 1, 0, 0) for u in up)
dense = encoder.dense_block
assert dense.depth == mag.dense_block.depth == pha.dense_block.depth
for block in (dense, mag.dense_block, pha.dense_block):
    for i, layer in enumerate(block.dense_block):
        assert layer[0].kernel_size == (3, 3) and layer[0].dilation == (2**i, 1) and layer[0].padding == (2**i, 1)
norms = [m for m in model.modules() if type(m).__name__ == "InstanceNorm2d"]
assert all(n.affine and not n.track_running_stats for n in norms) and len({n.eps for n in norms}) == 1
layer_norms = [m for m in model.modules() if type(m).__name__ == "LayerNorm"]
assert len({n.eps for n in layer_norms}) == 1
mamba = model.TSMamba[0].time_mamba.forward_blocks
assert mamba.activation == "silu" and mamba.in_proj.bias is None and mamba.out_proj.bias is None and mamba.conv1d.bias is not None

w = GGUFWriter(None, ARCH)


def add_array(key, values, element):
    """An array with its element type given, not inferred from its first value."""
    w.add_key_value(key, list(values), GGUFValueType.ARRAY, sub_type=element)


w.add_uint32("speech.layout", LAYOUT)
w.add_string("speech.requires", RELEASES[LAYOUT])
w.add_string("speech.task", "enhancement")

# The STFT at the reference rate, which every rate scales (make_even(n_fft * rate // sampling_rate) in the scripts),
# so that a second holds sampling_rate / hop_size frames at any rate.
w.add_uint32("semamba.stft.sampling_rate", int(sc["sampling_rate"]))
w.add_uint32("semamba.stft.n_fft", int(sc["n_fft"]))
w.add_uint32("semamba.stft.hop_size", int(sc["hop_size"]))
# The rates of the model card's inputs.
w.add_uint32("semamba.sample_rate_min", SAMPLE_RATES[0])
w.add_uint32("semamba.sample_rate_max", SAMPLE_RATES[1])

# inference_chunk.sh's settings, and the constants of inference_chunk.py's loop: the floor of its Hann window, the share
# of a frame's bins at zero from which the frame is zeroed, and the value pad_or_trim_to_match() pads a short inverse
# STFT with.
w.add_float32("semamba.chunk_size_in_seconds", CHUNK_SIZE_IN_SECONDS)
w.add_float32("semamba.hop_length_portion", HOP_LENGTH_PORTION)
w.add_float32("semamba.window_floor", 1e-7)
w.add_float32("semamba.zero_portion", 0.5)
w.add_float32("semamba.pad_value", 1e-8)

# SEMamba.forward()'s padding of the input with two frames and two bins of zeros, DenseBlock's depth, whose layer i is
# dilated 2**i frames, dense_conv_2's stride over the frames and the bins, which the decoders' up-sampling undoes, and
# the number of TFMamba blocks.
w.add_uint32("semamba.input_padding", 2)
w.add_uint32("semamba.dense_depth", int(dense.depth))
w.add_uint32("semamba.frame_stride", int(conv_2.stride[0]))
w.add_uint32("semamba.bin_stride", int(conv_2.stride[1]))
w.add_uint32("semamba.block_count", len(model.TSMamba))
w.add_float32("semamba.instance_norm_eps", float(norms[0].eps))
w.add_float32("semamba.layer_norm_eps", float(layer_norms[0].eps))

sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
used = set()


def add(name, key, data=None):
    used.add(key)
    w.add_tensor(name, np.ascontiguousarray(sd[key] if data is None else data, dtype=np.float32))


def add_conv(name, key):
    """A Conv2d, its InstanceNorm2d and its PReLU: the nn.Sequential the codec module builds of each."""
    add(name + ".conv.weight", key + ".0.weight")
    add(name + ".conv.bias", key + ".0.bias")
    add(name + ".norm.weight", key + ".1.weight")
    add(name + ".norm.bias", key + ".1.bias")
    add(name + ".prelu", key + ".2.weight")


def add_dense(name, key):
    for i in range(int(dense.depth)):
        add_conv(f"{name}.{i}", f"{key}.dense_block.{i}")


def add_up(name, key):
    """An SPConvTranspose2d, whose Conv2d sits inside it, then its norm and PReLU."""
    add(name + ".conv.weight", key + ".0.conv.weight")
    add(name + ".conv.bias", key + ".0.conv.bias")
    add(name + ".norm.weight", key + ".1.weight")
    add(name + ".norm.bias", key + ".1.bias")
    add(name + ".prelu", key + ".2.weight")


add_conv("enc.in", "dense_encoder.dense_conv_1")
add_dense("enc.dense", "dense_encoder.dense_block")
add_conv("enc.out", "dense_encoder.dense_conv_2")
for b in range(len(model.TSMamba)):
    for part in ("time", "freq"):
        src, dst = f"TSMamba.{b}.{part}_mamba.", f"blk.{b}.{part}."
        for direction, name in (("forward_blocks", "fw"), ("backward_blocks", "bw")):
            s, d = src + direction + ".", dst + name + "."
            for t in ("in_proj.weight", "x_proj.weight", "dt_proj.weight", "dt_proj.bias", "A_log", "D", "out_proj.weight",
                      "conv1d.bias"):
                add(d + t, s + t)
            kernel = sd[s + "conv1d.weight"]
            add(d + "conv1d.weight", s + "conv1d.weight", kernel.reshape(kernel.shape[0], kernel.shape[2]))
        add(dst + "proj.weight", src + "output_proj.weight")
        add(dst + "proj.bias", src + "output_proj.bias")
        add(dst + "norm.weight", src + "norm.weight")
        add(dst + "norm.bias", src + "norm.bias")
for name, key, decoder in (("mag", "mask_decoder", mag), ("pha", "phase_decoder", pha)):
    add_dense(name + ".dense", key + ".dense_block")
    add_up(name + ".up1", key + ".up_conv1")
    add_up(name + ".up2", key + ".up_conv2")
add("mag.out.weight", "mask_decoder.final_conv.weight")
add("mag.out.bias", "mask_decoder.final_conv.bias")
for part in ("r", "i"):
    add(f"pha.{part}.weight", f"phase_decoder.phase_conv_{part}.weight")
    add(f"pha.{part}.bias", f"phase_decoder.phase_conv_{part}.bias")
assert set(sd) == used, sorted(set(sd) - used)

# The model's identity in the GGUF specification's general keys, which name the file. They follow the tensors, whose
# parameters give the size label, and then go first after general.architecture, since GGUFWriter writes the keys in the
# order they were added.
base = MODELS["NVIDIA-NemotronLabs-RE-USE"]["repository"].split("/")[1]
name = pin["repository"].split("/")[1]
finetune = name[len(base) + 1 :] or None
label = size_label(*w.get_total_parameter_count())
repository = f"https://huggingface.co/{pin['repository']}"
w.add_name(name)
w.add_organization(pin["repository"].split("/")[0])
w.add_basename(base)
if finetune:
    w.add_finetune(finetune)
w.add_size_label(label)
w.add_license(LICENSE)
w.add_source_url(f"{repository}/tree/{pin['revision']}")
w.add_source_repo_url(repository)
w.add_file_type(LlamaFileType.ALL_F32)
w.kv_data[0] = dict(sorted(w.kv_data[0].items(), key=lambda item: not item[0].startswith("general.")))

path = os.path.join(args.out_dir, naming_convention(None, base, finetune, None, label, "f32") + ".gguf")
w.write_header_to_file(path)
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", path)
