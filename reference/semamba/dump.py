"""Runs a pinned RE-USE checkpoint as the official inference_chunk.py does, on CUDA in float32, and saves the tensors the
C++ port is checked against.

usage: uv run python dump.py <model> <out dir> --data <speech-bench data dir> [--only NAME...]

<model> is a key of pins.MODELS. Each input of inputs.py goes to <out dir>/<model>/<input>/. TensorFloat-32 is turned
off for matrix products and convolutions, and cuDNN made deterministic, so that the tensors are float32's. The chunk
loop is inference_chunk.py's, with its STFT, model, zeroing of frames, inverse STFT and windowed overlap-add, and the
stages are recorded with hooks. Spectrograms are saved frame by frame, [frames, bins], and the network's activations
as [channels, frames, bins], the orders in which ggml holds them with the bins fastest.

  audio            the input at its rate, [N]
  output           the enhanced audio, [N]
  chunk<k>/        the k-th chunk, from 0:
    input          its samples, [L]
    mag, pha       the STFT's compressed magnitude and its phase, [T, F]
    mag_raw        the magnitude decoder's output before the crop to [T, F], [T4, F2]
    phase_r        the phase decoder's real and imaginary parts before atan2 and the crop, [T4, F2]
    phase_i
    amp, angle     the magnitude after the frames were zeroed and the phase, cropped, [T, F]
    zeroed         1 for each frame that was zeroed, [T]
    output         the inverse STFT, padded or trimmed to the chunk's length, [L]
  and for the first chunk alone:
    encoded        the dense encoder's output, [64, T', F']
    mamba_out      the last TFMamba block's output, [64, T', F']
  and, for an input whose entry asks for them, the first chunk's
    blocks         every TFMamba block's output, [30, 64, T', F']
    time_in        the first block's bidirectional Mamba over time: its input, [F', T', 64], and its output
    time_out
    freq_in        the same over frequency, [T', F', 64]
    freq_out

meta.json holds the rate, the STFT's sizes, the chunk settings in samples, each chunk's first sample and the number of
frames zeroed. For an input of one chunk the output is asserted to equal inference.py's, which runs no window.
"""

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

from inputs import inputs
from pins import MODELS, code_dir, load

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
parser.add_argument("--data", required=True)
parser.add_argument("--only", nargs="*")
args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")

model, cfg = load(args.model, device)
code_dir()
from models.stfts import mag_phase_istft, mag_phase_stft  # noqa: E402
from utils.util import pad_or_trim_to_match  # noqa: E402

RELU = nn.ReLU()
n_fft, hop_size, win_size = cfg["stft_cfg"]["n_fft"], cfg["stft_cfg"]["hop_size"], cfg["stft_cfg"]["win_size"]
compress_factor = cfg["model_cfg"]["compress_factor"]
sampling_rate = cfg["stft_cfg"]["sampling_rate"]


def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


captured = {}


def keep(name, which="output"):
    def hook(module, inputs, output):
        captured[name] = (inputs[0] if which == "input" else output).detach().float().cpu().numpy()

    return hook


def save(folder, name, array):
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, name + ".npy"), np.ascontiguousarray(array))


def enhance_chunk(noisy, sizes):
    """One chunk through inference_chunk.py's steps, with what each step gave."""
    n_fft_s, hop_s, win_s = sizes
    mag, pha, _ = mag_phase_stft(noisy, n_fft=n_fft_s, hop_size=hop_s, win_size=win_s, compress_factor=compress_factor,
                                 center=True, addeps=False)
    amp_g, pha_g, _ = model(mag, pha)
    raw = amp_g.clone()
    m = torch.expm1(RELU(amp_g))
    zero_portion = torch.sum(m == 0, 1) / m.shape[1]
    zeroed = (zero_portion > 0.5)[0]
    amp_g[:, :, zeroed] = 0
    audio_g = mag_phase_istft(amp_g, pha_g, n_fft_s, hop_s, win_s, compress_factor)
    audio_g = pad_or_trim_to_match(noisy.detach(), audio_g, pad_value=1e-8)
    return {"mag": mag, "pha": pha, "raw": raw, "amp": amp_g, "angle": pha_g, "zeroed": zeroed, "output": audio_g}


def frames(t):
    """A [1, F, T] spectrogram as [T, F]."""
    return t[0].transpose(0, 1).float().cpu().numpy()


for name, entry in inputs(args.data).items():
    if args.only and name not in args.only:
        continue
    folder = os.path.join(args.out_dir, args.model, name)
    sr = entry["rate"]
    wav = torch.from_numpy(entry["audio"]).unsqueeze(0).to(device)
    chunk_size = int(entry["chunk_size_in_seconds"] * sr)
    hop_length = int(entry["hop_length_portion"] * chunk_size)
    window = torch.clamp(torch.hann_window(chunk_size).to(device), min=1e-7)
    sizes = (make_even(n_fft * sr // sampling_rate), make_even(hop_size * sr // sampling_rate), make_even(win_size * sr // sampling_rate))
    enhanced = torch.zeros_like(wav)
    window_sum = torch.zeros_like(wav)
    num_chunks = max(1, math.ceil((wav.shape[1] - chunk_size) / hop_length) + 1)

    handles = [
        model.dense_encoder.register_forward_hook(keep("encoded")),
        model.TSMamba[-1].register_forward_hook(keep("mamba_out")),
        model.mask_decoder.register_forward_hook(keep("mag_raw")),
        model.phase_decoder.phase_conv_r.register_forward_hook(keep("phase_r")),
        model.phase_decoder.phase_conv_i.register_forward_hook(keep("phase_i")),
    ]
    if entry["blocks"]:
        for i, block in enumerate(model.TSMamba):
            handles.append(block.register_forward_hook(keep(f"block{i}")))
        for part in ("time", "freq"):
            mamba = getattr(model.TSMamba[0], part + "_mamba")
            handles.append(mamba.register_forward_hook(keep(part + "_in", "input")))
            handles.append(mamba.register_forward_hook(keep(part + "_out")))

    starts, zeroed_frames = [], []
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * hop_length if i < num_chunks - 1 else max(0, wav.shape[1] - chunk_size)
            noisy = wav[:, start : start + chunk_size]
            captured.clear()
            got = enhance_chunk(noisy, sizes)
            length = got["output"].shape[1]
            enhanced[:, start : start + length] += got["output"] * window[:length]
            window_sum[:, start : start + length] += window[:length]
            starts.append(start)
            zeroed_frames.append(int(got["zeroed"].sum()))

            sub = os.path.join(folder, f"chunk{i}")
            save(sub, "input", noisy[0].cpu().numpy())
            save(sub, "mag", frames(got["mag"]))
            save(sub, "pha", frames(got["pha"]))
            save(sub, "mag_raw", captured["mag_raw"][0, 0])
            save(sub, "phase_r", captured["phase_r"][0, 0])
            save(sub, "phase_i", captured["phase_i"][0, 0])
            save(sub, "amp", frames(got["amp"]))
            save(sub, "angle", frames(got["angle"]))
            save(sub, "zeroed", got["zeroed"].cpu().numpy().astype(np.uint8))
            save(sub, "output", got["output"][0].cpu().numpy())
            if i == 0:
                save(sub, "encoded", captured["encoded"][0])
                save(sub, "mamba_out", captured["mamba_out"][0])
                if entry["blocks"]:
                    save(sub, "blocks", np.stack([captured[f"block{b}"][0] for b in range(len(model.TSMamba))]))
                    for key in ("time_in", "time_out", "freq_in", "freq_out"):
                        save(sub, key, captured[key])
    for h in handles:
        h.remove()

    output = enhanced / window_sum
    assert output.shape == wav.shape
    if num_chunks == 1:
        with torch.no_grad():
            whole = enhance_chunk(wav, sizes)["output"]
        difference = float((output - whole).abs().max())
        assert difference <= 1e-6 * float(whole.abs().max()), difference

    save(folder, "audio", entry["audio"])
    save(folder, "output", output[0].cpu().numpy())
    meta = {
        "rate": sr,
        "n_fft": sizes[0],
        "hop_size": sizes[1],
        "win_size": sizes[2],
        "chunk_size_in_seconds": entry["chunk_size_in_seconds"],
        "hop_length_portion": entry["hop_length_portion"],
        "chunk_size": chunk_size,
        "hop_length": hop_length,
        "starts": starts,
        "zeroed_frames": zeroed_frames,
        "samples": int(wav.shape[1]),
    }
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(name, sr, "Hz,", num_chunks, "chunks, zeroed frames", zeroed_frames, flush=True)
