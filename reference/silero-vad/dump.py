"""Runs Silero VAD's 16 kHz model on the CPU in float32 and saves the tensors and regions the C++ port is checked against.

usage: uv run python dump.py <out dir> [input...] [--data <folder>]

Each input, a name from inputs.py (every one when none is given), read from the clips under <folder> (by default
~/speech-bench-data), goes to <out dir>/silero-vad/<input>/. The official get_speech_timestamps() runs the model over
the whole input, chunk by chunk as the package does: the samples cut into chunks of 512, the last padded with zeros, each
given to the model with the 64 samples before it (zeros before the first) and the LSTM state the chunk before left
(zeros before the first, after reset_states()). A wrapper around the model records what each call took and gave; the
finer stages are then computed again from those inputs with the model's own submodules, and asserted to give the
official probability and state bit for bit. Float32 throughout, per chunk in the order data flows:

  audio      the samples, [N]
  input      the 64 samples before the chunk and the chunk's 512, which the model reflects by 64 at the end, [n, 576]
  stft       the magnitude of the STFT of the reflected input, 4 frames of 129 bins, [n, 129, 4]
  block0     the output of each encoder block, a convolution and a ReLU, [n, 128, 4], [n, 64, 2], [n, 64, 1] and
  block1     [n, 128, 1]
  block2
  block3
  lstm_h     the LSTM cell's state after the chunk, h (its output) and c, [n, 128] each
  lstm_c
  probs      the speech probability of each chunk, [n]

regions.json holds the regions, in samples, that the official get_speech_timestamps() gives the input with each set of
options in OPTION_SETS, each asserted to be what get_speech_timestamps_from_probs() gives for probs.npy, and meta.json
the versions, the pin, the input's pieces and the lengths.
"""

import argparse
import importlib.metadata
import inspect
import json
import os

import numpy as np
import torch
from silero_vad import get_speech_timestamps, get_speech_timestamps_from_probs

import inputs
from pins import MODEL, PACKAGE, load

# The options of get_speech_timestamps() a request sets, each set's regions dumped: the defaults; stricter and looser
# settings of the four options; and max_speech_duration_s, which cuts a region longer than it at the longest silence
# within it or, without one, where it reaches the limit, with the official defaults of min_silence_at_max_speech and
# use_max_poss_sil_at_max_speech. Read speech pauses for breath often enough that no region of the default options
# reaches 10 s on these inputs; a silence of 1 s before a region ends joins sentences into regions that do, which the
# limit then cuts.
OPTION_SETS = {
    "default": {},
    "strict": {"threshold": 0.7, "min_speech_duration_ms": 500, "min_silence_duration_ms": 300, "speech_pad_ms": 100},
    "loose": {"threshold": 0.3, "min_speech_duration_ms": 0, "min_silence_duration_ms": 0, "speech_pad_ms": 0},
    "max10": {"max_speech_duration_s": 10.0},
    "silence1000": {"min_silence_duration_ms": 1000},
    "max10-silence1000": {"max_speech_duration_s": 10.0, "min_silence_duration_ms": 1000},
    "max3": {"max_speech_duration_s": 3.0, "speech_pad_ms": 60},
}

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("inputs", nargs="*", metavar="input")
parser.add_argument("--data", default=os.path.expanduser("~/speech-bench-data"))
args = parser.parse_args()
for name in args.inputs:
    if name not in inputs.INPUTS:
        parser.error(f"{name} is not an input of inputs.py: {', '.join(sorted(inputs.INPUTS))}")

model = load()
vad = model._model
blocks = [getattr(vad.encoder, str(i)) for i in range(4)]


class Recording:
    """The model as get_speech_timestamps() calls it, recording each chunk's input, state and probability."""

    def __init__(self):
        self.calls = []

    def reset_states(self):
        model.reset_states()
        self.calls = []

    def __call__(self, chunk, sampling_rate):
        # The model keeps the input's last 64 samples as the next chunk's context, which is empty after reset_states().
        context = model._context if len(model._context) else torch.zeros(1, vad.context_size_samples)
        before = model._state
        prob = model(chunk, sampling_rate)
        self.calls.append((torch.cat([context, chunk.unsqueeze(0)], 1)[0].clone(), before.clone(), model._state.clone(), prob.clone()))
        return prob


def stages(calls):
    """Each chunk's stages computed again with the model's submodules from the recorded input and state."""
    out = {k: [] for k in ("input", "stft", "block0", "block1", "block2", "block3", "lstm_h", "lstm_c", "probs")}
    for x, before, after, prob in calls:
        magnitude = vad.stft(x.unsqueeze(0))
        y = magnitude
        out["input"].append(x)
        out["stft"].append(magnitude[0])
        for i, block in enumerate(blocks):
            y = block(y)
            out[f"block{i}"].append(y[0])
        decoded, state = vad.decoder(y, before)
        p = torch.unsqueeze(torch.mean(torch.squeeze(decoded, 1), [1]), 1)
        assert torch.equal(p, prob) and torch.equal(state, after), "the submodules do not give what the model gave"
        out["lstm_h"].append(state[0, 0])
        out["lstm_c"].append(state[1, 0])
        out["probs"].append(p[0, 0])
    return {k: torch.stack(v).numpy().astype(np.float32) for k, v in out.items()}


defaults = {k: v.default for k, v in inspect.signature(get_speech_timestamps).parameters.items() if v.default is not inspect.Parameter.empty}
for name in args.inputs or sorted(inputs.INPUTS):
    audio = inputs.read(args.data, name)
    folder = os.path.join(args.out_dir, "silero-vad", name)
    os.makedirs(folder, exist_ok=True)
    recording = Recording()
    regions, probs = {}, None
    for set_name, options in OPTION_SETS.items():
        official = get_speech_timestamps(torch.from_numpy(audio), recording, sampling_rate=16000, **options)
        chunk_probs = [float(c[3]) for c in recording.calls]
        assert probs is None or probs == chunk_probs, "the model gave other probabilities on another run"
        probs = chunk_probs
        assert official == get_speech_timestamps_from_probs(probs, sampling_rate=16000, audio_length_samples=len(audio), **options)
        regions[set_name] = {"options": options, "regions": [[r["start"], r["end"]] for r in official]}
    dumped = stages(recording.calls)
    np.save(os.path.join(folder, "audio.npy"), audio)
    for k, v in dumped.items():
        np.save(os.path.join(folder, f"{k}.npy"), v)
    with open(os.path.join(folder, "regions.json"), "w") as f:
        json.dump(regions, f, indent=1)
    meta = {
        "package": PACKAGE, "version": importlib.metadata.version(PACKAGE), "torch": torch.__version__, "model": MODEL,
        "input": name, "pieces": inputs.INPUTS[name], "samples": len(audio), "chunks": len(probs),
        "defaults": {k: v for k, v in defaults.items() if k in ("threshold", "min_speech_duration_ms", "min_silence_duration_ms", "speech_pad_ms")},
    }
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    counts = ", ".join(f"{k} {len(v['regions'])}" for k, v in regions.items())
    print(f"{name}: {len(audio) / 16000:.2f} s, {len(probs)} chunks, regions: {counts}")
