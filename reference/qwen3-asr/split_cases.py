"""Splits synthetic audio as qwen-asr's transcribe() splits audio too long for the model, and saves where, for
qwen3-asr-split-check.

usage: uv run python split_cases.py <out dir>

Each case goes to <out dir>/split/<case>/: audio.npy, the samples (float32, 16 kHz), and split.npy, where each part
starts and the audio ends in samples (int32, as dump.py saves integers), as split_audio_into_chunks() of qwen-asr 0.0.6
cuts them at 1200 s.
The audio is made from a seeded generator, so that the cases are the same on every run:

  noise-dips      2500 s of noise with quiet stretches of 0.3 s at three depths around 1200 s and 2400 s, 4 s
                  before, and 3 s and 6 s after: the second cut, which follows the first, reaches one the first does
                  not.
  speech-like     3700 s of noise under syllables of 0.12 to 0.3 s and pauses of 0.05 to 0.6 s: 4 parts.
  silence         1300 s of zeros: every window is as quiet as the first, which the cut goes into.
  short-tail      1200.2 s of noise whose quietest stretch is near the end: a last part shorter than 0.5 s, which
                  qwen-asr pads with zeros to 0.5 s and the C++ pads as every utterance is padded.
  exact           1200 s of noise, which is not split.
"""

import argparse
import os

import numpy as np

from official_utils import qwen_asr_utils

utils = qwen_asr_utils()
RATE = utils.SAMPLE_RATE


def noise(rng, seconds, level=0.1):
    return (rng.standard_normal(int(seconds * RATE)) * level).astype(np.float32)


def quiet(audio, at_seconds, seconds, factor):
    start = int(at_seconds * RATE)
    audio[start:start + int(seconds * RATE)] *= factor


def noise_dips(rng):
    audio = noise(rng, 2500)
    for cut in (1200, 2400):
        quiet(audio, cut + 3, 0.3, 0.01)
        quiet(audio, cut + 6, 0.3, 0.001)
        quiet(audio, cut - 4, 0.3, 0.05)
    return audio


def speech_like(rng):
    pieces, total = [], 0
    while total < 3700 * RATE:
        syllable = noise(rng, rng.uniform(0.12, 0.3), rng.uniform(0.05, 0.3))
        pause = noise(rng, rng.uniform(0.05, 0.6), 0.002)
        pieces += [syllable, pause]
        total += syllable.shape[0] + pause.shape[0]
    return np.concatenate(pieces)[:3700 * RATE]


def short_tail(rng):
    audio = noise(rng, 1200.2)
    quiet(audio, 1200.0, 0.15, 0.01)
    return audio


CASES = {
    "noise-dips": noise_dips,
    "speech-like": speech_like,
    "silence": lambda rng: np.zeros(1300 * RATE, dtype=np.float32),
    "short-tail": short_tail,
    "exact": lambda rng: noise(rng, 1200),
}

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
args = parser.parse_args()
for name, make in CASES.items():
    audio = make(np.random.default_rng(len(name)))
    parts = utils.split_audio_into_chunks(wav=audio, sr=RATE, max_chunk_sec=utils.MAX_ASR_INPUT_SECONDS)
    starts = [round(offset * RATE) for _, offset in parts]
    split = np.array(starts + [audio.shape[0]], dtype=np.int32)
    # The parts follow one another without a gap, each but a padded last one exactly as long as its bounds say.
    for k, (part, _) in enumerate(parts):
        assert part.shape[0] == max(split[k + 1] - split[k], int(utils.MIN_ASR_INPUT_SECONDS * RATE))
    folder = os.path.join(args.out_dir, "split", name)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "audio.npy"), audio)
    np.save(os.path.join(folder, "split.npy"), split)
    print(name, f"{audio.shape[0] / RATE:.2f} s", "split at", [f"{b / RATE:.4f}" for b in split[1:-1]], "parts",
          [part.shape[0] for part, _ in parts])
