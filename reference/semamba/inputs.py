"""The inputs dump.py runs: FLEURS clips that may be published, under babble made of other FLEURS clips or seeded white
noise, at the rates the model card lists, with the chunk settings a request can take.

The clips are read from a folder laid out as speech-bench keeps its data (fleurs/<revision>/<config>/test/<id>.wav,
FLEURS at the pinned revision of google/fleurs, 16 kHz), each checked against its SHA-256. Speech and noise are mixed
at 16 kHz at a signal-to-noise ratio from their RMS; an input at another rate resamples the speech with torchaudio's
default resampler and adds its white noise at that rate, so that the noise fills the band the rate has.
"""

import hashlib
import os

import numpy as np
import soundfile
import torch
import torchaudio

from pins import CHUNK_SIZE_IN_SECONDS, HOP_LENGTH_PORTION

RATE = 16000
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
CLIPS = {
    "ja_jp/10020345318418093976": "bd1b9533cfa176b8c2a71ecb4821a02bc86fa368d096373efe8d8dceafb35634",
    "en_us/10197164397713068203": "b174be7730f0aab6a26762130b66f2125392b50550645727bd057d8479fb6219",
    "en_us/10052240106321793346": "7603cae3294309ef70d234180cefccc3b13865ee525262228a636069b9b62841",
    "fr_fr/11644651343215870474": "15566b986a205b1b7f67b401bf0be9adf1f837bad2f01f1f7e55a583c8e28faf",
    "de_de/10347138311808731867": "1cba9b708b7a4157497f5e2d358aaa726137b4d1378ed3f74aaa29882fd979ee",
    "de_de/10504450071805589678": "db6b2e990e13245f95ba92227cf6c22d1903ce6a34e2b86074523a7aa512637d",
    "de_de/1144628591089987683": "5a822ac5a2080084f25eb34d3dda70e2d928f6f4fa066a7338d3de127101aa65",
}


def clip(data, name, start=0.0, seconds=None):
    """A FLEURS clip at 16 kHz from `start` seconds, `seconds` long or to its end, checked against its pin."""
    config, ident = name.split("/")
    path = os.path.join(data, "fleurs", FLEURS_REVISION, config, "test", ident + ".wav")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != CLIPS[name]:
        raise SystemExit(f"{path} has the SHA-256 {digest}, not the pinned {CLIPS[name]}")
    audio, rate = soundfile.read(path, dtype="float32")
    assert rate == RATE and audio.ndim == 1, (path, rate, audio.shape)
    begin = int(round(start * RATE))
    end = len(audio) if seconds is None else begin + int(round(seconds * RATE))
    assert end <= len(audio), (name, start, seconds)
    return audio[begin:end]


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def at_snr(speech, noise, snr_db):
    """`speech` with `noise` of the same length added at `snr_db` from their RMS."""
    return (speech + noise * (rms(speech) / rms(noise) / 10 ** (snr_db / 20))).astype(np.float32)


def babble(data, samples):
    """Three German speakers at once from FLEURS, each clip from a different point, `samples` long at 16 kHz."""
    voices = [
        clip(data, "de_de/10347138311808731867", 1.0),
        clip(data, "de_de/10504450071805589678", 4.0),
        clip(data, "de_de/1144628591089987683", 2.0),
    ]
    assert all(len(v) >= samples for v in voices)
    return sum(v[:samples] for v in voices).astype(np.float32)


def white(samples, seed):
    return np.random.default_rng(seed).standard_normal(samples).astype(np.float32)


def resample(audio, rate):
    if rate == RATE:
        return audio
    return torchaudio.functional.resample(torch.from_numpy(audio), RATE, rate).numpy().astype(np.float32)


def noisy(data):
    """12 s: a Japanese sentence under babble at 5 dB, 1 s of digital silence, an English sentence under white noise at
    10 dB."""
    first = clip(data, "ja_jp/10020345318418093976", 0.0, 5.5)
    second = clip(data, "en_us/10052240106321793346", 0.0, 5.5)
    return np.concatenate([
        at_snr(first, babble(data, len(first)), 5),
        np.zeros(RATE, dtype=np.float32),
        at_snr(second, white(len(second), 1), 10),
    ])


def at_rate(data, rate, seconds, seed):
    """An English sentence at `rate` under white noise of that rate at 10 dB, `seconds` long."""
    speech = resample(clip(data, "en_us/10197164397713068203", 0.0, seconds), rate)
    return at_snr(speech, white(len(speech), seed), 10)


def inputs(data):
    """Each input's audio, its rate, its chunk settings and whether its dump keeps the output of every Mamba block."""

    def entry(audio, rate=RATE, chunk=CHUNK_SIZE_IN_SECONDS, hop=HOP_LENGTH_PORTION, blocks=False):
        return {"audio": audio, "rate": rate, "chunk_size_in_seconds": chunk, "hop_length_portion": hop, "blocks": blocks}

    whole = noisy(data)
    short = clip(data, "ja_jp/10020345318418093976", 1.0, 1.2)
    fr = resample(clip(data, "fr_fr/11644651343215870474"), 48000)
    fr = np.concatenate([fr, np.zeros(7 * 48000 - len(fr), dtype=np.float32)])
    return {
        # 1.2 s under babble, one chunk, with every block's output.
        "short": entry(at_snr(short, babble(data, len(short)), 5), blocks=True),
        # 12 s in four chunks of the default settings, the last aligned with the end.
        "noisy": entry(whole),
        # The same audio in one chunk, which is what inference.py gives.
        "noisy-whole": entry(whole, chunk=30.0),
        # Its first 7 s in chunks a quarter apart.
        "noisy-quarter": entry(whole[: 7 * RATE], hop=0.25),
        # A French sentence and silence at 48 kHz under white noise of the full band, 7 s in two chunks.
        "white-48k": entry(at_snr(fr, white(len(fr), 2), 10), rate=48000),
        "rate-8000": entry(at_rate(data, 8000, 3.0, 3), rate=8000),
        "rate-22050": entry(at_rate(data, 22050, 3.0, 4), rate=22050),
        "rate-44100": entry(at_rate(data, 44100, 3.0, 5), rate=44100),
        # 400 samples, the fewest of the inputs: six STFT frames, two after the encoder.
        "tiny": entry(at_snr(short[:400], white(400, 6), 10)),
    }
