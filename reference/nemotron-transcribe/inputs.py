"""The inputs dump.py runs, made from FLEURS clips that may be published, silence and seeded noise.

Each input is a length and a list of pieces added into it at 16 kHz: a stretch of a clip from `start` for `seconds`
seconds (to its end without `seconds`), scaled by its gain and placed `at` seconds into the input, which overlaps
whatever else is there, or white noise of a given RMS from a fixed seed; the rest is digital silence. The clips are read
from a folder laid out as speech-bench keeps its data (fleurs/<revision>/<config>/test/<id>.wav, FLEURS at the pinned
revision of google/fleurs), each checked against its SHA-256. The meeting is reference/sortformer's, so that the speaker
branch, which is Nemotron 3 Diarization, hears what that port's dumps hold.
"""

import hashlib
import os

import numpy as np
import soundfile

SAMPLE_RATE = 16000
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
CLIPS = {
    "fleurs/ja_jp/9518252661993015549": "cd11223cdb82b9f739cf7fc8febb52807abc75d5090cc8bc8ebdf01379a888be",
    "fleurs/ja_jp/8922261396806111795": "133ebe49c5fa8049d4c614705a2616eb49f31d49892ba3dc01e436886c05dd9c",
    "fleurs/en_us/2880067776280655708": "ffc7125c2d9c8d4664b7e2caf7e32db3440c271f6787d3ead4e00d1891d82b34",
    "fleurs/en_us/2508559905589212221": "6dc676eae80109d088233f838226d4164f76befd8aafc772e1a6cb9eaa0d31df",
    "fleurs/ja_jp/12382486935508330615": "4efa91de96362664afdc029efd0d24e88872a95884e9ab1cd7a6655edd729dff",
}
A, B, C, D, E = CLIPS

# The gain that brings each clip's speech to an RMS of about 0.04 (-28 dBFS) in the meeting: FLEURS' recordings lie
# between 0.001 and 0.15. A piece's own gain replaces its clip's.
GAINS = {A: 50.0, B: 0.3, C: 25.0, D: 1.6, E: 0.3}

INPUTS = {
    # One Japanese voice as FLEURS recorded it, shorter than a window of the encoder.
    "ja": {"seconds": 10.0, "pieces": [{"clip": B, "at": 0.0, "start": 8.5, "seconds": 10.0, "gain": 1.0}]},
    # One English voice as FLEURS recorded it.
    "en": {"seconds": 9.0, "pieces": [{"clip": D, "at": 0.0, "start": 0.0, "seconds": 9.0, "gain": 1.0}]},
    # Five voices taking turns for 98 s, with pauses, overlaps of 2 and 1.5 s, and a fifth voice that first speaks at
    # 63.5 s. B's and E's clips are silent for their first 8.5 and 6 s, which their pieces skip. Its length, 98.0123 s,
    # is no whole number of hops, encoder frames or windows, and it takes three of the encoder's 40 s windows.
    "meeting": {
        "seconds": 98.0123,
        "pieces": [
            {"clip": A, "at": 0.5, "start": 0.0, "seconds": 7.0},
            {"clip": B, "at": 8.0, "start": 8.5, "seconds": 6.0},
            {"clip": A, "at": 14.5, "start": 7.0, "seconds": 6.5},
            {"clip": C, "at": 21.5, "start": 0.0, "seconds": 7.5},
            {"clip": B, "at": 27.0, "start": 14.5, "seconds": 3.0},
            {"clip": D, "at": 30.5, "start": 0.0, "seconds": 9.0},
            {"clip": A, "at": 40.0, "start": 13.5, "seconds": 6.5},
            {"clip": B, "at": 47.0, "start": 17.5, "seconds": 5.5},
            {"clip": D, "at": 51.0, "start": 9.0, "seconds": 2.0},
            {"clip": C, "at": 55.5, "start": 7.5, "seconds": 7.5},
            {"clip": E, "at": 63.5, "start": 6.0, "seconds": 8.0},
            {"clip": D, "at": 72.0, "start": 11.0, "seconds": 8.0},
            {"clip": A, "at": 80.5, "start": 20.0, "seconds": 7.0},
            {"clip": C, "at": 88.0, "start": 15.0},
        ],
    },
    # 3 s of digital silence and 3 s of faint white noise at -70 dBFS, without speech.
    "quiet": {"seconds": 6.0, "pieces": [{"noise": 3.0, "at": 3.0, "rms": 3e-4, "seed": 1}]},
}


def clip(data, name):
    """The samples of a pinned clip, checked against its SHA-256."""
    corpus, config, clip_id = name.split("/")
    path = os.path.join(data, corpus, FLEURS_REVISION, config, "test", clip_id + ".wav")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != CLIPS[name]:
        raise SystemExit(f"{path} has the SHA-256 {digest}, not the pinned {CLIPS[name]}")
    samples, rate = soundfile.read(path, dtype="float32")
    if rate != SAMPLE_RATE or samples.ndim != 1:
        raise SystemExit(f"{path} is not mono at {SAMPLE_RATE} Hz")
    return samples


def make(data, name):
    """The samples of an input, float32 at SAMPLE_RATE."""
    spec = INPUTS[name]
    out = np.zeros(round(spec["seconds"] * SAMPLE_RATE), dtype=np.float32)
    for piece in spec["pieces"]:
        at = round(piece["at"] * SAMPLE_RATE)
        if "clip" in piece:
            samples = clip(data, piece["clip"])
            first = round(piece["start"] * SAMPLE_RATE)
            last = len(samples) if "seconds" not in piece else first + round(piece["seconds"] * SAMPLE_RATE)
            if last > len(samples):
                raise SystemExit(f"{piece['clip']} has {len(samples) / SAMPLE_RATE:.2f} s, less than the piece asks")
            samples = samples[first:last] * np.float32(piece.get("gain", GAINS[piece["clip"]]))
        else:
            rng = np.random.default_rng(piece["seed"])
            samples = (rng.standard_normal(round(piece["noise"] * SAMPLE_RATE)) * piece["rms"]).astype(np.float32)
        if at + len(samples) > len(out):
            raise SystemExit(f"a piece of {name} ends after the input's {spec['seconds']} s")
        out[at : at + len(samples)] += samples
    return out
