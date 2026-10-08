"""The inputs dump.py runs, made from FLEURS and Common Voice clips that may be published, silence and seeded noise.

Each input is a list of pieces joined end to end at 16 kHz: a clip, whole or from `start` seconds for `samples`
samples, digital silence, or white noise of a given RMS from a fixed seed; a cut keeps the first seconds of what comes
before it. The clips are read from a folder laid out as speech-bench keeps its data
(fleurs/<revision>/<config>/test/<id>.wav, FLEURS at the pinned revision of google/fleurs, and
common-voice/8.0/ja/test/<name>.wav, Common Voice 8.0's Japanese test split), each checked against its SHA-256.
"""

import hashlib
import os

import numpy as np
import soundfile

SAMPLE_RATE = 16000
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
CLIPS = {
    "fleurs/ja_jp/10020345318418093976": "bd1b9533cfa176b8c2a71ecb4821a02bc86fa368d096373efe8d8dceafb35634",
    "fleurs/en_us/10197164397713068203": "b174be7730f0aab6a26762130b66f2125392b50550645727bd057d8479fb6219",
    "fleurs/en_us/10052240106321793346": "7603cae3294309ef70d234180cefccc3b13865ee525262228a636069b9b62841",
    "fleurs/fr_fr/11644651343215870474": "15566b986a205b1b7f67b401bf0be9adf1f837bad2f01f1f7e55a583c8e28faf",
    "fleurs/de_de/10347138311808731867": "1cba9b708b7a4157497f5e2d358aaa726137b4d1378ed3f74aaa29882fd979ee",
    "fleurs/de_de/10504450071805589678": "db6b2e990e13245f95ba92227cf6c22d1903ce6a34e2b86074523a7aa512637d",
    "fleurs/de_de/1144628591089987683": "5a822ac5a2080084f25eb34d3dda70e2d928f6f4fa066a7338d3de127101aa65",
    "common-voice/ja/common_voice_ja_19482480": "cd66e4c4fd4abad513dff819cc6bc414bf25eb71c6212b8b75c4b9642485fe8d",
    "common-voice/ja/common_voice_ja_19482501": "e1b623ab1ae24faf33105d62f5082e8ef8c9395dbba36088d3a12b1a257069aa",
}

INPUTS = {
    # Sentences of four speakers in three languages, with pauses of 0.7, 2.0 and 0.3 s between them, 30.7 s, whose
    # length is no whole number of chunks.
    "pauses": [
        {"clip": "fleurs/ja_jp/10020345318418093976"},
        {"silence": 0.7},
        {"clip": "fleurs/en_us/10197164397713068203"},
        {"silence": 2.0},
        {"clip": "common-voice/ja/common_voice_ja_19482480"},
        {"silence": 0.3},
        {"clip": "fleurs/fr_fr/11644651343215870474"},
    ],
    # 4 s of digital silence and 6 s of faint white noise at -70 dBFS, without speech.
    "quiet": [{"silence": 4.0}, {"noise": 6.0, "rms": 3e-4, "seed": 1}],
    # Faint noise around a sentence: silence, noise, the sentence, noise.
    "noise-around-speech": [
        {"silence": 2.0},
        {"noise": 3.0, "rms": 3e-4, "seed": 2},
        {"clip": "fleurs/en_us/10052240106321793346"},
        {"noise": 3.0, "rms": 3e-4, "seed": 3},
    ],
    # A Common Voice clip from 0.88 s, the first 20 ms whose RMS exceeds 0.02, so that the first chunk holds speech.
    "starts-with-speech": [{"clip": "common-voice/ja/common_voice_ja_19482501", "start": 0.88}],
    # 300 samples from within the same speech, less than one chunk of 512.
    "short": [{"clip": "common-voice/ja/common_voice_ja_19482501", "start": 1.5, "samples": 300}],
    # Three German clips of 18 to 23 s, each several sentences, joined with 50 ms between them and cut at 60 s: long
    # stretches of speech for max_speech_duration_s, and the minute the speed is measured on.
    "long": [
        {"clip": "fleurs/de_de/10347138311808731867"},
        {"silence": 0.05},
        {"clip": "fleurs/de_de/10504450071805589678"},
        {"silence": 0.05},
        {"clip": "fleurs/de_de/1144628591089987683"},
        {"cut": 60.0},
    ],
}


def clip_path(data, clip):
    source, config, name = clip.split("/")
    if source == "fleurs":
        return os.path.join(data, "fleurs", FLEURS_REVISION, config, "test", name + ".wav")
    return os.path.join(data, "common-voice", "8.0", config, "test", name + ".wav")


def read_clip(data, clip):
    path = clip_path(data, clip)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != CLIPS[clip]:
        raise SystemExit(f"{path} has the SHA-256 {digest}, not the pinned {CLIPS[clip]}")
    audio, rate = soundfile.read(path, dtype="float32")
    assert rate == SAMPLE_RATE and audio.ndim == 1, f"{path} is not mono at {SAMPLE_RATE} Hz"
    return audio


def read(data, name):
    """The samples of an input, mono float32 at 16 kHz."""
    pieces = []
    for piece in INPUTS[name]:
        if "clip" in piece:
            audio = read_clip(data, piece["clip"])
            start = round(piece.get("start", 0) * SAMPLE_RATE)
            audio = audio[start:start + piece["samples"]] if "samples" in piece else audio[start:]
        elif "cut" in piece:
            pieces = [np.concatenate(pieces)[:round(piece["cut"] * SAMPLE_RATE)]]
            continue
        elif "silence" in piece:
            audio = np.zeros(round(piece["silence"] * SAMPLE_RATE), dtype=np.float32)
        else:
            noise = np.random.default_rng(piece["seed"]).standard_normal(round(piece["noise"] * SAMPLE_RATE))
            audio = (noise * piece["rms"]).astype(np.float32)
        pieces.append(audio)
    return np.concatenate(pieces)
