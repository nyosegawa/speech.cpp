"""The inputs dump.py runs, made from FLEURS' test split at the pinned revision.

Each has the FLEURS configuration it comes from, its utterances by file name (or the first n of the split's TSV), the
language qwen-asr is told when the language is forced, the prompt (qwen-asr's context) of the requests that carry
one, and, where not every model runs it, the models that do. The prompts name the proper nouns and terms of the
utterance, as a caller who knows the topic would. An input is one utterance, a cut of one, or utterances joined end
to end.
"""

import io
import tarfile

import numpy as np
import soundfile

from pins import fleurs_test

SAMPLE_RATE = 16000

INPUTS = {
    # Under 8 s, so one attention window.
    "ja_jp-12677001980660723842": {
        "config": "ja_jp", "files": ["12677001980660723842.wav"], "language": "Japanese", "prompt": "群島、湖、ヨット"},
    "en_us-10197164397713068203": {
        "config": "en_us", "files": ["10197164397713068203.wav"], "language": "English", "prompt": "cabbage juice, alkaline"},
    "cmn_hans_cn-12933878060487367144": {
        "config": "cmn_hans_cn", "files": ["12933878060487367144.wav"], "language": "Chinese", "prompt": "分离、重组、变异"},
    # 8 to 20 s: two windows.
    "ja_jp-13903496305700695803": {
        "config": "ja_jp", "files": ["13903496305700695803.wav"], "language": "Japanese", "prompt": "光合成、養分、日陰"},
    "cmn_hans_cn-14716260585206763911": {
        "config": "cmn_hans_cn", "files": ["14716260585206763911.wav"], "language": "Chinese",
        "prompt": "邓迪大学（University of Dundee）、Pamela Ferguson"},
    "de_de-10009182821551087671": {
        "config": "de_de", "files": ["10009182821551087671.wav"], "language": "German", "prompt": "Dinosaurier, T. Rex"},
    # 20 to 30 s: three or four windows.
    "ja_jp-2630315561484880103": {
        "config": "ja_jp", "files": ["2630315561484880103.wav"], "language": "Japanese",
        "prompt": "米国地質調査所(USGS)、アリゾナ州フラッグスタッフ、ノーザンアリゾナ大学、グレン・クッシング"},
    "en_us-2880067776280655708": {
        "config": "en_us", "files": ["2880067776280655708.wav"], "language": "English",
        "prompt": "Dr. Malar Balasubramanian, Blue Ash, Ohio, Cincinnati"},
    # 群島 and a little of what follows: shorter than one chunk of 100 frames (1 s), which transformers' encoder pads
    # with zeros to 100 frames and qwen-asr's, given the utterance alone, does not.
    "ja_jp-12677001980660723842-cut": {
        "config": "ja_jp", "files": ["12677001980660723842.wav"], "start": 0.6, "seconds": 0.9, "language": "Japanese",
        "prompt": "群島、湖、ヨット"},
    # The 0.6 s before the speech of the first input begins, near silence (an RMS of 1e-4 in 0.1 s windows), in which
    # the 0.6B model hears no language and writes "language None", and the 1.7B model a Chinese filler.
    "ja_jp-12677001980660723842-silence": {
        "config": "ja_jp", "files": ["12677001980660723842.wav"], "start": 0.0, "seconds": 0.6, "language": "Japanese",
        "prompt": "群島、湖、ヨット"},
    # 1338.42 s, which qwen-asr splits near 1200 s; dumped only to its split and its text. Where the split falls does
    # not depend on the model, and the 1.7B model in float32 takes about 17 GB and over an hour for it on an Apple M5,
    # so only the 0.6B model runs it.
    "ja_jp-first-100": {"config": "ja_jp", "first": 100, "language": "Japanese", "prompt": None,
                        "models": ["Qwen3-ASR-0.6B"]},
}


def tsv_rows(path):
    """The rows of a FLEURS TSV: id, file name, raw transcription, normalized transcription, characters, samples, gender."""
    with open(path, encoding="utf-8") as f:
        return [line.split("\t") for line in f.read().splitlines() if line.strip()]


def read(name):
    """The samples of an input, mono float32 at 16 kHz as FLEURS stores them, and the TSV rows of its utterances."""
    spec = INPUTS[name]
    paths = fleurs_test(spec["config"])
    rows = tsv_rows(paths["tsv"])
    if "first" in spec:
        rows = rows[: spec["first"]]
    else:
        rows = [next(row for row in rows if row[1] == file) for file in spec["files"]]
    pieces = {}
    wanted = {row[1] for row in rows}
    with tarfile.open(paths["audio"], "r|gz") as tar:
        for member in tar:
            file = member.name.removeprefix("test/")
            if file in wanted:
                pieces[file] = tar.extractfile(member).read()
                if len(pieces) == len(wanted):
                    break
    samples = []
    for row in rows:
        audio, rate = soundfile.read(io.BytesIO(pieces[row[1]]), dtype="float32")
        assert rate == SAMPLE_RATE and audio.ndim == 1 and audio.shape[0] == int(row[5]), f"{row[1]} is not as its TSV says"
        samples.append(audio)
    audio = np.concatenate(samples)
    if "start" in spec:
        start = round(spec["start"] * SAMPLE_RATE)
        audio = audio[start : start + round(spec["seconds"] * SAMPLE_RATE)]
    return audio, rows
