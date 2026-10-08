# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow==25.0.1"]
# ///
"""Measures transcription by regions for each longest region (max_speech_duration_s) on long recordings joined from
public test clips, to choose the one its defaults give.

It cuts each clip of FLEURS ja_jp test (one reading of each sentence) and of the first clips of Common Voice 8.0 ja test
to where Silero VAD with its own defaults finds speech, brings each to an RMS of -23 dBFS, and joins them into
recordings of a few minutes, a pause of 0.3 to 1.0 s between two sentences and, between one in four, a silence of 2 to
8 s instead, the pauses and silences filled with white noise at -60 dBFS. For each longest region of 8, 10, 15, 20 and 30 s and none, `speech vad --split` with OpenAI's
server_vad defaults (threshold 0.5, speech_pad_ms 300, min_silence_duration_ms 500) writes the regions, and each model
recognizes them, Japanese forced; a region the caps share is recognized once. A row "whole" recognizes each recording
whole, without regions. For each it reports:

- the CER of the joined text against the clips' sentences, both written with NFKC, without punctuation, symbols or
  spaces, and in lower case, aligned within each group of clips and regions that overlap in time (the whole recording
  for "whole");
- the sentences dropped: clips of which half the characters or more are deleted in the alignment;
- the texts written where no one speaks: regions, or for "whole" the segments of a model that gives times, that overlap
  no clip and hold a character, with their characters.

Last, `speech asr --vad` with its defaults on every recording is compared with the regions' texts joined, which names
the longest region it gives. Everything it makes goes into the work folder, and a step that has its output there is not
run again.

usage: uv run --script measure/region_cap_compare.py <speech> <work dir> <FLEURS ja_jp test folder> <FLEURS ja_jp test.tsv>
                                                     <Common Voice 8.0 ja test folder> <Common Voice 8.0 ja test.parquet>
                                                     [--vad silero-vad] [--models reazonspeech-v2 parakeet-tdt_ctc-0.6b-ja qwen3-asr-0.6b]
"""

import array
import hashlib
import json
import os
import random
import struct
import subprocess
import sys
import unicodedata

import pyarrow.parquet

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checks", "smoke"))
from worker_client import joined_transcript  # noqa: E402

RATE = 16000
CAPS = [None, 30, 20, 15, 10, 8]
DETECTION = ["--threshold", "0.5", "--speech-pad-ms", "300", "--min-silence-duration-ms", "500"]
FLEURS_PER_RECORDING, COMMON_VOICE_PER_RECORDING, COMMON_VOICE_CLIPS = 16, 40, 600
BATCH = 300
# The RMS of each clip's speech, -23 dBFS, and of the noise of the pauses and silences, -60 dBFS.
LEVEL, NOISE = 10 ** (-23 / 20), 10 ** (-60 / 20)


def options(argv):
    named = {"--vad": ["silero-vad"], "--models": ["reazonspeech-v2", "parakeet-tdt_ctc-0.6b-ja", "qwen3-asr-0.6b"]}
    plain, at = [], 0
    while at < len(argv):
        if argv[at] in named:
            end = at + 1
            while end < len(argv) and not argv[end].startswith("--"):
                end += 1
            named[argv[at]] = argv[at + 1:end]
            at = end
        else:
            plain.append(argv[at])
            at += 1
    if len(plain) != 6:
        raise SystemExit("usage:" + __doc__.split("usage:")[1].rstrip())
    return plain, named["--vad"][0], named["--models"]


def speech_lines(speech, *args):
    r = subprocess.run([speech, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise SystemExit(f"speech {' '.join(args)[:200]} exited with {r.returncode}: {r.stderr.decode()[-800:]}")
    return [json.loads(line) for line in r.stdout.decode().splitlines()]


def in_batches(speech, args, files):
    out = []
    for at in range(0, len(files), BATCH):
        out += speech_lines(speech, *args, *files[at:at + BATCH])
    return out


def read_samples(path):
    with open(path, "rb") as f:
        data = f.read()
    at, fmt = 12, None
    while at < len(data):
        kind, size = data[at:at + 4], struct.unpack("<I", data[at + 4:at + 8])[0]
        if kind == b"fmt ":
            fmt = struct.unpack("<HHIIHH", data[at + 8:at + 24])
        elif kind == b"data":
            body = data[at + 8:at + 8 + size]
            assert fmt[1] == 1 and fmt[2] == RATE, (path, fmt)
            if fmt[0] == 3:
                return array.array("f", body).tolist()
            return [v / 32768 for v in array.array("h", body)]
        at += 8 + size + (size & 1)
    raise SystemExit(f"{path} has no data")


def write_wave(path, samples):
    pcm = array.array("h", [max(-32768, min(32767, round(x * 32768))) for x in samples]).tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16)
                + b"data" + struct.pack("<I", len(pcm)) + pcm)


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower() if unicodedata.category(c)[0] not in "PSZC")


def clips(fleurs_folder, fleurs_tsv, voice_folder, voice_parquet):
    """The clips of each set as (path, sentence): FLEURS's first reading of each sentence, and Common Voice's first clips."""
    fleurs, seen = [], set()
    with open(fleurs_tsv, encoding="utf-8") as f:
        for line in f:
            id, name, sentence = line.rstrip("\n").split("\t")[:3]
            if id not in seen:
                seen.add(id)
                fleurs.append((os.path.join(fleurs_folder, name), sentence))
    table = pyarrow.parquet.read_table(voice_parquet, columns=["audio", "transcription"]).to_pylist()
    voice = []
    for row in table:
        path = os.path.join(voice_folder, os.path.splitext(os.path.basename(row["audio"]["path"]))[0] + ".wav")
        if os.path.exists(path):
            voice.append((path, row["transcription"]))
        if len(voice) == COMMON_VOICE_CLIPS:
            break
    return {"fleurs": fleurs, "common-voice": voice}


def make_recordings(speech, vad, work, sets):
    """Recordings of each set and, for each, its clips' spans in seconds and sentences, kept in recordings.json."""
    manifest = os.path.join(work, "recordings.json")
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            return json.load(f)
    folder = os.path.join(work, "recordings")
    os.makedirs(folder, exist_ok=True)
    rng = random.Random(20261008)
    recordings = []
    for name, per in (("fleurs", FLEURS_PER_RECORDING), ("common-voice", COMMON_VOICE_PER_RECORDING)):
        found = in_batches(speech, ["vad", vad, "--format", "json"], [path for path, _ in sets[name]])
        speaking = [(path, sentence, r["regions"]) for (path, sentence), r in zip(sets[name], found) if r["regions"]]
        for at in range(0, len(speaking), per):
            samples, spans = [], []
            for k, (path, sentence, regions) in enumerate(speaking[at:at + per]):
                if k:
                    gap = rng.uniform(2, 8) if rng.random() < 0.25 else rng.uniform(0.3, 1.0)
                    samples += [rng.gauss(0, NOISE) for _ in range(round(gap * RATE))]
                clip = read_samples(path)[round(regions[0]["start"] * RATE):round(regions[-1]["end"] * RATE)]
                # A quiet clip, such as one of FLEURS that peaks at -42 dBFS, would otherwise sink into the noise of the pauses.
                gain = LEVEL / max(1e-9, (sum(x * x for x in clip) / len(clip)) ** 0.5)
                clip = [max(-1.0, min(1.0, x * gain)) for x in clip]
                spans.append({"start": len(samples) / RATE, "end": (len(samples) + len(clip)) / RATE, "sentence": sentence})
                samples += clip
            path = os.path.join(folder, f"{name}-{at // per + 1:02d}.wav")
            write_wave(path, samples)
            recordings.append({"set": name, "path": path, "seconds": len(samples) / RATE, "clips": spans})
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(recordings, f, ensure_ascii=False)
    return recordings


def cap_name(cap):
    return "none" if cap is None else f"{cap} s"


def find_regions(speech, vad, work, recordings, cap):
    """Each recording's regions with the longest region `cap`, each with the file --split wrote and its SHA-256."""
    folder = os.path.join(work, "regions", "none" if cap is None else str(cap))
    listed = os.path.join(folder, "regions.json")
    if os.path.exists(listed):
        with open(listed, encoding="utf-8") as f:
            return json.load(f)
    flags = DETECTION + ([] if cap is None else ["--max-speech-duration-s", str(cap)])
    found = speech_lines(speech, "vad", vad, "--format", "json", "--split", folder, *flags, *[r["path"] for r in recordings])
    out = []
    for recording, f in zip(recordings, found):
        stem, n = os.path.splitext(os.path.basename(recording["path"]))[0], len(f["regions"])
        regions = []
        for k, region in enumerate(f["regions"]):
            path = os.path.join(folder, f"{stem}-{str(k + 1).zfill(len(str(n)))}.wav")
            with open(path, "rb") as w:
                regions.append({**region, "path": path, "sha256": hashlib.sha256(w.read()).hexdigest()})
        out.append(regions)
    with open(listed, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def recognize(speech, work, model, files, extra=(), kept="texts"):
    """What the model answers for each file, by its SHA-256, kept in <kept>-<model>.json, each file recognized once."""
    cache_path = os.path.join(work, f"{kept}-{model.replace(':', '-')}.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    missing = {sha: path for sha, path in files.items() if sha not in cache}
    shas = list(missing)
    for at in range(0, len(shas), BATCH):
        batch = shas[at:at + BATCH]
        for sha, line in zip(batch, speech_lines(speech, "asr", model, "--format", "json", "--language", "ja", *extra, *[missing[s] for s in batch])):
            cache[sha] = line
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"  {model}: {min(at + BATCH, len(shas))} of {len(shas)} files recognized", file=sys.stderr)
    return cache


def align(ref, hyp):
    """The edit distance of `hyp` from `ref`, and for each character of `ref` whether the alignment deletes it."""
    n, m = len(ref), len(hyp)
    rows = [array.array("i", range(m + 1))]
    for i in range(1, n + 1):
        prev, row = rows[-1], array.array("i", [i]) * (m + 1)
        r = ref[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j - 1] + (r != hyp[j - 1]), prev[j] + 1, row[j - 1] + 1)
        rows.append(row)
    deleted = [False] * n
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and rows[i][j] == rows[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            i, j = i - 1, j - 1
        elif i > 0 and rows[i][j] == rows[i - 1][j] + 1:
            deleted[i - 1] = True
            i -= 1
        else:
            j -= 1
    return rows[n][m], deleted


def score_groups(groups):
    """Edits, reference characters, dropped clips and texts for silence of groups of (clip sentences, hypothesis texts)."""
    edits = chars = dropped = silent = silent_chars = 0
    for sentences, texts in groups:
        refs = [normalized(s) for s in sentences]
        hyp = "".join(normalized(t) for t in texts)
        if not refs:
            silent += sum(1 for t in texts if normalized(t))
            silent_chars += len(hyp)
            edits += len(hyp)
            continue
        e, deleted = align("".join(refs), hyp)
        edits += e
        chars += sum(len(r) for r in refs)
        at = 0
        for r in refs:
            dropped += len(r) > 0 and sum(deleted[at:at + len(r)]) * 2 >= len(r)
            at += len(r)
    return edits, chars, dropped, silent, silent_chars


def region_groups(recording, regions, texts):
    """The clips and regions of a recording grouped by overlap in time, each group's sentences and texts in order."""
    clips = recording["clips"]
    parent = list(range(len(clips) + len(regions)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, region in enumerate(regions):
        for i, clip in enumerate(clips):
            if region["start"] < clip["end"] and region["end"] > clip["start"]:
                parent[root(len(clips) + k)] = root(i)
    groups = {}
    for i, clip in enumerate(clips):
        groups.setdefault(root(i), ([], []))[0].append(clip["sentence"])
    for k in range(len(regions)):
        groups.setdefault(root(len(clips) + k), ([], []))[1].append(texts[k])
    return list(groups.values())


def main():
    (speech, work, fleurs_folder, fleurs_tsv, voice_folder, voice_parquet), vad, models = options(sys.argv[1:])
    os.makedirs(work, exist_ok=True)
    recordings = make_recordings(speech, vad, work, clips(fleurs_folder, fleurs_tsv, voice_folder, voice_parquet))
    sets = sorted({r["set"] for r in recordings}, key=lambda s: s != "fleurs")
    for s in sets:
        chosen = [r for r in recordings if r["set"] == s]
        print(f"{s}: {len(chosen)} recordings, {sum(len(r['clips']) for r in chosen)} clips, {sum(r['seconds'] for r in chosen) / 60:.1f} min")
    by_cap = {cap: find_regions(speech, vad, work, recordings, cap) for cap in CAPS}
    for cap, found in by_cap.items():
        lengths = [r["end"] - r["start"] for regions in found for r in regions]
        print(f"longest region {cap_name(cap)}: {len(lengths)} regions, the longest {max(lengths):.1f} s, "
              f"{sum(1 for x in lengths if x > 15)} over 15 s")
    files = {r["sha256"]: r["path"] for found in by_cap.values() for regions in found for r in regions}
    whole = {hashlib.sha256(open(r["path"], "rb").read()).hexdigest(): r["path"] for r in recordings}
    rows = []
    for model in models:
        info = speech_lines(speech, "info", model, "--json")[0]
        timed = any(o["name"] == "timestamps" for o in info["options"])
        texts = recognize(speech, work, model, files)
        wholes = recognize_whole(speech, work, model, whole, timed)
        for setting in ["whole"] + CAPS:
            row = {"model": model, "setting": "whole" if setting == "whole" else cap_name(setting)}
            for s in sets:
                groups, silent_segments = [], 0
                for at, recording in enumerate(recordings):
                    if recording["set"] != s:
                        continue
                    if setting == "whole":
                        got = wholes[hashlib.sha256(open(recording["path"], "rb").read()).hexdigest()]
                        groups.append(([c["sentence"] for c in recording["clips"]], [got["text"]]))
                        for segment in got.get("segments", []):
                            if normalized(segment["text"]) and not any(segment["start"] < c["end"] and segment["end"] > c["start"]
                                                                       for c in recording["clips"]):
                                silent_segments += 1
                    else:
                        regions = by_cap[setting][at]
                        groups += region_groups(recording, regions, [texts[r["sha256"]]["text"] for r in regions])
                edits, chars, dropped, silent, silent_chars = score_groups(groups)
                if setting == "whole":
                    silent = silent_segments if timed else None
                row[s] = {"cer": edits / chars, "dropped": dropped, "clips": sum(len(r["clips"]) for r in recordings if r["set"] == s),
                          "silent": silent, "silent_chars": silent_chars}
            rows.append(row)
        check_defaults(speech, vad, model, recordings, by_cap, texts)
    print()
    print("| Model | Longest region | " + " | ".join(f"{s} CER | {s} dropped" for s in sets) + " | texts where no one speaks |")
    print("|---|---|" + "---|---|" * len(sets) + "---|")
    for row in rows:
        cells = []
        silent = 0
        for s in sets:
            v = row[s]
            cells += [f"{v['cer'] * 100:.2f}%", f"{v['dropped']} of {v['clips']}"]
            silent = None if v["silent"] is None or silent is None else silent + v["silent"]
        print(f"| {row['model']} | {row['setting']} | " + " | ".join(cells) + f" | {'–' if silent is None else silent} |")


def recognize_whole(speech, work, model, files, timed):
    """Each recording recognized whole, with segments where the model gives times."""
    return recognize(speech, work, model, files, ["--timestamps"] if timed else [], "whole")


def check_defaults(speech, vad, model, recordings, by_cap, texts):
    """Compares speech asr --vad with its defaults with each longest region's texts joined, and says which it gives."""
    got = speech_lines(speech, "asr", model, "--vad", vad, "--format", "json", "--language", "ja", *[r["path"] for r in recordings])
    same = []
    for cap in CAPS:
        joined = [joined_transcript([(0, texts[r["sha256"]]) for r in regions], False)["text"] for regions in by_cap[cap]]
        if joined == [g["text"] for g in got]:
            same.append(cap_name(cap))
    print(f"{model}: speech asr --vad {vad} with its defaults gives the texts of the longest region "
          f"{', '.join(same) if same else 'of none of the caps'}", file=sys.stderr)


if __name__ == "__main__":
    main()
