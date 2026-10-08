"""Checks the `speech` command line against the worker, whose protocol worker_smoke.py and worker_recognition_smoke.py
check, and checks what every subcommand shares: --version, --help, the exit codes and the form of a failure.

For every model: `speech info` in text, in JSON equal to the model information of the worker's ready (without the
device and the threads), or for a detection model the worker's refusal, and with --meta, the identity in both as the general keys of --meta give it; `speech devices` in text and JSON; usage errors (exit 2) and a library error
(exit 1, "speech: <code> (<option>): <message>"). For a synthesis model: `speech tts`'s WAVE byte for byte against the
worker's audio of the same lines and seed, into a file, into a regular file through stdout, and appended to a file with
content and through a pipe into cat, where the header cannot be written again in place and keeps its sizes at
0xFFFFFFFF; that nothing but the WAVE reaches stdout and a failed run leaves no file; for Qwen3-TTS a stop at
--max-seconds reported on stderr, and boolean and sampling flags against the worker's members; --instructions against
the worker's member where the model takes it and its refusal where it does not; with --reference a voice file made by
`speech voice` that `speech tts` speaks with; and with --embedding one made of a speaker-inversion embedding. For a
recognition model: `speech asr` on WAVE files of the dumps of reference/fastconformer/dump.py or
reference/qwen3-asr/dump.py against the worker's text of the same samples, as text and as JSON with the stop and the
languages, the one qwen-asr parsed or none, and where the model takes timestamps as text with --timestamps one line per
segment and as JSON with them, the segments and tokens the worker's; and for a dump that holds requests with a forced
language and a prompt, or with a decoding other than the default, `speech asr --language --prompt` or `speech asr
--decoding` against the worker's text of the same request, and as JSON its languages; and with --vad a detection model,
`speech asr --vad` on the dumps' audio joined with silences between them against the worker's ends of the regions that
`speech vad --split` writes, each of which holds the samples of its region, joined as transcription by regions joins
them (as JSON, with times where the model takes timestamps), an empty text for a file of silence, and the refusals of a
model of another task for --vad and of a detection option the model does not take. For a detection model: `speech vad`
on 32-bit float WAVE files of the dumps of reference/silero-vad/dump.py against the regions in each dump's
regions.json, the official's, with the flags of each set of options, as text and as JSON; with --split a WAVE file of
each region holding its samples as 16-bit PCM, named so that they sort in order, and the refusals of two files of one
name and of a file named as another's region in the folder, left as it was; and `speech asr` of the detection model with
--vad, refused on a file without speech.

usage: python3 tools/speech_cli_smoke.py <speech> <work dir> <model.gguf> [dump folder... | --reference REF.wav]
                                         [--embedding E.speaker.safetensors] [--vad DETECTION.gguf] [-- load options...]
"""

import array
import ast
import base64
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import threading

from worker_client import Worker, dump_requests, joined_transcript, short

args = sys.argv[1:]
options = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
reference = None
if "--reference" in args:
    at = args.index("--reference")
    reference = args[at + 1]
    del args[at:at + 2]
embedding = None
if "--embedding" in args:
    at = args.index("--embedding")
    embedding = args[at + 1]
    del args[at:at + 2]
vad = None
if "--vad" in args:
    at = args.index("--vad")
    vad = args[at + 1]
    del args[at:at + 2]
speech, work, model, *dumps = args
os.makedirs(work, exist_ok=True)
added = [o.split("=", 1)[0] for i, o in enumerate(options) if i > 0 and options[i - 1] == "--add-voice"]
SUBCOMMANDS = ["tts", "asr", "vad", "voice", "info", "devices", "models", "pull", "rm", "quantize", "serve", "worker"]


def run(*command, input=None, stdout=subprocess.PIPE, code=0):
    r = subprocess.run([speech, *command], input=input, stdout=stdout, stderr=subprocess.PIPE)
    if r.returncode != code:
        raise SystemExit(f"speech {' '.join(command)[:120]} exited with {r.returncode}, not {code}: {r.stderr.decode()[-600:]}")
    return r


def failure(r, code, option):
    """The failure a run printed, checked to be "speech: <code> (<option>): <message>"."""
    line = r.stderr.decode().strip().splitlines()[-1]
    m = re.fullmatch(r"speech: ([a-z_]+)(?: \(([^)]+)\))?: (.+)", line)
    if not m or m.group(1) != code or m.group(2) != option:
        raise SystemExit(f"expected speech: {code} ({option}): ..., got {line!r}")
    return m.group(3)


version = run("--version").stdout.decode()
assert re.fullmatch(r"speech\.cpp \d+\.\d+\.\d+, C API \d+\.\d+\n", version), version
overview = run("--help").stdout.decode()
assert all(f"\n  {s} " in overview for s in SUBCOMMANDS), overview
run(code=2)
run("nope", code=2)
for s in SUBCOMMANDS:
    assert run(s, "--help").stdout.decode().startswith(f"usage: speech {s} "), s
print(f"{version.strip()}; --help lists the subcommands and each has its own; no or an unknown subcommand exits with 2")

info = json.loads(run("info", model, "--json").stdout)
if info["task"] != "detection":
    bare = Worker(speech, model, ["--no-warmup"] + [o for i, o in enumerate(options) if o == "--device" or (i and options[i - 1] == "--device")])
    loaded = bare.ready["model"]
    bare.close()
    assert info == {k: v for k, v in loaded.items() if k not in ("device", "threads")}, "speech info --json differs from the worker's ready"
meta = json.loads(run("info", model, "--json", "--meta").stdout)
assert {k: v for k, v in meta.items() if k != "meta"} == info and meta["meta"]["general.architecture"] == info["architecture"], meta.keys()
assert meta["meta"]["speech.layout"] == info["layout"] and meta["meta"]["general.name"] == info["name"]
identity = ["organization", "basename", "size_label", "finetune", "version", "license"]
assert all(info.get(m) == meta["meta"].get("general." + m) for m in identity), {m: info.get(m) for m in identity}
source = info["source"]
assert meta["meta"]["general.source.repo_url"] == source["repository"], source
assert meta["meta"]["general.source.url"] == f"{source['repository']}/tree/{source['revision']}", source
assert {0: "F32", 1: "F16", 7: "Q8_0"}[meta["meta"]["general.file_type"]] == info["weight_type"], info["weight_type"]
text = run("info", model, "--meta").stdout.decode()
assert text.splitlines()[0] == info["name"] and f"general.architecture = \"{info['architecture']}\"" in text, text[:300]
shown = [info[m] for m in identity if m in info] + [source["repository"], source["revision"], info["weight_type"]]
assert all(v in text.split("\n\n")[0] for v in shown), f"speech info does not show {[v for v in shown if v not in text]}"
longest = max((v for v in meta["meta"].values() if isinstance(v, list)), key=len, default=[])
if len(longest) > 8:
    assert f"({len(longest)} items)" in text, "a long array is not shortened"
devices = json.loads(run("devices", "--json").stdout)["devices"]
listed = [line.split()[0] for line in run("devices").stdout.decode().splitlines()]
assert [d["name"] for d in devices] == listed and all(d["kind"] in ("cpu", "gpu", "igpu") for d in devices), devices
print(f"info: {'the model information' if info['task'] == 'detection' else 'the worker' + chr(39) + 's model information without device and threads'}, with --meta {len(meta['meta'])} entries, the identity "
      f"{', '.join(str(v) for v in shown)} as the general keys give it; devices: {listed}")

for command in [["tts", model, "-o", "x.wav", "--steps", "4x", "あ"], ["tts", model, "あ"], ["tts", model, "-o", "x.wav", "--bogus", "あ"],
                ["tts", model, "-o", "x.wav", "--seed=abc", "あ"], ["asr", model, "--timestamps=1", "x.wav"],
                ["tts", model, "-o", "x.wav", "--do-sample=maybe", "あ"],
                ["tts", model, "-o", "x.wav", "--device", "cpu", "--device", "cpu", "あ"], ["info"], ["info", model, "--json=1"],
                ["worker", model, "--threads", "two"]]:
    r = run(*command, code=2)
    assert "Run speech " in r.stderr.decode(), r.stderr
print("usage errors: exit 2 with a pointer to --help")
r = run("worker", model, "--threads", "two", code=2)
fatal = json.loads(r.stdout)
assert fatal["type"] == "fatal" and fatal["error"]["code"] == "invalid_argument", fatal
r = run("worker", model, "--device", "no-such-device", code=1)
fatal = json.loads(r.stdout)
# The worker refuses a detection model, which its protocol has no messages for, before it touches a device.
refused = ("unsupported", None) if info["task"] == "detection" else ("device", "device")
assert fatal["type"] == "fatal" and (fatal["error"]["code"], fatal["error"]["option"]) == refused, fatal
failure(r, *refused)
print(f"the worker's usage error and its {refused[0]} error: fatal on stdout, exit 2 and 1")

if info["task"] == "synthesis":
    w = Worker(speech, model, options)
    # An added voice where there is one: the voice an Irodori-TTS file has of its own, none, speaks without a reference.
    voice, rate = added[0] if added else w.ready["model"]["voices"][0]["name"], info["sample_rate"]
    lines, seed = ["明日の東京は晴れです。", "二つ目の文です。"], 11
    pcm = bytearray()
    for i, line in enumerate(lines):
        w.request({"type": "synthesize", "id": str(i), "text": line, "voice": voice, "seed": seed})
        for m in w.until(str(i)):
            if m["type"] == "chunk":
                pcm += base64.b64decode(m["pcm"])
        assert m["type"] == "end", short(m)
    w.close()
    pcm = bytes(pcm)

    def wave(riff_size, data_size):
        return (b"RIFF" + struct.pack("<I", riff_size) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
                + b"data" + struct.pack("<I", data_size) + pcm)

    complete, streamed = wave(36 + len(pcm), len(pcm)), wave(0xFFFFFFFF, 0xFFFFFFFF)
    stdin = ("\n".join(lines) + "\n").encode()
    command = ["tts", model, "--seed", str(seed), "--voice", voice, *options]

    def expect(name, got, want):
        if got != want:
            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            raise SystemExit(f"{name}: {len(got)} bytes where {len(want)} were expected, first difference at byte {first}")
        print(f"{name}: {len(got)} bytes, the worker's audio")

    path = os.path.join(work, "speech-cli-smoke.wav")
    r = run(*command, "-o", path, input=stdin)
    assert r.stdout == b"", r.stdout[:100]
    with open(path, "rb") as f:
        expect("-o FILE", f.read(), complete)
    with open(path, "wb") as f:
        run(*command, "-o", "-", input=stdin, stdout=f)
    with open(path, "rb") as f:
        expect("-o - into a regular file", f.read(), complete)
    with open(path, "wb") as f:
        f.write(b"before")
    with open(path, "ab") as f:
        run(*command, "-o", "-", input=stdin, stdout=f)
    with open(path, "rb") as f:
        expect("-o - appended to a file", f.read(), b"before" + streamed)
    cat = shutil.which("cat")
    assert cat, "the pipe is checked through cat, which is not on PATH"
    reader = subprocess.Popen([cat], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    piped = bytearray()
    # cat's output is read while speech writes, or both stop once the pipe's buffer is full.
    drain = threading.Thread(target=lambda: piped.extend(reader.stdout.read()))
    drain.start()
    run(*command, "-o", "-", input=stdin, stdout=reader.stdin)
    reader.stdin.close()
    drain.join()
    reader.wait()
    expect("-o - through a pipe into cat", bytes(piped), streamed)
    os.remove(path)

    r = run("tts", model, "-o", path, "--voice", "no-such-voice", *options, "あ。", code=1)
    failure(r, "out_of_range", "voice")
    assert not os.path.exists(path), "a failed run left its WAVE file"
    print("a voice the model does not have: exit 1, speech: out_of_range (voice): ..., and no file left")
    # --instructions is the worker's member of the same name, and a model that takes no instruction refuses it.
    if any(o["name"] == "instructions" for o in info["options"]):
        told = "怒った口調で話してください。"
        w = Worker(speech, model, options)
        w.request({"type": "synthesize", "id": "i", "text": lines[0], "voice": voice, "seed": seed, "instructions": told})
        instructed = b"".join(base64.b64decode(m["pcm"]) for m in w.until("i") if m["type"] == "chunk")
        w.close()
        run("tts", model, "-o", path, "--voice", voice, "--seed", str(seed), "--instructions", told, *options, lines[0])
        with open(path, "rb") as f:
            expect("--instructions", f.read()[44:], instructed)
        os.remove(path)
    else:
        r = run("tts", model, "-o", path, "--voice", voice, "--instructions", "怒った口調で", *options, lines[0], code=1)
        failure(r, "unsupported", "instructions")
        print("--instructions on a model that takes none: exit 1, speech: unsupported (instructions): ...")
    if info["architecture"] == "qwen3-tts":
        r = run("tts", model, "-o", path, "--voice", voice, "--max-seconds", "0.5", *options, lines[0])
        assert "stopped there" in r.stderr.decode(), r.stderr
        os.remove(path)
        print("--max-seconds 0.5: the stop reported on stderr, exit 0")
        # A boolean option is false with "=false"; with neither stack drawing, the seed changes nothing.
        w = Worker(speech, model, options)
        w.request({"type": "synthesize", "id": "g", "text": lines[0], "voice": voice, "seed": 1, "do_sample": False,
                   "code_predictor_do_sample": False, "max_seconds": 2})
        greedy = b"".join(base64.b64decode(m["pcm"]) for m in w.until("g") if m["type"] == "chunk")
        w.close()
        run("tts", model, "-o", path, "--voice", voice, "--seed", "2", "--do-sample=false", "--code-predictor-do-sample=false",
            "--max-seconds", "2", *options, lines[0])
        with open(path, "rb") as f:
            expect("--do-sample=false --code-predictor-do-sample=false", f.read()[44:], greedy)
        os.remove(path)
        r = run("tts", model, "-o", path, "--voice", voice, "--do-sample=false", "--temperature", "0.5", *options, lines[0], code=1)
        failure(r, "invalid_argument", "temperature")
        print("--temperature with --do-sample=false: exit 1, speech: invalid_argument (temperature): ...")
    if reference:
        made = os.path.join(work, "speech-cli-smoke.voice.gguf")
        run("voice", model, reference, made)
        run("tts", model, "-o", path, "--add-voice", f"made={made}", "--voice", "made", *options, lines[0])
        os.remove(path)
        run("voice", model, reference, reference, made, "--lufs", "-23")
        run("tts", model, "-o", path, "--add-voice", f"made={made}", "--voice", "made", *options, lines[0])
        os.remove(path)
        r = run("voice", model, reference, made, "--lufs", "-23", "--keep-loudness", code=2)
        os.remove(made)
        print("speech voice made voice files of one reference and of two at -23 LUFS on the CPU, which speech tts speaks with")
    if embedding:
        made = os.path.join(work, "speech-cli-smoke.embedding.voice.gguf")
        run("voice", model, embedding, made)
        run("tts", model, "-o", path, "--add-voice", f"made={made}", "--voice", "made", *options, lines[0])
        os.remove(path)
        print("speech voice made a voice file of a speaker-inversion embedding, which speech tts speaks with")
        # A voice is one embedding: a second is a usage error, and an embedding beside a recording the library's refusal.
        run("voice", model, embedding, embedding, made, code=2)
        if reference:
            failure(run("voice", model, embedding, reference, made, code=1), "invalid_argument", "embedding")
        # Headers whose numbers would read outside the file or overflow the shape's size: offsets that wrap around
        # (Codex's case, which read 3072 bytes before the data), a negative offset, and a shape whose bytes pass 2^64.
        forged = os.path.join(work, "speech-cli-smoke.forged.speaker.safetensors")
        for shape, offsets, data in [([1, 768], [18446744073709548548, 4], b"\0" * 4), ([1, 768], [-3068, 4], b"\0" * 4),
                                     ([4611686018427387904, 4], [0, 0], b"")]:
            header = json.dumps({"speaker_embedding": {"dtype": "F32", "shape": shape, "data_offsets": offsets}}).encode()
            with open(forged, "wb") as f:
                f.write(struct.pack("<Q", len(header)) + header + data)
            failure(run("voice", model, forged, made, code=1), "invalid_argument", "embedding")
        os.remove(forged)
        os.remove(made)
        print("speech voice refused two embeddings (exit 2), an embedding with a recording and three forged headers (exit 1)")
else:
    def read_npy(path):
        with open(path, "rb") as f:
            data = f.read()
        header_length = struct.unpack("<H", data[8:10])[0]
        header = ast.literal_eval(data[10:10 + header_length].decode())
        values = array.array("f")
        values.frombytes(data[10 + header_length:])
        assert header["descr"] == "<f4"
        return values

    rate = info["sample_rate"]

if info["task"] == "detection":
    # 32-bit float WAVE files hold the dumps' samples as they are, which the official's regions were found in.
    files, sets = [], []
    for d in dumps:
        data = read_npy(os.path.join(d, "audio.npy")).tobytes()
        path = os.path.join(work, os.path.basename(os.path.normpath(d)) + ".wav")
        with open(path, "wb") as f:
            f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 1, rate, rate * 4, 4, 32)
                    + b"data" + struct.pack("<I", len(data)) + data)
        files.append(path)
        with open(os.path.join(d, "regions.json"), encoding="utf-8") as f:
            sets.append(json.load(f))
    for name in sets[0]:
        options_of = sets[0][name]["options"]
        flags = [x for k, v in options_of.items() for x in (f"--{k.replace('_', '-')}", str(v))]
        lines = [line.split("\t") for line in run("vad", model, *flags, *options, *files).stdout.decode().splitlines()]
        want = [[f, f"{start / rate:.3f}", f"{end / rate:.3f}"] for f, s in zip(files, sets) for start, end in s[name]["regions"]]
        assert lines == want, (name, lines[:3], want[:3])
        got = [json.loads(line) for line in run("vad", model, "--format", "json", *flags, *options, *files).stdout.decode().splitlines()]
        assert [g["file"] for g in got] == files, got
        assert all([[round(r["start"] * rate), round(r["end"] * rate)] for r in g["regions"]] == s[name]["regions"] for g, s in zip(got, sets)), name
    print(f"vad: {len(files)} files with {len(sets[0])} sets of options, the official's regions as text and as JSON")
    # --split writes each region of the first set of options as a 16-bit WAVE file holding the region's samples, scaled
    # by 32768 and rounded half away from zero, as the reader scales them.
    split = os.path.join(work, "speech-cli-smoke-split")
    shutil.rmtree(split, ignore_errors=True)
    flags = [x for k, v in sets[0][next(iter(sets[0]))]["options"].items() for x in (f"--{k.replace('_', '-')}", str(v))]
    got = [json.loads(line) for line in run("vad", model, "--format", "json", "--split", split, *flags, *options, *files).stdout.decode().splitlines()]
    names = sorted(os.listdir(split))
    # Each file's regions sort in their order; the files' own names sort as they will.
    for g in got:
        stem = os.path.splitext(os.path.basename(g["file"]))[0]
        want_names = [f"{stem}-{str(k + 1).zfill(len(str(len(g['regions']))))}.wav" for k in range(len(g["regions"]))]
        assert sorted(want_names) == want_names and all(n in names for n in want_names), (stem, want_names[:3])
    assert len(names) == sum(len(g["regions"]) for g in got), names[:5]
    for g, d in zip(got, dumps):
        samples = read_npy(os.path.join(d, "audio.npy"))
        stem = os.path.splitext(os.path.basename(g["file"]))[0]
        for k, region in enumerate(g["regions"]):
            name = os.path.join(split, f"{stem}-{str(k + 1).zfill(len(str(len(g['regions']))))}.wav")
            with open(name, "rb") as f:
                data = f.read()
            assert data[:4] == b"RIFF" and struct.unpack("<HHIIHH", data[20:36]) == (1, 1, rate, rate * 2, 2, 16), name
            cut = samples[round(region["start"] * rate):round(region["end"] * rate)]
            want = [max(-32768, min(32767, int(math.copysign(math.floor(abs(x) * 32768 + 0.5), x)))) for x in cut]
            assert array.array("h", data[44:]).tolist() == want, name
    print(f"vad --split: {len(names)} WAVE files holding the regions' samples, named so that they sort in order")
    shutil.rmtree(split)
    twin = os.path.join(work, "speech-cli-smoke-twin")
    os.makedirs(twin, exist_ok=True)
    shutil.copy(files[0], os.path.join(twin, os.path.basename(files[0])))
    run("vad", model, "--split", split, *options, files[0], os.path.join(twin, os.path.basename(files[0])), code=2)
    shutil.rmtree(twin)
    # A file named as another's region, in the folder of --split, is refused before it is written over.
    g = next(g for g in got if g["regions"])
    alias = os.path.join(work, "speech-cli-smoke-alias")
    os.makedirs(alias, exist_ok=True)
    first = os.path.join(alias, "a.wav")
    second = os.path.join(alias, f"a-{'1'.zfill(len(str(len(g['regions']))))}.wav")
    shutil.copy(g["file"], first)
    shutil.copy(g["file"], second)
    with open(second, "rb") as f:
        kept = f.read()
    run("vad", model, "--split", alias, *flags, *options, first, second, code=2)
    with open(second, "rb") as f:
        assert f.read() == kept, "vad --split wrote over a file it was given"
    shutil.rmtree(alias)
    print("vad --split with two files of one name, and with a file named as another's region in its folder: exit 2")
    silence = os.path.join(work, "speech-cli-smoke-silence.wav")
    with open(silence, "wb") as f:
        data = bytes(4 * rate)
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 1, rate, rate * 4, 4, 32)
                + b"data" + struct.pack("<I", len(data)) + data)
    failure(run("asr", model, "--vad", model, *options, silence, code=1), "unsupported", None)
    os.remove(silence)
    print("asr of a detection model with --vad, on a file without speech: exit 1, unsupported")
    failure(run("vad", model, "--threshold", "2", *options, files[0], code=1), "out_of_range", "threshold")
    failure(run("vad", model, "--language", "ja", *options, files[0], code=1), "unsupported", "language")
    failure(run("vad", model, *options, os.path.join(work, "no-such.wav"), code=1), "io", "audio")
    print("a threshold above 1, a language and a file that is not there: exit 1 with the library's form")
    for path in files:
        os.remove(path)
elif info["task"] == "recognition":
    files, pcms = [], []
    for d in dumps:
        samples = read_npy(os.path.join(d, "audio.npy"))
        pcm = array.array("h", [max(-32768, min(32767, round(x * 32768))) for x in samples]).tobytes()
        path = os.path.join(work, os.path.basename(os.path.normpath(d)) + ".wav")
        with open(path, "wb") as f:
            f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
                    + b"data" + struct.pack("<I", len(pcm)) + pcm)
        files.append(path)
        pcms.append(pcm)
    takes = {o["name"] for o in info["options"]}
    timed = {"timestamps": True} if "timestamps" in takes else {}
    w = Worker(speech, model, options)

    def worker_end(id, pcm, **members):
        w.expect(id)
        for k, at in enumerate(range(0, len(pcm), 32000)):
            w.send({"type": "chunk", "id": id, "seq": k, "pcm": base64.b64encode(pcm[at:at + 32000]).decode()})
        w.send({"type": "transcribe", "id": id, "sample_rate": rate, **members})
        return w.terminal(id, "end")

    ends = [worker_end(str(i), pcm, **timed) for i, pcm in enumerate(pcms)]
    requests = [dump_requests(speech, model, d) for d in dumps]
    # The worker's languages of each dump are the ones qwen-asr parsed, and the member is left out where there are none.
    assert [e.get("languages", []) for e in ends] == [r[0][3] for r in requests] and all(e.get("languages") != [] for e in ends), ends
    # The last request of each dump: its language forced and its prompt, or its decoding other than the default, where it
    # has one.
    asked = [(f, pcm, r[-1][1], r[-1][3]) for f, pcm, r in zip(files, pcms, requests)]
    asked_ends = [worker_end(f"asked-{i}", pcm, **members) for i, (f, pcm, members, _) in enumerate(asked) if members]
    w.close()

    def end_members(e):
        """The members of a worker's end that speech asr --format json writes after "file"."""
        return {k: v for k, v in e.items() if k not in ("type", "id", "_at")}

    texts = run("asr", model, *options, *files).stdout.decode().split("\n")
    assert texts == [e["text"] for e in ends] + [""], texts
    plain = [json.loads(line) for line in run("asr", model, "--format", "json", *options, *files).stdout.decode().splitlines()]
    assert plain == [{"file": f, **{k: v for k, v in end_members(e).items() if k not in ("segments", "tokens")}} for f, e in zip(files, ends)], plain[:2]
    if timed:
        rows = [line.split("\t") for line in run("asr", model, "--timestamps", *options, *files).stdout.decode().splitlines()]
        want_rows = [[f, f"{s['start']:.3f}", f"{s['end']:.3f}", s["text"]] for f, e in zip(files, ends) for s in e["segments"]]
        assert rows == want_rows, (rows[:3], want_rows[:3])
        timed_json = run("asr", model, "--format", "json", "--timestamps", *options, *files).stdout.decode().splitlines()
        want_json = [{"file": f, **end_members(e)} for f, e in zip(files, ends)]
        assert [json.loads(line) for line in timed_json] == want_json, timed_json[:1]
    for (f, _, members, languages), e in zip([a for a in asked if a[2]], asked_ends):
        flags = [x for name, value in members.items() for x in (f"--{name}", value)]
        assert run("asr", model, *flags, *options, f).stdout.decode() == e["text"] + "\n", (f, members)
        got = json.loads(run("asr", model, "--format", "json", *flags, *options, f).stdout)
        assert got == {"file": f, **end_members(e)} and got.get("languages", []) == languages, (got, languages)
    print(f"asr: {len(files)} files, the worker's texts as text and as JSON with the stop and the dumps' languages"
          + (", as segments with --timestamps and as JSON with times" if timed else "")
          + f"; {len(asked_ends)} with the flags of their options, --language and --prompt or --decoding, as the worker's requests with them")
    if vad:
        # The dumps' audio joined with a second of silence after each, which the detection cuts into regions again.
        joined = b"".join(pcm + b"\0" * 2 * rate for pcm in pcms)
        long = os.path.join(work, "speech-cli-smoke-long.wav")
        silence = os.path.join(work, "speech-cli-smoke-silence.wav")
        for path, pcm in ((long, joined), (silence, b"\0" * 6 * rate)):
            with open(path, "wb") as f:
                f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
                        + b"data" + struct.pack("<I", len(pcm)) + pcm)
        detection = ["--threshold", "0.5", "--speech-pad-ms", "300", "--min-silence-duration-ms", "500", "--max-speech-duration-s", "8"]
        split = os.path.join(work, "speech-cli-smoke-split")
        shutil.rmtree(split, ignore_errors=True)
        regions = json.loads(run("vad", vad, "--format", "json", "--split", split, *detection, long).stdout)["regions"]
        names = sorted(os.listdir(split))
        assert len(names) == len(regions) > len(pcms), (names, regions)
        w = Worker(speech, model, options)
        parts = []
        for k, (name, region) in enumerate(zip(names, regions)):
            with open(os.path.join(split, name), "rb") as f:
                pcm = f.read()[44:]
            first = round(region["start"] * rate)
            assert pcm == joined[2 * first:2 * round(region["end"] * rate)], name
            parts.append((first / rate, worker_end(f"region-{k}", pcm, **timed)))
        w.close()
        shutil.rmtree(split)
        want = {"file": long, **joined_transcript(parts, bool(timed))}
        got = json.loads(run("asr", model, "--vad", vad, "--format", "json", *detection, *(["--timestamps"] if timed else []), *options, long).stdout)
        assert got == want, (got, want)
        got = json.loads(run("asr", model, "--vad", vad, "--format", "json", *(["--timestamps"] if timed else []), *options, silence).stdout)
        assert got == {"file": silence, "text": "", "stop": "complete", **({"segments": [], "tokens": []} if timed else {})}, got
        print(f"asr --vad: {len(regions)} regions of {len(joined) / 2 / rate:.1f} s, the worker's ends of vad --split's files joined"
              f"{' with their times' if timed else ''}; a file of silence gives an empty text")
        run("asr", model, "--vad", model, *options, long, code=2)
        failure(run("asr", model, "--vad", vad, "--threshold", "2", *options, long, code=1), "out_of_range", "threshold")
        failure(run("asr", model, "--threshold", "0.5", *options, long, code=1), "unsupported", "threshold")
        print("asr --vad of a recognition model: exit 2; a threshold above 1 with --vad and a threshold without it: exit 1")
        os.remove(long)
        os.remove(silence)
    r = run("asr", model, "--language", "zz", *options, files[0], code=1)
    failure(r, "out_of_range", "language")
    r = run("asr", model, *options, os.path.join(work, "no-such.wav"), code=1)
    failure(r, "io", "audio")
    print("a language the model does not recognize and a file that is not there: exit 1 with the library's form")
    for path in files:
        os.remove(path)
print("ok")
