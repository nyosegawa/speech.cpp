"""Checks the `speech` command line against the worker, whose protocol worker_smoke.py and worker_recognition_smoke.py
check, and checks what every subcommand shares: --version, --help, the exit codes and the form of a failure.

For every model: `speech info` in text, in JSON equal to the model information of the worker's ready (without the
device and the threads) and with --meta; `speech devices` in text and JSON; usage errors (exit 2) and a library error
(exit 1, "speech: <code> (<option>): <message>"). For a synthesis model: `speech tts`'s WAVE byte for byte against the
worker's audio of the same lines and seed, into a file, into a regular file through stdout, and appended to a file with
content and through a pipe into cat, where the header cannot be written again in place and keeps its sizes at
0xFFFFFFFF; that nothing but the WAVE reaches stdout and a failed run leaves no file; for Qwen3-TTS a stop at
--max-seconds reported on stderr; and with --reference a voice file made by `speech voice` that `speech tts` speaks
with. For a recognition model: `speech asr` on WAVE files of the dumps of reference/fastconformer/dump.py against the
worker's text of the same samples, as text, as text with --timestamps one line per segment, and as JSON with and without
--timestamps, the segments and tokens the worker's.

usage: python3 tools/speech_cli_smoke.py <speech> <work dir> <model.gguf> [dump folder... | --reference REF.wav] [-- load options...]
"""

import array
import ast
import base64
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading

from worker_client import Worker, short

args = sys.argv[1:]
options = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
reference = None
if "--reference" in args:
    at = args.index("--reference")
    reference = args[at + 1]
    del args[at:at + 2]
speech, work, model, *dumps = args
SUBCOMMANDS = ["tts", "asr", "voice", "info", "devices", "serve", "worker"]


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

bare = Worker(speech, model, ["--no-warmup"] + [o for i, o in enumerate(options) if o == "--device" or (i and options[i - 1] == "--device")])
loaded = bare.ready["model"]
bare.close()
info = json.loads(run("info", model, "--json").stdout)
assert info == {k: v for k, v in loaded.items() if k not in ("device", "threads")}, "speech info --json differs from the worker's ready"
meta = json.loads(run("info", model, "--json", "--meta").stdout)
assert {k: v for k, v in meta.items() if k != "meta"} == info and meta["meta"]["general.architecture"] == info["architecture"], meta.keys()
assert meta["meta"]["speech.layout"] == info["layout"] and meta["meta"]["general.name"] == info["name"]
text = run("info", model, "--meta").stdout.decode()
assert text.splitlines()[0] == info["name"] and f"general.architecture = \"{info['architecture']}\"" in text, text[:300]
longest = max((v for v in meta["meta"].values() if isinstance(v, list)), key=len, default=[])
if len(longest) > 8:
    assert f"({len(longest)} items)" in text, "a long array is not shortened"
devices = json.loads(run("devices", "--json").stdout)["devices"]
listed = [line.split()[0] for line in run("devices").stdout.decode().splitlines()]
assert [d["name"] for d in devices] == listed and all(d["kind"] in ("cpu", "gpu", "igpu") for d in devices), devices
print(f"info: the worker's model information without device and threads, with --meta {len(meta['meta'])} entries; devices: {listed}")

for command in [["tts", model, "-o", "x.wav", "--steps", "4x", "あ"], ["tts", model, "あ"], ["tts", model, "-o", "x.wav", "--bogus", "あ"],
                ["tts", model, "-o", "x.wav", "--seed=abc", "あ"], ["asr", model, "--timestamps=1", "x.wav"],
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
assert fatal["type"] == "fatal" and fatal["error"]["code"] == "device" and fatal["error"]["option"] == "device", fatal
failure(r, "device", "device")
print("the worker's usage error and its device error: fatal on stdout, exit 2 and 1")

if info["task"] == "synthesis":
    w = Worker(speech, model, options)
    voice, rate = w.ready["model"]["voices"][0]["name"], info["sample_rate"]
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
    if info["architecture"] == "qwen3-tts":
        r = run("tts", model, "-o", path, "--voice", voice, "--max-seconds", "0.5", *options, lines[0])
        assert "stopped there" in r.stderr.decode(), r.stderr
        os.remove(path)
        print("--max-seconds 0.5: the stop reported on stderr, exit 0")
    if reference:
        made = os.path.join(work, "speech-cli-smoke.voice.gguf")
        run("voice", model, reference, made)
        run("tts", model, "-o", path, "--add-voice", f"made={made}", "--voice", "made", *options, lines[0])
        os.remove(path)
        os.remove(made)
        print("speech voice made a voice file on the CPU, which speech tts speaks with")
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
    w = Worker(speech, model, options)
    ends = []
    for i, pcm in enumerate(pcms):
        w.expect(str(i))
        for k, at in enumerate(range(0, len(pcm), 32000)):
            w.send({"type": "chunk", "id": str(i), "seq": k, "pcm": base64.b64encode(pcm[at:at + 32000]).decode()})
        w.send({"type": "transcribe", "id": str(i), "sample_rate": rate, "timestamps": True})
        ends.append(w.terminal(str(i), "end"))
    w.close()
    texts = run("asr", model, *options, *files).stdout.decode().split("\n")
    assert texts == [e["text"] for e in ends] + [""], texts
    rows = [line.split("\t") for line in run("asr", model, "--timestamps", *options, *files).stdout.decode().splitlines()]
    want_rows = [[f, f"{s['start']:.3f}", f"{s['end']:.3f}", s["text"]] for f, e in zip(files, ends) for s in e["segments"]]
    assert rows == want_rows, (rows[:3], want_rows[:3])
    plain = [json.loads(line) for line in run("asr", model, "--format", "json", *options, *files).stdout.decode().splitlines()]
    assert plain == [{"file": f, "text": e["text"]} for f, e in zip(files, ends)], plain[:2]
    timed = [json.loads(line) for line in run("asr", model, "--format", "json", "--timestamps", *options, *files).stdout.decode().splitlines()]
    assert timed == [{"file": f, "text": e["text"], "segments": e["segments"], "tokens": e["tokens"]} for f, e in zip(files, ends)], timed[:1]
    print(f"asr: {len(files)} files, the worker's texts as text, as segments with --timestamps, and as JSON with and without times")
    r = run("asr", model, "--language", "zz", *options, files[0], code=1)
    failure(r, "out_of_range", "language")
    r = run("asr", model, *options, os.path.join(work, "no-such.wav"), code=1)
    failure(r, "io", "audio")
    print("a language the model does not recognize and a file that is not there: exit 1 with the library's form")
    for path in files:
        os.remove(path)
print("ok")
