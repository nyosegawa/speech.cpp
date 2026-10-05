"""Checks speech-tts's WAVE byte for byte against speech-worker's audio for the same lines and seed: into a file,
into a regular file through stdout, and appended to a file with content and through a pipe into cat, where
the header cannot be written again in place and keeps its sizes at 0xFFFFFFFF. Also checks that nothing but
the WAVE reaches stdout.

usage: python3 tools/speech_tts_smoke.py <build dir> <work dir> <model.gguf> [model options...]
"""

import base64
import json
import os
import shutil
import struct
import subprocess
import sys
import threading

build, work, model, *options = sys.argv[1:]
exe = ".exe" if os.name == "nt" else ""
worker = os.path.join(build, "speech-worker" + exe)
tts = os.path.join(build, "speech-tts" + exe)
lines = ["明日の東京は晴れです。", "二つ目の文です。"]
seed = 11

proc = subprocess.Popen([worker, model, "--seed", str(seed), *options],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
ready = json.loads(proc.stdout.readline())
assert ready["type"] == "ready", ready
voice, rate = ready["voices"][0], ready["sampleRate"]
pcm = bytearray()
for i, text in enumerate(lines):
    proc.stdin.write((json.dumps({"id": str(i), "text": text, "voice": voice}, ensure_ascii=False) + "\n").encode())
    proc.stdin.flush()
    while (m := json.loads(proc.stdout.readline()))["type"] == "chunk":
        pcm += base64.b64decode(m["pcm"])
    assert m["type"] == "end", m
proc.stdin.close()
proc.wait(timeout=60)
pcm = bytes(pcm)


def wave(riff_size, data_size):
    return (b"RIFF" + struct.pack("<I", riff_size) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", data_size) + pcm)


complete, streamed = wave(36 + len(pcm), len(pcm)), wave(0xFFFFFFFF, 0xFFFFFFFF)
stdin = ("\n".join(lines) + "\n").encode()
command = [tts, model, "--seed", str(seed), "--voice-name", voice, *options]


def run(output, stdout):
    r = subprocess.run([*command, "-o", output], input=stdin, stdout=stdout, stderr=subprocess.PIPE)
    assert r.returncode == 0, r.stderr.decode()
    return r


def expect(name, got, want):
    if got != want:
        first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        raise SystemExit(f"{name}: {len(got)} bytes where {len(want)} were expected, first difference at byte {first}")
    print(f"{name}: {len(got)} bytes as expected")


path = os.path.join(work, "speech-tts-smoke.wav")
r = run(path, subprocess.PIPE)
assert r.stdout == b"", r.stdout[:100]
with open(path, "rb") as f:
    expect("-o FILE", f.read(), complete)

with open(path, "wb") as f:
    run("-", f)
with open(path, "rb") as f:
    expect("-o - into a regular file", f.read(), complete)

with open(path, "wb") as f:
    f.write(b"before")
with open(path, "ab") as f:
    run("-", f)
with open(path, "rb") as f:
    expect("-o - appended to a file", f.read(), b"before" + streamed)

cat = shutil.which("cat")
assert cat, "the pipe is checked through cat, which is not on PATH"
reader = subprocess.Popen([cat], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
piped = bytearray()
# cat's output is read while speech-tts writes, or both stop once the pipe's buffer is full.
drain = threading.Thread(target=lambda: piped.extend(reader.stdout.read()))
drain.start()
r = subprocess.run([*command, "-o", "-"], input=stdin, stdout=reader.stdin, stderr=subprocess.PIPE)
reader.stdin.close()
drain.join()
reader.wait()
piped = bytes(piped)
assert r.returncode == 0, r.stderr.decode()
expect("-o - through a pipe into cat", piped, streamed)
os.remove(path)
print("ok")
