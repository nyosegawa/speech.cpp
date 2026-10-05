"""Drives speech-worker the way a caller does: waits for ready, sends two requests in its first voice, cancels
the second after its first chunk, sends a third, and checks each answer and that every line on stdout is a
JSON object. Then checks that Irodori-TTS speaks a fixed length of 1 s in at most 1 s and that Qwen3-TTS
answers a speed and a length with an error, that both answer a speed out of range or not a number with
one, and that every line the worker cannot read gets an error. Writes the first answer to a WAV.

usage: python3 tools/worker_smoke.py <out.wav> <worker> <model.gguf> <codec.gguf> [worker options...]
"""

import base64
import json
import subprocess
import sys
import time
import wave

out_wav, *command = sys.argv[1:]
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
t0 = time.perf_counter()


def read():
    line = proc.stdout.readline()
    if not line:
        raise SystemExit("the worker exited")
    line = line.decode("utf-8").rstrip("\n")
    if line.endswith("\r"):
        raise SystemExit("a line ends with \\r")
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        message = None
    if not isinstance(message, dict):
        raise SystemExit(f"a line on stdout is not a JSON object: {line!r}")
    return message


def send(obj):
    proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
    proc.stdin.flush()


ready = read()
assert ready["type"] == "ready", ready
voice = ready["voices"][0]
print(f"ready in {time.perf_counter() - t0:.2f} s: {ready['model']} ({ready['architecture']}), rate {ready['sampleRate']}, "
      f"streaming by {ready['streaming']}, {len(ready['voices'])} voices, languages {ready['languages']}, "
      f"backend {ready['backend']}, speech.cpp {ready['version']}")

t1 = time.perf_counter()
send({"id": "a", "text": "明日の東京は晴れで、最高気温は二十四度の予報です。", "voice": voice, "speed": 1.0})
send({"id": "b", "text": "これは途中で止める長めの文です。止まったら終わりの知らせは来ません。", "voice": voice})
pcm_a, first_a, cancelled_b, seq_b = bytearray(), None, False, []
while True:
    m = read()
    if m["type"] == "chunk" and m["id"] == "a":
        if first_a is None:
            first_a = time.perf_counter() - t1
        pcm_a += base64.b64decode(m["pcm"])
    elif m["type"] == "end" and m["id"] == "a":
        assert m["samples"] * 2 == len(pcm_a), (m, len(pcm_a))
        print(f"a: first chunk {first_a:.3f} s, {m['samples'] / ready['sampleRate']:.2f} s of audio")
    elif m["type"] == "chunk" and m["id"] == "b":
        seq_b.append(m["seq"])
        if not cancelled_b:
            send({"type": "cancel", "id": "b"})
            send({"id": "c", "text": "三つ目です。", "voice": voice})
            cancelled_b = True
    elif m["type"] == "end" and m["id"] == "b":
        raise SystemExit("b ended although it was cancelled")
    elif m["type"] == "end" and m["id"] == "c":
        print(f"b: {len(seq_b)} chunk(s) before the cancel took effect, no end; c: {m['samples']} samples")
        break
    elif m["type"] in ("error", "fatal"):
        raise SystemExit(f"unexpected {m}")

send({"id": "d", "text": "声の名前が違います。", "voice": "no-such-voice"})
m = read()
assert m["type"] == "error" and m["id"] == "d", m
print(f"d: error as expected: {m['error']}")


def expect_error(request):
    send(request)
    m = read()
    assert m["type"] == "error" and m["id"] == request["id"], m
    print(f"{request['id']}: error as expected: {m['error']}")


if ready["architecture"] == "irodori-tts":
    send({"id": "e", "text": "三つ目です。", "voice": voice, "seconds": 1})
    samples = 0
    while (m := read())["type"] == "chunk":
        samples += len(base64.b64decode(m["pcm"])) // 2
    assert m["type"] == "end" and m["id"] == "e" and m["samples"] == samples <= ready["sampleRate"], m
    print(f"e: a length of 1 s gave {samples / ready['sampleRate']:.3f} s")
else:
    expect_error({"id": "e", "text": "速くしてください。", "voice": voice, "speed": 1.5})
    expect_error({"id": "e2", "text": "長さを決めてください。", "voice": voice, "seconds": 2})
expect_error({"id": "f", "text": "速すぎます。", "voice": voice, "speed": 5})
expect_error({"id": "g", "text": "数ではありません。", "voice": voice, "speed": "fast"})

# Lines the worker cannot read are answered too, with the id when one can be read.
for line, want_id in [
    ("this is not JSON", None),
    ('["id", "h"]', None),
    ('{"id": "h", "text": "あ。"} trailing', None),
    (json.dumps({"id": "i", "text": "あ。", "voice": voice, "language": None}), "i"),
    (json.dumps({"id": "j", "text": "あ。", "voice": voice, "speed": {"value": 1}}), "j"),
    (json.dumps({"text": "あ。", "voice": voice}), None),
    (json.dumps({"type": "pause", "id": "k"}), "k"),
]:
    proc.stdin.write((line + "\n").encode("utf-8"))
    proc.stdin.flush()
    m = read()
    assert m["type"] == "error" and m.get("id") == want_id, (line, m)
    print(f"{line[:40]!r}: error{' for ' + want_id if want_id else ' without an id'}: {m['error']}")

proc.stdin.close()
proc.wait(timeout=30)
rest = proc.stdout.read()
if rest:
    raise SystemExit(f"the worker wrote after its last answer: {rest[:200]!r}")
with wave.open(out_wav, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(ready["sampleRate"])
    w.writeframes(bytes(pcm_a))
print("ok, exit", proc.returncode)
