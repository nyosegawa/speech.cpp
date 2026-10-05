"""Drives a recognition speech-worker the way a caller does: waits for ready, sends the audio of each dump of
reference/fastconformer/dump.py as 16-bit chunks of one second and checks that the text is the dump's text.
Then checks that the chunks of two requests may interleave and that the requests are answered in the order of their
ends, that a cancel drops a request whether it is still taking chunks or waiting, that audio at three times the model's
rate is answered with a text, that a sample rate of 0, an unknown language and a request to speak are answered with an
error, that a refused chunk is the one answer of its request, that every line the worker cannot read gets an error, and
that every line on stdout is a JSON object.

usage: python3 tools/worker_recognition_smoke.py <worker> <model.gguf> <dump folder>... [-- worker options...]
"""

import array
import ast
import base64
import json
import os
import struct
import subprocess
import sys
import time

args = sys.argv[1:]
options = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
worker, model, *dumps = args
if not dumps:
    raise SystemExit(__doc__)
proc = subprocess.Popen([worker, model, *options], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
t0 = time.perf_counter()


def read():
    line = proc.stdout.readline()
    if not line:
        raise SystemExit("the worker exited")
    line = line.decode("utf-8").rstrip("\n")
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        message = None
    if not isinstance(message, dict):
        raise SystemExit(f"a line on stdout is not a JSON object: {line!r}")
    return message


def send_line(line):
    proc.stdin.write((line + "\n").encode("utf-8"))
    proc.stdin.flush()


def send(obj):
    send_line(json.dumps(obj, ensure_ascii=False))


def pcm16(samples):
    """Float samples as 16-bit little-endian PCM, scaled as the worker scales them back (by 32768)."""
    return array.array("h", [max(-32768, min(32767, round(x * 32768))) for x in samples]).tobytes()


def chunks(id, pcm, size=32000):
    return [{"type": "chunk", "id": id, "seq": i, "pcm": base64.b64encode(pcm[at:at + size]).decode()}
            for i, at in enumerate(range(0, len(pcm), size))]


def read_npy(path):
    with open(path, "rb") as f:
        data = f.read()
    header_length = struct.unpack("<H", data[8:10])[0]
    header = ast.literal_eval(data[10:10 + header_length].decode())
    assert header["descr"] == "<f4" and len(header["shape"]) == 1, header
    values = array.array("f")
    values.frombytes(data[10 + header_length:])
    return values


def expect_error(id, what):
    m = read()
    assert m["type"] == "error" and m.get("id") == id, (what, m)
    print(f"{what}: error as expected: {m['error']}")


ready = read()
assert ready["type"] == "ready" and ready["task"] == "recognition", ready
assert "voices" not in ready and "streaming" not in ready, ready
rate = ready["sampleRate"]
print(f"ready in {time.perf_counter() - t0:.2f} s: {ready['model']} ({ready['architecture']}), rate {rate}, "
      f"languages {ready['languages']}, backend {ready['backend']}, speech.cpp {ready['version']}")

audio, want_text = {}, {}
for d in dumps:
    name = os.path.basename(os.path.normpath(d))
    with open(os.path.join(d, "text.txt"), encoding="utf-8") as f:
        want = f.read()
    audio[name] = pcm16(read_npy(os.path.join(d, "audio.npy")))
    want_text[name] = want
    t1 = time.perf_counter()
    for c in chunks(name, audio[name]):
        send(c)
    send({"type": "end", "id": name, "sampleRate": rate, "language": ready["languages"][0]})
    m = read()
    assert m["type"] == "text" and m["id"] == name, m
    if m["text"] != want:
        raise SystemExit(f"{name}: the text differs from the dump's text\n  got  {m['text']}\n  want {want}")
    print(f"{name}: {len(audio[name]) / 2 / rate:.2f} s, the dump's text in {time.perf_counter() - t1:.3f} s")

first, last = sorted(audio, key=lambda n: len(audio[n]))[0], sorted(audio, key=lambda n: len(audio[n]))[-1]
short = audio[first]

# Two requests whose chunks interleave are answered in the order of their ends.
a, b = chunks("a", short), chunks("b", short)
for i in range(max(len(a), len(b))):
    for c in (a[i:i + 1] + b[i:i + 1]):
        send(c)
send({"type": "end", "id": "b", "sampleRate": rate})
send({"type": "end", "id": "a", "sampleRate": rate})
got = [read(), read()]
assert [(m["type"], m["id"]) for m in got] == [("text", "b"), ("text", "a")] and got[0]["text"] == got[1]["text"], got
print("interleaved chunks: answered in the order of the ends")

# A request cancelled while it takes chunks, and one cancelled while it waits behind another, send no text.
c = chunks("c", short)
send(c[0])
send({"type": "cancel", "id": "c"})
for rest in c[1:]:
    send(rest)
send({"type": "end", "id": "c", "sampleRate": rate})
for x in chunks("d", audio[last]):
    send(x)
send({"type": "end", "id": "d", "sampleRate": rate})
for x in chunks("e", short):
    send(x)
send({"type": "end", "id": "e", "sampleRate": rate})
send({"type": "cancel", "id": "e"})
for x in chunks("f", short):
    send(x)
send({"type": "end", "id": "f", "sampleRate": rate})
got = [read(), read()]
assert [(m["type"], m["id"]) for m in got] == [("text", "d"), ("text", "f")], got
print("cancelled while taking chunks and while waiting: no text; the others answered")

# A cancel during the recognition itself: the request sends nothing, and the next one is answered.
for x in chunks("g", audio[last]):
    send(x)
send({"type": "end", "id": "g", "sampleRate": rate})
time.sleep(0.05)
send({"type": "cancel", "id": "g"})
for x in chunks("h", short):
    send(x)
send({"type": "end", "id": "h", "sampleRate": rate})
m = read()
assert m["type"] == "text" and m["id"] == "h", m
print("cancelled while it ran: no text; the next one answered")

# The short audio with each sample three times, at three times the rate: the library resamples it to the model's.
tripled = array.array("h")
tripled.frombytes(short)
tripled = array.array("h", [x for x in tripled for _ in range(3)]).tobytes()
for x in chunks("i", tripled):
    send(x)
send({"type": "end", "id": "i", "sampleRate": rate * 3})
m = read()
assert m["type"] == "text" and m["id"] == "i", m
print(f"audio at {rate * 3} Hz: a text, {'equal to' if m['text'] == want_text[first] else 'unlike'} the dump's text")
for x in chunks("v", short):
    send(x)
send({"type": "end", "id": "v", "sampleRate": 0})
expect_error("v", "a sample rate of 0")
for x in chunks("j", short):
    send(x)
send({"type": "end", "id": "j", "sampleRate": rate, "language": "zz"})
expect_error("j", "a language the model does not recognize")
send({"type": "end", "id": "k", "sampleRate": rate})
expect_error("k", "an end without chunks")
send({"id": "l", "text": "明日の東京は晴れです。", "voice": "x"})
expect_error("l", "a request to speak")

# A refused chunk is its request's one answer: its other chunks and its end are dropped.
m_chunks = chunks("m", short)
send(m_chunks[1])
for x in m_chunks:
    send(x)
send({"type": "end", "id": "m", "sampleRate": rate})
expect_error("m", "a chunk out of order")
send({"type": "chunk", "id": "n", "seq": 0, "pcm": "not base64!"})
send({"type": "end", "id": "n", "sampleRate": rate})
expect_error("n", "a chunk that is not base64")
send({"type": "chunk", "id": "o", "seq": 0, "pcm": base64.b64encode(b"abc").decode()})
send({"type": "end", "id": "o", "sampleRate": rate})
expect_error("o", "a chunk of an odd number of bytes")
for x in chunks("p", short):
    send(x)
send({"type": "end", "id": "p"})
expect_error("p", "an end without sampleRate")
send({"type": "chunk", "id": "q", "seq": 0.5, "pcm": ""})
expect_error("q", "a seq that is not a whole number")

for line, want_id in [
    ("this is not JSON", None),
    ('["id", "r"]', None),
    (json.dumps({"type": "chunk", "seq": 0, "pcm": ""}), None),
    (json.dumps({"type": "pause", "id": "s"}), "s"),
    (json.dumps({"type": "end", "id": "t", "sampleRate": None}), "t"),
]:
    send_line(line)
    m = read()
    assert m["type"] == "error" and m.get("id") == want_id, (line, m)
    print(f"{line[:40]!r}: error{' for ' + want_id if want_id else ' without an id'}: {m['error']}")

# Nothing was left unanswered: the next request is answered at once.
for x in chunks("u", short):
    send(x)
send({"type": "end", "id": "u", "sampleRate": rate})
m = read()
assert m["type"] == "text" and m["id"] == "u", m

proc.stdin.close()
proc.wait(timeout=30)
rest = proc.stdout.read()
if rest:
    raise SystemExit(f"the worker wrote after its last answer: {rest[:200]!r}")
print("ok, exit", proc.returncode)
