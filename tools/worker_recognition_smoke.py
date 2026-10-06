"""Drives `speech worker` with a recognition model through protocol 2 the way a caller does, with every line checked by
worker_client.py: one JSON object per line, and one terminal message per request and nothing after it.

It sends the audio of each dump of reference/fastconformer/dump.py as 16-bit chunks of one second and checks that the
text is the dump's, and with timestamps that the segments and tokens join into it in time order; peeks at a request
while it collects chunks, with and without timestamps, and checks that the request then gets its one end; checks that
a peek of a request cancelled before the peek's turn is dropped, and that a peek of an id that is not collecting is a
partial with an error; that the chunks of two requests may interleave and the requests are answered in the order of
their transcribes; cancels while a request collects chunks (whose later lines are dropped and whose id a chunk 0 starts
again), while it waits and while it runs; that audio at three times the model's rate is recognized; each error with its
code and option (a chunk out of order, not base64 or of odd bytes, a sample rate missing or 0, an unknown language or
member, audio missing, the other task's messages); info and the model information of ready; progress for a recording
longer than a minute; and the error of a request still collecting chunks when stdin closes.

usage: python3 tools/worker_recognition_smoke.py <speech> <model.gguf> <dump folder>... [-- worker options...]
"""

import array
import ast
import base64
import os
import struct
import sys
import time

from worker_client import Worker, short

args = sys.argv[1:]
options = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
speech, model, *dumps = args
if not dumps:
    raise SystemExit(__doc__)
w = Worker(speech, model, options)
info = w.ready["model"]
assert w.ready["protocol"] == 2 and info["task"] == "recognition" and "voices" not in info, short(w.ready)
w.check_model_information()
rate = info["sample_rate"]
print(f"ready in {time.perf_counter() - w.started:.2f} s: {info['name']} on {info['device']}, rate {rate}, languages {info['languages']}, "
      f"model information equal to speech info --json")


def pcm16(samples):
    """Float samples as 16-bit little-endian PCM, scaled as the worker scales them back (by 32768)."""
    return array.array("h", [max(-32768, min(32767, round(x * 32768))) for x in samples]).tobytes()


def chunks(id, pcm, size=None):
    size = size or 2 * rate
    return [{"type": "chunk", "id": id, "seq": i, "pcm": base64.b64encode(pcm[at:at + size]).decode()}
            for i, at in enumerate(range(0, len(pcm), size))]


def send_audio(id, pcm, **members):
    """A whole recognition request: its chunks and its transcribe."""
    w.expect(id)
    for c in chunks(id, pcm):
        w.send(c)
    w.send({"type": "transcribe", "id": id, "sample_rate": members.pop("sample_rate", rate), **members})


def read_npy(path):
    with open(path, "rb") as f:
        data = f.read()
    header_length = struct.unpack("<H", data[8:10])[0]
    header = ast.literal_eval(data[10:10 + header_length].decode())
    assert header["descr"] == "<f4" and len(header["shape"]) == 1, header
    values = array.array("f")
    values.frombytes(data[10 + header_length:])
    return values


def check_times(m):
    """The segments and tokens of a message join into its text and run forward in time."""
    for key in ("segments", "tokens"):
        items = m[key]
        assert "".join(s["text"] for s in items) == m["text"], (key, short(m))
        assert all(0 <= s["start"] <= s["end"] for s in items), (key, short(m))
        assert all(a["start"] <= b["start"] for a, b in zip(items, items[1:])), (key, short(m))


def expect_error(id, code, option, what):
    m = w.terminal(id, "error", code, option)
    print(f"{what}: {code} ({option}) as expected: {m['error']['message'][:100]}")


audio, want_text = {}, {}
for d in dumps:
    name = os.path.basename(os.path.normpath(d))
    with open(os.path.join(d, "text.txt"), encoding="utf-8") as f:
        want_text[name] = f.read()
    audio[name] = pcm16(read_npy(os.path.join(d, "audio.npy")))
    t1 = time.perf_counter()
    send_audio(name, audio[name], language=info["languages"][0], timestamps=True)
    messages = w.until(name)
    m = messages[-1]
    assert m["type"] == "end", short(m)
    if m["text"] != want_text[name]:
        raise SystemExit(f"{name}: the text differs from the dump's text\n  got  {m['text']}\n  want {want_text[name]}")
    check_times(m)
    seconds = len(audio[name]) / 2 / rate
    progress = [p for p in messages if p["type"] == "progress"]
    gaps = [b["_at"] - a["_at"] for a, b in zip(progress, progress[1:])]
    assert all(g > 0.9 for g in gaps) and all(0 <= p["done"] <= 1 for p in progress), [short(p) for p in progress]
    if seconds > 60:
        assert progress, f"{name}: {seconds:.0f} s of audio sent no progress"
    print(f"{name}: {seconds:.2f} s, the dump's text in {time.perf_counter() - t1:.3f} s, {len(m['segments'])} segments and "
          f"{len(m['tokens'])} tokens joining into it, {len(progress)} progress messages")

by_length = sorted(audio, key=lambda n: len(audio[n]))
first, last = by_length[0], by_length[-1]
short_pcm, short_text = audio[first], want_text[first]

# A peek recognizes the audio so far and leaves the request open for more chunks and its one end.
c = chunks("p", short_pcm)
half = max(1, len(c) // 2)
w.expect("p")
for x in c[:half]:
    w.send(x)
w.send({"type": "peek", "id": "p", "sample_rate": rate})
first_partial = w.next_for("p")
assert first_partial["type"] == "partial" and isinstance(first_partial["text"], str) and "segments" not in first_partial, short(first_partial)
for x in c[half:]:
    w.send(x)
w.send({"type": "peek", "id": "p", "sample_rate": rate, "timestamps": True})
m2 = w.next_for("p")
assert m2["type"] == "partial" and m2["text"] == short_text, short(m2)
check_times(m2)
w.send({"type": "peek", "id": "p", "sample_rate": rate, "speed": 2})
m3 = w.next_for("p")
assert m3["type"] == "partial" and m3["error"]["code"] == "unsupported" and m3["error"]["option"] == "speed", short(m3)
w.send({"type": "transcribe", "id": "p", "sample_rate": rate})
assert w.terminal("p", "end")["text"] == short_text
w.send({"type": "peek", "id": "p", "sample_rate": rate})
m = w.next_for("p")
assert m["type"] == "partial" and m["error"]["option"] == "id", short(m)
print(f"peek: half the audio gave {first_partial['text'][:20]!r}...; the whole gave the dump's text with times; a peek's error left the request "
      f"open; the request ended once; a peek after it is a partial with an error")

# A peek of a request cancelled before the peek's turn is dropped without an answer.
send_audio("long", audio[last])
w.expect("q")
for x in chunks("q", short_pcm):
    w.send(x)
w.send({"type": "peek", "id": "q", "sample_rate": rate})
w.send({"type": "cancel", "id": "q"})
assert [m["type"] for m in w.until("q")] == ["cancelled"]
w.terminal("long", "end")
w.request({"type": "info", "id": "after-q"})
w.terminal("after-q", "end")
assert not any(m.get("id") == "q" for m in w.backlog), w.backlog
print("a peek of a request cancelled while it collected chunks was dropped; the request had its cancelled alone")

# Two requests whose chunks interleave are answered in the order of their transcribes.
a, b = chunks("a", short_pcm), chunks("b", short_pcm)
w.expect("a")
w.expect("b")
for i in range(max(len(a), len(b))):
    for x in a[i:i + 1] + b[i:i + 1]:
        w.send(x)
w.send({"type": "transcribe", "id": "b", "sample_rate": rate})
w.send({"type": "transcribe", "id": "a", "sample_rate": rate})
got = [w.read(), w.read()]
assert [(m["type"], m["id"]) for m in got] == [("end", "b"), ("end", "a")] and got[0]["text"] == got[1]["text"], [short(m) for m in got]
print("interleaved chunks: answered in the order of the transcribes")

# A cancel while collecting answers at once, and the request's later lines are dropped; a chunk 0 starts the id again.
c = chunks("c", short_pcm)
w.expect("c")
w.send(c[0])
w.send({"type": "cancel", "id": "c"})
assert w.terminal("c", "cancelled")
for x in c[1:]:
    w.send(x)
w.send({"type": "transcribe", "id": "c", "sample_rate": rate})
send_audio("c", short_pcm)
assert w.terminal("c", "end")["text"] == short_text
w.expect("r")
w.send(c[0] | {"id": "r"})
w.send({"type": "cancel", "id": "r"})
w.terminal("r", "cancelled")
send_audio("r", short_pcm)
assert w.terminal("r", "end")["text"] == short_text
print("cancelled while collecting: cancelled at once, its later lines dropped; the id answered again, with and without a transcribe "
      "after the cancel")

# A cancel while waiting behind another request, and one while running.
send_audio("d", audio[last])
send_audio("e", short_pcm)
w.send({"type": "cancel", "id": "e"})
w.terminal("e", "cancelled")
w.terminal("d", "end")
send_audio("g", audio[last])
time.sleep(0.05)
w.send({"type": "cancel", "id": "g"})
g = w.until("g")[-1]["type"]
send_audio("h", short_pcm)
w.terminal("h", "end")
print(f"cancelled while waiting: cancelled; while running: {g}; the next answered")

# The short audio with each sample three times, at three times the rate: the library resamples it to the model's.
tripled = array.array("h")
tripled.frombytes(short_pcm)
tripled = array.array("h", [x for x in tripled for _ in range(3)]).tobytes()
send_audio("i", tripled, sample_rate=rate * 3)
m = w.terminal("i", "end")
print(f"audio at {rate * 3} Hz: {'the dump' if m['text'] == short_text else 'another'}'s text")

send_audio("v", short_pcm, sample_rate=0)
expect_error("v", "invalid_argument", "sample_rate", "a sample rate of 0")
send_audio("v2", short_pcm, sample_rate=None)
expect_error("v2", "invalid_argument", "sample_rate", "no sample rate")
send_audio("j", short_pcm, language="zz")
expect_error("j", "out_of_range", "language", "a language the model does not recognize")
send_audio("j2", short_pcm, bogus=1)
expect_error("j2", "invalid_argument", "bogus", "a member transcribe does not have")
w.request({"type": "transcribe", "id": "k", "sample_rate": rate})
expect_error("k", "invalid_argument", "audio", "a transcribe without chunks")
w.request({"type": "synthesize", "id": "l", "text": "明日の東京は晴れです。", "voice": "x"})
expect_error("l", "unsupported", "type", "a synthesize")
w.request({"type": "count_tokens", "id": "l2", "text": "明日の東京は晴れです。"})
expect_error("l2", "unsupported", None, "count_tokens")
w.request({"type": "add_voice", "id": "l3", "name": "x", "path": "x.wav"})
expect_error("l3", "unsupported", None, "add_voice")
w.request({"type": "info", "id": "l4"})
assert w.terminal("l4", "end")["model"] == info

# A refused chunk is its request's one answer: its later chunks and its transcribe are dropped.
m_chunks = chunks("m", short_pcm)
w.expect("m")
w.send(m_chunks[1])
for x in m_chunks[2:]:
    w.send(x)
w.send({"type": "transcribe", "id": "m", "sample_rate": rate})
expect_error("m", "invalid_argument", "seq", "a chunk out of order")
w.request({"type": "chunk", "id": "n", "seq": 0, "pcm": "not base64!"})
w.send({"type": "transcribe", "id": "n", "sample_rate": rate})
expect_error("n", "invalid_argument", "pcm", "a chunk that is not base64")
w.request({"type": "chunk", "id": "o", "seq": 0, "pcm": base64.b64encode(b"abc").decode()})
w.send({"type": "transcribe", "id": "o", "sample_rate": rate})
expect_error("o", "invalid_argument", "pcm", "a chunk of an odd number of bytes")
w.request({"type": "chunk", "id": "o2", "seq": 0.5, "pcm": ""})
expect_error("o2", "invalid_argument", "seq", "a seq that is not a whole number")
w.request({"type": "chunk", "id": "o3", "seq": 0, "pcm": "", "language": "ja"})
expect_error("o3", "invalid_argument", "language", "a member chunk does not have")
send_audio("m", short_pcm)
assert w.terminal("m", "end")["text"] == short_text
w.send({"type": "peek", "id": "nothing", "sample_rate": rate})
m = w.next_for("nothing")
assert m["type"] == "partial" and m["error"]["option"] == "id", short(m)
print("a peek of an unknown id: a partial with an error")

send_audio("u", audio[last])
w.send({"type": "transcribe", "id": "u", "sample_rate": rate})
w.error_without_id("invalid_argument", "id")
w.send(chunks("u", short_pcm)[0])
w.error_without_id("invalid_argument", "id")
w.terminal("u", "end")
print("a second transcribe and a chunk of a request that waits: errors without an id; the request answered")
for line in ["this is not JSON", '{"type": "chunk", "seq": 0, "pcm": ""}']:
    w.send_line(line)
    m = w.error_without_id()
    print(f"{line[:44]!r}: an error without an id: {m['error']['message'][:80]}")

# A request still collecting chunks when stdin closes is answered with an error.
w.request(chunks("z", short_pcm)[0])
rest = w.close()
assert [(m["type"], m["id"]) for m in rest] == [("error", "z")], [short(m) for m in rest]
print("ok: every request had one terminal message; the one collecting chunks when stdin closed had an error; exit 0")
