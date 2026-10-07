"""Drives `speech worker` with a synthesis model through protocol 2 the way a caller does, with every line checked by
worker_client.py: one JSON object per line, and one terminal message per request and nothing after it.

It checks ready (protocol 2, the release, and the model information of `speech info --json` with the device, the
threads and the voices of --add-voice), info and count_tokens, also answered while a synthesis runs; that a seed repeats the audio and a drawn seed is
reported and repeats it too; cancels while a request waits, while it runs, of an unknown id, and after an end, whose id
a later request then reuses; a second request under an id in flight; each error with its code and option (members the
message does not have, values of the wrong type, out of range or not taken, a missing text, lines that are not JSON
objects or have no id or type, the other task's messages); a peek and chunks to a synthesis model; add_voice; for
Irodori-TTS a fixed length and the progress of a long sampler, and for Qwen3-TTS max_seconds and the sampling options.
Writes the first answer to a WAV.

usage: python3 tools/worker_smoke.py <out.wav> <speech> <model.gguf> [worker options...]
       A model that takes voice files needs --add-voice NAME=FILE, whose FILE add_voice adds again under another name.
"""

import base64
import sys
import time
import wave

from worker_client import Worker, short

out_wav, speech, model, *options = sys.argv[1:]
added = [o.split("=", 1) for i, o in enumerate(options) if i > 0 and options[i - 1] == "--add-voice"]
w = Worker(speech, model, options)
ready = w.ready
info = ready["model"]
assert ready["protocol"] == 2 and isinstance(ready["version"], str) and info["task"] == "synthesis", short(ready)
w.check_model_information([name for name, _ in added])
rate = info["sample_rate"]
voice = info["voices"][0]["name"]
irodori = info["architecture"] == "irodori-tts"
print(f"ready in {time.perf_counter() - w.started:.2f} s: {info['name']} on {info['device']}, protocol 2, speech.cpp {ready['version']}, "
      f"{len(info['voices'])} voices, model information equal to speech info --json")

TEXT = "明日の東京は晴れで、最高気温は二十四度の予報です。"
LONG = "これは途中で止める長めの文です。止まったら終わりの知らせの代わりに取り消しの知らせが来ます。"


def speak(id, text=TEXT, **members):
    """The audio and the end of a synthesize."""
    w.request({"type": "synthesize", "id": id, "text": text, "voice": voice, **members})
    pcm = bytearray()
    for m in w.until(id):
        if m["type"] == "chunk":
            pcm += base64.b64decode(m["pcm"])
    if m["type"] != "end":
        raise SystemExit(f"{id}: {short(m)}")
    assert m["samples"] * 2 == len(pcm) and m["stop"] in ("complete", "max_seconds", "model_limit"), short(m)
    return bytes(pcm), m


def expect_error(message, code, option):
    w.request(message)
    m = w.terminal(message["id"], "error", code, option)
    print(f"{message['id']}: {code} ({option}) as expected: {m['error']['message']}")


w.request({"type": "info", "id": "info"})
assert w.terminal("info", "end")["model"] == info, "info's model differs from ready's"
w.request({"type": "count_tokens", "id": "count", "text": TEXT})
count = w.terminal("count", "end")
assert isinstance(count["tokens"], int) and 0 < count["tokens"] <= info["max_text_tokens"], short(count)
print(f"info: the model of ready; count_tokens: {count['tokens']} tokens")

t0 = time.perf_counter()
pcm_a, end_a = speak("a", seed=11)
assert end_a["seed"] == 11 and end_a["stop"] == "complete", short(end_a)
pcm_a2, _ = speak("a2", seed=11)
assert pcm_a == pcm_a2, "the same seed gave other audio"
print(f"a: {len(pcm_a) // 2 / rate:.2f} s of audio in {time.perf_counter() - t0:.2f} s, seed 11; the same seed repeats it byte for byte")
pcm_d, end_d = speak("d")
assert 0 <= end_d["seed"] < 2 ** 53, short(end_d)
assert speak("d2", seed=end_d["seed"])[0] == pcm_d, "the reported seed did not repeat the audio"
print(f"d: drew the seed {end_d['seed']}, which repeats the audio")

# A cancel while the request runs ends it with cancelled; one while it waits, at once, with no chunk.
w.request({"type": "synthesize", "id": "b", "text": LONG, "voice": voice})
while (m := w.next_for("b"))["type"] == "progress":
    pass
assert m["type"] == "chunk", short(m)
w.send({"type": "cancel", "id": "b"})
w.request({"type": "synthesize", "id": "c", "text": "三つ目です。", "voice": voice})
w.request({"type": "synthesize", "id": "c2", "text": "四つ目です。", "voice": voice})
w.send({"type": "cancel", "id": "c2"})
b = w.until("b")[-1]["type"]
c2 = w.until("c2")
assert [m["type"] for m in c2] == ["cancelled"], [short(m) for m in c2]
w.terminal("c", "end")
assert b == "cancelled", b
print("b: cancelled while it ran; c2: cancelled while it waited, with no chunk; c answered")

# A cancel of an id no request has, and one after its request ended, change nothing later: the id is reused.
w.send({"type": "cancel", "id": "nothing"})
w.send({"type": "cancel", "id": "a"})
assert speak("a", seed=11)[0] == pcm_a, "the reused id gave other audio"
print("a cancel of an unknown id and one after the end changed nothing; the id a was answered again")

# A request under an id in flight is refused without an id, and the request in flight is answered.
w.request({"type": "synthesize", "id": "x", "text": LONG, "voice": voice})
w.send({"type": "synthesize", "id": "x", "text": "二つ目。", "voice": voice})
m = w.error_without_id("invalid_argument", "id")
w.terminal("x", "end")
print(f"a second request under x: an error without an id ({m['error']['message'][:60]}...); x answered")

# count_tokens and info read the model's information alone, so they are answered while a synthesis runs.
w.request({"type": "synthesize", "id": "busy", "text": LONG, "voice": voice})
while (m := w.next_for("busy"))["type"] == "progress":
    pass
assert m["type"] == "chunk", short(m)
w.request({"type": "count_tokens", "id": "busy-count", "text": TEXT})
w.request({"type": "info", "id": "busy-info"})
ended = []
while "busy" not in ended:
    m = w.read()
    if m["type"] in ("end", "error", "cancelled"):
        assert m["type"] == "end", short(m)
        ended.append(m["id"])
        if m["id"] == "busy-count":
            assert m["tokens"] == count["tokens"], short(m)
        if m["id"] == "busy-info":
            assert m["model"] == info, short(m)
assert ended == ["busy-count", "busy-info", "busy"], ended
print("count_tokens and info sent while a synthesis ran: answered before it ended")

expect_error({"type": "synthesize", "id": "e1", "text": "あ。", "voice": voice, "bogus": 1}, "invalid_argument", "bogus")
expect_error({"type": "synthesize", "id": "e2", "text": "あ。", "voice": voice, "speed": "fast"}, "invalid_argument", "speed")
expect_error({"type": "synthesize", "id": "e3", "text": "あ。", "voice": voice, "seed": 1.5}, "invalid_argument", "seed")
expect_error({"type": "synthesize", "id": "e4", "text": "あ。", "voice": "no-such-voice"}, "out_of_range", "voice")
expect_error({"type": "synthesize", "id": "e5", "voice": voice}, "invalid_argument", "text")
expect_error({"type": "synthesize", "id": "e6", "text": "", "voice": voice}, "invalid_argument", "text")
expect_error({"type": "synthesize", "id": "e7", "text": "あ。"}, "invalid_argument", "voice")
expect_error({"type": "synthesize", "id": "e8", "text": "あ。", "voice": voice, "timestamps": True}, "unsupported", "timestamps")
too_long, tokens = TEXT, count["tokens"]
while tokens <= info["max_text_tokens"]:
    too_long += too_long
    w.request({"type": "count_tokens", "id": "count-long", "text": too_long})
    tokens = w.terminal("count-long", "end")["tokens"]
expect_error({"type": "synthesize", "id": "e9", "text": too_long, "voice": voice}, "out_of_range", "text")
expect_error({"type": "pause", "id": "e10"}, "invalid_argument", "type")
expect_error({"type": "add_voice", "id": "e11", "name": "x"}, "invalid_argument", "path")
# A member set to null counts as left out, and an option at its neutral value is taken by every model.
speak("n", text="あ。", language=None, speed=1, duration_scale=1, timestamps=False)
print("n: null, a speed and a scale of 1 and timestamps false were taken")
if irodori:
    expect_error({"type": "synthesize", "id": "e12", "text": "あ。", "voice": voice, "speed": 5}, "out_of_range", "speed")
    expect_error({"type": "synthesize", "id": "e13", "text": "あ。", "voice": voice, "max_seconds": 1}, "unsupported", "max_seconds")
    expect_error({"type": "synthesize", "id": "e14", "text": "あ。", "voice": voice, "seconds": 2, "duration_scale": 1.5},
                 "invalid_argument", "seconds")
    expect_error({"type": "synthesize", "id": "e15", "text": "あ。", "voice": voice, "temperature": 0.5}, "unsupported", "temperature")
    pcm, end = speak("s", text="三つ目です。", seconds=1)
    assert end["samples"] <= rate, short(end)
    print(f"s: a length of 1 s gave {end['samples'] / rate:.3f} s")
    # A sampler of many steps passes no audio for seconds, and the worker reports its progress at most once a second.
    w.request({"type": "synthesize", "id": "p", "text": "あ。", "voice": voice, "steps": 160})
    messages = w.until("p")
    progress = [m for m in messages if m["type"] == "progress"]
    assert messages[-1]["type"] == "end" and progress, [short(m) for m in messages if m["type"] != "chunk"]
    done = [m["done"] for m in progress]
    gaps = [b["_at"] - a["_at"] for a, b in zip(progress, progress[1:])]
    assert done == sorted(done) and 0 <= done[0] and done[-1] <= 1 and all(g > 0.9 for g in gaps), (done, gaps)
    print(f"p: 160 steps, {len(progress)} progress messages ({', '.join(f'{d:.2f}' for d in done)}) at least a second apart")
else:
    expect_error({"type": "synthesize", "id": "e12", "text": "あ。", "voice": voice, "speed": 1.5}, "unsupported", "speed")
    expect_error({"type": "synthesize", "id": "e13", "text": "あ。", "voice": voice, "seconds": 2}, "unsupported", "seconds")
    expect_error({"type": "synthesize", "id": "e14", "text": "あ。", "voice": voice, "steps": 4}, "unsupported", "steps")
    pcm, end = speak("s", text=LONG, max_seconds=0.5)
    assert end["stop"] == "max_seconds" and end["samples"] <= 0.5 * rate, short(end)
    print(f"s: max_seconds 0.5 stopped it at {end['samples'] / rate:.3f} s ({end['stop']})")
    # Neither stack draws, so the seed changes nothing; a temperature of a stack that does not draw is refused.
    greedy = dict(text="あ。", do_sample=False, code_predictor_do_sample=False, max_seconds=2)
    assert speak("g1", seed=1, **greedy)[0] == speak("g2", seed=2, **greedy)[0], "a request that draws nothing changed with the seed"
    print("g1, g2: do_sample and code_predictor_do_sample false gave the same audio for the seeds 1 and 2")
    expect_error({"type": "synthesize", "id": "e15", "text": "あ。", "voice": voice, "do_sample": False, "temperature": 0.5},
                 "invalid_argument", "temperature")
    expect_error({"type": "synthesize", "id": "e16", "text": "あ。", "voice": voice, "code_predictor_top_k": -1}, "out_of_range",
                 "code_predictor_top_k")
    expect_error({"type": "synthesize", "id": "e17", "text": "あ。", "voice": voice, "do_sample": "no"}, "invalid_argument", "do_sample")

# The other task's messages: a chunk opens a recognition request answered unsupported, whose later lines are dropped.
w.request({"type": "chunk", "id": "r", "seq": 0, "pcm": ""})
w.terminal("r", "error", "unsupported", "type")
w.send({"type": "chunk", "id": "r", "seq": 1, "pcm": ""})
w.send({"type": "transcribe", "id": "r", "sample_rate": 16000})
w.send({"type": "peek", "id": "r2", "sample_rate": 16000})
m = w.next_for("r2")
assert m["type"] == "partial" and m["error"]["code"] == "unsupported", short(m)
w.request({"type": "transcribe", "id": "r3", "sample_rate": 16000})
w.terminal("r3", "error", "unsupported", "type")
print("chunk and transcribe: unsupported once, the request's later lines dropped; peek: a partial with the error")

# Lines that name no request are answered without an id.
for line in ["this is not JSON", '["id", "h"]', '{"type": "synthesize", "text": "あ。"}', '{"type": "synthesize", "id": 5}',
             '{"type": "info", "id": ""}', '{"type": "cancel", "id": "a", "now": true}']:
    w.send_line(line)
    m = w.error_without_id()
    print(f"{line[:44]!r}: an error without an id: {m['error']['message'][:90]}")

if info["voice_files"]:
    name, path = added[0]
    w.request({"type": "add_voice", "id": "v", "name": "added", "path": path})
    w.terminal("v", "end")
    w.request({"type": "info", "id": "v-info"})
    assert [v["name"] for v in w.terminal("v-info", "end")["model"]["voices"]] == [n for n, _ in added] + ["added"]
    voice = "added"
    speak("v-speak", text="あ。")
    expect_error({"type": "add_voice", "id": "v2", "name": "added", "path": path}, "invalid_argument", "name")
    expect_error({"type": "add_voice", "id": "v3", "name": "other", "path": "/no/such/voice.gguf"}, "io", "path")
    print("add_voice: the voice was added, info lists it, and a synthesis speaks with it")
else:
    expect_error({"type": "add_voice", "id": "v", "name": "x", "path": "x.wav"}, "unsupported", None)

# Requests queued when stdin closes are answered before the worker exits with 0.
w.request({"type": "synthesize", "id": "last", "text": "おしまい。", "voice": voice})
rest = w.close()
assert rest[-1]["type"] == "end" and rest[-1]["id"] == "last", short(rest[-1])
with wave.open(out_wav, "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(rate)
    f.writeframes(pcm_a)
print("ok: every request had one terminal message; the worker answered the last after stdin closed and exited with 0")
