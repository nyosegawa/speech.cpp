"""Drives `speech worker` with an enhancement model through its protocol the way a caller does, with every line checked by
worker_client.py: one JSON object per line, and one terminal message per request and nothing after it.

It sends the audio of the dumps of reference/semamba/dump.py it is given, as 16-bit chunks of half a second at the dump's
rate, and checks that the chunks the worker sends back are, sample for sample, the WAVE that `speech enhance` writes for
the same 16-bit audio with the same options, that end gives their number and the stop complete, and that progress or
chunks come for each request; that bwe resamples the audio and the chunks come at its rate; that the chunks of two
requests may interleave and the requests are answered in the order of their enhance lines; cancels while a request
collects chunks and while it runs; each error with its code and option (audio at a rate outside the model's range
without bwe, audio shorter than the model's STFT, a chunk shorter than it, a bwe out of range, an unknown member, a
missing sample_rate, and the other tasks' messages); the error of a request still collecting chunks when stdin closes;
and that --chunk-size-in-seconds given to the worker is the default of a request that does not set it.

usage: python3 checks/smoke/worker_enhancement_smoke.py <speech> <model.gguf> <dump folder>... [-- worker options...]
"""

import array
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import wave

from worker_client import Worker, pcm16, read_npy, short

args = sys.argv[1:]
flags = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
speech, model, *dumps = args
if not dumps:
    raise SystemExit(__doc__)
workdir = tempfile.mkdtemp(prefix="enhancement-smoke-")


def rate_of(dump):
    with open(os.path.join(dump, "meta.json")) as f:
        return json.load(f)["rate"]


def write_wav(path, pcm, rate):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def cli_enhanced(pcm, rate, options):
    """The 16-bit samples and rate of `speech enhance` for the 16-bit audio `pcm` at `rate` with `options` as flags."""
    source, out = os.path.join(workdir, "in.wav"), os.path.join(workdir, "out.wav")
    write_wav(source, pcm, rate)
    device = flags[flags.index("--device"):flags.index("--device") + 2] if "--device" in flags else []
    command = [speech, "enhance", model, *device, "-o", out, source]
    for name, value in options.items():
        command += ["--" + name.replace("_", "-"), str(value)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with wave.open(out, "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


def send_audio(w, id, pcm, rate, options=None, piece=None):
    """Sends `pcm` as chunks of half a second, then the enhance line; the request is in flight from its first chunk."""
    piece = piece or rate
    first = True
    for seq, at in enumerate(range(0, len(pcm), piece)):
        message = {"type": "chunk", "id": id, "seq": seq, "pcm": base64.b64encode(pcm[at:at + piece]).decode()}
        if first:
            w.request(message)
            first = False
        else:
            w.send(message)
    w.send({"type": "enhance", "id": id, "sample_rate": rate, **(options or {})})


def received(messages):
    """The PCM of a request's chunks, in order, and its end."""
    pcm = b""
    seq = 0
    for m in messages[:-1]:
        if m["type"] != "chunk":
            assert m["type"] == "progress" and 0 <= m["done"] <= 1, short(m)
            continue
        assert m["seq"] == seq, short(m)
        seq += 1
        pcm += base64.b64decode(m["pcm"])
    end = messages[-1]
    assert end["type"] == "end", short(end)
    assert set(end) - {"_at"} == {"type", "id", "samples", "stop"}, short(end)
    assert end["samples"] == len(pcm) // 2 and end["stop"] == "complete", short(end)
    return pcm


w = Worker(speech, model, flags)
info = w.ready["model"]
assert info["task"] == "enhancement" and info["sample_rate"] == 0 and info["languages"] == [], short(w.ready)
w.check_model_information()
options = {o["name"]: o for o in info["options"]}
assert set(options) >= {"chunk_size_in_seconds", "hop_length_portion", "bwe"}, options
lowest, highest = options["bwe"]["minimum"], options["bwe"]["maximum"]
print(f"ready in {time.perf_counter() - w.started:.2f} s: {info['name']} on {info['device']}, rates {lowest} to {highest} Hz")

# Each dump's audio, chunk by chunk, against speech enhance.
for d in dumps:
    rate = rate_of(d)
    pcm = pcm16(read_npy(os.path.join(d, "audio.npy")))
    want, want_rate = cli_enhanced(pcm, rate, {})
    t0 = time.perf_counter()
    send_audio(w, "e", pcm, rate)
    got = received(w.until("e"))
    assert want_rate == rate and got == want, f"{d}: the worker's {len(got) // 2} samples differ from speech enhance's {len(want) // 2}"
    print(f"{os.path.basename(d)}: {len(pcm) // 2} samples at {rate} Hz in {time.perf_counter() - t0:.2f} s, speech enhance's sample for sample")

# bwe resamples, and the chunks come at its rate.
d = dumps[0]
rate = rate_of(d)
pcm = pcm16(read_npy(os.path.join(d, "audio.npy")))
target = int(lowest) if rate != int(lowest) else int(highest)
want, want_rate = cli_enhanced(pcm, rate, {"bwe": target})
send_audio(w, "bwe", pcm, rate, {"bwe": target})
got = received(w.until("bwe"))
assert want_rate == target and got == want, "bwe: the worker's audio differs from speech enhance --bwe"
print(f"bwe {target}: {len(got) // 2} samples, speech enhance --bwe's")

# Two requests whose chunks interleave are answered in the order of their enhance lines.
short_pcm = pcm[:2 * rate]
w.request({"type": "chunk", "id": "one", "seq": 0, "pcm": base64.b64encode(short_pcm).decode()})
w.request({"type": "chunk", "id": "two", "seq": 0, "pcm": base64.b64encode(short_pcm).decode()})
w.send({"type": "enhance", "id": "two", "sample_rate": rate})
w.send({"type": "enhance", "id": "one", "sample_rate": rate})
two, one = w.terminal("two", "end"), w.terminal("one", "end")
assert two["_at"] < one["_at"], "the requests were not answered in the order of their enhance lines"
print("interleaved chunks: answered in the order of their enhance lines")

# Refusals, each with its code and the input at fault.
tiny = pcm[:100]
for id, audio, at, extra, code, option in [
    ("high", short_pcm, int(highest) * 2, {}, "out_of_range", "audio"),
    ("low", short_pcm, int(lowest) // 2, {}, "out_of_range", "audio"),
    ("tiny", tiny, rate, {}, "out_of_range", "audio"),
    ("chunk", short_pcm, rate, {"chunk_size_in_seconds": 0.001}, "out_of_range", "chunk_size_in_seconds"),
    ("bwe-out", short_pcm, rate, {"bwe": int(highest) + 1}, "out_of_range", "bwe"),
    ("portion", short_pcm, rate, {"hop_length_portion": 1.5}, "out_of_range", "hop_length_portion"),
]:
    send_audio(w, id, audio, at, extra)
    w.terminal(id, "error", code, option)
w.request({"type": "chunk", "id": "member", "seq": 0, "pcm": base64.b64encode(tiny).decode()})
w.send({"type": "enhance", "id": "member", "sample_rate": rate, "speed": 1.5})
w.terminal("member", "error", "unsupported", "speed")
w.request({"type": "chunk", "id": "rate", "seq": 0, "pcm": base64.b64encode(tiny).decode()})
w.send({"type": "enhance", "id": "rate"})
w.terminal("rate", "error", "invalid_argument", "sample_rate")
w.request({"type": "synthesize", "id": "speak", "text": "Hello."})
w.terminal("speak", "error", "unsupported", "type")
w.request({"type": "chunk", "id": "recognize", "seq": 0, "pcm": base64.b64encode(tiny).decode()})
w.send({"type": "transcribe", "id": "recognize", "sample_rate": rate})
w.terminal("recognize", "error", "unsupported", "type")
w.request({"type": "enhance", "id": "nothing", "sample_rate": rate})
w.terminal("nothing", "error", "invalid_argument", "audio")
print("refusals: each with its code and option")

# Cancels: while a request collects chunks, and while it runs.
w.request({"type": "chunk", "id": "collecting", "seq": 0, "pcm": base64.b64encode(short_pcm).decode()})
w.send({"type": "cancel", "id": "collecting"})
w.terminal("collecting", "cancelled")
w.send({"type": "chunk", "id": "collecting", "seq": 1, "pcm": base64.b64encode(short_pcm).decode()})
w.send({"type": "enhance", "id": "collecting", "sample_rate": rate})
send_audio(w, "running", pcm, rate, {"chunk_size_in_seconds": 0.25})
first = w.next_for("running")
w.send({"type": "cancel", "id": "running"})
w.backlog.insert(0, first)
end = w.terminal("running")
assert end["type"] in ("cancelled", "end"), short(end)
print(f"cancels: while collecting chunks, and while running ({end['type']})")

# A request still collecting chunks when stdin closes gets an error.
w.request({"type": "chunk", "id": "open", "seq": 0, "pcm": base64.b64encode(tiny).decode()})
rest = w.close()
assert any(m.get("id") == "open" and m["type"] == "error" for m in rest), rest
print("stdin closed: the request still collecting chunks got its error")

# A request option given to the worker is the default of a request that does not set it.
w = Worker(speech, model, [*flags, "--chunk-size-in-seconds", "0.5"])
assert w.ready["defaults"] == {"chunk_size_in_seconds": 0.5}, short(w.ready)
want, _ = cli_enhanced(short_pcm, rate, {"chunk_size_in_seconds": 0.5})
send_audio(w, "default", short_pcm, rate)
assert received(w.until("default")) == want, "--chunk-size-in-seconds given to the worker was not the request's default"
w.close()
print("--chunk-size-in-seconds: the default of a request that does not set it")
print("worker enhancement smoke passed")
