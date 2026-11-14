"""Drives `speech worker` with a dialogue model through its protocol the way a caller does, with every line checked by
worker_client.py: one JSON object per line, and one terminal message per request and nothing after it.

Given a dump of reference/haru/dump.py, it adds the dump's voice reference with add_voice and holds conversations that
hear the dump's user audio as 16-bit chunks, and checks: that the model information of ready is that of `speech info
--json`; that a conversation's first chunk comes before any of its audio is sent; that the chunks the worker sends back,
a frame each and numbered from 0, are sample for sample the WAVE that `speech talk` writes for the same 16-bit audio and
options, and their text what it writes on stdout; that end gives the frames and the samples sent, the audio short of a
frame dropped; that chunks of another length give the same frames; that a conversation whose chunks and hang_up arrive
while another runs hears them once it runs, after that one's end; cancels while a conversation waits and while it runs,
after which its chunks and its hang_up are dropped; each error with its code and option (max_seconds reached, a chunk
out of order while the conversation runs, pcm that is not base64 while it waits, a missing sample_rate, an unknown
member, an option the model does not take, a chunk and a hang_up with no conversation, a hang_up with a member it does
not have and the other tasks' messages); and that stdin closing ends a running conversation as its hang_up does.

usage: python3 checks/smoke/worker_dialogue_smoke.py <speech> <model.gguf> <dump folder> [-- worker options...]
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
import wave

from worker_client import Worker, pcm16, read_npy, short

args = sys.argv[1:]
flags = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
if len(args) != 3:
    raise SystemExit(__doc__)
speech, model, dump = args
workdir = tempfile.mkdtemp(prefix="dialogue-smoke-")
with open(os.path.join(dump, "meta.json")) as f:
    meta = json.load(f)
rate, frame = meta["sample_rate"], meta["frame_samples"]
# 30 whole frames of the user's audio and part of another, which the conversation drops.
WHOLE = 30
user = pcm16(read_npy(os.path.join(dump, "user_audio.npy")))[:(WHOLE * frame + frame // 3) * 2]
OPTIONS = {"language": "ja", "name": "ハル", "persona": "落ち着いた話し方をする大学生。", "backchannel": "mid", "voice": "ref", "seed": 7}


def write_wav(path, pcm, rate):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


reference = os.path.join(workdir, "reference.wav")
write_wav(reference, pcm16(read_npy(os.path.join(dump, "reference_audio.npy"))), rate)


def chunks(w, id, pcm, piece, start=0):
    """Sends `pcm` as chunks of `piece` samples, numbered from `start`; returns the next number."""
    seq = start
    for at in range(0, len(pcm), piece * 2):
        w.send({"type": "chunk", "id": id, "seq": seq, "pcm": base64.b64encode(pcm[at:at + piece * 2]).decode()})
        seq += 1
    return seq


def frames(messages):
    """The PCM and text of a conversation's chunks, checked to be a frame each, numbered from 0, and its terminal message."""
    pcm, text = b"", ""
    for seq, m in enumerate(messages[:-1]):
        if m["type"] != "chunk" or m["seq"] != seq or not isinstance(m.get("text"), str):
            raise SystemExit(f"chunk {seq} of a conversation is {short(m)}")
        piece = base64.b64decode(m["pcm"])
        if len(piece) != frame * 2:
            raise SystemExit(f"chunk {seq} holds {len(piece) // 2} samples, not a frame of {frame}")
        pcm += piece
        text += m["text"]
    return pcm, text, messages[-1]


def converse(w, id, pcm, piece, options=OPTIONS):
    """A conversation that hears `pcm` in chunks of `piece` samples and hangs up; checks that its first chunk comes first."""
    w.request({"type": "dialogue", "id": id, "sample_rate": rate, **options})
    first = w.next_for(id)
    if first["type"] != "chunk" or first["seq"] != 0:
        raise SystemExit(f"a conversation began with {short(first)}, not its first frame")
    chunks(w, id, pcm, piece)
    w.send({"type": "hang_up", "id": id})
    return frames([first] + w.until(id))


def nothing_more(w, why):
    """Checks, through an info answered at once, that the lines sent before it got no answer."""
    w.request({"type": "info", "id": "sentinel"})
    w.terminal("sentinel", "end")
    stray = [m for m in w.backlog if "id" not in m or m["id"] not in w.in_flight]
    if stray:
        raise SystemExit(f"{why}, and the worker answered {short(stray[0])}")


def cli_talk(pcm):
    """The 16-bit samples and the text of `speech talk` hearing the 16-bit audio `pcm` with OPTIONS as flags."""
    heard, out = os.path.join(workdir, "user.wav"), os.path.join(workdir, "reply.wav")
    write_wav(heard, pcm, rate)
    device = flags[flags.index("--device"):flags.index("--device") + 2] if "--device" in flags else []
    command = [speech, "talk", model, *device, "--add-voice", "ref=" + reference, "-o", out, heard]
    for name, value in OPTIONS.items():
        command += ["--" + name.replace("_", "-"), str(value)]
    r = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    with wave.open(out, "rb") as f:
        assert f.getframerate() == rate, f.getframerate()
        return f.readframes(f.getnframes()), r.stdout.decode("utf-8")


w = Worker(speech, model, flags)
w.check_model_information()
assert w.ready["model"]["task"] == "dialogue" and w.ready["model"]["frame_samples"] == frame, short(w.ready)
w.request({"type": "add_voice", "id": "v", "name": "ref", "path": reference})
w.terminal("v", "end")
print(f"ready: {w.ready['model']['name']}, the model information of speech info --json; the voice ref added from the dump's reference")

pcm, text, end = converse(w, "a", user, rate // 2)
want = WHOLE + 1
if end["type"] != "end" or end["frames"] != want or end["samples"] != want * frame or len(pcm) != want * frame * 2:
    raise SystemExit(f"a conversation of {WHOLE} whole frames ended with {short(end)} after {len(pcm) // 2 // frame} frames")
talked, said = cli_talk(user)
if talked != pcm:
    raise SystemExit("speech talk's WAVE differs from the worker's chunks of the same audio")
if said != text + ("\n" if text else ""):
    raise SystemExit(f"speech talk wrote {said!r}, the worker's chunks {text!r}")
print(f"a conversation of {WHOLE} frames and a third: its first chunk before any audio, {end['frames']} chunks sample for sample speech talk's WAVE, "
      f"their text its stdout ({text.strip()!r})")

again, again_text, _ = converse(w, "b", user, 1234)
if (again, again_text) != (pcm, text):
    raise SystemExit("chunks of 1234 samples gave other frames than chunks of half a second")
print("chunks of 1234 samples: the same frames and text")

# A conversation that waits behind another hears the chunks and the hang_up that arrived while it waited.
w.request({"type": "dialogue", "id": "c", "sample_rate": rate, **OPTIONS})
w.next_for("c")
w.request({"type": "dialogue", "id": "d", "sample_rate": rate, **OPTIONS})
chunks(w, "d", user, rate // 2)
w.send({"type": "hang_up", "id": "d"})
chunks(w, "c", user[:frame * 2 * 3], frame)
w.send({"type": "hang_up", "id": "c"})
c_end = w.until("c")[-1]
d = w.until("d")
waited, waited_text, _ = frames(d)
if c_end["type"] != "end" or d[0]["_at"] < c_end["_at"] or (waited, waited_text) != (pcm, text):
    raise SystemExit(f"a conversation that waited gave other frames, or began before the one before it ended ({short(c_end)})")
print("a conversation whose chunks and hang_up arrived while another ran: begun after that one's end, the same frames")

# Cancels: of a conversation that waits, at once, and of one that runs; their later chunks and hang_up are dropped.
w.request({"type": "dialogue", "id": "e", "sample_rate": rate, **OPTIONS})
w.next_for("e")
next_seq = chunks(w, "e", user[:frame * 2 * 4], frame)
w.request({"type": "dialogue", "id": "f", "sample_rate": rate, **OPTIONS})
w.send({"type": "cancel", "id": "f"})
w.terminal("f", "cancelled")
w.send({"type": "chunk", "id": "f", "seq": 0, "pcm": base64.b64encode(user[:frame * 2]).decode()})
w.send({"type": "hang_up", "id": "f"})
w.send({"type": "cancel", "id": "e"})
w.terminal("e", "cancelled")
chunks(w, "e", user[:frame * 2], frame, next_seq)
w.send({"type": "hang_up", "id": "e"})
nothing_more(w, "chunks and a hang_up followed cancelled conversations")
print("cancels: of a waiting conversation at once, of a running one at its next frame; their later chunks and hang_up dropped")

# Errors that end a conversation: max_seconds reached, a chunk out of order while it runs, pcm not base64 while it waits.
w.request({"type": "dialogue", "id": "g", "sample_rate": rate, **OPTIONS, "max_seconds": 1})
seq = chunks(w, "g", user, frame)
w.terminal("g", "error", "out_of_range", "max_seconds")
w.send({"type": "chunk", "id": "g", "seq": seq, "pcm": base64.b64encode(user[:frame * 2]).decode()})
w.send({"type": "hang_up", "id": "g"})
w.request({"type": "dialogue", "id": "h", "sample_rate": rate, **OPTIONS})
w.next_for("h")
chunks(w, "h", user[:frame * 2], frame)
w.request({"type": "dialogue", "id": "i", "sample_rate": rate, **OPTIONS})
w.send({"type": "chunk", "id": "i", "seq": 0, "pcm": "not base64!"})
w.terminal("i", "error", "invalid_argument", "pcm")
w.send({"type": "chunk", "id": "h", "seq": 5, "pcm": base64.b64encode(user[:frame * 2]).decode()})
w.terminal("h", "error", "invalid_argument", "seq")
w.send({"type": "hang_up", "id": "h"})
w.send({"type": "hang_up", "id": "i"})
nothing_more(w, "chunks and a hang_up followed conversations that ended with errors")
print("errors: max_seconds reached (out_of_range), a chunk out of order while running (seq), pcm not base64 while waiting (pcm); later lines dropped")

for message, code, option in [({"type": "dialogue", "id": "j"}, "invalid_argument", "sample_rate"),
                              ({"type": "dialogue", "id": "k", "sample_rate": rate, "bogus": 1}, "invalid_argument", "bogus"),
                              ({"type": "dialogue", "id": "l", "sample_rate": rate, **OPTIONS, "speed": 1.5}, "unsupported", "speed"),
                              ({"type": "synthesize", "id": "m", "text": "あ"}, "unsupported", "type"),
                              ({"type": "transcribe", "id": "n", "sample_rate": rate}, "unsupported", "type"),
                              ({"type": "enhance", "id": "o", "sample_rate": rate}, "unsupported", "type")]:
    w.request(message)
    w.terminal(message["id"], "error", code, option)
for message, code, option in [({"type": "chunk", "id": "nobody", "seq": 0, "pcm": ""}, "invalid_argument", "id"),
                              ({"type": "hang_up", "id": "nobody"}, "invalid_argument", "id"),
                              ({"type": "hang_up", "id": "nobody", "now": True}, "invalid_argument", "now")]:
    w.send(message)
    w.error_without_id(code, option)
print("refusals: no sample_rate, an unknown member, an option the model does not take, the other tasks' requests; "
      "a chunk and a hang_up with no conversation and a hang_up with another member, each without an id")

# stdin closing ends a running conversation as its hang_up does.
w.request({"type": "dialogue", "id": "p", "sample_rate": rate, **OPTIONS})
w.next_for("p")
chunks(w, "p", user[:frame * 2 * 5], frame)
rest = w.close()
ended = [m for m in rest if m.get("id") == "p"]
if ended[-1]["type"] != "end" or ended[-1]["frames"] != 6:
    raise SystemExit(f"stdin's end gave the conversation {short(ended[-1])}, not its end after 6 frames")
print("stdin closed: the running conversation heard its chunks and ended with end; the worker exited with 0")
