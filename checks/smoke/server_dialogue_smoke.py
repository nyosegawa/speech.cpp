"""Holds conversations with `speech serve`'s dialogue model over its WebSocket at /v1/dialogue the way a client does, with
the standard library alone.

Given a dump of reference/haru/dump.py, it starts the server with the model and the dump's voice reference added by
--add-voice, and checks: that /v1/models and the page's /speech/models give the dialogue model; that a conversation
answers its start with started, the model's rate and frame, and its first frame before any audio is sent; that the
frames it sends while it hears the dump's user audio in binary messages of 16-bit PCM, a frame each, are sample for
sample the WAVE that `speech talk` writes for the same audio and options, and their text what it writes on stdout; that
messages of another length give the same frames, and audio at 16 kHz what speech talk gives the same samples at 16 kHz;
that hang_up gets end with the frames and the samples sent, the audio short of a frame dropped, and then the close;
that a second conversation while one runs is refused with 409, and that a client gone in the middle of one frees the
model for the next; each refusal and error with its code and param; and a voice the page adds to the dialogue model,
in which a conversation speaks as in the same voice added by --add-voice.

usage: python3 checks/smoke/server_dialogue_smoke.py <speech> <dialogue.gguf> <dump folder> [-- serve options...]
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
import wave

from server_client import Server, expect_error
from worker_client import pcm16, read_npy

args = sys.argv[1:]
flags = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
if len(args) != 3:
    raise SystemExit(__doc__)
speech, model, dump = args
workdir = tempfile.mkdtemp(prefix="dialogue-server-smoke-")
with open(os.path.join(dump, "meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
rate, frame = meta["sample_rate"], meta["frame_samples"]
# 30 whole frames of the user's audio and part of another, which the conversation drops.
WHOLE = 30
user = pcm16(read_npy(os.path.join(dump, "user_audio.npy")))[:(WHOLE * frame + frame // 3) * 2]
OPTIONS = {"language": "ja", "name": "ハル", "persona": "落ち着いた話し方をする大学生。", "backchannel": "mid", "voice": "ref", "seed": 7}


def write_wav(path, pcm, at):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(at)
        w.writeframes(pcm)


reference = os.path.join(workdir, "reference.wav")
write_wav(reference, pcm16(read_npy(os.path.join(dump, "reference_audio.npy"))), rate)
server = Server(speech, [model, "--add-voice", "ref=" + reference, *flags])


class Conversation:
    """A conversation on a WebSocket of its own, whose messages a thread reads as they come, so that neither end waits on
    the other while the audio goes one way and the frames the other."""

    def __init__(self, options=OPTIONS, sample_rate=None):
        self.ws = server.websocket("/v1/dialogue")
        assert not isinstance(self.ws, tuple), f"the server refused the conversation: {self.ws}"
        self.messages = []
        self.closed = threading.Event()
        self.ws.send({"type": "start", **({"sample_rate": sample_rate} if sample_rate else {}), **options})
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            while (m := self.ws.receive()) is not None:
                self.messages.append(m if isinstance(m, bytes) else json.loads(m))
        except (ConnectionError, OSError):
            pass
        self.closed.set()

    def wait_for(self, wanted, what):
        for _ in range(6000):
            found = [m for m in self.messages if wanted(m)]
            if found:
                return found[0]
            if self.closed.is_set():
                raise SystemExit(f"the server closed the conversation before {what}: {self.summary()}")
            self.closed.wait(0.1)
        raise SystemExit(f"no {what} in ten minutes: {self.summary()}")

    def summary(self):
        return [f"<{len(m)} bytes>" if isinstance(m, bytes) else m for m in self.messages][:8]

    def send_audio(self, pcm, piece):
        for at in range(0, len(pcm), piece * 2):
            self.ws.send_binary(pcm[at:at + piece * 2])

    def hang_up(self):
        """Hangs up; returns the frames' PCM and text, and end, checking that the server then closes."""
        self.ws.send({"type": "hang_up"})
        end = self.wait_for(lambda m: isinstance(m, dict) and m["type"] in ("end", "error"), "end")
        if end["type"] != "end":
            raise SystemExit(f"a hang-up got {end}")
        if not self.closed.wait(30):
            raise SystemExit("the server did not close the conversation after its end")
        return self.frames() + (end,)

    def frames(self):
        pcm, text = b"", ""
        for m in self.messages:
            if isinstance(m, bytes):
                if len(m) != frame * 2:
                    raise SystemExit(f"a binary message holds {len(m) // 2} samples, not a frame of {frame}")
                pcm += m
            elif m["type"] == "text":
                text += m["text"]
        return pcm, text

    def error(self, code, param, what):
        """Waits for an error event with `code` and `param`, then for the close, and prints it."""
        e = self.wait_for(lambda m: isinstance(m, dict) and m["type"] == "error", "an error")["error"]
        if e.get("code") != code or e.get("param") != param or not e.get("message"):
            raise SystemExit(f"{what}: expected {code} ({param}), got {e}")
        if not self.closed.wait(30):
            raise SystemExit(f"{what}: the server did not close the conversation after its error")
        print(f"{what}: {code} ({param}) as expected: {e['message'][:90]}")


def talk(pcm, piece, options=OPTIONS, sample_rate=None):
    """A conversation that hears `pcm` in binary messages of `piece` samples and hangs up; checks its first frame comes first."""
    c = Conversation(options, sample_rate)
    started = c.wait_for(lambda m: isinstance(m, dict), "started")
    if started != {"type": "started", "sample_rate": rate, "frame_samples": frame}:
        raise SystemExit(f"a conversation began with {started}")
    c.wait_for(lambda m: isinstance(m, bytes), "the first frame")
    c.send_audio(pcm, piece)
    return c.hang_up()


def cli_talk(pcm, at=rate, options=OPTIONS):
    """The 16-bit samples and the text of `speech talk` hearing the 16-bit audio `pcm` at `at` with `options` as flags."""
    heard, out = os.path.join(workdir, "user.wav"), os.path.join(workdir, "reply.wav")
    write_wav(heard, pcm, at)
    device = flags[flags.index("--device"):flags.index("--device") + 2] if "--device" in flags else []
    command = [speech, "talk", model, *device, "--add-voice", "ref=" + reference, "-o", out, heard]
    for name, value in options.items():
        command += ["--" + name.replace("_", "-"), str(value)]
    r = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    with wave.open(out, "rb") as f:
        assert f.getframerate() == rate, f.getframerate()
        return f.readframes(f.getnframes()), r.stdout.decode("utf-8")


status, _, body = server.call("GET", "/v1/models")
listed = [m["speech"]["task"] for m in json.loads(body)["data"]]
status_page, _, page_body = server.call("GET", "/speech/models")
held = json.loads(page_body)["dialogue"]["held"]
if status != 200 or listed != ["dialogue"] or status_page != 200 or held["model"]["task"] != "dialogue" or held["model"]["frame_samples"] != frame:
    raise SystemExit(f"/v1/models gave {listed}, /speech/models {page_body[:300]!r}")
print(f"the dialogue model in /v1/models and in /speech/models: {held['model']['name']}, frames of {frame} samples at {rate} Hz")

pcm, text, end = talk(user, rate // 2)
want = WHOLE + 1
if end != {"type": "end", "frames": want, "samples": want * frame} or len(pcm) != want * frame * 2:
    raise SystemExit(f"a conversation of {WHOLE} whole frames ended with {end} after {len(pcm) // 2 // frame} frames")
talked, said = cli_talk(user)
if talked != pcm:
    raise SystemExit("speech talk's WAVE differs from the frames of /v1/dialogue for the same audio")
if said != text + ("\n" if text else ""):
    raise SystemExit(f"speech talk wrote {said!r}, /v1/dialogue's text events {text!r}")
print(f"a conversation of {WHOLE} frames and a third: started and its first frame before any audio, {end['frames']} frames sample for sample "
      f"speech talk's WAVE, their text its stdout ({text.strip()!r}), then end and the close")

again = talk(user, 1234)
if again[:2] != (pcm, text):
    raise SystemExit("messages of 1234 samples gave other frames than messages of half a second")
print("messages of 1234 samples: the same frames and text")

slow = 16000
low, low_text, low_end = talk(user, slow // 10, sample_rate=slow)
low_talked, low_said = cli_talk(user, slow)
if low != low_talked or low_said != low_text + ("\n" if low_text else ""):
    raise SystemExit("audio sent at 16 kHz gave other frames than speech talk on the same samples at 16 kHz")
print(f"audio at 16 kHz: {low_end['frames']} frames sample for sample speech talk's on the same samples at 16 kHz")

# One conversation at a time: a second is refused while the first runs, and a client gone frees the model.
first = Conversation()
first.wait_for(lambda m: isinstance(m, bytes), "the first frame")
expect_error(server.websocket("/v1/dialogue"), 409, "conversation_in_progress", None, "/v1/dialogue while a conversation runs")
first.send_audio(user[:frame * 2 * 3], frame)
first.ws.sock.close()
server.wait_for_line(lambda line: line.startswith("dialogue from") and line.rstrip().endswith("gone"), "a conversation whose client went away")
_, _, after = talk(user[:frame * 2 * 2], frame)
if after["frames"] != 3:
    raise SystemExit(f"the conversation after a client went away ended with {after}")
print("one conversation at a time: a second refused with 409 while one runs; a client gone frees the model for the next")

expect_error(server.websocket("/v1/dialogue?model=x"), 400, "unknown_parameter", "model", "/v1/dialogue with a query")
expect_error(server.websocket("/v1/dialogue", {"Origin": "http://elsewhere.test"}), 403, "origin_not_allowed", None, "/v1/dialogue from another Origin")
expect_error(server.websocket("/v1/dialogue", {"Host": "elsewhere.test"}), 403, "host_not_allowed", None, "/v1/dialogue for another Host")


def refused_start(start, code, param, what):
    ws = server.websocket("/v1/dialogue")
    c = Conversation.__new__(Conversation)
    c.ws, c.messages, c.closed = ws, [], threading.Event()
    if isinstance(start, bytes):
        ws.send_binary(start)
    else:
        ws.send(start)
    threading.Thread(target=c._read, daemon=True).start()
    c.error(code, param, what)


refused_start(b"\x00\x00", "invalid_value", None, "audio before the start")
refused_start({"type": "hang_up"}, "invalid_value", "type", "a hang-up before the start")
refused_start({"type": "start", "bogus": 1}, "unknown_parameter", "bogus", "a member the start does not have")
refused_start({"type": "start", "sample_rate": 1.5}, "invalid_value", "sample_rate", "a sample_rate that is not a whole number")
refused_start({"type": "start", "voice": 3}, "invalid_type", "voice", "an option of another type")
refused_start({"type": "start", "speed": 1.5}, "unsupported_parameter", "speed", "an option the model does not take")
refused_start({"type": "start", "voice": "nobody"}, "unsupported_value", "voice", "a voice the model does not have")

c = Conversation()
c.wait_for(lambda m: isinstance(m, bytes), "the first frame")
c.ws.send_binary(b"\x00\x00\x00")
c.error("invalid_value", None, "audio of an odd number of bytes")
c = Conversation()
c.wait_for(lambda m: isinstance(m, bytes), "the first frame")
c.ws.send({"type": "commit"})
c.error("invalid_value", "type", "a text message other than hang_up")
c = Conversation({**OPTIONS, "max_seconds": 1})
c.wait_for(lambda m: isinstance(m, bytes), "the first frame")
c.send_audio(user, frame)
c.error("unsupported_value", "max_seconds", "audio past max_seconds")

# A voice the page makes from the same recording speaks as the one --add-voice made.
boundary = uuid.uuid4().hex
with open(reference, "rb") as f:
    recording = f.read()
form = b""
for name, value in (("name", "page"), ("task", "dialogue")):
    form += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
form += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"recording.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
form += recording + f"\r\n--{boundary}--\r\n".encode()
status, _, body = server.call("POST", "/speech/voices", form, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
voices = next(o["choices"] for o in json.loads(body)["held"]["model"]["options"] if o["name"] == "voice") if status == 200 else []
if status != 200 or "page" not in voices:
    raise SystemExit(f"POST /speech/voices with task dialogue answered {status} {body[:300]!r}")
paged = talk(user, rate // 2, {**OPTIONS, "voice": "page"})
if paged[:2] != (pcm, text):
    raise SystemExit("a conversation in the voice the page made gave other frames than in the same voice added by --add-voice")
bad = form.replace(b"\r\n\r\ndialogue\r\n", b"\r\n\r\nsinging\r\n")
expect_error(server.call("POST", "/speech/voices", bad, {"Content-Type": f"multipart/form-data; boundary={boundary}"}), 400, "invalid_value", "task",
             "POST /speech/voices with another task")
print(f"a voice the page made from the reference ({', '.join(voices)}): the same frames as the voice of --add-voice")

print("ok")
