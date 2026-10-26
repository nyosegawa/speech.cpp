"""Checks the speakers that `speech serve` gives with a diarization model against `speech diarize` of the same samples
with the same options, which must agree to the millisecond: on /v1/realtime, with server_vad, each completed of a
session with session.audio.input.diarization carries the speakers within its utterance, from speech_started's
audio_start_ms to speech_stopped's audio_end_ms, labeled A, B and on from the first in each session alone while two
sessions run at once, one of them with the inputs one after another; a session.update with other diarization options
starts a diarization of the audio from there, whose speakers take the labels after those of the one before, one with
diarization null ends it, and one that starts it again goes on after both, while the same options in another order keep
the diarization and its labels; a session without one gets no speakers, and its transcription does not wait for a
commit of another session whose speakers wait for audio that has not come. On
/v1/audio/transcriptions, diarized_json gives each region of chunking_strategy the speakers of the whole file's
diarization within it, and its speaker the one who speaks the most there. It also checks the refusals: diarized_json
without chunking_strategy, a diarization member the model does not take or a value out of its range, and diarization on
a server without a diarization model.

The inputs are folders of reference/sortformer/dump.py, whose audio.npy is 16 kHz; the script brings them to the 24000 Hz
16-bit PCM that /v1/realtime takes and writes them as WAVE files for `speech diarize` and the HTTP requests. Options
after -- go to every `speech serve` and `speech diarize`, such as --device cpu.

usage: python3 checks/smoke/server_diarization_smoke.py <speech> <recognition.gguf> <detection.gguf> <diarization.gguf> <input folder>... [-- options...]
"""

import array
import base64
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time

from server_client import Server, WebSocket, expect_error
from worker_client import main_speaker, read_npy, realtime_speakers, speaker_label, speakers_cut

args = sys.argv[1:]
options = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
if len(args) < 5:
    raise SystemExit("usage:" + __doc__.split("usage:")[1].rstrip())
speech, recognizer, vad, diarizer, *inputs = args
RATE = 24000


def pcm24(samples):
    """16 kHz samples at 24000 Hz, by linear interpolation, as 16-bit PCM."""
    out = array.array("h")
    for k in range(len(samples) * 3 // 2):
        at = k * 2 / 3
        i = int(at)
        x = samples[i] if i + 1 >= len(samples) else samples[i] + (samples[i + 1] - samples[i]) * (at - i)
        out.append(max(-32768, min(32767, round(x * 32767))))
    return out.tobytes()


def wav(pcm):
    return b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, RATE, 2 * RATE, 2, 16) + b"data" + \
        struct.pack("<I", len(pcm)) + pcm


work = tempfile.mkdtemp(prefix="speech-diarization-smoke-")
# Two seconds of silence after each input end its last utterance.
pieces = [pcm24(read_npy(os.path.join(d, "audio.npy")).tolist()) + bytes(2 * 2 * RATE) for d in inputs]
joined = b"".join(pieces)
files = []
for name, pcm in [(f"input{k}", p) for k, p in enumerate(pieces)] + [("joined", joined), ("two", pieces[0] + pieces[-1])]:
    files.append(os.path.join(work, name + ".wav"))
    with open(files[-1], "wb") as f:
        f.write(wav(pcm))
two_file = files.pop()
joined_file = files.pop()


def diarize(path, *flags):
    """The segments of `speech diarize`, (start, end, speaker) each."""
    done = subprocess.run([speech, "diarize", diarizer, "--format", "json", *flags, *options, path], capture_output=True)
    if done.returncode != 0:
        raise SystemExit(f"speech diarize failed: {done.stderr.decode(errors='replace')}")
    return [(s["start"], s["end"], s["speaker"]) for s in json.loads(done.stdout)["segments"]]


class Socket(WebSocket):
    """A WebSocket whose frames one thread sends while another reads, as it answers pings."""

    def __init__(self, sock):
        super().__init__(sock)
        self.lock = threading.Lock()

    def _frame(self, opcode, payload):
        with self.lock:
            super()._frame(opcode, payload)


class Session:
    """A Realtime transcription session with server_vad whose server events a thread keeps as they arrive."""

    def __init__(self, server, diarization, turns={"type": "server_vad"}):
        self.ws = Socket.open("127.0.0.1", server.port, "/v1/realtime", {})
        assert self.ws.event()["type"] == "session.created"
        self.events, self.changed = [], threading.Condition()
        self.turns = turns
        self.update(diarization)
        threading.Thread(target=self.read, daemon=True).start()

    def update(self, diarization):
        """Sets the diarization, as in session.update, before the reader runs or while it does."""
        member = {} if diarization is False else {"diarization": diarization}
        self.ws.send({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"turn_detection": self.turns, **member}}}})

    def read(self):
        while True:
            try:
                e = self.ws.event()
            except (OSError, ConnectionError, ValueError, AssertionError):
                return
            with self.changed:
                self.events.append(e)
                self.changed.notify_all()

    def append(self, pcm):
        for at in range(0, len(pcm), 4800):
            self.ws.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[at:at + 4800]).decode()})

    def completed(self):
        with self.changed:
            return sum(e["type"] == "conversation.item.input_audio_transcription.completed" for e in self.events)

    def wait_for(self, count, what, seconds=600, kind="conversation.item.input_audio_transcription.completed"):
        """Waits for the count-th event of `kind`."""
        ended = time.time() + seconds
        with self.changed:
            while sum(e["type"] == kind for e in self.events) < count:
                if not self.changed.wait(ended - time.time()):
                    raise SystemExit(f"{what}: no {kind} within {seconds} s")

    def settle(self, mark):
        """Waits for the events of the audio appended so far: those before the answer to an event that is always
        refused, then each commit's completed."""
        self.ws.send({"type": "no.such.event", "event_id": mark})
        ended = time.time() + 600
        with self.changed:
            while not any(e["type"] == "error" and e["error"].get("event_id") == mark for e in self.events) or \
                    sum(e["type"] == "input_audio_buffer.committed" for e in self.events) > \
                    sum(e["type"].endswith((".completed", ".failed")) for e in self.events):
                if not self.changed.wait(ended - time.time()):
                    raise SystemExit(f"session {mark}: no completed for every commit within 600 s")

    def utterances(self, since=0):
        """Each completed after the first `since` events, with its utterance's start and end in milliseconds."""
        with self.changed:
            events = list(self.events)
        started = {e["item_id"]: e["audio_start_ms"] for e in events if e["type"] == "input_audio_buffer.speech_started"}
        stopped = {e["item_id"]: e["audio_end_ms"] for e in events if e["type"] == "input_audio_buffer.speech_stopped"}
        # The errors that answer settle()'s marks are the only ones expected.
        failed = [e for e in events[since:] if e["type"].endswith(".failed") or (e["type"] == "error" and not e["error"].get("event_id"))]
        assert not failed, failed[:2]
        return [(started[e["item_id"]], stopped[e["item_id"]], e) for e in events[since:]
                if e["type"] == "conversation.item.input_audio_transcription.completed"]

    def updated(self):
        with self.changed:
            return [e for e in self.events if e["type"] == "session.updated"]

    def close(self):
        self.ws.close()


def check_speakers(utterances, segments, what, offset=0.0, first=0):
    assert utterances, what
    for start, end, e in utterances:
        want = realtime_speakers(segments, start, end, offset, first)
        got = e.get("speakers")
        assert got == want, (what, start, end, got, want)


def numbers(segments):
    return max((who for _, _, who in segments), default=-1) + 1


streamed = [diarize(path, "--streaming-mode", "low_latency") for path in files]
server = Server(speech, [recognizer, vad, diarizer, *options])

# Two sessions at once: the first hears the inputs one after another, the second the last input alone, and both label
# their speakers from A; a third without diarization gets none.
first, second, third = Session(server, {}), Session(server, {}), Session(server, False)
threads = [threading.Thread(target=s.append, args=(pcm,)) for s, pcm in ((first, joined), (second, pieces[-1]), (third, pieces[0]))]
for t in threads:
    t.start()
for t in threads:
    t.join()
for k, s in enumerate((first, second, third)):
    s.settle(f"settle{k}")
assert first.updated()[0]["session"]["audio"]["input"]["diarization"] == {}, first.updated()[0]
assert third.updated()[0]["session"]["audio"]["input"]["diarization"] is None, third.updated()[0]
check_speakers(second.utterances(), streamed[-1], "the second session")
check_speakers(first.utterances(), diarize(joined_file, "--streaming-mode", "low_latency"), "the first session")
assert all("speakers" not in e for _, _, e in third.utterances()), third.utterances()[:1]
print(f"realtime: {len(first.utterances())} and {len(second.utterances())} utterances of two sessions at once carry the speakers of "
      f"speech diarize in low_latency, each session's labeled alone; {len(third.utterances())} of a session without diarization carry none")
for s in (first, second, third):
    s.close()

# Other options start a diarization of the audio from the update on, whose speakers take the labels after the first's;
# null ends it, and a diarization started again goes on after both.
session = Session(server, {})
session.append(pieces[0])
session.settle("before")
heard = len(pieces[0]) // 2 / RATE
mark = len(session.events)
session.update({"onset": 0.6})
session.append(pieces[-1])
session.settle("after")
later = diarize(files[-1], "--streaming-mode", "low_latency", "--onset", "0.6")
check_speakers(session.utterances(mark), later, "the utterances after other options", heard, numbers(streamed[0]))
mark = len(session.events)
session.update(None)
session.append(pieces[0])
session.settle("off")
assert session.updated()[-1]["session"]["audio"]["input"]["diarization"] is None
assert all("speakers" not in e for _, _, e in session.utterances(mark)), "an utterance after diarization null"
heard += (len(pieces[-1]) + len(pieces[0])) // 2 / RATE
mark = len(session.events)
session.update({})
session.append(pieces[0])
session.settle("again")
check_speakers(session.utterances(mark), streamed[0], "the utterances of a diarization started again", heard, numbers(streamed[0]) + numbers(later))
print("realtime: a session.update with other options labels the speakers of the audio after it after those before, null ends the "
      "diarization, and one started again goes on after both")

# The same options in another order keep the diarization under way and its labels.
session.close()
session = Session(server, {"onset": 0.5, "offset": 0.5})
session.append(pieces[0])
session.settle("in order")
session.update({"offset": 0.5, "onset": 0.5})
session.append(pieces[-1])
session.settle("reordered")
check_speakers(session.utterances(), diarize(two_file, "--streaming-mode", "low_latency", "--onset", "0.5", "--offset", "0.5"),
               "the utterances after the same options in another order")
print("realtime: a session.update with the same options in another order keeps the diarization and its labels")

# Commits whose speakers wait for audio that has not come hold up no other session: the second commit, which took its
# place among the recognizer's requests before the other session's, is recognized while the first waits, the other
# session's transcription completes, and the first session's two once audio follows.
waiting, other = Session(server, {}, None), Session(server, False, None)
# The second commit is 100 ms long, too short for the diarization to settle the first's audio.
for at, end in ((0, 4 * RATE), (4 * RATE, 4 * RATE + RATE // 10)):
    waiting.append(pieces[0][2 * at:2 * end])
    waiting.ws.send({"type": "input_audio_buffer.commit"})
# The other session commits once both commits have taken their places, after them.
waiting.wait_for(2, "the session that waits for its speakers", 60, "input_audio_buffer.committed")
other.append(pieces[0][:2 * 4 * RATE])
other.ws.send({"type": "input_audio_buffer.commit"})
other.wait_for(1, "the session without diarization", 120)
assert waiting.completed() == 0, "a commit's speakers came before the audio that settles them"
waiting.append(bytes(2 * 2 * RATE))
waiting.wait_for(2, "the session that waits for its speakers")
done = [e for e in waiting.events if e["type"] == "conversation.item.input_audio_transcription.completed"]
assert all("speakers" in e for e in done) and done[0]["speakers"], done
print("realtime: commits whose speakers wait for audio hold up no other session's transcription, and complete in order once audio follows")
for s in (waiting, other):
    s.close()

# The session's refusals, each an error with the member at fault.
for member, code, param in (({"threshold": 0.5}, "unsupported_parameter", "session.audio.input.diarization.threshold"),
                            ({"onset": 2}, "unsupported_value", "session.audio.input.diarization.onset"),
                            ({"no_such_option": 1}, "unknown_parameter", "session.audio.input.diarization.no_such_option"),
                            (7, "invalid_type", "session.audio.input.diarization")):
    mark = len(session.events)
    session.update(member)
    session.settle(f"refused {param}")
    error = next(e for e in session.events[mark:] if e["type"] == "error")
    assert error["error"]["code"] == code and error["error"]["param"] == param, (member, error)
print("realtime: a diarization option the model does not take, a value out of its range, an unknown member and a number "
      "are refused with the member at fault")
session.close()

# diarized_json: each region of chunking_strategy with the speakers of the whole file's diarization, offline.
offline = diarize(files[0])
status, _, body = server.transcribe([("response_format", "diarized_json"), ("chunking_strategy", "auto")], [("file", "a.wav", wav(pieces[0]))])
assert status == 200, body
got = json.loads(body)
assert got["task"] == "transcribe" and got["usage"] == {"type": "duration", "seconds": got["duration"]} and got["segments"], got
for k, s in enumerate(got["segments"]):
    want = [{"speaker": speaker_label(who), "start": round(b, 3), "end": round(e, 3)} for b, e, who, _, _ in speakers_cut(offline, s["start"], s["end"])]
    main = main_speaker(want) or ""
    assert s["type"] == "transcript.text.segment" and s["id"] == f"seg_{k}" and s["speakers"] == want and s["speaker"] == main, (s, want)
assert got["text"] == json.loads(server.transcribe([("chunking_strategy", "auto")], [("file", "a.wav", wav(pieces[0]))])[2])["text"]
print(f"diarized_json: {len(got['segments'])} regions, each with the speakers of speech diarize within it and its main speaker")
expect_error(server.transcribe([("response_format", "diarized_json")], [("file", "a.wav", wav(pieces[0]))]), 400, "missing_required_parameter",
             "chunking_strategy", "diarized_json without chunking_strategy")
server.stop()

# A server without a diarization model refuses both, saying to give it one.
alone = Server(speech, [recognizer, vad, *options])
expect_error(alone.transcribe([("response_format", "diarized_json"), ("chunking_strategy", "auto")], [("file", "a.wav", wav(pieces[0]))]), 400,
             "unsupported_value", "response_format", "diarized_json on a server without a diarization model")
ws = alone.websocket("/v1/realtime")
assert ws.event()["type"] == "session.created"
ws.send({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"diarization": {}}}}})
refused = ws.event()
assert refused["type"] == "error" and refused["error"]["param"] == "session.audio.input.diarization", refused
ws.close()
alone.stop()
print(f"realtime diarization on a server without a diarization model: error {refused['error']['code']}: {refused['error']['message'][:80]}")
for path in files + [joined_file, two_file]:
    os.remove(path)
os.rmdir(work)
