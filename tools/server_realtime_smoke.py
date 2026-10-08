"""Speaks OpenAI's Realtime transcription to `speech serve` over the WebSocket at /v1/realtime, the way a client of OpenAI's
API does, with the standard library alone: the session.created and session.updated of a transcription session, two
commits of 24000 Hz PCM appended in pieces, each committed in order and transcribed into the deltas and the completed of
the text, the languages and the duration of the same samples sent as a WAV file of that rate, a cleared buffer, a commit
with transcription null left untranscribed, an error event for each member and value it does not take with the client's
event_id, a session.update that changes only the members it gives, the bound on the audio a session holds before it is
transcribed, two commits and an HTTP request run in the order they arrived, a client gone while its commit is
transcribed, and the refusals of the upgrade: another model, another query member, another Origin and another Host.

usage: python3 tools/server_realtime_smoke.py <speech> <recognition.gguf> <dump folder>... [-- serve options...]
"""

import array
import base64
import json
import os
import struct
import sys
import time

from server_client import Server, expect_error, read_npy

args = sys.argv[1:]
options = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
if len(args) < 3:
    raise SystemExit("usage:" + __doc__.split("usage:")[1].rstrip())
speech, model, *dumps = args
ORIGIN = "http://localhost:5173"

server = Server(speech, [model, "--cors-origin", ORIGIN, *options])
transcribe = server.transcribe
info = json.loads(server.call("GET", "/v1/models")[2])["data"][0]["speech"]
assert info["task"] == "recognition", info["task"]
rate = info["sample_rate"]

# OpenAI's Realtime transcription at /v1/realtime: 16-bit PCM at 24000 Hz, which the server resamples to the model's
# rate as it does a WAV file of that rate, so a commit's text is that of the same samples sent as a file.
def pcm24(samples):
    n = len(samples) * 24000 // rate
    out = []
    for i in range(n):
        at = i * rate / 24000
        j = int(at)
        x = samples[j] + (samples[min(j + 1, len(samples) - 1)] - samples[j]) * (at - j)
        out.append(max(-32768, min(32767, round(x * 32768))))
    return array.array("h", out).tobytes()

def wav24(pcm):
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)

def appended(ws, pcm, piece=9600):
    for at in range(0, len(pcm), piece):
        ws.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[at:at + piece]).decode()})

def realtime_error(ws, code, param, what, event_id=None):
    e = ws.event()
    assert e["type"] == "error" and e["error"]["code"] == code and e["error"]["param"] == param and e["error"]["event_id"] == event_id, (what, e)
    assert e["error"]["type"] == "invalid_request_error" and e["error"]["message"] and e["event_id"].startswith("event_"), e
    print(f"realtime {what}: error {code} ({param}) as expected: {e['error']['message'][:80]}")

def session(transcription, turn_detection=None, **members):
    return {"type": "session.update", **members, "session": {"type": "transcription", "audio": {"input": {
        "format": {"type": "audio/pcm", "rate": 24000}, "transcription": transcription, "turn_detection": turn_detection}}}}

name = info["name"]
pcms = [pcm24(read_npy(os.path.join(d, "audio.npy"))) for d in (dumps * 2)[:2]]
wants = []
for pcm in pcms:
    status, _, body = transcribe([("response_format", "verbose_json")], [("file", "x.wav", wav24(pcm))])
    verbose = json.loads(body)
    wants.append((verbose["text"], verbose.get("language", "").split(",") if verbose.get("language") else []))
ws = server.websocket(f"/v1/realtime?model={name}")
created = ws.event()
assert created["type"] == "session.created" and created["event_id"].startswith("event_"), created
configured = created["session"]
assert configured["type"] == "transcription" and configured["object"] == "realtime.transcription_session" and configured["id"], configured
assert configured["audio"]["input"] == {"format": {"type": "audio/pcm", "rate": 24000}, "transcription": {"model": name, "language": None},
                                        "noise_reduction": None, "turn_detection": None}, configured
ws.send(session({"model": name}))
updated = ws.event()
assert updated["type"] == "session.updated" and updated["session"] == configured, updated
# Two commits in a row, each of its appends: both are committed at once, the second after the first, and each is
# transcribed in turn into one delta and its completed with the duration.
items = []
for pcm in pcms:
    appended(ws, pcm)
    ws.send({"type": "input_audio_buffer.commit"})
events = []
while sum(e["type"].endswith(".completed") for e in events) < len(pcms):
    events.append(ws.event())
committed = [e for e in events if e["type"] == "input_audio_buffer.committed"]
items = [e["item_id"] for e in committed]
assert [e["previous_item_id"] for e in committed] == [None, items[0]] and len(set(items)) == 2, committed
for item, pcm, (text, languages) in zip(items, pcms, wants):
    mine = [e for e in events if e.get("item_id") == item and e["type"] != "input_audio_buffer.committed"]
    deltas, done = mine[:-1], mine[-1]
    assert "".join(d["delta"] for d in deltas) == text and all(d["type"].endswith(".delta") and d["content_index"] == 0 for d in deltas), mine
    assert done["type"] == "conversation.item.input_audio_transcription.completed" and done["transcript"] == text, done
    assert done["usage"] == {"type": "duration", "seconds": len(pcm) / 2 / 24000} and done["content_index"] == 0, done
    assert [x["code"] for x in done.get("languages", [])] == languages and done["stop"] == "complete", done
assert [e["type"] for e in events].index("conversation.item.input_audio_transcription.completed") > [e["type"] for e in events].index(
    "input_audio_buffer.committed"), events
print(f"realtime: session.created and session.updated as configured, two commits committed in order and transcribed as their "
      f"samples sent as a 24000 Hz file: {wants[0][0][:40]!r}")
# A buffer cleared leaves nothing to commit; a session that transcribes nothing commits without transcribing.
appended(ws, pcms[0][:4800])
ws.send({"type": "input_audio_buffer.clear"})
assert ws.event()["type"] == "input_audio_buffer.cleared"
ws.send({"type": "input_audio_buffer.commit", "event_id": "empty"})
realtime_error(ws, "input_audio_buffer_commit_empty", None, "commit of a cleared buffer", "empty")
ws.send(session(None))
assert ws.event()["session"]["audio"]["input"]["transcription"] is None
appended(ws, pcms[0])
ws.send({"type": "input_audio_buffer.commit"})
assert ws.event()["type"] == "input_audio_buffer.committed"
ws.send({"type": "input_audio_buffer.clear"})
assert ws.event()["type"] == "input_audio_buffer.cleared", "a session with transcription null transcribed a commit"
ws.send(session({"model": name, "language": info["languages"][0]}))
assert ws.event()["session"]["audio"]["input"]["transcription"]["language"] == info["languages"][0]
print("realtime: a cleared buffer, and a commit with transcription null, which is committed and not transcribed")
# A session.update changes only the members it gives, and a member given as null takes its value away.
ws.send({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"transcription": {"model": name}}}}})
kept = ws.event()["session"]["audio"]["input"]["transcription"]
assert kept == {"model": name, "language": info["languages"][0]}, kept
ws.send({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"transcription": {"language": None}}}}})
kept = ws.event()["session"]["audio"]["input"]["transcription"]
assert kept == {"model": name, "language": None}, kept
print("realtime: a session.update of the model alone keeps the language, and a language of null takes it away")
for event, code, param, what in (
        (session({"model": name}) | {"session": {"type": "realtime"}}, "unsupported_value", "session.type", "a conversation session"),
        (session({"model": name}, {"type": "server_vad", "silence_duration_ms": 500}), "unsupported_value",
         "session.audio.input.turn_detection", "turn_detection server_vad"),
        (session({"model": name}, {"type": "semantic_vad"}), "unsupported_value", "session.audio.input.turn_detection", "semantic_vad"),
        (session({"model": "gpt-4o-transcribe"}), "model_not_found", "session.audio.input.transcription.model", "another model"),
        (session({"model": name, "language": "zz"}), "unsupported_value", "session.audio.input.transcription.language", "a language"),
        (session({"model": name, "languages": ["ja", "en"]}), "unsupported_value", "session.audio.input.transcription.languages",
         "two languages"),
        (session({"model": name, "keywords": ["渋谷"]}), "unsupported_parameter", "session.audio.input.transcription.keywords", "keywords"),
        (session({"model": name, "delay": "low"}), "unsupported_parameter", "session.audio.input.transcription.delay", "delay"),
        ({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"noise_reduction": {"type": "near_field"}}}}},
         "unsupported_parameter", "session.audio.input.noise_reduction", "noise reduction"),
        ({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"format": {"type": "audio/pcmu"}}}}},
         "unsupported_value", "session.audio.input.format.type", "G.711"),
        ({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}}},
         "invalid_value", "session.audio.input.format.rate", "16000 Hz"),
        ({"type": "session.update", "session": {"type": "transcription", "include": ["item.input_audio_transcription.logprobs"]}},
         "unsupported_value", "session.include", "logprobs"),
        ({"type": "session.update", "session": {"type": "transcription", "model": name}}, "unknown_parameter", "session.model",
         "a member of a conversation session"),
        ({"type": "session.update"}, "missing_required_parameter", "session", "no session"),
        ({"type": "input_audio_buffer.append", "audio": "not base64!"}, "invalid_value", "audio", "audio that is not base64"),
        ({"type": "input_audio_buffer.append", "audio": base64.b64encode(b"abc").decode()}, "invalid_value", "audio", "an odd byte"),
        ({"type": "input_audio_buffer.append"}, "missing_required_parameter", "audio", "no audio"),
        ({"type": "response.create"}, "unsupported_value", "type", "response.create"),
        ({"type": "input_audio_buffer.clear", "item_id": "x"}, "unknown_parameter", "item_id", "a member clear does not have"),
        ({"event_id": "x"}, "missing_required_parameter", "type", "no type")):
    ws.send(event | {"event_id": event.get("event_id", "e-" + what)})
    realtime_error(ws, code, param, what, event.get("event_id", "e-" + what))
ws.send("{not json")
realtime_error(ws, "invalid_json", None, "a message that is not JSON")
ws.send_binary(b"\0\0")
realtime_error(ws, "invalid_value", None, "a binary message")
ws.send({"type": "input_audio_buffer.clear"})
assert ws.event()["type"] == "input_audio_buffer.cleared", "the session did not go on after its errors"
# The audio a session holds before it is transcribed, buffered or committed, is bounded: an append past the bound is an
# error event and keeps nothing of itself, the buffer before it stays whole, and a commit that waits or runs counts until
# its transcription ends.
piece = b"\0" * (1 << 20)


def fill(limit_pieces):
    """Appends pieces of a mebibyte until an append is refused, each followed by an event that is always refused; the
    number of pieces taken, or None when none was refused."""
    for k in range(limit_pieces):
        ws.send({"type": "input_audio_buffer.append", "event_id": f"a{k}", "audio": base64.b64encode(piece).decode()})
        ws.send({"type": "no.such.event", "event_id": f"p{k}"})
        e = ws.event()
        if e["error"]["event_id"] == f"a{k}":
            assert e["error"]["param"] == "audio" and e["error"]["code"] == "input_audio_buffer_full", e
            assert ws.event()["error"]["event_id"] == f"p{k}"
            return k
        assert e["error"]["event_id"] == f"p{k}", e
    return None


taken = fill(512)
assert taken, "512 MiB of audio appended without a commit were all taken"
ws.send({"type": "input_audio_buffer.commit"})
assert ws.event()["type"] == "input_audio_buffer.committed"
done = ws.event()
while not done["type"].endswith((".completed", ".failed")):
    done = ws.event()
assert done["type"].endswith(".completed") and done["usage"]["seconds"] == taken * (1 << 20) / 2 / 24000, done
ws.send({"type": "input_audio_buffer.clear"})
assert ws.event()["type"] == "input_audio_buffer.cleared"
appended(ws, piece * (taken * 3 // 5), 1 << 20)
ws.send({"type": "input_audio_buffer.commit"})
assert ws.event()["type"] == "input_audio_buffer.committed"
assert fill(taken) == taken - taken * 3 // 5, "a commit that waits or runs did not count toward the bound"
while not ws.event()["type"].endswith(".completed"):
    pass
ws.send({"type": "input_audio_buffer.clear"})
assert ws.event()["type"] == "input_audio_buffer.cleared"
print(f"realtime: {taken} MiB of audio held, the next append refused and the buffer committed whole; a commit that waits "
      f"or runs counts until it is transcribed")
# A commit takes its turn on the model when it is accepted, so an HTTP request that arrives after two commits runs
# after both, as the server runs every request in the order it arrives.
appended(ws, pcms[0] * 4)
ws.send({"type": "input_audio_buffer.commit"})
appended(ws, pcms[1])
ws.send({"type": "input_audio_buffer.commit"})
first, second = ws.event(), ws.event()
assert first["type"] == second["type"] == "input_audio_buffer.committed", (first, second)
assert transcribe([], [("file", "x.wav", wav24(pcms[1]))])[0] == 200
done = []
while len(done) < 2:
    e = ws.event()
    if e["type"].endswith((".completed", ".failed")):
        done.append(e)
assert [(e["type"], e["item_id"]) for e in done] == [("conversation.item.input_audio_transcription.completed", item) for item in
                                                    (first["item_id"], second["item_id"])], done
http = f"transcription: {len(pcms[1]) / 2 / 24000:.2f} s of audio at 24000 Hz"
for _ in range(50):
    lines = list(server.lines)
    order = [k for k, line in enumerate(lines) if line.startswith((f"realtime {first['item_id']}:", f"realtime {second['item_id']}:", http))]
    if len(order) >= 3:
        break
    time.sleep(0.1)
ran = [lines[k].split(":")[0] for k in order][-3:]
assert ran == [f"realtime {first['item_id']}", f"realtime {second['item_id']}", "transcription"], ran
print("realtime: two commits and then an HTTP transcription run in that order")
# A client that goes away while its commit is transcribed stops the transcription, gives up the turn of the commit
# that waits, and the model serves the next request.
for _ in range(2):
    appended(ws, pcms[0])
    ws.send({"type": "input_audio_buffer.commit"})
    assert ws.event()["type"] == "input_audio_buffer.committed"
ws.sock.close()
status, _, body = transcribe([], [("file", "x.wav", wav24(pcms[0]))])
assert status == 200 and json.loads(body)["text"] == wants[0][0], body
print("realtime: a session that goes on after its errors, and a client gone while its commit was transcribed")
expect_error(server.websocket("/v1/realtime?model=gpt-realtime"), 404, "model_not_found", "model", "/v1/realtime for another model")
expect_error(server.websocket("/v1/realtime?intent=transcription"), 400, "unknown_parameter", "intent", "/v1/realtime with intent")
expect_error(server.websocket("/v1/realtime", {"Origin": "http://elsewhere.test"}), 403, "origin_not_allowed", None,
             "/v1/realtime from another origin")
expect_error(server.websocket("/v1/realtime", {"Host": "elsewhere.test"}), 403, "host_not_allowed", None, "/v1/realtime for another Host")
ws = server.websocket("/v1/realtime", {"Origin": ORIGIN})
assert ws.event()["type"] == "session.created"
ws.close()
print("realtime: /v1/realtime from --cors-origin's origin opens")

out = server.stop()
if out:
    raise SystemExit(f"the server wrote to stdout: {out[:200]!r}")
print("ok")
