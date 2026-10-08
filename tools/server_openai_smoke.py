# /// script
# requires-python = ">=3.10"
# dependencies = ["openai[realtime]==3.26.0"]
# ///
"""Drives `speech serve` with OpenAI's own Python client, openai-python, pinned by this script's lock, to show that a
program written for OpenAI's API works by changing the address alone.

It lists the models; transcribes the first dump's audio with chunking_strategy "auto" and with a server_vad object, as
the client sends them, against the same requests sent by hand; and opens a Realtime transcription session with
client.realtime.connect(model=...), whose events the client parses into its own types: session.created, a
session.update of a transcription session answered by session.updated, PCM appended in pieces and committed, answered
by input_audio_buffer.committed, the delta and the completed of the text that the same samples give as a 24000 Hz WAV
file, an error event for turn_detection server_vad, and input_audio_buffer.cleared.

usage: uv run --script tools/server_openai_smoke.py <speech> <recognition.gguf> <detection.gguf> <dump folder>
"""

import array
import ast
import base64
import io
import json
import os
import struct
import sys
import uuid

import openai
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_client import Server  # noqa: E402

if len(sys.argv) != 5:
    raise SystemExit("usage:" + __doc__.split("usage:")[1].rstrip())
speech, model, vad, dump = sys.argv[1:]
server = Server(speech, [model, vad])
client = OpenAI(base_url=f"http://127.0.0.1:{server.port}/v1", api_key="unused")

listed = [m.id for m in client.models.list()]
name = listed[0]
print(f"openai {openai.__version__}: models.list() gives {listed}")


def read_npy(path):
    with open(path, "rb") as f:
        data = f.read()
    header_length = struct.unpack("<H", data[8:10])[0]
    assert ast.literal_eval(data[10:10 + header_length].decode())["descr"] == "<f4"
    values = array.array("f")
    values.frombytes(data[10 + header_length:])
    return values


def wav(pcm, rate):
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)


def by_hand(fields, content):
    boundary = uuid.uuid4().hex
    body = b"".join(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode() for k, v in fields)
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\n\r\n").encode() + content
    body += f"\r\n--{boundary}--\r\n".encode()
    status, _, answer = server.call("POST", "/v1/audio/transcriptions", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert status == 200, answer
    return json.loads(answer)["text"]


samples = read_npy(os.path.join(dump, "audio.npy"))
rate = json.loads(server.call("GET", f"/v1/models/{name}")[2])["speech"]["sample_rate"]
pcm = array.array("h", [max(-32768, min(32767, round(x * 32768))) for x in samples]).tobytes()
for strategy, fields in (("auto", [("chunking_strategy", "auto")]),
                         ({"type": "server_vad", "threshold": 0.6, "silence_duration_ms": 700},
                          [("chunking_strategy[type]", "server_vad"), ("chunking_strategy[threshold]", "0.6"),
                           ("chunking_strategy[silence_duration_ms]", "700")])):
    got = client.audio.transcriptions.create(model=name, file=("x.wav", io.BytesIO(wav(pcm, rate)), "audio/wav"), chunking_strategy=strategy)
    assert got.text == by_hand(fields, wav(pcm, rate)) and got.text, (got.text, strategy)
print(f"audio.transcriptions.create() with chunking_strategy auto and a server_vad object: the text of the same forms sent by hand, "
      f"{got.text[:30]!r}")

# The dump's samples at 24000 Hz, the one rate of audio/pcm, by linear interpolation.
n = len(samples) * 24000 // rate
at24 = [samples[int(i * rate / 24000)] + (samples[min(int(i * rate / 24000) + 1, len(samples) - 1)] - samples[int(i * rate / 24000)])
        * (i * rate / 24000 - int(i * rate / 24000)) for i in range(n)]
pcm24 = array.array("h", [max(-32768, min(32767, round(x * 32768))) for x in at24]).tobytes()
want = by_hand([], wav(pcm24, 24000))

with client.realtime.connect(model=name) as connection:
    created = connection.recv()
    assert type(created).__name__ == "SessionCreatedEvent" and created.session.type == "transcription", created
    connection.session.update(session={"type": "transcription", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
                                                                                     "transcription": {"model": name}, "turn_detection": None}}})
    updated = connection.recv()
    assert type(updated).__name__ == "SessionUpdatedEvent" and updated.session.audio.input.transcription.model == name, updated
    for start in range(0, len(pcm24), 9600):
        connection.input_audio_buffer.append(audio=base64.b64encode(pcm24[start:start + 9600]).decode())
    connection.input_audio_buffer.commit()
    events = [connection.recv()]
    while events[-1].type != "conversation.item.input_audio_transcription.completed":
        events.append(connection.recv())
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["InputAudioBufferCommittedEvent", "ConversationItemInputAudioTranscriptionDeltaEvent",
                     "ConversationItemInputAudioTranscriptionCompletedEvent"], kinds
    committed, delta, completed = events
    assert delta.item_id == completed.item_id == committed.item_id and delta.delta == completed.transcript == want, (delta, completed, want)
    assert completed.usage.type == "duration" and completed.usage.seconds == len(pcm24) / 2 / 24000, completed.usage
    connection.session.update(session={"type": "transcription", "audio": {"input": {"turn_detection": {"type": "server_vad"}}}})
    refused = connection.recv()
    assert type(refused).__name__ == "RealtimeErrorEvent" and refused.error.param == "session.audio.input.turn_detection", refused
    connection.input_audio_buffer.clear()
    assert type(connection.recv()).__name__ == "InputAudioBufferClearedEvent"
print(f"realtime.connect(): {', '.join(kinds)} and RealtimeErrorEvent for server_vad, parsed by the client; the transcript of the "
      f"same samples as a 24000 Hz file: {want[:30]!r}")
out = server.stop()
assert out == b"", out[:200]
print("ok")
