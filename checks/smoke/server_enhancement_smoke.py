"""Checks speech serve with an enhancement model the way a client uses it, over HTTP and the Realtime WebSocket.

POST /v1/audio/enhancements of a dump's audio of reference/semamba/dump.py, as a 16-bit WAV file: the wav answer is, sample
for sample, the WAVE that `speech enhance` writes for the same file, with X-Sample-Rate and X-Speech-Stop; the pcm answer
streams the same samples; a form field bwe resamples, its rate in X-Sample-Rate; and the refusals: no file, a field it does
not have, a response_format it does not give, another model, audio shorter than the model's STFT and a rate outside its
range. GET /v1/models lists the recognition, detection and enhancement models in that order. A transcription with
noise_reduction near_field or far_field gives the text that `speech asr --enhance` gives the same file, and with
chunking_strategy auto that of `speech asr --vad --enhance`; another value is refused. A Realtime session with
session.audio.input.noise_reduction near_field echoes it in session.updated, and the completion of a commit of the audio at
24 kHz is the text `speech asr --enhance` gives the same samples as a file; a type it does not have is refused, and null
takes it away. A server without an enhancement model answers /v1/audio/enhancements with a 404, and refuses
noise_reduction in a transcription and in a session, naming the field.

usage: python3 checks/smoke/server_enhancement_smoke.py <speech> <enhancer.gguf> <recognizer> <vad> <dump folder>
       [-- options of speech serve and of the commands, such as --device cpu]
"""

import array
import base64
import json
import os
import struct
import subprocess
import sys
import tempfile
import uuid

from server_client import Server, expect_error
from worker_client import read_npy

args = sys.argv[1:]
options = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
if len(args) != 5:
    raise SystemExit("usage:" + __doc__.split("usage:")[1].rstrip())
speech, enhancer, recognizer, vad, dump = args
work = tempfile.mkdtemp(prefix="speech-enhancement-smoke-")
with open(os.path.join(dump, "meta.json")) as f:
    rate = json.load(f)["rate"]
samples = read_npy(os.path.join(dump, "audio.npy"))


def pcm16(values):
    return array.array("h", [max(-32768, min(32767, round(x * 32768))) for x in values]).tobytes()


def pcm24(values):
    """The samples at 24000 Hz, by linear interpolation, as 16-bit PCM."""
    n = len(values) * 24000 // rate
    out = []
    for i in range(n):
        at = i * rate / 24000
        j = int(at)
        x = values[j] + (values[min(j + 1, len(values) - 1)] - values[j]) * (at - j)
        out.append(max(-32768, min(32767, round(x * 32768))))
    return array.array("h", out).tobytes()


def wav(pcm, at):
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, at, 2 * at, 2, 16) + b"data" +
            struct.pack("<I", len(pcm)) + pcm)


def saved(name, content):
    path = os.path.join(work, name)
    with open(path, "wb") as f:
        f.write(content)
    return path


def run(*command):
    done = subprocess.run([speech, *command], capture_output=True)
    if done.returncode != 0:
        raise SystemExit(f"speech {command[0]} failed: {done.stderr.decode(errors='replace')[-600:]}")
    return done.stdout


def data_of(wave_bytes):
    """The rate and the samples of a 16-bit mono WAV file of a canonical 44-byte header."""
    assert wave_bytes[:4] == b"RIFF" and wave_bytes[36:40] == b"data", wave_bytes[:44]
    return struct.unpack("<I", wave_bytes[24:28])[0], wave_bytes[44:]


def form(fields, files):
    boundary = uuid.uuid4().hex
    body = b""
    for name, value in fields:
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    for name, filename, content in files:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                 f"Content-Type: audio/wav\r\n\r\n").encode() + content + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def enhance(server, fields, content, stream=False):
    body, headers = form(fields, [("file", "x.wav", content)] if content is not None else [])
    return server.call("POST", "/v1/audio/enhancements", body, headers, stream=stream)


source = wav(pcm16(samples), rate)
source_path = saved("source.wav", source)
run("enhance", enhancer, *options, "-o", os.path.join(work, "out.wav"), source_path)
want_rate, want = data_of(open(os.path.join(work, "out.wav"), "rb").read())
bwe = 8000 if rate != 8000 else 16000
run("enhance", enhancer, *options, "--bwe", str(bwe), "-o", os.path.join(work, "bwe.wav"), source_path)
bwe_rate, bwe_want = data_of(open(os.path.join(work, "bwe.wav"), "rb").read())
assert want_rate == rate and bwe_rate == bwe, (want_rate, bwe_rate)
heard = run("asr", recognizer, *options, "--enhance", enhancer, source_path).decode().strip()
heard_by_regions = run("asr", recognizer, *options, "--vad", vad, "--enhance", enhancer, source_path).decode().strip()
pcm_24k = pcm24(samples)
heard_24k = run("asr", recognizer, *options, "--enhance", enhancer, saved("source24.wav", wav(pcm_24k, 24000))).decode().strip()

server = Server(speech, [recognizer, vad, enhancer, *options])
status, _, body = server.call("GET", "/v1/models")
tasks = [m["speech"]["task"] for m in json.loads(body)["data"]]
assert status == 200 and tasks == ["recognition", "detection", "enhancement"], tasks
print(f"/v1/models: {tasks}")

status, headers, body = enhance(server, [], source)
got_rate, got = data_of(body)
assert status == 200 and headers["content-type"] == "audio/wav" and headers["x-sample-rate"] == str(rate) and got_rate == rate, (status, headers)
assert headers["x-speech-stop"] == "complete" and got == want, "the wav answer differs from speech enhance"
print(f"/v1/audio/enhancements wav: {len(got) // 2} samples at {rate} Hz, speech enhance's sample for sample")
response = enhance(server, [("response_format", "pcm")], source, stream=True)
streamed = response.read()
assert response.status == 200 and response.getheader("Content-Type") == "audio/pcm" and response.getheader("X-Sample-Rate") == str(rate), response.status
assert streamed == want, "the pcm stream differs from speech enhance"
print("/v1/audio/enhancements pcm: streamed, the same samples")
status, headers, body = enhance(server, [("bwe", str(bwe))], source)
assert status == 200 and headers["x-sample-rate"] == str(bwe) and data_of(body)[1] == bwe_want, (status, headers)
print(f"/v1/audio/enhancements bwe {bwe}: speech enhance --bwe's")

expect_error(enhance(server, [("response_format", "wav")], None), 400, "missing_required_parameter", "file", "no file")
expect_error(enhance(server, [("speed_of_light", "1")], source), 400, "unknown_parameter", "speed_of_light", "a field it does not have")
expect_error(enhance(server, [("response_format", "mp3")], source), 400, "unsupported_value", "response_format", "mp3")
expect_error(enhance(server, [("model", "another")], source), 404, "model_not_found", "model", "another model")
expect_error(enhance(server, [], wav(pcm16(samples[:100]), rate)), 400, "unsupported_value", "file", "audio shorter than the STFT")
expect_error(enhance(server, [], wav(pcm16(samples), 96000)), 400, "unsupported_value", "file", "a rate above the model's")

for reduction in ("near_field", "far_field"):
    status, _, body = server.transcribe([("noise_reduction", reduction)], [("file", "x.wav", source)])
    assert status == 200 and json.loads(body)["text"] == heard, (status, body[:300], heard)
print(f"transcription noise_reduction: {heard!r}, speech asr --enhance's")
status, _, body = server.transcribe([("noise_reduction", "near_field"), ("chunking_strategy", "auto")], [("file", "x.wav", source)])
assert status == 200 and json.loads(body)["text"] == heard_by_regions, (status, body[:300], heard_by_regions)
print(f"transcription noise_reduction with chunking_strategy auto: {heard_by_regions!r}, speech asr --vad --enhance's")
expect_error(server.transcribe([("noise_reduction", "loud")], [("file", "x.wav", source)]), 400, "invalid_value", "noise_reduction",
             "transcription noise_reduction of another value")


def session_update(ws, reduction):
    ws.send({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"noise_reduction": reduction, "turn_detection": None}}}})
    return ws.event()


ws = server.websocket("/v1/realtime")
assert ws.event()["type"] == "session.created"
updated = session_update(ws, {"type": "near_field"})
assert updated["type"] == "session.updated" and updated["session"]["audio"]["input"]["noise_reduction"] == {"type": "near_field"}, updated
for at in range(0, len(pcm_24k), 9600):
    ws.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm_24k[at:at + 9600]).decode()})
ws.send({"type": "input_audio_buffer.commit"})
while True:
    e = ws.event()
    if e["type"].endswith((".completed", ".failed")):
        break
assert e["type"] == "conversation.item.input_audio_transcription.completed" and e["transcript"] == heard_24k, (e, heard_24k)
print(f"realtime noise_reduction near_field: {heard_24k!r}, speech asr --enhance's of the same samples")
refused = session_update(ws, {"type": "loud"})
assert refused["type"] == "error" and refused["error"]["param"] == "session.audio.input.noise_reduction.type", refused
updated = session_update(ws, None)
assert updated["type"] == "session.updated" and updated["session"]["audio"]["input"]["noise_reduction"] is None, updated
ws.close()
print("realtime noise_reduction: a type it does not have refused, and null takes it away")
server.stop()

plain = Server(speech, [recognizer, *options])
expect_error(enhance(plain, [], source), 404, None, None, "/v1/audio/enhancements without an enhancement model")
expect_error(plain.transcribe([("noise_reduction", "near_field")], [("file", "x.wav", source)]), 400, "unsupported_value", "noise_reduction",
             "transcription noise_reduction without an enhancement model")
ws = plain.websocket("/v1/realtime")
assert ws.event()["type"] == "session.created"
refused = session_update(ws, {"type": "near_field"})
assert refused["type"] == "error" and refused["error"]["param"] == "session.audio.input.noise_reduction", refused
ws.close()
print("realtime noise_reduction without an enhancement model: refused, naming session.audio.input.noise_reduction")
plain.stop()
print("server enhancement smoke passed")
