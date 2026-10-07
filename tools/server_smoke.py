"""Drives `speech serve` over HTTP the way a client of OpenAI's audio API does, with the standard library alone.

It checks GET /health, GET /v1/models and /v1/models/{id} (OpenAI's model object with the release and the model
information of `speech info --json` with the device, the threads and the voices of --add-voice), a model it does not
serve, the endpoint of the other task, and CORS. For a synthesis model: a wav with its X-Sample-Rate, X-Speech-Seed and
X-Speech-Stop, a pcm stream and an SSE stream of the same seed giving the same samples, the SSE stream's
speech.audio.done with the seed, the samples and the stop reason, a drawn seed that repeats the audio, and each error
with its status, type, param and code: the server's own (a member it does not have, a value of the wrong type, no
input, a format it does not give, a body that is not JSON) and the library's mapped by category (a value it does not
take, out of range or not taken, an empty or too long input, a voice left out), on a wav and on a stream. For a
recognition model: json and text with X-Speech-Stop for each dump of reference/fastconformer/dump.py or
reference/qwen3-asr/dump.py, each other request a Qwen3-ASR dump holds with its language and its prompt as form fields,
verbose_json with its segments and a segment granularity where the model takes timestamps and its refusal where it does
not, WAV at three times the model's rate, and the errors of a file that is not WAV or cannot be read, a language, a
prompt, a member and a granularity it does not take, and audio the library cannot take.

The audio goes as 16-bit samples, which move the near-silent input of reference/qwen3-asr/dump.py far enough to change
the text a forced language makes of it; leave that dump out.

usage: python3 tools/server_smoke.py <speech> <model.gguf> [dump folder...] [-- serve options...]
"""

import array
import ast
import base64
import http.client
import json
import os
import socket
import struct
import subprocess
import sys
import time
import uuid

from worker_client import check_model_information, dump_requests

args = sys.argv[1:]
options = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
speech, model, *dumps = args
added = [o.split("=", 1)[0] for i, o in enumerate(options) if i > 0 and options[i - 1] == "--add-voice"]
ORIGIN = "http://localhost:5173"

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
server = subprocess.Popen([speech, "serve", model, "--port", str(port), "--cors-origin", ORIGIN, *options],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def call(method, path, body=None, headers=None, stream=False):
    """The status, the headers and the body, or the open response of a stream."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    if stream:
        return r
    data = r.read()
    c.close()
    return r.status, {k.lower(): v for k, v in r.getheaders()}, data


def post_json(path, member):
    return call("POST", path, json.dumps(member, ensure_ascii=False).encode(), {"Content-Type": "application/json"})


def expect_error(got, status, code, param, what):
    s, _, body = got
    error = json.loads(body).get("error", {}) if body else {}
    kind = "server_error" if status >= 500 else "invalid_request_error"
    if s != status or error.get("code") != code or error.get("param") != param or error.get("type") != kind or not error.get("message"):
        raise SystemExit(f"{what}: expected {status} {code} ({param}), got {s} {body[:300]!r}")
    print(f"{what}: {status} {kind} {code} ({param}) as expected: {error['message'][:90]}")


t0 = time.perf_counter()
while True:
    if server.poll() is not None:
        raise SystemExit(f"the server exited with {server.returncode}")
    try:
        status, _, body = call("GET", "/health")
        break
    except OSError:
        time.sleep(0.2)
assert status == 200 and json.loads(body) == {"status": "ok"}, body
status, _, body = call("GET", "/v1/models")
listed = json.loads(body)
assert status == 200 and listed["object"] == "list" and len(listed["data"]) == 1, body
entry = listed["data"][0]
info = entry["speech"]
assert entry["object"] == "model" and entry["owned_by"] == "speech.cpp" and entry["id"] == info["name"], entry
assert isinstance(entry["created"], int) and isinstance(entry["version"], str), entry
check_model_information(speech, model, info, added)
status, _, body = call("GET", "/v1/models/" + info["name"])
assert status == 200 and json.loads(body) == entry, body
print(f"up in {time.perf_counter() - t0:.2f} s: {info['name']} on {info['device']}, speech.cpp {entry['version']}; /health, /v1/models and "
      f"/v1/models/{{id}} as expected, the model information equal to speech info --json")
expect_error(call("GET", "/v1/models/tts-1"), 404, "model_not_found", "model", "GET /v1/models/tts-1")
status, headers, _ = call("OPTIONS", "/v1/audio/speech", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST"})
assert status == 204 and headers.get("access-control-allow-origin") == ORIGIN, (status, headers)
status, headers, _ = call("GET", "/health", headers={"Origin": ORIGIN})
assert "X-Speech-Stop" in headers.get("access-control-expose-headers", ""), headers
status, headers, _ = call("GET", "/health", headers={"Origin": "http://elsewhere.test"})
assert "access-control-allow-origin" not in headers, headers
print("CORS: the preflight answered and the headers exposed for the origin given, none for another")
rate = info["sample_rate"]


def wav_data(body):
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE" and body[36:40] == b"data", body[:44]
    assert struct.unpack("<I", body[24:28])[0] == rate and struct.unpack("<I", body[40:44])[0] == len(body) - 44
    return body[44:]


def sse_events(r):
    events = []
    for line in r.read().decode().split("\n\n"):
        if line:
            assert line.startswith("data: "), line[:100]
            events.append(json.loads(line[6:]))
    return events


if info["task"] == "synthesis":
    form = b'--x\r\nContent-Disposition: form-data; name="language"\r\n\r\nja\r\n--x--\r\n'
    expect_error(call("POST", "/v1/audio/transcriptions", form, {"Content-Type": "multipart/form-data; boundary=x"}), 404, None, None,
                 "POST /v1/audio/transcriptions to a synthesis model")
    voice = info["voices"][0]["name"]
    TEXT = "明日の東京は晴れで、最高気温は二十四度の予報です。"
    asked = {"model": info["name"], "input": TEXT, "voice": voice, "seed": 11}
    status, headers, body = post_json("/v1/audio/speech", asked)
    assert status == 200 and headers["content-type"] == "audio/wav", (status, body[:200])
    assert headers["x-sample-rate"] == str(rate) and headers["x-speech-seed"] == "11" and headers["x-speech-stop"] == "complete", headers
    pcm = wav_data(body)
    print(f"wav: {len(pcm) // 2 / rate:.2f} s, X-Sample-Rate {rate}, X-Speech-Seed 11, X-Speech-Stop complete")
    r = call("POST", "/v1/audio/speech", json.dumps(asked | {"response_format": "pcm"}).encode(), {"Content-Type": "application/json"}, stream=True)
    assert r.status == 200 and r.getheader("Content-Type") == "audio/pcm" and r.getheader("X-Speech-Seed") == "11", r.getheaders()
    assert r.read() == pcm, "the pcm stream differs from the wav"
    r = call("POST", "/v1/audio/speech", json.dumps(asked | {"response_format": "pcm", "stream_format": "sse"}).encode(),
             {"Content-Type": "application/json"}, stream=True)
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream", r.getheaders()
    events = sse_events(r)
    done = events[-1]
    assert all(e["type"] == "speech.audio.delta" for e in events[:-1]), events[0]
    assert b"".join(base64.b64decode(e["audio"]) for e in events[:-1]) == pcm, "the SSE stream differs from the wav"
    assert done == {"type": "speech.audio.done", "seed": 11, "samples": len(pcm) // 2, "stop": "complete"}, done
    print(f"pcm and sse: the wav's samples; done {done}")
    status, headers, body = post_json("/v1/audio/speech", {"input": TEXT, "voice": voice, "seed": None})
    drawn = int(headers["x-speech-seed"])
    assert 0 <= drawn < 2 ** 53 and wav_data(post_json("/v1/audio/speech", {"input": TEXT, "voice": voice, "seed": drawn})[2]) == wav_data(body)
    print(f"a request without a seed drew {drawn}, which repeats the audio")
    if info["architecture"] == "qwen3-tts":
        status, headers, body = post_json("/v1/audio/speech", {"input": TEXT, "voice": voice, "max_seconds": 0.5})
        assert status == 200 and headers["x-speech-stop"] == "max_seconds" and len(wav_data(body)) <= rate, headers
        print("max_seconds 0.5: X-Speech-Stop max_seconds")
        expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": voice, "speed": 1.5}), 400, "unsupported_parameter", "speed",
                     "speed on Qwen3-TTS")
    else:
        expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": voice, "speed": 5}), 400, "unsupported_value", "speed",
                     "speed 5")
        expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": voice, "seconds": 2, "duration_scale": 2}), 400, "invalid_value",
                     "seconds", "seconds with a duration scale")
    expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": voice, "instructions": "x"}), 400, "unknown_parameter",
                 "instructions", "an unknown member")
    expect_error(post_json("/v1/audio/speech", {"input": 5, "voice": voice}), 400, "invalid_type", "input", "input of another type")
    expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": voice, "steps": 1.5}), 400, "invalid_type", "steps",
                 "steps with a fraction")
    expect_error(post_json("/v1/audio/speech", {"voice": voice}), 400, "missing_required_parameter", "input", "no input")
    expect_error(post_json("/v1/audio/speech", {"model": "tts-1", "input": "あ。", "voice": voice}), 404, "model_not_found", "model",
                 "another model")
    expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": voice, "response_format": "mp3"}), 400, "unsupported_value",
                 "response_format", "mp3")
    expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": voice, "stream_format": "sse"}), 400, "unsupported_value",
                 "response_format", "sse with wav")
    expect_error(call("POST", "/v1/audio/speech", b"{not json", {"Content-Type": "application/json"}), 400, None, None, "a body that is not JSON")
    expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": "no-such-voice"}), 400, "unsupported_value", "voice", "an unknown voice")
    expect_error(post_json("/v1/audio/speech", {"input": "", "voice": voice}), 400, "invalid_value", "input", "an empty input")
    expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": voice, "timestamps": True}), 400, "unsupported_parameter",
                 "timestamps", "timestamps")
    for format in ("wav", "pcm"):
        expect_error(post_json("/v1/audio/speech", {"input": "あ。", "response_format": format}), 400, "invalid_value", "voice",
                     f"no voice, {format}")
        expect_error(post_json("/v1/audio/speech", {"input": TEXT * 2000, "voice": voice, "response_format": format}), 400,
                     "unsupported_value", "input", f"an input too long, {format}")
else:
    expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": "x"}), 404, None, None, "POST /v1/audio/speech to a recognition model")

    def read_npy(path):
        with open(path, "rb") as f:
            data = f.read()
        header_length = struct.unpack("<H", data[8:10])[0]
        header = ast.literal_eval(data[10:10 + header_length].decode())
        values = array.array("f")
        values.frombytes(data[10 + header_length:])
        assert header["descr"] == "<f4"
        return values

    def wav_file(samples, at_rate):
        pcm = array.array("h", [max(-32768, min(32767, round(x * 32768))) for x in samples]).tobytes()
        return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, at_rate, at_rate * 2, 2, 16)
                + b"data" + struct.pack("<I", len(pcm)) + pcm)

    def transcribe(fields, files):
        boundary = uuid.uuid4().hex
        body = b""
        for name, value in fields:
            body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        for name, filename, content in files:
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                     f"Content-Type: audio/wav\r\n\r\n").encode() + content + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        return call("POST", "/v1/audio/transcriptions", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})

    takes = {o["name"] for o in info["options"]}
    first = None
    for d in dumps:
        name = os.path.basename(os.path.normpath(d))
        samples = read_npy(os.path.join(d, "audio.npy"))
        wav = wav_file(samples, rate)
        requests = dump_requests(speech, model, d)
        [(_, _, want, _)] = [r for r in requests if r[0] == "auto"]
        if want and not first:
            first = (wav, want, samples)
        status, headers, body = transcribe([("model", info["name"])], [("file", name + ".wav", wav)])
        assert status == 200 and json.loads(body) == {"text": want} and headers["x-speech-stop"] == "complete", (name, body)
        status, headers, body = transcribe([("response_format", "text")], [("file", name + ".wav", wav)])
        assert status == 200 and headers["content-type"].startswith("text/plain") and body.decode() == want, (name, body)
        for request, members, want_request, _ in requests:
            if request != "auto":
                status, _, body = transcribe(list(members.items()), [("file", name + ".wav", wav)])
                assert status == 200 and json.loads(body) == {"text": want_request}, (name, request, body)
        verbose_form = [("response_format", "verbose_json"), ("timestamp_granularities[]", "segment")]
        if "timestamps" not in takes:
            expect_error(transcribe(verbose_form, [("file", name + ".wav", wav)]), 400, "unsupported_parameter", "timestamps",
                         f"{name}: verbose_json to a model without timestamps")
            print(f"{name}: json and text give the dump's text and stop, and each of its {len(requests)} requests its text")
            continue
        status, _, body = transcribe(verbose_form, [("file", name + ".wav", wav)])
        verbose = json.loads(body)
        assert status == 200 and verbose["task"] == "transcribe" and verbose["text"] == want, body
        assert abs(verbose["duration"] - len(samples) / rate) < 1e-9, verbose["duration"]
        segments = verbose["segments"]
        assert [s["id"] for s in segments] == list(range(len(segments))) and "".join(s["text"] for s in segments) == want, segments
        assert all(set(s) == {"id", "start", "end", "text"} and 0 <= s["start"] <= s["end"] for s in segments), segments
        print(f"{name}: json, text and verbose_json give the dump's text and stop, {len(segments)} segments joining into it")
    wav, want, samples = first
    status, _, body = transcribe([], [("file", "x.wav", wav_file([x for x in samples for _ in range(3)], rate * 3))])
    assert status == 200, body
    print(f"WAV at {rate * 3} Hz: {'the dump' if json.loads(body)['text'] == want else 'another'}'s text")
    expect_error(transcribe([], [("file", "x.mp3", b"ID3" + bytes(100))]), 400, "unsupported_value", "file", "a file that is not WAV")
    expect_error(transcribe([], [("file", "x.wav", b"RIFF\x00\x00\x00\x00WAVEjunk")]), 400, "invalid_value", "file", "a WAV that cannot be read")
    expect_error(transcribe([], [("file", "x.wav", wav_file(samples[:16], 44101))]), 400, "invalid_value", "file",
                 "a rate the library cannot resample from")
    # FastConformer takes no less than two of its mel frames; Qwen3-ASR pads short audio with zeros.
    if info["architecture"] == "fastconformer":
        expect_error(transcribe([], [("file", "x.wav", wav_file(samples[:1], rate))]), 400, "unsupported_value", "file", "audio too short")
    expect_error(transcribe([("language", "zz")], [("file", "x.wav", wav)]), 400, "unsupported_value", "language", "an unknown language")
    if "prompt" in takes:
        expect_error(transcribe([("prompt", " word" * 70000)], [("file", "x.wav", wav)]), 400, "unsupported_value", "prompt",
                     "a prompt longer than the model takes")
    else:
        expect_error(transcribe([("prompt", "x")], [("file", "x.wav", wav)]), 400, "unsupported_parameter", "prompt", "a prompt not taken")
    expect_error(transcribe([("temperature", "0")], [("file", "x.wav", wav)]), 400, "unknown_parameter", "temperature", "an unknown member")
    expect_error(transcribe([("response_format", "srt")], [("file", "x.wav", wav)]), 400, "unsupported_value", "response_format", "srt")
    expect_error(transcribe([("response_format", "verbose_json"), ("timestamp_granularities[]", "word")], [("file", "x.wav", wav)]), 400,
                 "unsupported_value", "timestamp_granularities[]", "word timestamps")
    expect_error(transcribe([("timestamp_granularities[]", "segment")], [("file", "x.wav", wav)]), 400, "invalid_value",
                 "timestamp_granularities[]", "a granularity without verbose_json")
    expect_error(transcribe([("model", "whisper-1")], [("file", "x.wav", wav)]), 404, "model_not_found", "model", "another model")
    expect_error(transcribe([("language", info["languages"][0])], []), 400, "missing_required_parameter", "file", "no file")

server.terminate()
server.wait(timeout=30)
out = server.stdout.read()
if out:
    raise SystemExit(f"the server wrote to stdout: {out[:200]!r}")
print("ok")
