"""Drives `speech serve` over HTTP the way a client of OpenAI's audio API does, with the standard library alone.

It checks GET /health, GET /v1/models and /v1/models/{id} (OpenAI's model object with the release and the model
information of `speech info --json` with the device, the threads and the voices of --add-voice), a model it does not
serve, the endpoint of the other task, and CORS. For a synthesis model: a wav with its X-Sample-Rate, X-Speech-Seed and
X-Speech-Stop, a pcm stream and an SSE stream of the same seed giving the same samples, the SSE stream's
speech.audio.done with the seed, the samples and the stop reason, a drawn seed that repeats the audio, and each error
with its status, type, param and code: the server's own (a member it does not have, a value of the wrong type, no
input, a format it does not give, a body that is not JSON) and the library's mapped by category (a value it does not
take, out of range or not taken, an empty or too long input, a voice left out), on a wav and on a stream, Qwen3-TTS's
sampling options, and OpenAI's instructions followed by a model that takes them and refused by one that does not. For a
recognition model: json and text with X-Speech-Stop for each dump of reference/fastconformer/dump.py or
reference/qwen3-asr/dump.py, each other request a Qwen3-ASR dump holds with its language and its prompt as form fields
and each decoding other than the default that a FastConformer dump holds as the form field decoding, json without any
member but the text, verbose_json of each request with the language qwen-asr parsed by its tag or without one
(FastConformer, audio without speech), and where the model takes timestamps with its segments, the same with a segment
granularity, and that granularity's refusal where it does not; WAV at three times the model's rate, and the errors of a
file that is not WAV or cannot be read, a language, a prompt, a decoding, a member and a granularity it does not take,
and audio the library cannot take.

With --vad a detection model, which the server holds beside the model: /v1/models lists it after the model, and for a
recognition model chunking_strategy "auto" on the dumps' audio joined with silences gives the text, the languages and
the segments of `speech asr --vad` with its defaults, the members of server_vad give those of the flags they map to, a
file of silence gives an empty text, and each form of chunking_strategy it does not take is refused. Without --vad,
chunking_strategy is refused with a pointer to the detection model. A detection model alone is listed, and both
endpoints answer that the server holds no model of their task.

The audio goes as 16-bit samples, which move the near-silent input of reference/qwen3-asr/dump.py far enough to change
the text a forced language makes of it; leave that dump out.

For a recognition model it also speaks OpenAI's Realtime transcription over the WebSocket at /v1/realtime: the
session.created and session.updated of a transcription session, two commits of 24000 Hz PCM appended in pieces, each
committed in order and transcribed into the deltas and the completed of the text, the languages and the duration of the
same samples sent as a WAV file of that rate, a cleared buffer, a commit with transcription null left untranscribed, an
error event for each member and value it does not take with the client's event_id, a client gone while its commit is
transcribed, and the refusals of the upgrade: another model, another query member, another Origin and another Host.

usage: python3 tools/server_smoke.py <speech> <model.gguf> [dump folder...] [--vad DETECTION.gguf] [-- serve options...]
"""

import array
import ast
import base64
import json
import os
import struct
import subprocess
import sys
import tempfile
import uuid

from server_client import Server, expect_error
from worker_client import check_model_information, dump_requests

args = sys.argv[1:]
options = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
vad = None
if "--vad" in args:
    at = args.index("--vad")
    vad = args[at + 1]
    del args[at:at + 2]
speech, model, *dumps = args
added = [o.split("=", 1)[0] for i, o in enumerate(options) if i > 0 and options[i - 1] == "--add-voice"]
ORIGIN = "http://localhost:5173"

server = Server(speech, [model, *([vad] if vad else []), "--cors-origin", ORIGIN, *options])
call, post_json = server.call, server.post_json
status, _, body = call("GET", "/health")
assert status == 200 and json.loads(body) == {"status": "ok"}, body
status, _, body = call("GET", "/v1/models")
listed = json.loads(body)
assert status == 200 and listed["object"] == "list" and len(listed["data"]) == (2 if vad else 1), body
entry = listed["data"][0]
if vad:
    detector = listed["data"][1]
    check_model_information(speech, vad, detector["speech"])
    assert detector["id"] == detector["speech"]["name"] and detector["speech"]["task"] == "detection", detector
    assert json.loads(call("GET", "/v1/models/" + detector["id"])[2]) == detector
    print(f"/v1/models lists {detector['id']} after {entry['id']}, with the model information of speech info --json")
info = entry["speech"]
assert entry["object"] == "model" and entry["owned_by"] == "speech.cpp" and entry["id"] == info["name"], entry
assert isinstance(entry["created"], int) and isinstance(entry["version"], str), entry
check_model_information(speech, model, info, added)
status, _, body = call("GET", "/v1/models/" + info["name"])
assert status == 200 and json.loads(body) == entry, body
print(f"up in {server.started:.2f} s: {info['name']} on {info['device']}, speech.cpp {entry['version']}; /health, /v1/models and "
      f"/v1/models/{{id}} as expected, the model information equal to speech info --json")
expect_error(call("GET", "/v1/models/tts-1"), 404, "model_not_found", "model", "GET /v1/models/tts-1")
status, headers, _ = call("OPTIONS", "/v1/audio/speech", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST"})
assert status == 204 and headers.get("access-control-allow-origin") == ORIGIN, (status, headers)
status, headers, _ = call("GET", "/health", headers={"Origin": ORIGIN})
assert "X-Speech-Stop" in headers.get("access-control-expose-headers", ""), headers
expect_error(call("GET", "/health", headers={"Origin": "http://elsewhere.test"}), 403, "origin_not_allowed", None, "GET /health from another origin")
expect_error(call("OPTIONS", "/v1/audio/speech", headers={"Origin": "http://elsewhere.test", "Access-Control-Request-Method": "POST"}), 403,
             "origin_not_allowed", None, "a preflight from another origin")
print("CORS: the preflight answered and the headers exposed for the origin given, and another origin refused")
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


if info["task"] == "detection":
    form = b'--x\r\nContent-Disposition: form-data; name="language"\r\n\r\nja\r\n--x--\r\n'
    expect_error(call("POST", "/v1/audio/transcriptions", form, {"Content-Type": "multipart/form-data; boundary=x"}), 404, None, None,
                 "POST /v1/audio/transcriptions to a server with a detection model alone")
    expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": "x"}), 404, None, None, "POST /v1/audio/speech to it")
elif info["task"] == "synthesis":
    form = b'--x\r\nContent-Disposition: form-data; name="language"\r\n\r\nja\r\n--x--\r\n'
    expect_error(call("POST", "/v1/audio/transcriptions", form, {"Content-Type": "multipart/form-data; boundary=x"}), 404, None, None,
                 "POST /v1/audio/transcriptions to a synthesis model")
    # An added voice where there is one: the voice an Irodori-TTS file has of its own, none, speaks without a reference.
    voice = added[0] if added else info["voices"][0]["name"]
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
        greedy = {"input": "あ。", "voice": voice, "do_sample": False, "code_predictor_do_sample": False, "max_seconds": 2}
        assert wav_data(post_json("/v1/audio/speech", greedy | {"seed": 1})[2]) == wav_data(post_json("/v1/audio/speech", greedy | {"seed": 2})[2])
        print("do_sample and code_predictor_do_sample false: the same audio for the seeds 1 and 2")
        expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": voice, "do_sample": False, "temperature": 0.5}), 400,
                     "invalid_value", "temperature", "a temperature that does not draw")
        expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": voice, "top_p": 2}), 400, "unsupported_value", "top_p", "top_p 2")
    else:
        expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": voice, "speed": 5}), 400, "unsupported_value", "speed",
                     "speed 5")
        expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": voice, "seconds": 2, "duration_scale": 2}), 400, "invalid_value",
                     "seconds", "seconds with a duration scale")
        expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": voice, "temperature": 0.5}), 400, "unsupported_parameter",
                     "temperature", "temperature on Irodori-TTS")
    # OpenAI's instructions is the option of the same name: a model that takes one speaks the same seed otherwise with
    # it, and one that takes none refuses it unless it is empty.
    told = {"input": "あ。", "voice": voice, "seed": 3}
    plain = wav_data(post_json("/v1/audio/speech", told | {"instructions": ""})[2])
    if any(o["name"] == "instructions" for o in info["options"]):
        assert wav_data(post_json("/v1/audio/speech", told | {"instructions": "怒った口調で話してください。"})[2]) != plain
        print("instructions: other audio for the same seed")
    else:
        expect_error(post_json("/v1/audio/speech", told | {"instructions": "怒った口調で"}), 400, "unsupported_parameter", "instructions",
                     "instructions on a model that takes none")
    expect_error(post_json("/v1/audio/speech", {"input": "あ。", "voice": voice, "bogus": "x"}), 400, "unknown_parameter", "bogus",
                 "an unknown member")
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
        # verbose_json carries the language the model heard, the tags joined with commas, and leaves it out where there is
        # none; it carries segments where the model gives times, whose granularity is OpenAI's default.
        for request, members, want_request, want_languages in requests:
            if request != "auto":
                status, _, body = transcribe(list(members.items()), [("file", name + ".wav", wav)])
                assert status == 200 and json.loads(body) == {"text": want_request}, (name, request, body)
            status, _, body = transcribe(list(members.items()) + [("response_format", "verbose_json")], [("file", name + ".wav", wav)])
            verbose = json.loads(body)
            want_members = {"task", "duration", "text"} | ({"language"} if want_languages else set())
            want_members |= {"segments"} if "timestamps" in takes else set()
            assert status == 200 and set(verbose) == want_members and verbose["task"] == "transcribe", (name, request, body)
            assert verbose["text"] == want_request and verbose.get("language") == (",".join(want_languages) or None), (name, request, body)
            assert abs(verbose["duration"] - len(samples) / rate) < 1e-9, verbose["duration"]
            if request == "auto":
                plain_verbose = verbose
        verbose_form = [("response_format", "verbose_json"), ("timestamp_granularities[]", "segment")]
        if "timestamps" not in takes:
            expect_error(transcribe(verbose_form, [("file", name + ".wav", wav)]), 400, "unsupported_parameter", "timestamps",
                         f"{name}: a segment granularity to a model without timestamps")
            print(f"{name}: json and text give the dump's text and stop, and each of its {len(requests)} requests its text, and verbose_json "
                  f"its text and language ({','.join(requests[0][3]) or 'none'}) without segments")
            continue
        status, _, body = transcribe(verbose_form, [("file", name + ".wav", wav)])
        granular = json.loads(body)
        assert status == 200 and granular == plain_verbose, (body, plain_verbose)
        segments = granular["segments"]
        assert [s["id"] for s in segments] == list(range(len(segments))) and "".join(s["text"] for s in segments) == want, segments
        assert all(set(s) == {"id", "start", "end", "text"} and 0 <= s["start"] <= s["end"] for s in segments), segments
        others = f"; each of its {len(requests) - 1} other requests its text" if len(requests) > 1 else ""
        print(f"{name}: json, text and verbose_json give the dump's text and stop, verbose_json no language and {len(segments)} segments joining "
              f"into it, with or without a segment granularity{others}")
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
    if "decoding" in takes:
        expect_error(transcribe([("decoding", "no-such-decoding")], [("file", "x.wav", wav)]), 400, "unsupported_value", "decoding",
                     "a decoding the model does not have")
    else:
        expect_error(transcribe([("decoding", "greedy")], [("file", "x.wav", wav)]), 400, "unsupported_parameter", "decoding",
                     "a decoding not taken")
    expect_error(transcribe([("temperature", "0")], [("file", "x.wav", wav)]), 400, "unknown_parameter", "temperature", "an unknown member")
    expect_error(transcribe([("response_format", "srt")], [("file", "x.wav", wav)]), 400, "unsupported_value", "response_format", "srt")
    expect_error(transcribe([("response_format", "verbose_json"), ("timestamp_granularities[]", "word")], [("file", "x.wav", wav)]), 400,
                 "unsupported_value", "timestamp_granularities[]", "word timestamps")
    expect_error(transcribe([("timestamp_granularities[]", "segment")], [("file", "x.wav", wav)]), 400, "invalid_value",
                 "timestamp_granularities[]", "a granularity without verbose_json")
    expect_error(transcribe([("model", "whisper-1")], [("file", "x.wav", wav)]), 404, "model_not_found", "model", "another model")
    expect_error(transcribe([("language", info["languages"][0])], []), 400, "missing_required_parameter", "file", "no file")
    if not vad:
        expect_error(transcribe([("chunking_strategy", "auto")], [("file", "x.wav", wav)]), 400, "unsupported_parameter", "chunking_strategy",
                     "chunking_strategy without a detection model")
    else:
        # The dumps' audio joined with a second of silence after each, which the detection cuts into regions again; the
        # server's answers are speech asr --vad's for the same audio and options, which tools/speech_cli_smoke.py checks
        # against the worker.
        joined = array.array("f", [x for d in dumps for x in list(read_npy(os.path.join(d, "audio.npy"))) + [0.0] * rate])
        long, silence = wav_file(joined, rate), wav_file([0.0] * (6 * rate), rate)
        load = [o for i, o in enumerate(options) if o in ("--device", "--threads") or (i and options[i - 1] in ("--device", "--threads"))]
        with tempfile.TemporaryDirectory() as work:
            path = os.path.join(work, "long.wav")
            with open(path, "wb") as f:
                f.write(long)
            timed = ["--timestamps"] if "timestamps" in takes else []

            def cli(*flags):
                r = subprocess.run([speech, "asr", model, "--vad", vad, "--format", "json", *timed, *flags, *load, path],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                assert r.returncode == 0, r.stderr.decode()[-600:]
                return json.loads(r.stdout)

            for form, flags in (([("chunking_strategy", "auto")], []),
                                ([("chunking_strategy[type]", "server_vad"), ("chunking_strategy[threshold]", "0.6"),
                                  ("chunking_strategy[prefix_padding_ms]", "200"), ("chunking_strategy[silence_duration_ms]", "700")],
                                 ["--threshold", "0.6", "--speech-pad-ms", "200", "--min-silence-duration-ms", "700"])):
                want = cli(*flags)
                status, headers, body = transcribe(form, [("file", "long.wav", long)])
                assert status == 200 and json.loads(body) == {"text": want["text"]} and headers["x-speech-stop"] == want["stop"], (form, body)
                status, _, body = transcribe(form + [("response_format", "verbose_json")], [("file", "long.wav", long)])
                verbose = json.loads(body)
                assert verbose["text"] == want["text"] and verbose.get("language") == (",".join(want.get("languages", [])) or None), body
                if timed:
                    assert verbose["segments"] == [{"id": i, **s} for i, s in enumerate(want["segments"])], body
            assert json.loads(transcribe([("chunking_strategy", "auto")], [("file", "silence.wav", silence)])[2]) == {"text": ""}
        print(f"chunking_strategy: auto and server_vad give speech asr --vad's text, language{' and segments' if timed else ''} "
              f"for {len(joined) / rate:.1f} s of the dumps joined, and silence an empty text")
        for form, code, param, what in (
                ([("chunking_strategy", "server_vad")], "unsupported_value", "chunking_strategy", "a chunking_strategy other than auto"),
                ([("chunking_strategy", '{"type":"server_vad"}')], "unsupported_value", "chunking_strategy", "chunking_strategy as JSON"),
                ([("chunking_strategy[type]", "semantic_vad")], "unsupported_value", "chunking_strategy[type]", "a type other than server_vad"),
                ([("chunking_strategy[threshold]", "0.5")], "missing_required_parameter", "chunking_strategy[type]", "a member without the type"),
                ([("chunking_strategy", "auto"), ("chunking_strategy[type]", "server_vad")], "invalid_value", "chunking_strategy", "both forms"),
                ([("chunking_strategy[type]", "server_vad"), ("chunking_strategy[threshold]", "high")], "invalid_type", "chunking_strategy[threshold]",
                 "a threshold that is no number"),
                ([("chunking_strategy[type]", "server_vad"), ("chunking_strategy[silence_duration_ms]", "0.5")], "invalid_type",
                 "chunking_strategy[silence_duration_ms]", "a silence with a fraction"),
                ([("chunking_strategy[type]", "server_vad"), ("chunking_strategy[threshold]", "2")], "unsupported_value", "chunking_strategy[threshold]",
                 "a threshold above 1"),
                ([("chunking_strategy[type]", "server_vad"), ("chunking_strategy[prefix_padding_ms]", "-1")], "unsupported_value",
                 "chunking_strategy[prefix_padding_ms]", "a negative padding"),
                ([("chunking_strategy[type]", "server_vad"), ("chunking_strategy[create_response]", "true")], "unknown_parameter",
                 "chunking_strategy[create_response]", "a member of the Realtime API's server_vad")):
            expect_error(transcribe(form, [("file", "x.wav", wav)]), 400, code, param, what)

if info["task"] == "recognition":
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
    # A client that goes away while its commit is transcribed stops the transcription, and the model serves the next.
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
elif info["task"] == "synthesis":
    expect_error(server.websocket("/v1/realtime"), 404, None, None, "/v1/realtime without a recognition model")

out = server.stop()
if out:
    raise SystemExit(f"the server wrote to stdout: {out[:200]!r}")
print("ok")
