"""Times llama.cpp's llama-server with Qwen3-ASR on the audio of dumps of reference/qwen3-asr/dump.py, asked as ASIST
asks it: the audio as a 16-bit WAV in /v1/chat/completions, with the language left to the model and forced through the
prefill `language <Name><asr_text>` as each dump's forced request forces it, greedy, with the prompt cache off. For
each, the median over the runs of the request's time and of what the server reports: the prompt's time, which holds
the encoder's, and the decoding's, with its steps per second, a step being a token fed back to the decoder. One request
of the first dump goes first, untimed. It prints one JSON object per line, as qwen3-asr-timing does for speech.cpp.

usage: python3 checks/llama-server-timing.py <llama-server> <model.gguf> <mmproj.gguf> <device> <runs> <dump dir>...
(the device as llama-server names it: MTL0 on a Mac, Vulkan0 on Windows)
"""

import ast
import base64
import http.client
import json
import os
import socket
import statistics
import struct
import subprocess
import sys
import time

if len(sys.argv) < 7:
    raise SystemExit(__doc__)
server, model, mmproj, device, runs, *dumps = sys.argv[1:]
runs = int(runs)


def read_npy(path):
    with open(path, "rb") as f:
        data = f.read()
    header_length = struct.unpack("<H", data[8:10])[0]
    header = ast.literal_eval(data[10:10 + header_length].decode())
    assert header["descr"] == "<f4" and len(header["shape"]) == 1, header
    return struct.unpack(f"<{header['shape'][0]}f", data[10 + header_length:])


def wav(samples):
    pcm = struct.pack(f"<{len(samples)}h", *(round(max(-1.0, min(1.0, s)) * 32767) for s in samples))
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)


with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
process = subprocess.Popen([server, "--model", model, "--mmproj", mmproj, "--device", device, "--n-gpu-layers", "99", "--ctx-size", "4096",
                            "--parallel", "1", "--host", "127.0.0.1", "--port", str(port), "--no-webui", "--offline", "--log-verbosity", "2"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def post(body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    start = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    answer = json.loads(c.getresponse().read())
    return time.perf_counter() - start, answer


def ask(audio, language):
    messages = [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": base64.b64encode(audio).decode(), "format": "wav"}}]}]
    if language:
        messages.append({"role": "assistant", "content": f"language {language}<asr_text>"})
    return post({"messages": messages, "temperature": 0, "max_tokens": 4096, "cache_prompt": False})


try:
    while True:
        if process.poll() is not None:
            raise SystemExit(f"llama-server exited with {process.returncode}")
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            c.request("GET", "/health")
            if c.getresponse().status == 200:
                break
        except OSError:
            pass
        time.sleep(0.3)
    ask(wav(read_npy(os.path.join(dumps[0], "audio.npy"))), None)
    for dump in dumps:
        samples = read_npy(os.path.join(dump, "audio.npy"))
        audio = wav(samples)
        with open(os.path.join(dump, "forced", "meta.json"), encoding="utf-8") as f:
            language = json.load(f)["language"]
        for forced in (None, language):
            results = [ask(audio, forced) for _ in range(runs)]
            timings = [r[1]["timings"] for r in results]
            steps = timings[0]["predicted_n"] - 1
            print(json.dumps({"input": os.path.basename(os.path.normpath(dump)), "seconds": round(len(samples) / 16000, 2), "forced": forced,
                              "total": round(statistics.median(r[0] for r in results), 3),
                              "prompt": round(statistics.median(t["prompt_ms"] for t in timings) / 1000, 3), "prompt_rows": timings[0]["prompt_n"],
                              "decode": round(statistics.median(t["predicted_ms"] for t in timings) / 1000, 3), "tokens": timings[0]["predicted_n"],
                              "steps_per_second": round(statistics.median(steps / t["predicted_ms"] * 1000 for t in timings), 1),
                              "text": results[0][1]["choices"][0]["message"]["content"]}, ensure_ascii=False), flush=True)
finally:
    process.terminate()
    process.wait()
