"""Times Irodori-TTS through `speech worker` with and without instructions (the runtime's caption), as the speed table
of docs/models/irodori-tts.md times it: each sentence of a speech-bench prompts file one request at a time after the
worker is ready, in one voice, first without instructions and then with each caption given. For each, the median and the
90th percentile of the first audio, the time from the request to its first chunk, and the real-time factor, the
request's time over its audio's. It prints one JSON object per line.

usage: python3 checks/irodori-caption-timing.py <speech> <model.gguf> <prompts.json> <voice NAME=FILE> [--steps N]
                                                 [--caption TEXT]... [-- worker options...]
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from worker_client import Worker  # noqa: E402

# The example of a caption in docs/models/irodori-tts.md, the one the design's estimate of the cost was made for.
CAPTION = "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。"

args = sys.argv[1:]
options = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
if len(args) < 4:
    raise SystemExit(__doc__)
speech, model, prompts_path, voice, *rest = args
steps = None
captions = []
while rest:
    flag, value, *rest = rest
    if flag == "--steps":
        steps = int(value)
    elif flag == "--caption":
        captions.append(value)
    else:
        raise SystemExit(f"unknown option {flag}\n{__doc__}")
name, _ = voice.split("=", 1)
prompts = json.load(open(prompts_path, encoding="utf-8"))["prompts"]
worker = Worker(speech, model, ["--add-voice", voice, *options])
rate = worker.ready["model"]["sample_rate"]


def run(instructions):
    first, factors = [], []
    for i, prompt in enumerate(prompts):
        request = {"type": "synthesize", "id": str(i), "text": prompt["text"], "voice": name, "seed": 1}
        if steps:
            request["steps"] = steps
        if instructions:
            request["instructions"] = instructions
        sent = time.perf_counter()
        worker.request(request)
        messages = worker.until(str(i))
        end = messages[-1]
        if end["type"] != "end":
            raise SystemExit(f"{prompt['id']}: {end}")
        chunks = [m for m in messages if m["type"] == "chunk"]
        first.append(chunks[0]["_at"] - sent)
        factors.append((end["_at"] - sent) / (end["samples"] / rate))
    return {"instructions": instructions, "sentences": len(prompts), "steps": steps, "first_audio_median": statistics.median(first),
            "first_audio_p90": statistics.quantiles(first, n=10)[-1], "real_time_factor": statistics.median(factors)}


for instructions in ["", *(captions or [CAPTION])]:
    print(json.dumps(run(instructions), ensure_ascii=False), flush=True)
