"""Runs Haru's official text sampling, haru.sampling's sample_token() and DryState, on the logits of a dump of dump.py
and on token sequences made for the purpose, and saves what each computes for haru-sampling-check.

usage: uv run python sampling_cases.py <dump dir> <out dir>

Writes under <out dir>:
  <case>/meta.json          the settings: temperature, top_k and top_p, and "must_speak" for a case whose rows have the
                            logits of PAD, EOP and EOS at minus infinity, as the session bans them
  <case>/ids.npy            the tokens each draw takes from, in sample_token()'s order, [rows, candidates]
  <case>/probabilities.npy  the probability of each, 0 where top_p drops it, [rows, candidates]
  logits.npy                the dump's sampled_logits, the logits each draw takes, [rows, vocab]
  dry.jsonl                 one line per sequence: "tokens", pushed one at a time, and "penalties", after each push the
                            penalty of each token DryState subtracts, as [token, penalty] pairs by rising token

sample_token() draws with torch.multinomial(), which nothing outside torch reproduces, so the draws themselves are not
compared: the probabilities are the tensor it hands torch.multinomial(), with the ids its torch.topk() or torch.sort()
gave. The settings are the official defaults, which the model file holds, and others that take each branch: a top_p
below 1, a lower temperature with a top_k of 1, a wide top_k, and the defaults with the silent tokens banned. The DRY sequences are the dump's tokens and draws from a
few tokens, with the silent ones among them, long enough to pass DRY's lookback.
"""

import json
import os
import random
import sys
from unittest import mock

import numpy as np
import torch

from haru.config import HaruConfig
from haru.sampling import DryConfig, DryState, sample_token
from haru.session import DuplexConfig

import pins

dump_dir, out_dir = sys.argv[1:3]
os.makedirs(out_dir, exist_ok=True)
config = HaruConfig.from_file(os.path.join(pins.snapshot(), "config.json"))
logits = np.load(os.path.join(dump_dir, "sampled_logits.npy"))
defaults = DuplexConfig()
CASES = {
    "file": {"temperature": defaults.temperature, "top_k": defaults.top_k, "top_p": defaults.top_p},
    "top-p": {"temperature": 0.7, "top_k": 25, "top_p": 0.5},
    "cold-top-k-1": {"temperature": 0.3, "top_k": 1, "top_p": 1.0},
    "wide": {"temperature": 1.2, "top_k": 1000, "top_p": 0.9},
    "must-speak": {"temperature": defaults.temperature, "top_k": defaults.top_k, "top_p": defaults.top_p, "must_speak": True},
}


def draw(row, settings):
    """The ids and the probabilities sample_token() hands torch.multinomial() for `row`."""
    seen = {}
    topk, sort, multinomial = torch.topk, torch.sort, torch.multinomial

    def recording_topk(*args, **kwargs):
        result = topk(*args, **kwargs)
        seen["ids"] = result[1]
        return result

    def recording_sort(*args, **kwargs):
        result = sort(*args, **kwargs)
        seen["ids"] = result[1]
        return result

    def recording_multinomial(probs, *args, **kwargs):
        seen["probabilities"] = probs.clone()
        return multinomial(probs, *args, **kwargs)

    with mock.patch.object(torch, "topk", recording_topk), mock.patch.object(torch, "sort", recording_sort), \
            mock.patch.object(torch, "multinomial", recording_multinomial):
        sample_token(torch.from_numpy(row)[None], settings["temperature"], settings["top_k"], settings["top_p"])
    return seen["ids"][0].numpy().astype(np.int32), seen["probabilities"][0].numpy().astype(np.float32)


np.save(os.path.join(out_dir, "logits.npy"), logits.astype(np.float32))
# The session's bans while the model must speak, which a case's "must_speak" puts on every row.
banned = [config.pad_token_id, config.eop_token_id, config.eos_token_id]
for name, settings in CASES.items():
    folder = os.path.join(out_dir, name)
    os.makedirs(folder, exist_ok=True)
    source = logits.copy()
    if settings.get("must_speak"):
        source[:, banned] = -np.inf
    rows = [draw(row, settings) for row in source]
    np.save(os.path.join(folder, "ids.npy"), np.stack([r[0] for r in rows]))
    np.save(os.path.join(folder, "probabilities.npy"), np.stack([r[1] for r in rows]))
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=1)
    print(f"{name}: {len(rows)} rows of {len(rows[0][0])} candidates")

silent = [config.pad_token_id, config.eop_token_id, config.eos_token_id]
dry = DryConfig()
rng = random.Random(0)
sequences = [[int(t) for t in np.load(os.path.join(dump_dir, "tokens.npy"))]]
for alphabet, length in (([11, 12, 13], 300), ([5, 6, 7, 8, 9] + silent, 1300), ([100, 200], 1200)):
    sequences.append([rng.choice(alphabet) for _ in range(length)])
# A phrase repeated, with a different token after it each time, and long runs of one token.
phrase = [21, 22, 23, 24, 25]
sequences.append([t for i in range(60) for t in phrase + [30 + i % 4]])
sequences.append([42] * 80 + [43] + [42] * 80)
with open(os.path.join(out_dir, "dry.jsonl"), "w", encoding="utf-8") as f:
    for tokens in sequences:
        state = DryState(dry, set(silent))
        penalties = []
        for t in tokens:
            state.update(t)
            penalties.append(sorted([int(k), float(v)] for k, v in state.penalties.items()))
        f.write(json.dumps({"tokens": tokens, "penalties": penalties}) + "\n")
print(f"dry.jsonl: {len(sequences)} sequences")
