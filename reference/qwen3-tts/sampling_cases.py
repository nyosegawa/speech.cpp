"""Runs transformers' logits processors, as the official generate() builds them for the talker and the code predictor,
on the logits of a dump of dump.py, and saves what they leave for sampler-check.

usage: uv run python sampling_cases.py <model dir> <dump dir> <out dir>

Writes one folder per case under <out dir>, each with:
  meta.json   the stack ("talker" or "code_predictor") and its settings, or "settings": "file" where the official
              defaults of generation_config.json apply, which the model file holds
  logits.npy  the dump's logits, [rows, vocab]
  history.npy the codes the repetition penalty sees for each row, padded with -1, [rows, longest]
  scores.npy  the logits after the processors, -inf where a token is removed, [rows, vocab]

The talker's rows are the dump's talker_logits, each after the first codes of the frames before it, with the official
suppress_tokens and min_new_tokens; the code predictor's are its logits of the first frames, each after the codes of its
frame before it, as its generate() starts from embeddings alone. The processors come from transformers'
_get_logits_processor() itself, so their order and the conditions under which each applies are transformers' own.
torch.multinomial draws with torch's generator, which nothing outside torch reproduces, so the draws themselves are not
compared.
"""

import json
import os
import sys

import numpy as np
import torch
from transformers import GenerationConfig
from transformers.generation.utils import GenerationMixin

model_dir, dump_dir, out_dir = sys.argv[1:4]
config = json.load(open(os.path.join(model_dir, "config.json")))
defaults = json.load(open(os.path.join(model_dir, "generation_config.json")))
talker_config = config["talker_config"]
vocab = talker_config["vocab_size"]
eos = talker_config["codec_eos_token_id"]
cp_penalty = talker_config["code_predictor_config"]["repetition_penalty"]

talker_logits = np.load(os.path.join(dump_dir, "talker_logits.npy"))
cp_logits = np.load(os.path.join(dump_dir, "cp_logits.npy"))
codes = np.load(os.path.join(dump_dir, "codes.npy"))
cp_frames = 20


class Generator(GenerationMixin):
    """What _get_logits_processor() needs of a model: GenerationMixin's own methods, without weights."""


def processors(settings, talker):
    """The processors generate() builds for one stack, from the official kwargs of the talker's or the code predictor's."""
    kwargs = dict(do_sample=settings["do_sample"], temperature=settings["temperature"], top_k=settings["top_k"],
                  top_p=settings["top_p"], repetition_penalty=settings["repetition_penalty"])
    if talker:
        # Qwen3TTSForConditionalGeneration.generate()'s talker_kwargs.
        kwargs.update(min_new_tokens=2, eos_token_id=eos,
                      suppress_tokens=[i for i in range(vocab - 1024, vocab) if i != eos])
    generation_config = GenerationConfig(**kwargs)
    generation_config._eos_token_tensor = torch.tensor([eos]) if talker else None
    # The talker and the code predictor generate from embeddings alone, so the prompt adds no ids.
    return GenerationMixin._get_logits_processor(Generator(), generation_config, input_ids_seq_length=0, device="cpu")


def rows(talker):
    """The logits and the history of each row of a stack."""
    if talker:
        return [(talker_logits[f], codes[:f, 0]) for f in range(talker_logits.shape[0])]
    return [(cp_logits[f, g], codes[f, 1:g + 1]) for f in range(cp_frames) for g in range(cp_logits.shape[1])]


def official(stack):
    """The settings the official package passes when the caller sets none, from generation_config.json."""
    if stack == "talker":
        keys = ["do_sample", "temperature", "top_k", "top_p", "repetition_penalty"]
        return {k: defaults[k] for k in keys}
    return {"do_sample": defaults["subtalker_dosample"], "temperature": defaults["subtalker_temperature"],
            "top_k": defaults["subtalker_top_k"], "top_p": defaults["subtalker_top_p"], "repetition_penalty": cp_penalty}


CASES = {
    "talker-file": ("talker", None),
    "talker-top-p": ("talker", dict(do_sample=True, temperature=1.3, top_k=0, top_p=0.8, repetition_penalty=1.2)),
    "talker-top-p-wide": ("talker", dict(do_sample=True, temperature=2.5, top_k=0, top_p=0.99, repetition_penalty=1.05)),
    "talker-top-k-1": ("talker", dict(do_sample=True, temperature=0.7, top_k=1, top_p=1.0, repetition_penalty=1.05)),
    "talker-top-p-0": ("talker", dict(do_sample=True, temperature=0.9, top_k=0, top_p=0.0, repetition_penalty=1.05)),
    "talker-penalty-below-1": ("talker", dict(do_sample=True, temperature=1.0, top_k=100, top_p=0.95, repetition_penalty=0.8)),
    "talker-greedy": ("talker", dict(do_sample=False, temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.5)),
    "code-predictor-file": ("code_predictor", None),
    "code-predictor-top-p": ("code_predictor", dict(do_sample=True, temperature=0.6, top_k=0, top_p=0.5, repetition_penalty=1.0)),
    "code-predictor-top-p-wide": ("code_predictor", dict(do_sample=True, temperature=2.0, top_k=200, top_p=0.95, repetition_penalty=1.0)),
    "code-predictor-top-k-beyond-vocab": ("code_predictor", dict(do_sample=True, temperature=1.1, top_k=5000, top_p=1.0,
                                                                 repetition_penalty=1.0)),
    "code-predictor-penalty": ("code_predictor", dict(do_sample=True, temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.3)),
    "code-predictor-greedy": ("code_predictor", dict(do_sample=False, temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.0)),
}

for name, (stack, settings) in CASES.items():
    talker = stack == "talker"
    applied = settings or official(stack)
    process = processors(applied, talker)
    logits, histories, scores = [], [], []
    for row, history in rows(talker):
        ids = torch.tensor(history, dtype=torch.long).unsqueeze(0)
        out = process(ids, torch.tensor(row, dtype=torch.float32).unsqueeze(0))
        logits.append(row)
        histories.append(history)
        scores.append(out[0].numpy())
    longest = max(len(h) for h in histories)
    padded = np.full((len(histories), max(longest, 1)), -1, dtype=np.int32)
    for i, h in enumerate(histories):
        padded[i, :len(h)] = h
    folder = os.path.join(out_dir, name)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "logits.npy"), np.stack(logits).astype(np.float32))
    np.save(os.path.join(folder, "history.npy"), padded)
    np.save(os.path.join(folder, "scores.npy"), np.stack(scores).astype(np.float32))
    meta = {"stack": stack, "processors": [type(p).__name__ for p in process]}
    meta.update({"settings": "file", "official": applied} if settings is None else applied)
    json.dump(meta, open(os.path.join(folder, "meta.json"), "w"), indent=1)
    print(name, len(logits), "rows,", ", ".join(meta["processors"]))
