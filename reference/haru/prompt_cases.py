"""Renders character cards with the official haru.prompt and tokenizes them as the official session does, and saves each
for haru-prompt-check.

usage: uv run python prompt_cases.py <dump dir> <out dir>

Writes <out dir>/prompt-cases.jsonl, one card per line: "language", "card", its fields as the official card takes
them, "text", what render_card() writes with the language's template, and "ids", what system_prompt_ids() gives for
it: the card as a ChatML system turn, tokenized by the checkpoint's tokenizer.json with special tokens taken literally.
For each of the nine languages: the official example card, a card of every field with each choice of gender and
backchannel, a card of the name alone, and a card of no field, which is the lead sentence alone.

It also writes reference-<n>-in.npy and reference-<n>-out.npy: a voice reference before and after prepare_reference(),
which cuts its leading and trailing silence and keeps its last 180 seconds: the dump's reference with silence and
quiet noise around it, the same repeated past 180 seconds, silence alone, and less than one window.
"""

import itertools
import json
import os
import sys
from importlib import resources

import numpy as np
from tokenizers import Tokenizer

from haru.audio import SAMPLE_RATE, prepare_reference
from haru.prompt import CARD_FIELDS, card_templates, render_card, system_prompt_ids

import pins

dump_dir, out_dir = sys.argv[1:3]
os.makedirs(out_dir, exist_ok=True)
tokenizer = Tokenizer.from_file(os.path.join(pins.snapshot(), "tokenizer.json"))
templates = card_templates()["templates"]

cases = []
for language, template in templates.items():
    example = json.loads((resources.files("haru") / "assets" / "cards" / f"{language}.json").read_text(encoding="utf-8"))["chat"]
    cards = [example, {}, {"name": example["name"]}]
    for gender, backchannel in itertools.product(sorted(template["gender"]), sorted(template["backchannel"])):
        card = {key: f"{example.get(key) or example['name']} ({key}) 🙂\n  {gender}" for key in CARD_FIELDS}
        card["gender"], card["backchannel"] = gender, backchannel
        cards.append(card)
    for card in cards:
        text = render_card(card, template)
        cases.append({"language": language, "card": card, "text": text, "ids": system_prompt_ids(tokenizer, text)})

with open(os.path.join(out_dir, "prompt-cases.jsonl"), "w", encoding="utf-8") as f:
    for case in cases:
        f.write(json.dumps(case, ensure_ascii=False) + "\n")
print(f"{len(cases)} cards in {len(templates)} languages")

reference = np.load(os.path.join(dump_dir, "reference_audio.npy"))
rng = np.random.default_rng(0)
quiet = lambda seconds, level: (rng.standard_normal(int(seconds * SAMPLE_RATE)) * level).astype(np.float32)
inputs = [
    np.concatenate([np.zeros(SAMPLE_RATE, np.float32), reference, quiet(0.7, 1e-4)]),
    np.concatenate([quiet(0.3, 3e-3), reference, np.zeros(SAMPLE_RATE // 2, np.float32)]),
    np.concatenate([quiet(1.0, 1e-4)] + [reference] * 19 + [quiet(1.0, 1e-4)]),
    np.zeros(2 * SAMPLE_RATE, np.float32),
    reference[:300].copy(),
]
for n, audio in enumerate(inputs):
    np.save(os.path.join(out_dir, f"reference-{n}-in.npy"), audio)
    np.save(os.path.join(out_dir, f"reference-{n}-out.npy"), np.ascontiguousarray(prepare_reference(audio)))
print(f"{len(inputs)} references")
