"""Writes <out dir>/guardrail-cases.tsv, the cases checks/nemotron-transcribe/nemotron-transcribe-guardrail-check
compares the family's generation-loop guardrail with: one line per text, its UTF-8 in hex, a tab, the UTF-8 in hex of
the text the model card's cleanup_generation_loops() gives it, run here with the card's own regular expressions, a tab,
and 1 where either expression, anchored at the end, matches a run that ends the text, 0 where neither does.

usage: uv run python guardrail_cases.py <out dir>
"""

import os
import random
import re
import sys

# The model card's guardrail, as its "Generation-Loop Guardrail" section gives it.
CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)
ENDS = [re.compile(r.pattern + r"\Z", re.DOTALL) for r in (CHAR_RUN, BLOCK_RUN)]


def cleanup_generation_loops(text: str) -> str:
    text = CHAR_RUN.sub(lambda m: m.group(1), text)
    return BLOCK_RUN.sub(lambda m: m.group(1), text)


texts = [
    "",
    "a",
    "米国体操連盟、USA Gymnasticsは、選手全員のための安全な環境づくりを約束。",
    "a" * 20,
    "a" * 21,
    "a" * 100 + "b" * 20 + "c" * 21,
    "ab" * 19,
    "ab" * 20,
    "ab" * 25 + "a",
    "x" + "はい、" * 20 + "そうです。",
    "はい、" * 19 + "そうです。",
    "\n" * 30 + "end",
    "the cat " * 40,
    "aab" * 20 + "aab" * 3,
    "<spk:0> yes <spk:1> no " * 22,
    "🙂" * 25 + "😀🙂" * 21,
    "ab" * 20 + "ba" * 20,
    "abc" * 10 + "abcabc" * 20,
    "a" * 19,
    "x" + "a" * 20,
    "ab" * 20 + "a",
    "ab" * 20 + "c",
    "intro " + "彼らの行動は、大地震のたびに起こる。 " * 19 + "彼らの行動は",
    "intro " + "彼らの行動は、大地震のたびに起こる。 " * 20,
]
rng = random.Random(0)
for _ in range(200):
    pieces = []
    for _ in range(rng.randint(1, 6)):
        block = "".join(rng.choice("ab xyはい、。\n🙂") for _ in range(rng.randint(1, 6)))
        pieces.append(block * rng.choice([1, 2, 5, 19, 20, 21, 30]))
    texts.append("".join(pieces))

os.makedirs(sys.argv[1], exist_ok=True)
path = os.path.join(sys.argv[1], "guardrail-cases.tsv")
with open(path, "w") as f:
    for text in texts:
        ends = any(e.search(text) for e in ENDS)
        f.write(text.encode("utf-8").hex() + "\t" + cleanup_generation_loops(text).encode("utf-8").hex() + "\t" + str(int(ends)) + "\n")
print(len(texts), "cases ->", path)
