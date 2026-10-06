"""Writes texts as the model might write them, with the text qwen-asr's parse_asr_output() makes of each, for
qwen3-asr-decoder-check.

usage: uv run python parse_cases.py <out dir>

Writes <out dir>/parse-cases.jsonl, one {"raw", "forced", "text"} per line: runs of one character just under, at and
over the repetition fix's threshold, patterns of 1 to 21 characters repeated around it, one after a prefix and before
a suffix, two in a row, the whitespace Python's str.strip() removes and some it keeps, outputs with and without
<asr_text>, the language None of audio without speech, and the same with the language forced, whose output is the
text alone. The dumps hold none of these: the model wrote no repetition in them.
"""

import argparse
import json
import os

from official_utils import qwen_asr_utils

utils = qwen_asr_utils()
threshold = 20

texts = ["", "あ", "language None<asr_text>", "language None<asr_text>はい。", "language Japanese<asr_text>", "<asr_text>",
         "language English<asr_text>a<asr_text>b", "no tag at all", "language Japanese\n<asr_text>改行の後"]
for n in (threshold - 1, threshold, threshold + 1, 2 * threshold, 100):
    texts += ["x" * n, "前" + "あ" * n + "後", "ab" + "c" * n]
for k in (1, 2, 3, 7, 20, 21):
    pattern = "".join(chr(ord("a") + i % 26) for i in range(k))
    for times in (threshold - 1, threshold, threshold + 5):
        texts += [pattern * times, "始め" + pattern * times + "終わり", pattern * times + "z" + pattern * times]
texts += ["はい、" * 30 + "そうです。" * 25, "ありがとうございます。" * 21, "ねえ" * 10 + "ね" * 30, "abab" * 15 + "ab" * 3]
spaces = ["\u3000", "\x85", "\u2028", "\u00a0", "\t\n\r\x0b\x0c", "\x1c\x1f", "\u200b", "\ufeff"]
texts += [f"{s}language Japanese<asr_text>{s}text{s}" for s in spaces]

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
with open(os.path.join(args.out_dir, "parse-cases.jsonl"), "w", encoding="utf-8") as f:
    for raw in texts:
        for forced in (None, "Japanese"):
            f.write(json.dumps({"raw": raw, "forced": forced is not None, "text": utils.parse_asr_output(raw, user_language=forced)[1]},
                               ensure_ascii=False) + "\n")
print(f"wrote {2 * len(texts)} cases")
