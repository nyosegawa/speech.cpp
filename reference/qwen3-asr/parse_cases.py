"""Writes texts as the model might write them, with the language and the text qwen-asr's parse_asr_output() makes of
each, for qwen3-asr-decoder-check.

usage: uv run python parse_cases.py <out dir>

Writes <out dir>/parse-cases.jsonl, one {"raw", "forced", "language", "text"} per line, "forced" the language's name
or null: runs of one character just under, at and over the repetition fix's threshold, patterns of 1 to 21 characters
repeated around it, one after a prefix and before a suffix, two in a row, the whitespace Python's str.strip() removes
and some it keeps, outputs with and without <asr_text>, the language None of audio without speech, and the same with
the language forced, whose output is the text alone. Before <asr_text>: names in other cases, names that are none of
the model's languages, None in other cases and places, a language line after another line, the line breaks of
str.splitlines(), a name the repetition fix rejoins, and the code points outside ASCII that str.upper() and
str.lower() turn into ASCII. The dumps hold none of these: the model wrote no repetition in them, and only the names
of its languages.
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
texts += ["language japanese<asr_text>x", "language JAPANESE<asr_text>x", "LANGUAGE English<asr_text>x", "Language  \t German \n<asr_text>x",
          "language\tJapanese<asr_text>x", "language <asr_text>x", "xlanguage Japanese<asr_text>x", "note\nlanguage German<asr_text>x",
          "language German\nlanguage French<asr_text>x", "Language NONE<asr_text>x", "language Nonexistent<asr_text>x",
          "language Japanese\nlanguage None<asr_text>x", "it is language none\nlanguage Japanese<asr_text>x",
          "language Klingon<asr_text>x", "language Sichuanese<asr_text>x", "language Chinese,English<asr_text>x",
          "language Japanese Japanese<asr_text>x", "language Ja" + "pa" * 25 + "nese<asr_text>x", "language \u0131talian<asr_text>x",
          "language \u017fpanish<asr_text>x", "language Tur\u212aish<asr_text>x", "language H\u0130ndi<asr_text>x",
          "language \u00dfwedish<asr_text>x", "language \ufb01nnish<asr_text>x", "language \u212aorean<asr_text>x"]
texts += [f"language{b}Japanese<asr_text>x" for b in ("\n", "\r\n")] + [f"language English{b}<asr_text>x" for b in
                                                                        ("\r", "\r\n", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")]
texts += [f"x{b}language English<asr_text>y" for b in ("\x1c", "\x85", "\u2029", "\u200b")]

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
with open(os.path.join(args.out_dir, "parse-cases.jsonl"), "w", encoding="utf-8") as f:
    for raw in texts:
        for forced in (None, "Japanese"):
            language, text = utils.parse_asr_output(raw, user_language=forced)
            f.write(json.dumps({"raw": raw, "forced": forced, "language": language, "text": text}, ensure_ascii=False) + "\n")
print(f"wrote {2 * len(texts)} cases")
