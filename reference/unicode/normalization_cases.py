"""Writes <out dir>/normalization-cases.tsv, the cases checks/unicode-check compares src/common/unicode.cpp's
normalization with: one line per text, its UTF-8 in hex, then, each after a tab and in hex, its NFC and NFKC by the
tokenizers library (Unicode 9.0) and its NFC and NFKC by this Python's unicodedata (Unicode 13.0).

usage: uv run python normalization_cases.py <out dir>

A text holds several pieces, each after U+0001, which neither composes nor reorders with anything:

- every code point that has a decomposition, a non-zero combining class or a part in a composition, every Hangul
  syllable and every conjoining jamo, alone,
- every mark with a combining class between two of a mark of each class, which it moves before, after or neither
  depending on its class in each version,
- the two code points of every canonical decomposition of two, composed or excluded, alone and with a mark of class
  220 and of 230 between them, which blocks the composition or not,
- every leading and vowel jamo, alone, with each trailing jamo, and as a syllable with each trailing jamo,
- texts that are not in NFC, as macOS writes Japanese file names, and random sequences of the code points above.
"""

import os
import random
import sys
import unicodedata

from tokenizers import normalizers

SEP = "\x01"
LEADS, VOWELS, TRAILS = range(0x1100, 0x1113), range(0x1161, 0x1176), range(0x11A8, 0x11C3)

decomposed = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF and unicodedata.decomposition(chr(cp))]
marks = [cp for cp in range(0x110000) if unicodedata.combining(chr(cp))]
pairs = []
for cp in decomposed:
    parts = unicodedata.decomposition(chr(cp)).split()
    if len(parts) == 2 and not parts[0].startswith("<"):
        pairs.append((int(parts[0], 16), int(parts[1], 16)))
jamo = [*range(0x1100, 0x1200), *range(0xA960, 0xA980), *range(0xD7B0, 0xD800)]
pieces_alone = sorted({*decomposed, *marks, *(cp for pair in pairs for cp in pair), *jamo, *range(0xAC00, 0xD7A4)})
reference = {}
for cp in marks:
    reference.setdefault(unicodedata.combining(chr(cp)), cp)

texts = []
for i in range(0, len(pieces_alone), 64):
    texts.append([chr(cp) for cp in pieces_alone[i:i + 64]])
for cp in marks:
    texts.append([chr(r) + chr(cp) + chr(r) for _, r in sorted(reference.items())])
for i in range(0, len(pairs), 16):
    texts.append([chr(a) + m + chr(b) for a, b in pairs[i:i + 16] for m in ("", "\u0316", "\u0305")])
for lead in LEADS:
    for vowel in VOWELS:
        syllable = chr(0xAC00 + ((lead - 0x1100) * 21 + vowel - 0x1161) * 28)
        texts.append([chr(lead), chr(vowel), chr(lead) + chr(vowel)] + [chr(lead) + chr(vowel) + chr(t) for t in TRAILS] +
                     [syllable + chr(t) for t in TRAILS])
texts.append([
    "\u30ab\u3099\u30ad\u3099\u30af\u3099\u30cf\u309a\u30d2\u309a.txt",
    "\u30bf\u3099\u30a6\u30f3\u30ed\u30fc\u30c8\u3099\u306e\u30d5\u30a9\u30eb\u30bf\u3099",
    "Cafe\u0301 nai\u0308ve re\u0301sume\u0301 A\u030a a\u0301\u0323 q\u0307\u0323",
    "\u1112\u1161\u11ab\u1100\u1173\u11af \u110b\u1161\u11ab\u1102\u1167\u11bc",
    "\u212b \u2126 \uf900 \u0958 \u2000 >\u0338 =\u0338",
    "a\u1df8\u0301 a\u0301\u1df8 a\u1df8\u0316 \U00011935\U00011930",
])
rng = random.Random(0)
draw = pieces_alone + [ord(c) for c in "aeiouAEIOU <>=\u3042\u304b\u30ab\u30cf"] * 20
for _ in range(2000):
    texts.append(["".join(chr(rng.choice(draw)) for _ in range(rng.randint(2, 12))) for _ in range(4)])

forms = [normalizers.NFC(), normalizers.NFKC()]
os.makedirs(sys.argv[1], exist_ok=True)
path = os.path.join(sys.argv[1], "normalization-cases.tsv")
with open(path, "w") as f:
    for pieces in texts:
        text = SEP + SEP.join(pieces)
        outs = [n.normalize_str(text) for n in forms] + [unicodedata.normalize(form, text) for form in ("NFC", "NFKC")]
        f.write("\t".join(s.encode("utf-8").hex() for s in [text] + outs) + "\n")
print(len(texts), "texts ->", path)
