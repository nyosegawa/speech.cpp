"""Writes src/common/unicode-data.inc: the tables src/common/unicode.cpp normalizes text with, NFC and NFKC, in the
Unicode version of each reference, and those of Python's str.isspace().

usage: uv run python gen_unicode.py ../../src/common/unicode-data.inc

The references run three versions of Unicode, found on 2026-10-06:

- Irodori-TTS's runtime normalizes its text with unicodedata.normalize("NFKC") in Python 3.10 (reference/irodori-tts),
  whose tables are Unicode 13.0.0.
- The Qwen3-TTS and Qwen3-ASR references run Python 3.12, Unicode 15.0.0, whose unicodedata never sees their text: their
  tokenizers are the tokenizers library's (Qwen2TokenizerFast in transformers 4.57.3, Qwen2Tokenizer in 5.18.0), with
  the NFC normalizer that transformers gives them when it builds them from vocab.json and merges.txt.
- tokenizers 0.22.2 (reference/qwen3-tts) and 0.23.2 (reference/qwen3-asr) both link the unicode-normalization-
  alignments crate 0.1.12, whose tables are Unicode 9.0.0: they give the combining marks that 9.0 added (U+08D4, U+1DFB,
  Adlam, Newa, Bhaiksuki) their classes, and those that 10.0 added (U+1DF6 to U+1DF9, U+0D3B, Zanabazar Square) none.

Each normalization of every code point alone, and every combining class, compared between them: NFC is the same in all
three. Python 3.12 differs from 3.10 in NFKC on 121 code points (U+A7F2 to U+A7F4, 56 of U+10781 to U+107BA, U+1E030 to
U+1E06D) and in the class of 50, all characters that 14.0 and 15.0 added. The crate differs from Python 3.10 in NFD on
U+11938 and in NFKC on 13 code points (U+32FF, U+AB69, U+1F16C, U+1FBF0 to U+1FBF9), characters of 12.0 and 13.0, and in
the class of 58 marks, those of 10.0 to 13.0, which it takes for starters. Texts with them differ in NFC too: U+11935
U+11930 composes to U+11938 in 13.0 and not in the crate, and a mark of 10.0 or later is reordered among the marks
around it in 13.0, where the crate leaves it in place and composes nothing across it. So each normalization keeps the
version of its reference: NFC that of the crate for the Qwen tokenizers, and NFKC that of Python 3.10 for Irodori-TTS.

One set of tables serves both, because the Unicode Standard keeps the decomposition and the combining class of a
character, and whether it composes, once the character is assigned: the tables of 9.0 are those of 13.0 without the
characters assigned since. This script takes the tables from this Python's unicodedata, 13.0, and finds the entries the
crate lacks by running its normalizers, which it checks against the tables of both versions on every code point. Each
entry is marked with the first version that has it, 9 or 13:

  kDecompositions     code point, offset and length into kDecompositionPool of its decomposition mapping, one level
                      as UnicodeData.txt gives it, whether the mapping is a compatibility one, and the version; for
                      every code point that has one other than the Hangul syllables, whose decomposition is computed
  kCombiningClasses   ranges of code points with the same non-zero canonical combining class and version
  kCompositions       the pairs that canonical composition joins, what they join into and its version, other than
                      Hangul
  kWhitespace         ranges of the code points str.isspace() accepts, which Python 3.10 and 3.12 accept alike
"""

import sys
import unicodedata

from tokenizers import normalizers

assert unicodedata.unidata_version == "13.0.0", unicodedata.unidata_version
HANGUL_FIRST, HANGUL_LAST = 0xAC00, 0xD7A3
VERSIONS = (9, 13)
CRATE = {name: getattr(normalizers, name)() for name in ("NFD", "NFKD", "NFC")}


def crate(form, texts):
    return [CRATE[form].normalize_str(text) for text in texts]


def ranges(values):
    """(first, last, value) runs of consecutive code points with the same value."""
    out = []
    for cp, value in values:
        if out and out[-1][1] == cp - 1 and out[-1][2] == value:
            out[-1][1] = cp
        else:
            out.append([cp, cp, value])
    return out


code_points = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
crate_nfd = dict(zip(code_points, crate("NFD", [chr(cp) for cp in code_points])))
crate_nfkd = dict(zip(code_points, crate("NFKD", [chr(cp) for cp in code_points])))

mappings = {}
for cp in code_points:
    parts = unicodedata.decomposition(chr(cp)).split()
    if parts and not HANGUL_FIRST <= cp <= HANGUL_LAST:
        compatibility = parts[0].startswith("<")
        since = 9 if crate_nfkd[cp] != chr(cp) else 13
        mappings[cp] = ([int(p, 16) for p in parts[1 if compatibility else 0:]], compatibility, since)

classes = {}
for cp in code_points:
    c = unicodedata.combining(chr(cp))
    if c:
        classes[cp] = c
# A mark the crate knows is reordered before U+0300 (230) when its class is lower, and after U+0334 (1) when higher. A
# mark with a canonical decomposition is decomposed before its class is read, and its class has its mapping's version.
canonical = {cp for cp, (_, compatibility, _) in mappings.items() if not compatibility}
probes = [cp for cp in classes if cp not in canonical]
texts = ["\u0300" + chr(cp) if classes[cp] < 230 else chr(cp) + "\u0334" for cp in probes]
known = {cp for cp, text, out in zip(probes, texts, crate("NFD", texts)) if out != text}
class_since = {cp: mappings[cp][2] if cp in canonical else 9 if cp in known else 13 for cp in classes}


def combining(cp, version):
    return classes[cp] if cp in classes and class_since[cp] <= version else 0


def decompose(cp, compatibility, version):
    if HANGUL_FIRST <= cp <= HANGUL_LAST:
        return [ord(c) for c in unicodedata.normalize("NFD", chr(cp))]
    if cp not in mappings or mappings[cp][2] > version or (mappings[cp][1] and not compatibility):
        return [cp]
    return [x for part in mappings[cp][0] for x in decompose(part, compatibility, version)]


def ordered(text, version):
    out = list(text)
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and combining(out[j], version):
            j += 1
        out[i:j] = sorted(out[i:j], key=lambda c: combining(c, version))
        i = max(j, i + 1)
    return "".join(map(chr, out))


for cp in code_points:
    for version, nfd, nfkd in ((13, unicodedata.normalize("NFD", chr(cp)), unicodedata.normalize("NFKD", chr(cp))),
                               (9, crate_nfd[cp], crate_nfkd[cp])):
        assert ordered(decompose(cp, False, version), version) == nfd, (hex(cp), version)
        assert ordered(decompose(cp, True, version), version) == nfkd, (hex(cp), version)

compositions = []
for cp, (mapping, compatibility, since) in mappings.items():
    if compatibility or len(mapping) != 2:
        continue
    # Exclusions and non-starter decompositions are the pairs that NFC does not join back.
    if unicodedata.normalize("NFC", chr(mapping[0]) + chr(mapping[1])) == chr(cp):
        compositions.append((mapping[0], mapping[1], cp))
compositions.sort()
joined = crate("NFC", [chr(a) + chr(b) for a, b, _ in compositions])
compositions = [(a, b, cp, 9 if out == chr(cp) else 13) for (a, b, cp), out in zip(compositions, joined)]
assert all(since == mappings[cp][2] for _, _, cp, since in compositions)

decompositions, pool = [], []
for cp in sorted(mappings):
    mapping, compatibility, since = mappings[cp]
    decompositions.append((cp, len(pool), len(mapping), "true" if compatibility else "false", since))
    pool.extend(mapping)
combining_ranges = ranges((cp, (classes[cp], class_since[cp])) for cp in sorted(classes))
whitespace = ranges((cp, 1) for cp in range(0x110000) if chr(cp).isspace())


def rows(items, per_line, fmt):
    lines = []
    for i in range(0, len(items), per_line):
        lines.append("    " + " ".join(fmt(x) + "," for x in items[i:i + per_line]))
    return "\n".join(lines)


with open(sys.argv[1], "w") as f:
    f.write(f"// Generated by reference/unicode/gen_unicode.py from Unicode {unicodedata.unidata_version} and the tokenizers "
            "library's 9.0.0.\n")
    f.write("static const Decomposition kDecompositions[] = {\n")
    f.write(rows(decompositions, 4, lambda d: "{0x%X, %d, %d, %s, %d}" % d) + "\n};\n")
    f.write("static const uint32_t kDecompositionPool[] = {\n")
    f.write(rows(pool, 10, lambda c: "0x%X" % c) + "\n};\n")
    f.write("static const CombiningClassRange kCombiningClasses[] = {\n")
    f.write(rows(combining_ranges, 4, lambda r: "{0x%X, 0x%X, %d, %d}" % (r[0], r[1], *r[2])) + "\n};\n")
    f.write("static const Composition kCompositions[] = {\n")
    f.write(rows(compositions, 4, lambda c: "{0x%X, 0x%X, 0x%X, %d}" % c) + "\n};\n")
    f.write("static const CodepointRange kWhitespace[] = {\n")
    f.write(rows(whitespace, 4, lambda r: "{0x%X, 0x%X}" % tuple(r[:2])) + "\n};\n")
for version in VERSIONS:
    print(f"{version}: {sum(d[4] <= version for d in decompositions)} decompositions, "
          f"{sum(s <= version for s in class_since.values())} combining classes, "
          f"{sum(c[3] <= version for c in compositions)} compositions")
print(len(pool), "code points of mappings,", len(combining_ranges), "ranges of classes,", len(whitespace), "of whitespace")
