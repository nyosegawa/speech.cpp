"""Writes a SentencePiece BPE tokenizer's data and the ids sentencepiece encodes a set of texts to, which
checks/fastconformer/fastconformer-sentencepiece-check encodes with src/families/fastconformer/sentencepiece.cpp and
compares, as NeMo's SentencePieceTokenizer.text_to_ids() encodes the words of a custom vocabulary.

usage: uv run python sentencepiece_cases.py <tokenizer.model> <out dir> [<text file or folder>...]

The tokenizer.model is the one a .nemo checkpoint holds (`tar -xf <model>.nemo --wildcards '*tokenizer.model'`), such
as Haas-ja-v1.0's, and checks/run.py finds the cases in out/<the model's general.name>/sentencepiece/
(out/Haas-ja-v1.0/sentencepiece/). A model of another type than BPE, with pieces of another type than normal, unknown
and byte, or with treat_whitespace_as_suffix is refused, as the port refuses them or does not read the setting. The out
dir gets:

  normalizer.tsv  add_dummy_prefix, remove_extra_whitespaces and escape_whitespaces of the normalizer, one per line: the
                  name, a tab, and 1 or 0
  charsmap.bin    the normalizer's precompiled_charsmap as the model holds it
  pieces.tsv      one line per piece in the order of the ids: its type (the value of ModelProto.SentencePiece.Type), a
                  tab, its score as float.hex() writes it, a tab, and its UTF-8 in hex
  cases.tsv       one line per text: its kind, a tab, its UTF-8 in hex, a tab, the UTF-8 in hex of what the normalizer
                  makes of it (SentencePieceProcessor.normalize()), a tab, and the ids encode() gives it, separated by
                  spaces

and variants/<name>/ the same files for the same texts, from the model with one of its settings changed, which
sentencepiece encodes with the same code: each of the three settings of the normalizer the other way, no byte fallback
(the byte pieces removed), and no charsmap.

The texts are words and phrases in kana, kanji and katakana, ASCII words and names, full-width letters and digits,
half-width katakana, spaces, tabs and newlines at the start, the middle and the end, characters the charsmap rewrites,
characters outside the vocabulary (which byte fallback writes as bytes), combining marks, other scripts, controls, the
empty text, a few long texts, 600 random strings drawn from them, every normal piece with its "▁" as a space, runs of
each unit whose double is a piece (whose overlapping pairs score the same), and the Japanese lines of each text file or
folder given: each line of a .txt or .md file, and each string of a .json or .jsonl file, that holds a kana or a kanji,
such as the texts and words of reference/*/out/'s dumps. NeMo gives sentencepiece a str; texts that are not well-formed
UTF-8 are given as bytes, which sentencepiece takes as well.
"""

import json
import os
import random
import sys

from sentencepiece import SentencePieceProcessor
from sentencepiece import sentencepiece_model_pb2 as pb

Piece = pb.ModelProto.SentencePiece
ALLOWED = (Piece.NORMAL, Piece.UNKNOWN, Piece.BYTE)
# Hiragana, katakana, the CJK ideographs and their extension A, and half-width katakana.
JAPANESE = [(0x3041, 0x3096), (0x30A1, 0x30FA), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xFF66, 0xFF9F)]
IDEOGRAPHIC = "\N{IDEOGRAPHIC SPACE}"
NBSP = "\N{NO-BREAK SPACE}"
ZWJ = "\N{ZERO WIDTH JOINER}"
VS16 = "\N{VARIATION SELECTOR-16}"
ACUTE = "\N{COMBINING ACUTE ACCENT}"
VOICED = "\N{COMBINING KATAKANA-HIRAGANA VOICED SOUND MARK}"
SEMI_VOICED = "\N{COMBINING KATAKANA-HIRAGANA SEMI-VOICED SOUND MARK}"

model_path, out_dir = sys.argv[1:3]
sources = sys.argv[3:]
model = pb.ModelProto()
with open(model_path, "rb") as f:
    model.ParseFromString(f.read())
assert model.trainer_spec.model_type == pb.TrainerSpec.BPE, "the tokenizer is not a BPE model"
assert not model.trainer_spec.treat_whitespace_as_suffix
assert all(p.type in ALLOWED for p in model.pieces), "the tokenizer has pieces other than normal, unknown and byte"
assert model.trainer_spec.byte_fallback == any(p.type == Piece.BYTE for p in model.pieces)

kana = ["ありがとう", "こんにちは", "おはようございます", "すみません", "いただきます", "ごちそうさまでした",
        "よろしくおねがいします", "ちょっと", "きゃりーぱみゅぱみゅ", "ぁぃぅぇぉっゃゅょゎ", "ゔ", "を", "ん", "っ"]
kanji = ["東京", "大阪府", "北海道", "日本語", "音声認識", "機械学習", "人工知能", "自然言語処理", "株式会社", "議事録",
         "会議室", "厚生労働省", "取締役会", "量子計算機", "東京特許許可局", "生麦生米生卵", "今日", "明日", "一昨日", "々",
         "〆切"]
katakana = ["コンピューター", "スマートフォン", "インターネット", "データベース", "アルゴリズム", "ソフトウェア",
            "マイクロソフト", "ヴァイオリン", "ウィキペディア", "フィードバック", "ディープラーニング", "トランスフォーマー",
            "ヱヴァンゲリヲン", "ヶ月", "パーティー", "シェア", "ツイッター", "チェックイン"]
ascii_words = ["hello", "world", "Hello", "WORLD", "Python", "NVIDIA", "OpenAI", "GPU", "API", "iPhone", "macOS",
               "ChatGPT", "speech.cpp", "Haas", "John Smith", "McDonald's", "e-mail", "C++", "x86_64", "README.md",
               "3.14", "100%", "2026", "a", "I", "Q&A", "user@example.com", "https://example.com/path?q=1", "#hashtag",
               "@mention", "<unk>", "<0x41>", "<0xE3><0x81><0x82>", "<s>", "</s>"]
fullwidth = ["ＡＢＣ", "ａｂｃ", "ｐｙｔｈｏｎ", "ＮＨＫ", "１２３", "２０２６年１０月３１日", "Ｗｉ－Ｆｉ", "！？", "＃＄％＆",
             "（括弧）", "［角括弧］", "～", "＝"]
halfwidth = ["ｶﾀｶﾅ", "ﾃﾞｰﾀ", "ｺﾝﾋﾟｭｰﾀｰ", "ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ", "ｱｲｳｴｵ", "ｶﾞｷﾞｸﾞｹﾞｺﾞ", "ｳﾞ", "ﾞ", "ﾟ", "｡｢｣､･"]
spaces = [" ", "  ", "   ", " 東京", "東京 ", " 東京 ", "東京 大阪", "東京  大阪", "東京   大阪", "\t東京", "東京\t",
          "東京\t大阪", "東京\n大阪", "東京\r\n大阪", "\n", "\t", "\r", " \t\n ", IDEOGRAPHIC, "東京" + IDEOGRAPHIC + "大阪",
          IDEOGRAPHIC + "東京" + IDEOGRAPHIC, NBSP, "東京" + NBSP + "大阪", "\N{EN SPACE}\N{EM SPACE}\N{THIN SPACE}",
          "a b", "a  b", " a", "a ", "▁", "▁▁", "▁東京", "東京▁大阪", IDEOGRAPHIC * 2]
charsmap = ["①②③", "⑩⑳", "㈱", "㈲", "㍻", "㌔", "㌢", "㎏", "㎡", "℃", "№", "™", "ﬁ", "ﬂ", "ﬀ", "Ⅳ", "ⅷ", "½", "¼",
            "…", "‥", "〜", "－", "―", "‐", "–", "—", "\N{ANGSTROM SIGN}", "\N{OHM SIGN}", "\N{MICRO SIGN}", "ｍ", "Ⓐ",
            "⒜", "㊤", "㊥", "ㇰ", "ゟ", "ヿ", "︙", "﹁", "﹏", "ﾞﾟ", "\N{SOFT HYPHEN}", "\N{ZERO WIDTH SPACE}",
            "\N{ZERO WIDTH NON-JOINER}", ZWJ, "\N{WORD JOINER}", "\N{ZERO WIDTH NO-BREAK SPACE}", "´", "¨", "ª", "²³",
            "ⁿ", "₁₂", "ǅ", "ǈ", "ĳ", "ŉ", "Ⅻ", "ⅰⅱ", "㋀", "㏠", "㍿", "丨", "⺟", "⼀",
            "\N{CJK COMPATIBILITY IDEOGRAPH-F900}", "\N{CJK COMPATIBILITY IDEOGRAPH-F9DC}", "𝐀𝐁", "𝟏𝟐"]
outside = ["𠮷野家", "鬱", "躑躅", "薔薇", "魑魅魍魎", "𩸽", "龘", "齉", "😀", "👍🏽", "🇯🇵", "👨" + ZWJ + "👩" + ZWJ + "👧",
           "❤" + VS16, "葛\N{VARIATION SELECTOR-17}城", "🀄", "🎌", "♨" + VS16, "〠", "㉑", "龍", "鷗", "髙橋", "﨑", "彅",
           "𠀋"]
combining = ["e" + ACUTE, "か" + VOICED, "は" + SEMI_VOICED, "a\N{COMBINING GRAVE ACCENT}" + ACUTE, ACUTE, VOICED,
             "ア" + VOICED, "ウ" + VOICED, "Z\N{COMBINING SHORT STROKE OVERLAY}\N{COMBINING LONG STROKE OVERLAY}",
             "n\N{COMBINING TILDE}", "o\N{COMBINING DIAERESIS}", "ﾊ" + SEMI_VOICED, "\N{COMBINING ENCLOSING CIRCLE}",
             "1" + VS16 + "\N{COMBINING ENCLOSING KEYCAP}"]
scripts = ["안녕하세요", "你好世界", "Привет", "Γειά σου", "مرحبا", "שלום", "สวัสดี", "नमस्ते", "Ünïcödé café",
           "Ångström", "ß", "İı", "ﬆ"]
controls = ["\x00", "a\x00b", "\x01", "\x07", "\x1b[0m", "\x7f", "\x85", "\N{LINE SEPARATOR}",
            "\N{PARAGRAPH SEPARATOR}", "\U0000e000", "\U000f0000", "\N{REPLACEMENT CHARACTER}", "\U0000fffe",
            "\U0000ffff", "\U0010ffff"]
malformed = [b"\xff", b"\xfe", b"\x80", b"\xc0\xaf", b"\xe3\x81", b"\xe3", b"abc\x80def", b"\xed\xa0\x80",
             b"\xf4\x90\x80\x80", b"\xf8\x88\x80\x80\x80", b"\xe3\x81\x82\xe3", "東京".encode() + b"\x81" + "大阪".encode(),
             b"\xc3", b" \xff ", b"\xef\xbf\xbd", b"\xe0\x80\x80", b"\xf0\x80\x80\x80"]

pools = {"kana": kana, "kanji": kanji, "katakana": katakana, "ascii": ascii_words, "fullwidth": fullwidth,
         "halfwidth": halfwidth, "spaces": spaces, "charsmap": charsmap, "outside": outside, "combining": combining,
         "scripts": scripts, "controls": controls}
cases = [(kind, text) for kind, pool in pools.items() for text in pool]
cases += [("malformed", text) for text in malformed]
cases.append(("empty", ""))
phrases = ["東京都の会議室で、機械学習の議事録を作成しました。", "明日の天気は晴れ、最高気温は２４度です。",
           "スマートフォンのアプリでＡＩと話す", "NVIDIAのGPUでspeech.cppを動かす", "ｶﾀｶﾅとカタカナと片仮名",
           "①番線に電車が参ります", "𠮷野家で牛丼を食べた😀", "  先頭と  中間と末尾の空白  "]
cases += [("phrases", text) for text in phrases]
cases += [("long", "".join(phrases) * 40), ("long", "あ" * 3000), ("long", "ー" * 500), ("long", " " * 200),
          ("long", "東京 " * 300), ("long", "😀" * 200), ("long", "ab" * 1000)]

# Random strings: words of the pools and random code points of the ranges below, joined by nothing or by spaces, tabs
# and newlines, with spaces at either end now and then.
ranges = [(0x20, 0x7E), (0x3041, 0x3096), (0x30A1, 0x30FC), (0x4E00, 0x9FFF), (0xFF01, 0xFF5E), (0xFF61, 0xFF9F),
          (0x0300, 0x036F), (0x3099, 0x309C), (0x1F300, 0x1F64F), (0x2460, 0x24FF), (0x3300, 0x33FF), (0xAC00, 0xD7A3),
          (0x00A0, 0x00FF), (0x2000, 0x206F), (0x20000, 0x2A6DF), (0x01, 0x1F), (0xFE00, 0xFE0F), (0x2581, 0x2581)]
separators = ["", "", "", " ", "  ", "\t", "\n", IDEOGRAPHIC, " \t "]
words = [w for pool in pools.values() for w in pool]
rng = random.Random(0)
for _ in range(600):
    parts = []
    for _ in range(rng.randint(1, 6)):
        if rng.random() < 0.6:
            parts.append(rng.choice(words))
        else:
            lo, hi = rng.choice(ranges)
            parts.append("".join(chr(rng.randint(lo, hi)) for _ in range(rng.randint(1, 5))))
    text = "".join(p + rng.choice(separators) for p in parts)
    if rng.random() < 0.2:
        text = " " * rng.randint(1, 3) + text
    cases.append(("random", text))

normal = [p.piece.replace("▁", " ") for p in model.pieces if p.type == Piece.NORMAL]
cases += [("pieces", p) for p in normal]
# Runs of a unit whose double is a piece, such as Haas's "いい" and "00": in a run of three, two pairs of the same score
# overlap, and the leftmost merges first.
doubled = sorted({p[:len(p) // 2] for p in normal if len(p) % 2 == 0 and p[:len(p) // 2] == p[len(p) // 2:]})
cases += [("runs", unit * n) for unit in doubled + ["あ", "ー", "w"] for n in range(2, 8)]
cases += [("runs", "東京" + unit * 3 + "です") for unit in doubled]


def japanese(text):
    """Whether the text holds a kana or a kanji."""
    return any(lo <= ord(c) <= hi for c in text for lo, hi in JAPANESE)


def strings(value):
    """Every string of a JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from strings(v)


def texts_of(path):
    """The lines of a .txt or .md file, or the strings of a .json or .jsonl file."""
    with open(path, encoding="utf-8") as f:
        if path.endswith((".txt", ".md")):
            return f.read().splitlines()
        if path.endswith(".jsonl"):
            return [s for line in f if line.strip() for s in strings(json.loads(line))]
        return list(strings(json.load(f)))


files = []
for source in sources:
    if os.path.isfile(source):
        files.append(source)
        continue
    for folder, dirs, names in os.walk(source):
        dirs.sort()
        files += [os.path.join(folder, n) for n in sorted(names) if n.endswith((".txt", ".md", ".json", ".jsonl"))]
seen = {text for _, text in cases}
for path in files:
    for text in texts_of(path):
        if japanese(text) and text not in seen:
            seen.add(text)
            cases.append(("corpus", text))


def write(model, folder):
    """Writes the model's data and its encoding of every case into the folder."""
    os.makedirs(folder, exist_ok=True)
    sp = SentencePieceProcessor()
    sp.LoadFromSerializedProto(model.SerializeToString())
    ns = model.normalizer_spec
    with open(os.path.join(folder, "normalizer.tsv"), "w") as f:
        for name in ("add_dummy_prefix", "remove_extra_whitespaces", "escape_whitespaces"):
            f.write(f"{name}\t{int(getattr(ns, name))}\n")
    with open(os.path.join(folder, "charsmap.bin"), "wb") as f:
        f.write(ns.precompiled_charsmap)
    with open(os.path.join(folder, "pieces.tsv"), "w") as f:
        for p in model.pieces:
            f.write(f"{p.type}\t{p.score.hex()}\t{p.piece.encode('utf-8').hex()}\n")
    with open(os.path.join(folder, "cases.tsv"), "w") as f:
        for kind, text in cases:
            data = text if isinstance(text, bytes) else text.encode("utf-8")
            ids = sp.encode(text, out_type=int)
            normalized = sp.normalize(data)
            if not isinstance(text, bytes):
                assert sp.encode(data, out_type=int) == ids
            f.write(f"{kind}\t{data.hex()}\t{normalized.hex()}\t{' '.join(map(str, ids))}\n")
    print(len(cases), "cases ->", folder)


write(model, out_dir)


def variant(change):
    v = pb.ModelProto()
    v.CopyFrom(model)
    change(v)
    return v


def no_byte_fallback(v):
    kept = [p for p in v.pieces if p.type != Piece.BYTE]
    del v.pieces[:]
    v.pieces.extend(kept)
    v.trainer_spec.byte_fallback = False


ns = model.normalizer_spec
variants = {
    "dummy-prefix": lambda v: setattr(v.normalizer_spec, "add_dummy_prefix", not ns.add_dummy_prefix),
    "extra-whitespaces": lambda v: setattr(v.normalizer_spec, "remove_extra_whitespaces",
                                           not ns.remove_extra_whitespaces),
    "escape-whitespaces": lambda v: setattr(v.normalizer_spec, "escape_whitespaces", not ns.escape_whitespaces),
    "no-charsmap": lambda v: v.normalizer_spec.ClearField("precompiled_charsmap"),
}
if model.trainer_spec.byte_fallback:
    variants["no-byte-fallback"] = no_byte_fallback
for name, change in variants.items():
    write(variant(change), os.path.join(out_dir, "variants", name))
