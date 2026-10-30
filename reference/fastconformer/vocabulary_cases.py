"""Writes <out dir>/vocabulary-cases.json, the cases checks/fastconformer/fastconformer-vocabulary-check compares the
port of Haas-ASR's custom vocabulary (src/families/fastconformer/vocabulary.cpp and guard.cpp) with: what Haas-ASR's own
Vocabulary, spellings(), guarded() and has_speech() give.

usage: uv run python vocabulary_cases.py <Haas-ASR checkout> <out dir>

The checkout is of the commit haas/pyproject.toml pins. Its src/haas_asr/vocabulary.py and boosting.py are imported by
path, with audio.py, which boosting.py imports, and without the package's __init__.py, which imports NeMo; boosting.py
imports NeMo only inside functions this script does not call. The texts hold characters of Unicode 13.0 alone, none of
those whose case or digits Python 3.10, whose tables the port has, and 3.11, which runs this, tell apart.

The file holds one JSON object:

  frame, rms     has_speech()'s frame and loudness, SPEECH_FRAME and SPEECH_RMS
  vocabularies   entries as Vocabulary.parse() reads them and an alpha, the words, readings and spellings of the
                 Vocabulary they make, and transcripts with what its find() and format() give of each
  refused        entries Vocabulary refuses with a ValueError
  spellings      a written form and readings, and spellings() of them
  guarded        a vocabulary (its index in vocabularies), texts decoded without and with it, the opcodes of
                 difflib.SequenceMatcher(None, plain, boosted, autojunk=False) and guarded() of them
  has_speech     samples (float32, little-endian, in hex), a loudness and a frame, and has_speech() of them

The cases are those of Haas-ASR's tests/test_vocabulary.py, asserted to give the tests' values; vocabularies of
Japanese names and terms, ASCII, full-width and spaced words, mixed scripts, Greek, U+0130 and combining marks, with
duplicates, readings that are the written form and empty readings, and transcripts that write their words in the other
kana, width or case and with spaces; pairs of random texts over small alphabets, for SequenceMatcher's ties and long
equal runs; and signals whose loudest frame is just at or just under the loudness, of lengths around a frame's end.
"""

import difflib
import importlib.util
import json
import os
import random
import sys
import types

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


src = os.path.join(sys.argv[1], "src", "haas_asr")
# boosting.py imports haas_asr.audio and haas_asr.vocabulary, which are found under these names beneath a package
# that runs no code of its own.
sys.modules["haas_asr"] = types.ModuleType("haas_asr")
load("haas_asr.audio", os.path.join(src, "audio.py"))
vocabulary = load("haas_asr.vocabulary", os.path.join(src, "vocabulary.py"))
boosting = load("haas_asr.boosting", os.path.join(src, "boosting.py"))
Vocabulary = vocabulary.Vocabulary

rng = random.Random(0)
vocabularies, refused, spellings, guarded, speech = [], [], [], [], []


def variant(s):
    """`s` as the model might write it: each character in the other kana, width or case at random, and spaces between
    some."""
    out = []
    for c in s:
        o, r = ord(c), rng.random()
        if 0x30A1 <= o <= 0x30F6 and r < 0.4:
            c = chr(o - 0x60)
        elif 0x3041 <= o <= 0x3096 and r < 0.4:
            c = chr(o + 0x60)
        elif 0x21 <= o <= 0x7E and r < 0.3:
            c = chr(o + 0xFEE0)
        elif 0xFF01 <= o <= 0xFF5E and r < 0.4:
            c = chr(o - 0xFEE0)
        if rng.random() < 0.3:
            flipped = c.upper() if c.islower() else c.lower()
            if len(flipped) == 1:
                c = flipped
        out.append(c)
        if rng.random() < 0.1:
            out.append(rng.choice([" ", "\u3000", "  ", "\t"]))
    return "".join(out)


FILLERS = ["です", "の", "、", "。", " ", "さん", "は", "a", "the ", "1", "x", "ー", "ス", "ハ", "AI", "ai", "Σ",
           "ς", "'", ".", "-", "\u0301", "İ", "i", "\U00020bb7", "ﾊ", "ａ", "\u3000", "z", "Z", "aa",
           "々", "\u3099", "Ｓ"]


def transcripts(v, count):
    pieces = [s for text, readings in v.words.items() for s in (text, *readings)] + list(v.spellings)
    out = []
    for _ in range(count):
        parts = [variant(rng.choice(pieces)) if pieces and rng.random() < 0.6 else rng.choice(FILLERS)
                 for _ in range(rng.randint(1, 5))]
        out.append(rng.choice(["", "", " ", "、"]).join(parts))
    return out


def add_vocabulary(entries, texts=(), alpha=2.0, random_texts=30):
    v = Vocabulary.parse(entries, alpha=alpha)
    texts = list(texts) + transcripts(v, random_texts)
    vocabularies.append({
        "entries": entries,
        "alpha": alpha,
        "words": [[text, list(readings)] for text, readings in v.words.items()],
        "spellings": list(v.spellings),
        "transcripts": [{"text": t, "find": [list(f) for f in v.find(t)], "format": v.format(t)} for t in texts],
    })
    return len(vocabularies) - 1, v


def add_refused(entries):
    try:
        Vocabulary.parse(entries)
    except ValueError:
        refused.append(entries)
        return
    raise AssertionError(f"{entries} is not refused")


def add_guarded(index, v, plain, boosted):
    opcodes = [list(op) for op in difflib.SequenceMatcher(None, plain, boosted, autojunk=False).get_opcodes()]
    out = boosting.guarded(plain, boosted, v)
    guarded.append({"vocabulary": index, "plain": plain, "boosted": boosted, "opcodes": opcodes, "guarded": out})
    return out


def add_speech(samples, rms, frame):
    samples = np.asarray(samples, np.float32)
    saved = boosting.SPEECH_FRAME
    boosting.SPEECH_FRAME = frame
    try:
        out = boosting.has_speech(samples, rms)
    finally:
        boosting.SPEECH_FRAME = saved
    speech.append({"samples": samples.astype("<f4").tobytes().hex(), "rms": rms, "frame": frame, "speech": out})
    return out


# tests/test_vocabulary.py.
WORDS = ["Haas=ハース", "有難う=ありがとう", "AIモデル"]
_, v = add_vocabulary([" Haas =ハース, ハース,Haas,", "有難う=ありがとう"])
assert v.words == Vocabulary({" Haas ": ["ハース", " ハース", "Haas", ""], "有難う": "ありがとう"}).words
assert v.words == {"Haas": ("ハース",), "有難う": ("ありがとう",)}
_, v = add_vocabulary([])
assert not v and not Vocabulary({})
add_refused([" "])
_, v = add_vocabulary(["Haas=ハース,はーす", "AIモデル", "Haas=ハウス"], alpha=3.0)
assert v.words == {"Haas": ("ハース", "はーす", "ハウス"), "AIモデル": ()} and v.alpha == 3.0
spellings.append({"text": "有難う", "readings": ["ありがとう"], "spellings": vocabulary.spellings("有難う", ["ありがとう"])})
assert spellings[-1]["spellings"] == ["ありがとう", "アリガトウ", "有難う"]
_, v = add_vocabulary(["Haas=ハース"])
assert v.spellings == ("ハース", "haas")
words_index, words = add_vocabulary(WORDS, ["ＨＡＡＳ、アリガトウ", "ai モデル", "haasten", "𠮷ハース", "はーすさん、ありがとう。ai モデルです"])
assert words.find("ＨＡＡＳ、アリガトウ") == [(0, 4, "Haas"), (5, 10, "有難う")]
assert words.find("ai モデル") == [(0, 6, "AIモデル")]
assert words.find("haasten") == []
assert words.find("𠮷ハース") == [(1, 4, "Haas")]
assert words.format("はーすさん、ありがとう。ai モデルです") == "Haasさん、有難う。AIモデルです"
_, v = add_vocabulary(["Haas=ハース", "Haas-ja-v1.0=ハースjaバージョン1.0"], ["ハースjaバージョン1.0とハース"])
assert v.format("ハースjaバージョン1.0とハース") == "Haas-ja-v1.0とHaas"
haas_index, haas = add_vocabulary(["Haas=ハース"], ["はすです", "ハースです", "ハスの最新", "そうハー"])
for plain, boosted, want in (("はすです", "ハースです", "ハースです"), ("ハスの最新", "ハース最新", "ハース最新"),
                             ("そうですね", "そうハー", "そうですね"), ("ハスのデモです", "ハースです", "ハースのデモです")):
    assert add_guarded(haas_index, haas, plain, boosted) == want
assert not add_speech(np.full(16000, 0.005, np.float32), boosting.SPEECH_RMS, boosting.SPEECH_FRAME)
loud = np.zeros(16000, np.float32)
loud[8000:8480] = 0.1
assert add_speech(loud, boosting.SPEECH_RMS, boosting.SPEECH_FRAME)

# Vocabularies of every kind of word.
add_vocabulary(["逆瀬川=さかせがわ,サカセガワ", "東京都=とうきょうと", "佐々木=ささき", "大和=やまと", "有明=ありあけ,アリアケ",
                "スマートフォン=すまほ", "ヴァイオリン=ゔぁいおりん,ばいおりん", "ヶ月=かげつ,ゖげつ", "音声認識=おんせいにんしき",
                "ゟ=より", "゠=だぶるはいふん"],
               ["さかせがわさんはとうきょうとにすんでいます", "ササキさん、ヤマトの有明", "ばいおりんとゔぁいおりん", "3かげつ"])
add_vocabulary(["ChatGPT=チャットジーピーティー", "Claude Code=クロードコード,くろーど こーど", "Third Intelligence=サードインテリジェンス",
                "speech.cpp=スピーチシーピーピー", "ＡＩ=エーアイ", "Ｆｕｌｌ ｗｉｄｔｈ", "New  York=ニュー ヨーク,にゅー\u3000よーく",
                "GPT-4o=ジーピーティーフォーオー", "iPhone 17=アイフォーン17,アイフォーン１７", "AI", "MAIN", "v1.0=ブイワンテンゼロ",
                "C++=シープラプラ", "R&D"],
               ["chatgpt と claude code", "ChatGPTs", "aimain", "AI MAIN", "ai2", "the AI's", "v 1 . 0", "speech . cpp",
                "full width", "ｆｕｌｌｗｉｄｔｈ", "new york", "newyork", "iphone17", "gpt - 4 o", "c + +", "r&d", "R & D"])
add_vocabulary(["ΣΟΦΙΑ=ソフィア", "Σίσυφος=シーシュポス", "ΟΔΥΣΣΕΥΣ", "ΑΣ.Β", "ΑΣ'", "ΑΣ\u0301", "Σ", "ΣΑ", "ΑΣΑ",
                "Straße=シュトラーセ", "ẞ", "Ǆemal", "ǅ=どぜ", "Ꭰ Cherokee=チェロキー", "Kelvin=ケルビン",
                "Ångström=オングストローム", "Ωmega", "ı", "ﬃ", "ŉ", "ΑΣ:Β", "AΣ\u00ad"],
               ["σοφια", "ΣΟΦΙΑ", "οδυσσευς", "οδυσσευσ", "ας.β", "ασ.β", "kelvin", "KELVIN", "straße", "STRASSE", "ß",
                "ǆemal", "ǅ", "ꭰ cherokee", "ångström", "ωmega", "ΩMEGA"])
add_vocabulary(["İstanbul=イスタンブール", "İ", "Dİ=ディー", "KİM", "i\u0307stanbul"],
               ["İstanbul", "i\u0307stanbul", "istanbul", "İSTANBUL", "dİ", "Dİx", "aİ", "kim", "KİM", "kİm"])
add_vocabulary(["cafe\u0301=カフェ", "café=キャフェ", "Pokémon=ポケモン", "ハ\u3099ス=ばす", "e\u0301\u0301", "a\u0308\u0304",
                "か\u3099っこう=学校", "\u0301", "Ａ\u030a"],
               ["café", "cafe\u0301", "バス", "ハ\u3099ス", "がっこう", "pokemon", "pokémon", "ǟ", "Å"])
add_vocabulary(["Haas=ハース", "Haas=ハース", " Haas=ハウス,Haas, Haas ", "Haas=", "AI=,,", "x=, ,\u3000", "東京=東京,とうきょう,東京",
                "a=b=c", "Haas=はーす", "y=,", "z=z", "q=Q"],
               ["ハウス", "はーす", "b=c", "とうきょう", "Q", "q", "x", "y z"])
add_vocabulary(["2026年=にせんにじゅうろくねん", "第3回=だいさんかい", "①番=いちばん", "x²=エックスじじょう", "٣=さん",
                "Ｗｉ－Ｆｉ=ワイファイ", "COVID-19=コビッドナインティーン", "ハースjaバージョン1.0", "5G=ファイブジー", "１０月"],
               ["２０２６年", "2026 年", "第 3 回", "① 番", "x ²", "wi-fi", "Wi - Fi", "covid 19", "covid-19", "5g", "5 G",
                "10月", "１０ 月"])
add_vocabulary(["Haas=ハース", "ハウス=ハース", "House=ハウス"], ["ハース", "ハウス", "house", "haas"])
add_vocabulary(["AI=エーアイ", "AIモデル=エーアイモデル", "モデル"], ["エーアイモデル", "エーアイ モデル", "ai", "モデル", "aiモデルモデル"])
add_vocabulary(["ﾊｰｽ=ハース", "ｱｲ", "ﾃﾞｰﾀ=でーた"], ["ﾊｰｽ", "ハース", "ｱｲ", "ﾃﾞｰﾀ", "データ"])
add_vocabulary(["Haas=ハ ース,ハ\tース,ハ\u3000ース", "Big Data=ビッグ データ", "a b c", "東 京"],
               ["ハース", "ハ ース", "big data", "bigdata", "abc", "a  b\u3000c", "東京", "東\u3000京"])
# Each of every vocabulary above alone, so that a word is found without the others.
for entries in [e for case in list(vocabularies) for e in case["entries"]]:
    add_vocabulary([entries], random_texts=4)
add_refused(["=ハース"])
add_refused(["Haas=ハース", "\u3000=x"])
add_refused(["\t\n"])
add_refused(["Haas", "="])

for text, readings in (("Haas", ["ハース", "はーす", "ハース"]), ("ΣΟΦΙΑ", ["ΣΟΦΙΑΣ", "ΑΣ ΑΣ"]), ("İ", ["İİ", "i"]),
                       ("x", []), ("ゔ", ["ゔぁ", "ゕゖ", "ゟ"]), ("ア", ["あ", "ア"]), ("Ａ", ["ａ", "A", "a"]),
                       ("東京", ["とうきょう", "トウキョウ", "とうキョウ", "とう きょう"])):
    spellings.append({"text": text, "readings": readings, "spellings": vocabulary.spellings(text, readings)})

# Plain and boosted texts: random ones over small alphabets, the boosted often the plain edited, and long ones.
ALPHABETS = [("ab", ["ab=ba"]), ("abc", ["ab", "c=cc"]), ("ハース", ["Haas=ハース"]), ("ハスのー", ["Haas=ハース", "ノハ=のは"]),
             ("ab ハ", ["aハ=ハa", "b"]), ("アイウ", ["イ", "アウ=ウア"]), ("aab", ["aab"]), ("あア ", ["ア=あ"])]
indexed = [(alphabet, *add_vocabulary(entries, random_texts=5)) for alphabet, entries in ALPHABETS]


def random_text(alphabet, n):
    return "".join(rng.choice(alphabet) for _ in range(n))


def edited(text, alphabet, words, edits):
    for _ in range(edits):
        at = rng.randint(0, len(text))
        piece = rng.choice(words) if words and rng.random() < 0.3 else random_text(alphabet, rng.randint(1, 3))
        kind = rng.random()
        if kind < 0.4:
            text = text[:at] + piece + text[at:]
        elif kind < 0.7:
            text = text[:at] + text[at + rng.randint(1, 3):]
        else:
            text = text[:at] + piece + text[at + rng.randint(1, 3):]
    return text


for _ in range(800):
    alphabet, index, v = rng.choice(indexed)
    plain = random_text(alphabet, rng.randint(0, 20))
    if rng.random() < 0.4:
        boosted = random_text(alphabet, rng.randint(0, 20))
    else:
        boosted = edited(plain, alphabet, list(v.spellings), rng.randint(1, 4))
    add_guarded(index, v, plain, boosted)
for _ in range(60):
    alphabet, index, v = rng.choice(indexed)
    plain = random_text(alphabet, rng.randint(100, 300))
    add_guarded(index, v, plain, edited(plain, alphabet, list(v.spellings), rng.randint(3, 10)))
for plain, boosted in (("きょうはいいてんきですね", "きょうはハースいいてんきですね"), ("ai モデルを使う", "AIモデルを使う"),
                       ("はすさん、ありがとう", "ハースさん、アリガトウ"), ("ありがと", "ありがとう"), ("", "ハース"), ("ハース", ""),
                       ("はーす", "ハース"), ("あいもでる", "AIモデル"), ("haas", "ハース"), ("ハスのハスの", "ハースハース"),
                       ("ハスのです", "ハースのです"), ("のハス", "ハース"), ("ハ", "ハース"), ("ハースの", "ハース")):
    add_guarded(words_index, words, plain, boosted)

# Signals whose loudest frame is just at, and just under, the loudness they are given, for frames of every size NumPy's
# pairwise sum treats apart, and lengths around a frame's end.
gen = np.random.default_rng(0)


def at_the_edge(samples, frame):
    padded = np.pad(samples, (0, -samples.size % frame))
    loudest = np.sqrt((padded.reshape(-1, frame) ** 2).mean(axis=1)).max()
    assert add_speech(samples, float(loudest), frame)
    assert not add_speech(samples, float(np.nextafter(loudest, np.float32(np.inf))), frame)


for frame in (1, 3, 7, 8, 9, 16, 100, 128, 129, 136, 255, 256, 480, 481, 1000, 1024):
    for n in (1, frame - 1, frame, frame + 1, 3 * frame + 5):
        if n > 0:
            at_the_edge((gen.standard_normal(n) * gen.uniform(0.001, 0.1)).astype(np.float32), frame)
for n in (16000, 16001, 32000):
    samples = (gen.standard_normal(n) * 0.004).astype(np.float32)
    samples[gen.integers(0, n - 480):][:480] *= np.float32(3)
    at_the_edge(samples, boosting.SPEECH_FRAME)
rms, frame = boosting.SPEECH_RMS, boosting.SPEECH_FRAME
threshold = np.float32(rms)
for amplitude in (0.0119, np.nextafter(threshold, np.float32(0)), threshold, np.nextafter(threshold, np.float32(1)), 0.0121, 0.5):
    for n in (1, 479, 480, 481, 960, 1000):
        add_speech(np.full(n, amplitude, np.float32), rms, frame)
for n in (481, 959, 961, 1439, 16001):
    for loud in (1, 2, 10):
        for scale in (1 - 1e-7, 1, 1 + 1e-7):
            samples = np.zeros(n, np.float32)
            samples[n - loud:] = np.float32(rms * np.sqrt(frame / loud) * scale)
            add_speech(samples, rms, frame)
add_speech(np.zeros(0, np.float32), rms, frame)
add_speech(np.zeros(100, np.float32), 0.0, frame)
add_speech(np.zeros(100, np.float32), rms, frame)
add_speech(np.array([np.nan] * 480 + [0.0] * 480, np.float32), rms, frame)
add_speech(np.array([np.nan] * 480 + [0.5] * 480, np.float32), rms, frame)
add_speech(np.array([np.inf] + [0.0] * 479, np.float32), rms, frame)

os.makedirs(sys.argv[2], exist_ok=True)
path = os.path.join(sys.argv[2], "vocabulary-cases.json")
cases = {"frame": boosting.SPEECH_FRAME, "rms": boosting.SPEECH_RMS, "vocabularies": vocabularies, "refused": refused,
         "spellings": spellings, "guarded": guarded, "has_speech": speech}
with open(path, "w", encoding="utf-8") as f:
    json.dump(cases, f, ensure_ascii=False)
print(len(vocabularies), "vocabularies with", sum(len(v["transcripts"]) for v in vocabularies), "transcripts,", len(refused),
      "refused,", len(spellings), "spellings,", len(guarded), "guarded,", len(speech), "has_speech ->", path)
