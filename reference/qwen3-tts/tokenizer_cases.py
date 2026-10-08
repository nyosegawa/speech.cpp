"""Writes tokenizer cases for checks/common/tokenizer-check from the model's own tokenizer.json, which transformers
builds from a Qwen3-TTS or Qwen3-ASR checkpoint.

The encoding cases are one line per text: the text's UTF-8 in hex, a tab, and the ids the tokenizer gives the text as
it is, which its normalizer brings to NFC first. The decoding cases are one line per sequence of ids: the ids, a tab,
and the UTF-8 in hex of the text the tokenizer decodes them to: every prefix and some slices of the ids of the texts, so
that characters split between tokens are cut at either end, the ids of special tokens among text, an id that has no
token, and random sequences, most of which are not UTF-8. For Qwen3-TTS, whose text speech.cpp tokenizes without its
added tokens, decoding keeps the special tokens. For Qwen3-ASR, whose tokenizer splits a text at its added tokens
first, the texts also hold added tokens, and decoding drops the special tokens, as the family's tokenizer does.

usage: uv run python tokenizer_cases.py <Qwen3-TTS checkpoint dir> <cases.tsv> <decode-cases.tsv>
  or, in reference/qwen3-asr, whose environment has the transformers and tokenizers of Qwen3-ASR's reference:
       uv run python ../qwen3-tts/tokenizer_cases.py <Qwen3-ASR checkpoint dir> <cases.tsv> <decode-cases.tsv>
"""

import json
import os
import random
import sys
import unicodedata

from transformers import AutoTokenizer

model_dir, out_path, decode_path = sys.argv[1:4]
family = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))["model_type"]
assert family in ("qwen3_tts", "qwen3_asr"), family
recognition = family == "qwen3_asr"
# Loaded without fix_mistral_regex, so the pre-tokenizer is the one in tokenizer.json.
tok = AutoTokenizer.from_pretrained(model_dir)

texts = [
    "明日の東京は晴れで、最高気温は二十四度の予報です。",
    "ドル円は今、百四十八円二十銭前後で、昨日より少し円安に動いています。",
    "GitHubのプルリクエストのCIが落ちているか確認して",
    "ASISTの設定でVOICEVOXの話者を変えたい",
    "It's 3:45 PM, and I'll be there in 10 minutes—don't worry!",
    "WE'RE HERE. They've gone. I'd say it's fine; she'll see.",
    "https://asist-agent.com/en/privacy/ and C:\\Users\\name\\file.txt",
    "  leading spaces,   inner   spaces, and trailing spaces   ",
    "line one\nline two\r\n\nline four\n",
    "tabs\tand\t\tmore\t",
    "数字123と４５６、1,234.56円、2026年9月29日",
    "絵文字😀👍🏽と記号★☆※、「かぎかっこ」『二重』",
    "Ünïcödé café naïve résumé — Ελληνικά Русский 한국어 中文",
    "e\u0301 combining and ｶﾀｶﾅ half-width",
    "a",
    " ",
    "\n",
    "!!!???...",
    "x'y 'tis O'Neil ' spaced ' quote",
]
# Texts that are not in NFC: Japanese in the NFD that macOS gives file names, its voiced marks apart, Latin with its
# accents apart and two of them out of canonical order, Hangul as conjoining jamo, singletons and an excluded
# composition that NFC replaces, '>' and '=' with U+0338, which NFC joins, and a mark of Unicode 10.0 (U+1DF8) and a
# pair (U+11935 U+11930) of 13.0 that the tokenizers library's Unicode 9.0 neither reorders nor composes.
texts += [
    unicodedata.normalize("NFD", text) for text in
    ["ダウンロードしたファイル「ガイド_パンフレット.pdf」", "がぎぐげご ぱぴぷぺぽ ヴォイス", "Café naïve résumé Ångström",
     "한국어 텍스트입니다"]
]
texts += [
    "a" + chr(0x301) + chr(0x323) + " q" + chr(0x307) + chr(0x323),
    chr(0x212B) + chr(0x2126) + " " + chr(0xF900) + chr(0xF9DC) + " " + chr(0x958) + chr(0x95C),
    "a>" + chr(0x338) + "b =" + chr(0x338) + " c",
    "a" + chr(0x1DF8) + chr(0x301) + " " + chr(0x11935) + chr(0x11930),
]
if recognition:
    # Texts with added tokens, which the tokenizers library finds in the text as it is given, before its normalizer:
    # the prompt qwen-asr writes, with a context not in NFC, and the '>' of an added token followed by U+0338.
    texts += [
        "<|im_start|>system\n" + unicodedata.normalize("NFD", "ガイドのパンフレット、ヴォイス") + "<|im_end|>\n" +
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n" +
        "language Japanese<asr_text>",
        "<|im_end|>" + chr(0x338) + "<asr_text>" + chr(0x338) + "x>" + chr(0x338),
    ]
# Texts that only the decoding cases use: characters of three and four bytes that the vocabulary splits between
# tokens.
decode_texts = [
    "今天北京的天气很好，最高气温二十四度，适合出去散步。",
    "龘靐齉爩鱻麤，𠮷野家の𩸽と𠀋",
    "👨‍👩‍👧‍👦🏳️‍🌈🇯🇵🇺🇸 family, flags and 🧑🏿‍💻",
    "สวัสดีครับ नमस्ते مرحبا Привет 안녕하세요",
    "混在 mixed テキスト with 中文 and 한국어 😀!",
]

with open(out_path, "w") as f:
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        f.write(t.encode("utf-8").hex() + "\t" + " ".join(map(str, ids)) + "\n")
print(len(texts), "cases ->", out_path)

sequences = []
for t in texts + decode_texts:
    ids = tok(t, add_special_tokens=False)["input_ids"]
    sequences += [ids[:k] for k in range(1, len(ids) + 1)]
    sequences += [ids[k:] for k in range(1, len(ids))]
# Special tokens, but for the fourth, which is not.
special = tok.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<asr_text>", "<|audio_pad|>"]
                                    if recognition else
                                    ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "<tts_pad>"])
body = tok("お天気です", add_special_tokens=False)["input_ids"]
sequences += [
    [special[0]] + tok("assistant\n", add_special_tokens=False)["input_ids"] + body + [special[1]],
    special,
    body[:1] + [special[3]] + body[1:],
    body + [len(tok)] + body,
    [len(tok) + 100],
]
rng = random.Random(0)
for _ in range(300):
    sequences.append([rng.randrange(len(tok)) for _ in range(rng.randint(1, 12))])
with open(decode_path, "w") as f:
    for ids in sequences:
        text = tok.decode(ids, skip_special_tokens=recognition, clean_up_tokenization_spaces=False)
        f.write(" ".join(map(str, ids)) + "\t" + text.encode("utf-8").hex() + "\n")
print(len(sequences), "decoding cases ->", decode_path)
