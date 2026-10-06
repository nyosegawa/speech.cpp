"""Writes tokenizer cases for checks/tokenizer-check from the model's own tokenizer.json.

The encoding cases are one line per text: the text's UTF-8 in hex, a tab, and the ids the tokenizer gives. The
decoding cases are one line per sequence of ids: the ids, a tab, and the UTF-8 in hex of the text the tokenizer decodes
them to, special tokens kept: every prefix and some slices of the ids of the texts, so that characters split between
tokens are cut at either end, the ids of special tokens among text, an id that has no token, and random sequences,
most of which are not UTF-8.

usage: uv run python tokenizer_cases.py <model dir> <cases.tsv> <decode-cases.tsv>
"""

import random
import sys
import unicodedata

from transformers import AutoTokenizer

model_dir, out_path, decode_path = sys.argv[1:4]
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
        t = unicodedata.normalize("NFC", t)
        ids = tok(t, add_special_tokens=False)["input_ids"]
        f.write(t.encode("utf-8").hex() + "\t" + " ".join(map(str, ids)) + "\n")
print(len(texts), "cases ->", out_path)

sequences = []
for t in texts + decode_texts:
    ids = tok(unicodedata.normalize("NFC", t), add_special_tokens=False)["input_ids"]
    sequences += [ids[:k] for k in range(1, len(ids) + 1)]
    sequences += [ids[k:] for k in range(1, len(ids))]
special = tok.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "<tts_pad>"])
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
        text = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        f.write(" ".join(map(str, ids)) + "\t" + text.encode("utf-8").hex() + "\n")
print(len(sequences), "decoding cases ->", decode_path)
