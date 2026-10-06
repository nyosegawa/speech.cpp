# Text is normalized as each reference normalizes it, in its version of Unicode

Decided 2026-10-06.

## Context

Irodori-TTS's runtime brings its text to NFKC with the unicodedata of Python 3.10, whose tables are Unicode 13.0. The
tokenizers of Qwen3-TTS and Qwen3-ASR, which transformers builds from each checkpoint's vocab.json and merges.txt,
bring the text to NFC before they split it; they run in the tokenizers library (0.22.2 and 0.23.2), whose
unicode-normalization-alignments crate 0.1.12 has the tables of Unicode 9.0, and the Python 3.12 of their references
(Unicode 15.0) never sees the text. speech.cpp took NFKC for Irodori-TTS and no normalization for the Qwen
tokenizers, so text not in NFC, such as Japanese from a macOS file name, whose voiced marks stand apart, got other
tokens than the official's.

Run on every code point on 2026-10-06, NFC of a code point alone is the same in all three versions, but the crate
lacks 14 decompositions and the combining classes of 58 marks, all of characters that Unicode 10.0 to 13.0 added, and
texts with them normalize otherwise: U+11935 U+11930 composes to U+11938 in 13.0 and not in 9.0, and a mark of 10.0 or
later is reordered among the marks around it in 13.0, where 9.0 takes it for a starter. The tokenizers library also
finds an added token that is not "normalized", as none of Qwen3-ASR's is, in the text as given, before its normalizer
runs: "<|im_end|>" followed by U+0338 stays the added token and U+0338, which the NFC of the whole would join into
"<|im_end|" and U+226F.

## Decision

- **Each normalization follows its reference in its version of Unicode**: NFKC by the tables of 13.0 for Irodori-TTS,
  and NFC by those of 9.0 for the Qwen2 tokenizer, which brings every text to NFC before it splits it. Qwen3-ASR's
  tokenizer finds its added tokens in the text as given and brings the text between them to NFC.
- **One set of tables holds every version**, in src/common, each entry marked with the first version that has it:
  the Unicode Standard never changes a character's decomposition, combining class or composition once the character
  is assigned, so the tables of 9.0 are those of 13.0 without the characters assigned since. reference/unicode/ writes
  them from Python 3.10's unicodedata and the tokenizers library's normalizers, and checks/unicode-check compares the
  library's NFC and NFKC of both versions with theirs.

The alternatives were turned down:

- One version for every family, 13.0 or the newest: the Qwen tokenizers would give other ids than the official's on
  texts with the characters above.
- A set of tables per version: the same entries twice, but for the characters assigned between the versions.
- The NFC of a whole Qwen3-ASR prompt before its added tokens are found, as transformers' slow Qwen2Tokenizer does in
  `prepare_for_tokenization()`: the references load the fast tokenizer.

## Consequences

Irodori-TTS's normalization gives what it gave before on every code point. A Qwen prompt or text not in NFC gets the
official's ids. A family whose reference normalizes with another version adds that version's marks to the generator,
taken from that reference's own normalizer, and the cases of unicode-check.
