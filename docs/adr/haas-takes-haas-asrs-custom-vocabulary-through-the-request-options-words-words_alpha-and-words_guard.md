# Haas takes Haas-ASR's custom vocabulary through the request options words, words_alpha and words_guard

## Context

Haas-ASR, the inference package of Haas-ja-v1.0's model card, transcribes with a custom vocabulary:
`transcribe(model, audios, batch_size=64, vocabulary=None, guard=True)`, where `Vocabulary(words, alpha=2.0)` maps each
word's written form ("Haas", "Third Intelligence") to its readings, the spellings Haas writes for it ("ハース"), and
`Vocabulary.parse()` reads the form of its examples' `--words`, "written form=reading,reading". With a vocabulary it:

- boosts each word's spellings, each reading as given, in katakana too for a reading in kana, and the written form,
  all in lower case, while decoding: their token sequences, with and without the leading word-start piece that
  SentencePiece gives a text, become a tree of NeMo's phrase boosting (`ContextGraph` with a bonus of 1 a token and a
  depth scaling of 1, read into a `GPUBoostingTreeModel` whose token that starts no word scores 0), and greedy TDT adds
  alpha times the tree's score to each token's logit wherever the decoding emits a token rather than a blank, the tree
  taking back the bonus of a word left unfinished;
- decodes the same encoder output twice, without the words and with them, and keeps the plain text where the audio has
  no 30 ms frame at least 0.012 in RMS (`has_speech()`);
- with `guard`, keeps a boosted change only where it holds a whole listed word, by the opcodes of
  `difflib.SequenceMatcher` between the two texts (`guarded()`);
- writes each word found in the text, in either kana, any width, letter case or spacing, in its written form
  (`Vocabulary.format()`).

NeMo encodes a spelling with SentencePiece, which normalizes the text with the tokenizer's precompiled character map and
merges its pieces by their scores, neither of which a FastConformer file held.

## Decision

- **Three request options, added in 3.7, taken by a model whose file says it offers Haas-ASR's vocabulary**
  (`fastconformer.vocabulary.kind` haas-asr, Haas-ja-v1.0):
  - `words`, a string: the words, one a line, each its written form followed, where it has readings, by "=" and the
    readings separated by commas, as `Vocabulary.parse()` takes each entry; lines with nothing but whitespace are
    skipped, and "" (the neutral value) boosts nothing. A written form that is empty once stripped is refused, as
    `Vocabulary` refuses it.
  - `words_alpha`, a number above 0: `Vocabulary`'s alpha, its default from the file (2).
  - `words_guard`, a boolean: `transcribe()`'s guard, true by default.

  `words_alpha` and `words_guard` without words are refused when the request runs, since they would change nothing.
  `batch_size` is not offered: a model serves one request at a time.
- **A recognition with words decodes the encoder's output twice**, with the request's decoding device, and gives the
  text Haas-ASR gives: the plain decoding's text where `has_speech()` finds no speech, the guarded or the boosted text
  otherwise, with the words in their written forms. Its text is not a decoding's tokens, so a request that sets
  timestamps gets the text alone, and `speech_result_untimed()` says why.
- **Layout 4 holds what the vocabulary needs**: under the kind haas-asr, the tree's bonus, depth scaling and unknown
  score, the default alpha and `has_speech()`'s threshold and frame, each from Haas-ASR's code and NeMo's defaults, and
  the tokenizer's piece scores, precompiled character map and normalizer options, with which the C++ encodes a spelling
  as SentencePiece encodes it for a BPE model. The reader brings an earlier file up with the kind none.
- **The text is compared as Python compares it**: NFC, `str.lower()`, `str.isdigit()` and `str.isspace()` by the
  tables of Python 3.10's `unicodedata` (Unicode 13.0) that the library's other normalizations use. Haas-ASR runs on
  Python 3.11 (Unicode 14.0): a character assigned in 14.0 is handled as unassigned.

The alternatives were turned down:

- The words through `prompt` or `instructions`, or a replacement of the text alone. Neither boosts the tokens while
  decoding, takes back an unfinished word or guards the change, so neither gives Haas-ASR's text.
- An option of a list of words. The C API's options are strings, numbers and booleans; a line a word is the form a
  command line, a JSON message and a form field all carry.
- The vocabulary for parakeet and ReazonSpeech. Haas-ASR's spellings, guard and formatting are its package's, for
  Haas; NeMo's own boosting of those checkpoints takes other settings (a depth scaling of 2, no readings, no guard).
- The boosted decoding's tokens and times with the guarded text. The guard keeps parts of either text, and the
  formatting rewrites them, so neither decoding's tokens make the text.
- The boosted decoding of audio without speech. Its text is not used.

## Consequences

Haas-ja-v1.0's files are converted again in layout 4, which 0.9.4 reads first; 0.9.3's files of it read as before and
take no words. A recognition with words takes the time of a second decoding and of the guard's comparison besides the
encoder's one pass. A word Haas writes in characters outside Unicode 13.0 can be found otherwise than Haas-ASR finds
it.
