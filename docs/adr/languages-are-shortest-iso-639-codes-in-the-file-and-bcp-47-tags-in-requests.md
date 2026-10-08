# Languages are shortest ISO 639 codes in the file and BCP 47 tags in requests

## Context

The official implementations name languages each their own way: Irodori-TTS by BCP 47 tags (`ja`), Qwen3-TTS by its
checkpoint's names (`japanese`), with `beijing_dialect` and `sichuan_dialect` listed beside the languages although only
two of its speakers speak them, and Qwen3-ASR by names it writes before its text (`Japanese`). A caller that had to name
a language as each family lists it would have to know which family it talks to.

The GGUF specification has `general.languages` and asks for ISO 639-1 codes of two letters in it. Qwen3-ASR recognizes
Cantonese and Filipino, which have no such code. BCP 47 names a language by its shortest ISO 639 code: two letters where
ISO 639-1 has one, and otherwise three, from ISO 639-2 or 639-3 (`yue`, `fil`).

OpenAI's clients send `language` as a hint, also to a recognizer of one language.

## Decision

- **The file holds the languages in `general.languages` alone**, each as its shortest ISO 639 code, sorted. The reader
  takes two or three lowercase letters. Where a family needs its checkpoint's own ids or names, a key of the family maps
  them from these codes, in the same order: Qwen3-TTS's `qwen3-tts.language_ids` and Qwen3-ASR's
  `qwen3-asr.language_names`, which each converter takes from one table.
- **A request's `language` is a BCP 47 tag** that names one of the model's languages, by itself or with a region or
  script (`ja-JP`, `zh-Hant`), compared without case, or `auto`, which leaves the language to the model. Anything else
  is `out_of_range`.
- **A model of detection has no languages**: its file has neither `general.languages` nor `speech.language_use`, and
  it takes the language `auto` alone ([the record of detection](detecting-speech-is-a-task-of-its-own-run-by-speech-detect-with-silero-vads-options-and-f32-files.md)).
- **A language steers the model or is only checked**, as the file's `speech.language_use` and the model information
  say. Qwen3-TTS and Qwen3-ASR take it as an input; Irodori-TTS, which has one language, and FastConformer, whose models
  take no language, check it against their languages and do not use it.
- **Qwen3-TTS's dialects get no tag.** As in the official implementation, `dylan` and `eric` speak theirs when the
  language is `zh` or left to the model (`qwen3-tts.dialect_ids` and `qwen3-tts.dialect_language`).
- **A result gives the languages Qwen3-ASR heard as tags of `general.languages`**
  ([the record of recognition results](a-recognition-result-carries-its-stop-reason-times-segments-and-languages.md)).

The alternatives were turned down:

- Each family's own names in requests. A caller would have to know the family, and Qwen3-TTS's dialects would read as
  languages.
- Leaving out the languages that have no two-letter code. It would hide languages the model takes.
- A code of speech.cpp's own for them, or a key of speech.cpp's own for the languages beside the specification's. Either
  is an invented name for a known fact, which tools that read GGUF metadata do not show.
- Refusing a language on a model that does not steer by it. OpenAI's clients would fail on the language they send as a
  hint.
