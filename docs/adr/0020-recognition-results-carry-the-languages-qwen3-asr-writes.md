# Recognition results carry the languages Qwen3-ASR writes

Decided 2026-10-07.

## Context

Qwen3-ASR, when the language is left to it, writes the language it hears before the text (`language Japanese<asr_text>…`),
and `language None` for audio without speech. speech.cpp parsed its output as qwen-asr 0.0.6's `parse_asr_output()`
does (docs/adr/0018) and kept the text alone, so a caller that let the model find the language could not learn which it
found. qwen-asr's `transcribe()` returns the language with the text, as the name it parsed:

- the forced language for a request that forced one, whose prefill writes it, but "" for an output that is empty once
  stripped, which the parse returns before it looks at the forced language;
- "" for an output without `<asr_text>` and for one whose part before it holds `language none` in any case;
- otherwise the name on the first line of that part that begins with `language `, its first letter raised and the rest
  lowered, whether or not it is one of the 30 names qwen-asr supports;
- for audio over 1200 s, which it cuts into parts, the parts' names joined with commas by `merge_languages()`, in order,
  one for each run of parts with the same name, and without the parts that gave "".

FastConformer writes no language. parakeet-tdt-0.6b-v3 tells its 25 languages apart without being told, but NeMo 3.0.0
gives a transducer's languages only with an `AggregateTokenizer` and `compute_langs`, and its checkpoint has one
SentencePiece tokenizer; its decoding strips no language tags, and none of the 672 ids of its 12 dumps is one of the 262
pieces of the form `<|…|>` its vocabulary holds, `<|en|>` and the other language tags among them.

OpenAI's API reference (github.com/openai/openai-openapi at commit 31af4fc, 2026-10-05) requires `language` in a
`verbose_json` transcription, a string it describes as the language of the input audio, its example Whisper's
`english`, and takes a request's `language` as an ISO 639-1 code. Its `json` answer of `gpt-transcribe` carries
`languages`, a list of codes, empty where no language was detected. speech.cpp's `verbose_json` set the option
`timestamps`, so Qwen3-ASR, which gives no times, refused it.

## Decision

- **A result carries a list of languages**, as BCP 47 tags of the model's languages: `speech_result_language_count()`
  and `speech_result_language()`, added in C API 3.1. A name becomes the tag of `general.languages` that
  `qwen3-asr.language_names`, the converter's one table, gives it, after qwen-asr's normalization of the name. The table
  is layout 1's, so the files converted for 0.7 give the language as they are.
- **The list is qwen-asr's**: the language each part wrote, or the forced one, merged as `merge_languages()` merges
  names, so that a single part gives one language at most. It is empty where qwen-asr gives "": audio without speech, a
  forced request whose output is empty, and a cancelled request.
- **A name outside the model's languages gives no language**, and a warning in the log names it. The model wrote one of
  its 30 names on every dump, and a name the table does not hold has no tag to give.
- **FastConformer gives none.** A language a request gives it is only checked, and the model did not hear it.
- **The worker's `end` and `partial` and `speech asr --format json` carry `languages`**, left out where the list is
  empty, so FastConformer's answers stay as they were. The worker's protocol stays 2: README says an added member does
  not raise it, since a caller that does not know it ignores it and keeps working.
- **The server's `verbose_json` carries `language`**: the tag, the form a request's `language` takes, and where the
  parts of long audio gave several, their tags joined with commas in order, as qwen-asr joins their names; left out
  where the list is empty, as the members of OpenAI's segment without a value are. `verbose_json` sets `timestamps` only
  for a model that gives times, or when the request names a granularity, so a model that gives none answers it without
  segments, which OpenAI's schema does not require, and refuses a granularity named. `json` and `text` stay as they
  were.

The alternatives were turned down:

- One language per result, the first part's or the most frequent. Either drops what qwen-asr keeps for audio whose
  parts differ, and the most frequent needs a rule for ties and for parts of unequal length that qwen-asr does not
  have.
- A language per segment. Qwen3-ASR gives no segments, which are timed text (docs/adr/0016).
- Refusing a name outside the model's languages with an error. The text the model wrote would be lost for the label in
  front of it, which qwen-asr passes on as it is.
- The name itself where the table holds none. It is no tag, so no caller could compare it with the model's languages
  or send it back as a request's language.
- An empty `languages` in every answer of a model that writes none. It would change every FastConformer answer and say
  nothing.
- A key of the file for `language none`, the parse's mark of audio without speech. It is the text of the parse's code,
  as its stripping and its line breaks are, which the converter finds in that code; a key would need a layout 2 and an
  upgrade of every file converted for layout 1 to supply it.
- The name in `verbose_json` as Whisper writes it (`japanese`), as OpenAI's example does. No request takes that form,
  and Qwen3-ASR's names are not Whisper's for every language (`Filipino`, where Whisper writes `tagalog`).
- `""` in `verbose_json` for no language, which the schema's required member would have. It names no language, and a
  client cannot tell it from one.

## Consequences

A caller that leaves the language to Qwen3-ASR can answer in the language it heard, or force it on the next request.
A request that asks nothing new gives the text and segments it gave before. The reader of layout 1 now requires the
names and the prefix in ASCII, the names with no capital but the first letter, which the parse's comparison needs and
every converted file has. Qwen3-ASR answers `verbose_json`, which docs/adr/0017 described with segments, the form it
keeps for a model that gives times. The checks compare the language of every dumped request and of 262 outputs,
written by `parse_cases.py` with qwen-asr's parse, with qwen-asr's, and `parse-cases.jsonl` written before this has no
languages, so `parse_cases.py` writes it again.
