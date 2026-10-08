# A recognition result carries its stop reason, times, segments and languages

## Context

Qwen3-ASR writes its text token by token up to a limit of its file, and a text cut at the limit reads like one that
ended. FastConformer's decoders know on which encoder frame they emitted each token. NeMo's
`transcribe(timestamps=True)` groups the tokens into words at spaces and ends a segment at a word ending in `.`, `!` or
`?`, so a Japanese recording, which has no spaces, is one segment however long.

Qwen3-ASR, when the language is left to it, writes the language it hears before the text
(`language Japanese<asr_text>…`), and `language None` for audio without speech. qwen-asr 0.0.6's `transcribe()` returns
the language with the text, as the name its `parse_asr_output()` parses:

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
`verbose_json` transcription, a string it describes as the language of the input audio, its example Whisper's `english`,
and takes a request's `language` as an ISO 639-1 code. Its `json` answer of `gpt-transcribe` carries `languages`, a list
of codes, empty where no language was detected.

## Decision

- **A recognition reports why it stopped**: complete, at the model's limit, or cancelled. A Qwen3-ASR recognition that
  reaches the 4096 tokens of its file returns what it wrote and says so, as a synthesis does at its limit.
- **A recognition returns its text, and when the request sets `timestamps` its tokens and segments with times.** A
  token's time is the encoder frame its decoder emitted it on, and for TDT the frames its predicted duration covers, as
  NeMo computes them, a punctuation mark taking the end of the token before it. For the beam search NeMo records its
  search step, which is the frame plus the number of tokens before it and runs past the end of the audio (386 s for a
  311 s input); speech.cpp gives the frame. Segments follow NeMo's rule for the separators of the model's file
  (`fastconformer.segment.separators`, the checkpoint's or NeMo's default `.`, `?`, `!`): a segment ends after a word
  whose last mark is one, words being split at spaces. A model whose languages are written without spaces also lists
  breaks (`fastconformer.segment.breaks`: `。`, `？`, `！`, `?`, `!` for the Japanese models), which end a segment after
  any token that ends in one. Qwen3-ASR gives no times and does not take `timestamps`.
- **A result carries a list of languages**, as BCP 47 tags of the model's languages (`speech_result_language_count()`
  and `speech_result_language()`). A name becomes the tag of `general.languages` that `qwen3-asr.language_names`, the
  converter's one table, gives it, after qwen-asr's normalization of the name.
- **The list is qwen-asr's**: the language each part wrote, or the forced one, merged as `merge_languages()` merges
  names, so that a single part gives one language at most. It is empty where qwen-asr gives "": audio without speech, a
  forced request whose output is empty, and a cancelled request.
- **A name outside the model's languages gives no language**, and a warning in the log names it. The model wrote one of
  its 30 names on every dump, and a name the table does not hold has no tag to give.
- **FastConformer gives none.** A language a request gives it is only checked, and the model did not hear it.
- **The worker's `end` and `partial` and `speech asr --format json` carry `languages`**, left out where the list is
  empty.
- **The server's `verbose_json` carries `language`**: the tag, the form a request's `language` takes, and where the
  parts of long audio gave several, their tags joined with commas in order, as qwen-asr joins their names; left out
  where the list is empty, as the members of OpenAI's segment without a value are. `verbose_json` sets `timestamps` only
  for a model that gives times, or when the request names a granularity, so a model that gives none answers it without
  segments, which OpenAI's schema does not require, and refuses a granularity named. `json` and `text` carry the text
  alone.

The alternatives were turned down:

- NeMo's segments as they are. A Japanese recording of minutes would be one segment.
- One list of marks that end a segment after any token. It cuts Japanese, but also European text inside numbers and
  abbreviations ("1.000", "z.B.") where NeMo does not.
- One language per result, the first part's or the most frequent. Either drops what qwen-asr keeps for audio whose parts
  differ, and the most frequent needs a rule for ties and for parts of unequal length that qwen-asr does not have.
- A language per segment. Qwen3-ASR gives no segments, which are timed text.
- Refusing a name outside the model's languages with an error. The text the model wrote would be lost for the label in
  front of it, which qwen-asr passes on as it is.
- The name itself where the table holds none. It is no tag, so no caller could compare it with the model's languages or
  send it back as a request's language.
- An empty `languages` in every answer of a model that writes none. It would add a member to every FastConformer answer
  that says nothing.
- A key of the file for `language none`, the parse's mark of audio without speech. It is the text of the parse's code,
  as its stripping and its line breaks are, which the converter finds in that code; a key would need a new layout and
  an upgrade of every file of the current one to supply it.
- The name in `verbose_json` as Whisper writes it (`japanese`), as OpenAI's example does. No request takes that form,
  and Qwen3-ASR's names are not Whisper's for every language (`Filipino`, where Whisper writes `tagalog`).
- `""` in `verbose_json` for no language, which the schema's required member would have. It names no language, and a
  client cannot tell it from one.
- `languages` in the `json` answer, as `gpt-transcribe`'s carries it. Given by every model, a FastConformer answer's
  empty list would say that no language was detected where none was looked for; given only by a model that writes
  languages, it would need a fact of the model information saying which do. The default answer stays `{"text"}`, and a
  client that wants the language asks for `verbose_json`, which OpenAI's SDK reads.

## Consequences

A caller that leaves the language to Qwen3-ASR can answer in the language it heard, or force it on the next request. The
reader requires Qwen3-ASR's names and prefix in ASCII, the names with no capital but the first letter, which the parse's
comparison needs. The checks compare the language of every dumped request, and of 262 outputs that `parse_cases.py`
writes with qwen-asr's parse, with qwen-asr's.
