# Transcription by regions recognizes each region where someone speaks alone, in the tools, and joins the texts

## Context

A FastConformer model (reazonspeech-nemo-v2, parakeet), NeMo's own implementation included, drops whole sentences of a
stretch of 15 to 20 s that holds several, and writes words for audio in which no one speaks ("うん。", "ピッチャー。").
The page of `speech serve` cuts what it sends into pieces of at most 20 s at the quietest point, so a piece still holds
several sentences, and a piece without speech is still recognized.

`speech_detect()` gives the regions of a request's audio where someone speaks, by the rule and with the options of Silero
VAD's `get_speech_timestamps()` ([the record of detection](detecting-speech-is-a-task-of-its-own-run-by-speech-detect-with-silero-vads-options-and-f32-files.md)).
With Silero VAD's defaults (`min_silence_duration_ms` 100, `speech_pad_ms` 30) a region ends at a breath or a comma:
three read sentences in 20 s gave 8 regions. With `min_silence_duration_ms` 500 the same three became one region of
13.3 s. OpenAI's transcription API takes `chunking_strategy`, `"auto"` or a `server_vad` object of `threshold`,
`prefix_padding_ms` and `silence_duration_ms`, which its reference (github.com/openai/openai-openapi at commit 234829e,
2026-10-07, `VadConfig`) defaults to 0.5, 300 ms and 200 ms and leaves `"auto"` to the server; its Realtime API's
`server_vad` takes the same three, defaulting to 0.5, 300 ms and 500 ms.

A library request runs on one model. The library recognizes Qwen3-ASR audio over 1200 s in parts and joins their texts
without a separator, as qwen-asr does.

## Decision

- **Transcription by regions lives in `tools/common`, beside the library**, for the command line and the server alike:
  one detection request finds the regions of the whole audio, each region's samples become a recognition request of
  their own, and the results are joined. A region is cut from the audio as given, at its start and end times the rate,
  rounded.
- **The detection options default to OpenAI's Realtime `server_vad`, for every way in**: `threshold` 0.5,
  `speech_pad_ms` 300 for its `prefix_padding_ms`, and `min_silence_duration_ms` 500 for its `silence_duration_ms`;
  `min_speech_duration_ms` keeps Silero VAD's 250, and `max_speech_duration_s`, which OpenAI's forms have no member for,
  is 10 s, so that a long run of speech without a pause of half a second is cut at its longest silence.
  `chunking_strategy` `"auto"` takes them all, and a `server_vad` object those of the members it leaves out.
  `speech asr --vad MODEL` takes the options of `speech vad` as flags of their names: those the detection model takes and
  the recognition model does not go to the detection, the others to every region's recognition.
- **The texts are joined with a space, unless either side is written without spaces or already has one**: Han,
  Hiragana and Katakana with their radicals, symbols, punctuation and full-width forms, the scripts the page's join
  leaves without a space. The space begins the next region's first segment and first token, so that the segments, and
  the tokens, still join into the text. Their times move to the whole audio's by the region's first sample; the
  languages follow in order, a run of one language counted once, as qwen-asr merges those of its parts; the stop is
  `model_limit` where any region stopped there.
- **The longest region is 10 s, as measured** (below): it gave the lowest CER on average over the three models, and the
  FastConformer models dropped fewer sentences in all than with any longer limit, while it cost Qwen3-ASR, which reads a
  long region better, less than a shorter one.
- **Audio in which no one speaks gives "" and recognizes nothing**, while the recognition options are still checked
  against the model, so that a request is refused the same way whatever its audio holds.
- **`speech vad --split DIR` writes each region as a 16-bit WAVE file**, cut as transcription by regions cuts it, so that
  the regions it recognizes can be heard and given to other programs.

`measure/region_cap_compare.py` measured each longest region on 105.6 minutes of recordings joined from public test
clips (Apple M5, Metal, 2026-10-08): FLEURS ja_jp test, one reading of each of its 321 sentences, in 21 recordings, and
the first 600 clips of Common Voice 8.0 ja test in 15, each clip cut to its speech and brought to -23 dBFS, a pause of
0.3 to 1.0 s between two clips and a silence of 2 to 8 s between one in four, filled with noise at -60 dBFS; Japanese
forced; the CER without punctuation or spaces after NFKC; a sentence counted dropped when half its characters or more
are deleted. "Whole" recognizes each recording without regions. The cells are FLEURS / Common Voice.

| Longest region | Regions | reazonspeech-v2 CER | dropped | parakeet-tdt_ctc-0.6b-ja CER | dropped | qwen3-asr-0.6b CER | dropped |
|---|---|---|---|---|---|---|---|
| whole | | 25.29% / 86.01% | 67 / 522 | 83.39% / 44.87% | 264 / 266 | 9.80% / 14.48% | 1 / 1 |
| none | 854 | 11.97% / 16.81% | 12 / 42 | 6.39% / 10.82% | 3 / 19 | 10.55% / 14.21% | 0 / 2 |
| 30 s | 854 | 11.97% / 16.81% | 12 / 42 | 6.39% / 10.82% | 3 / 19 | 10.55% / 14.21% | 0 / 2 |
| 20 s | 872 | 11.32% / 16.81% | 10 / 42 | 6.06% / 10.82% | 1 / 19 | 10.65% / 14.21% | 0 / 2 |
| 15 s | 903 | 10.90% / 16.96% | 9 / 43 | 5.99% / 10.36% | 2 / 16 | 10.71% / 14.11% | 0 / 2 |
| **10 s** | 1041 | 10.25% / 17.17% | 4 / 46 | 5.75% / 9.97% | 0 / 11 | 11.44% / 14.17% | 0 / 2 |
| 8 s | 1171 | 10.31% / 17.38% | 2 / 44 | 6.21% / 9.58% | 0 / 8 | 11.63% / 14.28% | 0 / 1 |

The six CERs average 11.79% without a limit, 11.51% at 15 s, 11.46% at 10 s and 11.57% at 8 s. No region and no segment
of a whole recognition that lies where no one speaks held a word, so the noise of these recordings does not show the
words a recognizer writes for silence. reazonspeech-v2 drops 42 to 46 Common Voice sentences at every limit: 43 of the
46 at 10 s are one of two or three short clips, read by different speakers, that share a region no limit here cuts
apart. `speech asr --vad` gave the texts of the same regions recognized one by one.

The alternatives were turned down:

- Transcription by regions in the library, as an option of recognition or as a call that takes two models. A request
  runs on one model and a family reads one file, and a caller that wants the regions alone, as the page does, would need
  a recognizer; the record of detection turned down detection as an option of recognition for the same reason.
- Joining without a separator, as the library joins Qwen3-ASR's parts. A region ends at every pause of half a second,
  so the words of every language written with spaces (parakeet-tdt-0.6b-v3's, and Qwen3-ASR's English) would run
  together at every pause, where a part of 1200 s ends once in twenty minutes.
- Silero VAD's own defaults. They cut a sentence at every breath, and an OpenAI client that asks for `"auto"` or leaves a
  member of `server_vad` out expects OpenAI's values.
- 200 ms of silence for `chunking_strategy`, as `VadConfig` gives it, beside 500 ms for the Realtime API. The same audio
  would be cut into other regions by a file sent whole than by the same audio streamed, and by the command line.
- No longest region, or 15 to 30 s. Regions of up to 27.7 s let reazonspeech-v2 drop three times as many FLEURS
  sentences, and their CERs were higher on average.
- 8 s. It dropped fewer sentences still, but cut Qwen3-ASR's and parakeet's FLEURS sentences to a higher CER.
- Pieces of a fixed length cut at the quietest point, as the page cuts them. A piece still holds several sentences, and
  one without speech is still recognized.

## Consequences

A program that wants the text of a long recording runs a detection model beside the recognizer: `speech asr --vad`, or
`speech serve` given one, which answers `chunking_strategy` without it with an error rather than a transcription of the
whole audio. The page, which cuts its own pieces, can send whole recordings once it asks for regions.
