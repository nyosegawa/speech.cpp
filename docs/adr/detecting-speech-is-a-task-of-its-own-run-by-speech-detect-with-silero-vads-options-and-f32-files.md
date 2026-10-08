# Detecting speech is a task of its own, run by speech_detect() with Silero VAD's options, in F32 files

## Context

The recognizers work best on one utterance at a time. Given a stretch of 13 to 20 s that holds several sentences,
ReazonSpeech and parakeet drop whole sentences, as NeMo's own implementation does on the same audio, and given silence
ReazonSpeech writes words ("うん。"). The page of `speech serve` and long files need to be cut into utterances first.

Silero VAD (snakers4/silero-vad, MIT) gives a speech probability for each chunk of 512 samples at 16 kHz, with the 64
samples before it as context and an LSTM state carried from chunk to chunk; the package's `get_speech_timestamps()` turns
the probabilities into regions by `threshold`, `min_speech_duration_ms`, `min_silence_duration_ms`, `speech_pad_ms` and
`max_speech_duration_s`. Its other settings, `neg_threshold` (the threshold less 0.15, at least 0.01),
`min_silence_at_max_speech` (98 ms) and `use_max_poss_sil_at_max_speech` (true), only refine where the official cuts. A
region longer than `max_speech_duration_s` is cut at its longest silence, which is what keeps a recognizer's input to
one or a few sentences. The JIT file the package loads holds a 16 kHz model and an 8 kHz one; the 16 kHz model has 309K
parameters, 1.2 MB in F32. whisper.cpp offers the same settings under its own flags (`--vad-max-speech-duration-s`).

The C API had two tasks, synthesis and recognition, a call of each, and segments with a start, an end and a text in a
recognition's result.

## Decision

- **Detection is a third task**, `SPEECH_TASK_DETECTION`, run by `speech_detect()` on a request's audio, which is
  resampled to the model's rate as a recognition's is. The result's regions are its segments, read through
  `speech_result_segment_count()` and `speech_result_segment()`, each with its start and end in seconds and an empty
  text; the result has no text, tokens or languages. The calls of the other tasks are `unsupported` on a detection model,
  and `speech_detect()` on the others.
- **A request sets the options under the official names**: `threshold`, `min_speech_duration_ms`,
  `min_silence_duration_ms`, `speech_pad_ms` and `max_speech_duration_s`, with the defaults of `get_speech_timestamps()`
  from the file. `max_speech_duration_s` has no default, the official's infinity being no value a request sets, and a
  request without it has no limit. `neg_threshold`, `min_silence_at_max_speech` and `use_max_poss_sil_at_max_speech` are
  no options: the file holds the constants the official computes the first two with, and the library runs the rule of
  the third's default. The regions are the official's for the same probabilities, sample for sample.
- **A detection model takes no language.** Its file has neither `general.languages` nor `speech.language_use`, its
  information lists none, and it takes the language `auto` alone, as every model does.
- **The file holds the 16 kHz model alone, in F32 alone.** Every input reaches the model at 16 kHz through the library's
  resampler, so nothing would choose the 8 kHz model. `speech quantize` writes no other type of it: the layout keeps every
  tensor in F32, and a file of 1.2 MB gains nothing from a smaller type that the probabilities would pay for.
- **`speech vad` is the command line's way in**; `speech asr --vad` and `speech serve`'s `chunking_strategy` transcribe
  by its regions in the tools. The worker refuses a detection model when it starts, since its protocol has no messages
  for regions.

The alternatives were turned down:

- Detection as an option of recognition, a recognizer cutting its audio with a VAD model before it recognizes. A request
  runs on one model and a family reads one file, and a caller that wants the regions alone, as the page does to cut what
  it sends, would load a recognizer for them.
- A detection model described as a recognizer whose text is empty. A caller choosing a model by its task would send it
  audio and wait for text.
- Accessors of the result's own for regions. A region is a segment without text, and the segments' accessors already
  give its start and end.
- Names of speech.cpp's own for the options, or whisper.cpp's. The official names are the ones a user of Silero VAD
  knows, as Qwen3-TTS's sampling keeps transformers' names
  ([the record of its sampling](qwen3-tts-requests-set-sampling-under-transformers-names.md)).
- F16 and Q8_0 files. They would save less than a megabyte, and on the CPU in F32 the probabilities lie within 4e-6 of
  the official's (Apple M5, 2026-10-08).

## Consequences

A caller that wants utterances runs a detection model and gives each region to a recognizer. The worker's protocol
changes before it carries detection.
