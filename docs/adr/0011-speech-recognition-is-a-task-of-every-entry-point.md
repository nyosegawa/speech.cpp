# Speech recognition is a task of every entry point

Superseded in part by docs/adr/0014, 0015, 0016 and 0017: the request objects and loading, the codec in the model file, resampling and the result object, and the worker's messages, the server and the `speech` executable.

Decided 2026-10-05.

## Context

docs/adr/0009 brought FastConformer into speech.cpp as a family that turns audio into text, and left its entry
point for later. Every way into the library assumed synthesis: `speech_model_params` asked for a codec and voices,
`speech_synthesize()` was the only request, `speech_model_load()` chose the family with an `if` per architecture,
the worker's requests were texts, the server had only `/v1/audio/speech`, and the one command-line tool was
`speech-tts`. More families of both kinds are coming (TDT decoding, parakeet-tdt-0.6b-v3, reazonspeech-nemo-v2, other
speech synthesis), so the shape chosen now has to take them without a special case each.

## Decision

A model does one task, synthesis or recognition, and every entry point says which and takes the requests of that
task alone.

- **The family table.** `src/speech.cpp` holds one table of families: the `general.architecture` of their GGUF
  files, the task, whether they need a codec and take voices, a context and sampler steps, and the engine's
  constructor. Loading looks the architecture up in it and refuses, naming the field, a codec missing where the
  family needs one and any field it does not take that is not at its default. `speech_model_architecture()` and
  `speech_model_task()` read the table, so the task is written in one place.
- **The C API.** `speech_model_task()` says what a model does. `speech_transcribe()` takes a
  `speech_transcription_request` (mono float samples, their rate and a language), started from
  `speech_transcription_request_default()`, and passes the text to a callback, which is called once by FastConformer
  and may stop the request by returning nonzero. `speech_cancel()` and the one-request-at-a-time rule apply as to
  synthesis. Calling `speech_synthesize()` on a recognition model, or `speech_transcribe()` on a synthesis model, is
  an error that names the model's task. The getters that mean something for synthesis alone say what they give a
  recognition model: no voices, no steps, and `SPEECH_STREAMING_NONE`. The sample rate is the rate the model makes or
  takes, and a request's language is checked against the model's languages wherever the model is not told it.
- **No resampling.** A recognition request at a rate other than the model's is an error that names both rates. A
  resampler's filter changes what the model hears, the official implementations differ in theirs (NeMo's
  `transcribe()` resamples a file with librosa, a caller may use ffmpeg or soxr), and no reference dump would check
  ours; a caller that has audio at another rate resamples it with the tool it trusts.
- **SPEECH_API_VERSION stays 2.** No declaration an existing caller uses changes and no struct changes its layout:
  the task, the request, the callback, the function and `SPEECH_STREAMING_NONE` are additions, which the header's
  rule does not count. A program built against version 2 runs against this library and loads the same models.
- **The worker.** `ready` carries `"task"`. A recognition worker takes a request's audio in the chunks a
  synthesis worker sends, `{"type":"chunk","id","seq","pcm"}` with base64 16-bit PCM and `seq` from 0, and the
  request itself in `{"type":"end","id","sampleRate","language"}`, and answers `{"type":"text","id","text"}` or one
  `error`. A refused chunk is its request's one answer, and the request's other lines are dropped. A request to
  speak sent to a recognition worker, and a chunk sent to a synthesis worker, are answered with an error.
- **The server.** `/v1/models` carries `task`. A recognition model serves OpenAI's `POST /v1/audio/transcriptions`
  (multipart/form-data with `file`, and `model`, `language` and `response_format` `json` or `text`), reads WAV
  alone and refuses any other file with an OpenAI error; the endpoint of the other task is a 404 that names the
  right one.
- **The command line.** `speech-asr` writes the text of WAVE files to stdout, one line per file, with
  `speech-tts`'s names for the options both have (`--device`, `--language`, `-v`, `--devices`).

The alternatives were turned down:

- A handle per task (`speech_synthesizer`, `speech_recognizer`) with their own load, free, getters and cancel.
  Loading, devices, cancellation, the one-request rule and most getters are the same for both, and two handles
  would double the API for a difference one getter states.
- A codec and voices that a recognition model ignores. The caller would not learn that what it passed means
  nothing, the fallback AGENTS.md refuses.
- The text as an out string the caller frees with a function of its own. A callback keeps every string the library
  returns its own, matches synthesis, and lets a family that recognizes in segments pass each as it is done.
- One worker line with the whole audio of a request. Chunks keep lines small, let a caller send the microphone's
  audio while it records, and mirror what a synthesis worker sends.
- Float samples on the worker's protocol. The 16-bit PCM of the chunks a synthesis worker sends gives the dumps'
  CTC text exactly, and one form of audio on the protocol is simpler for every caller.
- A mode of `speech-tts` instead of a tool of its own. The two share `--device` and `--language` and nothing else:
  one writes a WAVE file from text, the other text from WAVE files.
- Decoding MP3, FLAC and the other formats OpenAI takes in the server. It would need a decoder as a second
  dependency besides cpp-httplib, or a guess at the format.

## Consequences

The worker's `ready` and `/v1/models` gain a member, a recognition model is loaded without a codec, and new tools,
messages and an endpoint appear; the next release raises the minor version. Synthesis is unchanged byte for byte:
the worker, the server and `speech-tts` give the same audio for the same request and seed. A context or steps
given to a family without them is now refused, where Irodori-TTS used to ignore a context. FastConformer stops for
a cancel only before its encoder or once the encoder has run, which on the CPU is seconds for long audio; and its
encoder attends over the whole utterance, so its time and memory grow with the square of the audio's length.
