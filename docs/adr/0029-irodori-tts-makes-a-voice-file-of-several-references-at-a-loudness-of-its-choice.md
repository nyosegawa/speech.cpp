# Irodori-TTS makes a voice file of several references at a loudness of its choice

Decided 2026-10-07.

## Context

The official runtime's request takes several reference recordings (`ref_wavs`), which it encodes one by one, each
brought to its loudness on its own, and joins in order before the speaker encoder; it cuts the joined latent at
`max_ref_seconds` (120 s, 3000 frames). It brings each to `ref_normalize_db`, -16 LUFS by default, or leaves the
loudness as recorded with `None`, scaling down a peak above 1 (`ref_ensure_max`). speech.cpp's voice was one
recording at the model's loudness, and its voice file (layout 1) recorded one length and one rate.

A voice is a property of the speaker rather than of a request: the worker loads its voices when it starts, ASIST
carries voice files (docs/adr/0002), and `--add-voice` and the worker's `add_voice` take one path.

## Decision

- **Several references and the loudness are a voice file's**, made by `speech voice MODEL REF.wav... VOICE.gguf`
  with `--lufs` or `--keep-loudness`, and by the C API's voice parameters (`speech_voice_params_new()`,
  `_add_reference()`, `_set_loudness()`, `_keep_loudness()`, `speech_voice_make_from()`). The references are encoded
  as the runtime encodes its `ref_wavs`, and one reference at the model's loudness gives the bytes of
  `speech_voice_make()`. `--add-voice`, `add_voice` and `speech_voice_add()` keep taking one file.
- **Joined references past the model's `irodori-tts.reference.max_seconds` are refused**, naming `references`, where
  the runtime cuts them.
- **Layout 2 of the voice file** records each recording's length and rate, in the order joined, and whether they were
  brought to a loudness and which, and its `speech.requires` is 0.8.0. A layout 1 voice file reads as one of one
  recording brought to -16 LUFS, the loudness every layout 1 model file gives, so ASIST's voice files keep working.

The alternatives were turned down:

- Several references and the loudness as options of a request, or `add_voice` taking several paths. The worker would
  encode them at every start or with every request, where a voice file encodes them once, and the protocol's voice
  would be a list.
- One function with the references as an array and the loudness as a number, NaN for none. A caller would pass the
  model's -16 to keep the default, and NaN would mean something.
- Cutting joined references at 120 s, as the runtime does. The voice would lose the end of what the caller gave.

## Consequences

The six functions are added in the C API's 3.1. Releases before 0.8.0 refuse layout 2 voice files naming 0.8.0;
a program that shares voice files with them makes them with 0.7.x.
