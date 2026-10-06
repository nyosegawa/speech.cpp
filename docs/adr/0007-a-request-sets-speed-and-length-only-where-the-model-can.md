# A request sets speed and length only where the model can

Superseded in part by docs/adr/0014, docs/adr/0016 and docs/adr/0017: speed and length are options of a request object that each model declares, the scaled length is no longer kept within the bounds, Qwen3-TTS takes `max_seconds`, and the worker names the scale `duration_scale` as the C API does.

Decided 2026-10-05.

## Context

The worker took a `speed` in a request and did nothing with it, a choice made when ASIST was its only
caller and sent one. To any other program that is a silent fallback: it asks for faster speech and gets the
model's own rate without a word. speech.cpp now serves any program, through the C API and the worker.

The models differ. Irodori-TTS sets a sentence's length before it makes it, and its official runtime lets a
request fix the length (`seconds`) or scale the prediction (`duration_scale`); Irodori-TTS-Server, by the
same author, serves OpenAI's speech API on it and turns OpenAI's `speed` (0.25 to 4) into those two by
dividing both by it. Qwen3-TTS's official implementation has no control of the rate or of the length: the
talker decides frame by frame when the speech ends, and `generate_custom_voice()` takes no such argument.
Only the 1.7B model takes a natural-language instruction, which speech.cpp does not implement and which
states no rate a caller could ask for.

## Decision

A request carries `speed`, `seconds` and `duration_scale` (`durationScale` in the worker), and a model
follows them only where its official implementation can; otherwise the request is an error.

- Irodori-TTS follows the runtime for `seconds` and `duration_scale` and Irodori-TTS-Server for `speed`,
  which divides whichever length applies. Where the runtime goes on quietly, speech.cpp refuses: seconds
  outside 0.5 to 30 s (the runtime clamps them), seconds together with a duration scale (the runtime ignores
  the scale), and a speed outside the server's 0.25 to 4.
- Qwen3-TTS refuses any speed but 1 and any length.

In the C API the three are fields of `speech_request`, and `speech_request_default()` gives a request with
their defaults (speed 1, no fixed length, scale 1), as `speech_model_default_params()` does for loading.
`SPEECH_API_VERSION` rises to 2.

The alternatives were turned down:

- Changing Qwen3-TTS's rate after synthesis by resampling or a time stretch. Resampling moves the pitch with
  the rate, and a time stretch (WSOLA, a phase vocoder) adds its own artifacts to every request that uses it
  and is no part of the model; a caller that wants either can apply it to the audio it receives.
- Keeping `speed` and ignoring it where a model cannot follow it. The caller cannot tell that its request
  was not followed.
- Clamping seconds into the bounds as the runtime does. The caller gets a length other than the one it asked
  for, told only in a log.
- Refusing `speed` together with a length. Irodori-TTS-Server combines them by division, and a caller moving
  from the server gets the same length here.
- Making zero mean the default for every new field, so that old code with a brace initializer still
  compiles. The compiled struct is larger either way, so a program built against version 1 has to be
  rebuilt, and a speed or scale of 0 would be read as the default instead of being refused.

## Consequences

A program built against API version 1 passes a smaller `speech_request` and must be rebuilt; it starts
its requests from `speech_request_default()`. The worker's protocol gains `seconds` and `durationScale`, and
a request with a value that is not a number, out of range, or one the model cannot follow is answered with
an `error`. A worker request without the three is spoken as before, with the same audio for the same seed.
