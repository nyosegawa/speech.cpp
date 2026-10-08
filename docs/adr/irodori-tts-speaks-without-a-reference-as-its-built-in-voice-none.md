# Irodori-TTS speaks without a reference as its built-in voice none

## Context

The official runtime's request may set `no_ref` instead of a reference: it gives the speaker encoder a zero latent with
every position masked, so that the DiT attends to no speaker, turns the speaker's guidance off, and predicts the length
with the duration predictor's learned null speaker. With a caption this is VoiceDesign, a voice described in words;
without one the model chooses a voice. A request that sets neither a reference nor `no_ref` is an error there.

speech.cpp's `voice` is required for every synthesis model that takes it, Irodori-TTS's voices are otherwise those added
since loading, and OpenAI's speech API, which `speech serve` speaks, requires `voice` in every request.

## Decision

A file that holds the null speaker (layout 2) has one voice of its own, `none`, which the model information lists
before the voices added, and a request that names it speaks without a reference. `voice` stays required, so a request
that forgets its voice is still refused, as the runtime refuses one without a reference. `speech_voice_add()` refuses
the name `none` on such a file, as it refuses any name the model has. A layout 1 file has no such voice, and a request
for it is refused naming the null speaker the file lacks.

With `none` the speaker's guidance does not run, as in the runtime: `cfg_scale_speaker` other than its default and 0,
`speaker_uncond_mode` `noise` and the speaker's scaling (`speaker_kv_scale`) are refused, since they would have no
effect, where the runtime ignores them with a message.

The alternatives were turned down:

- `voice` made optional, a request without one speaking without a reference. A caller that forgets the voice would get a
  voice it did not choose and no error, where the runtime refuses that request, and an OpenAI client cannot leave
  `voice` out.
- A boolean option `no_reference` beside `voice`. The model information could not say that one of the two is required,
  and an OpenAI client would send both and be refused.
- The name `auto`, as `language` names the model's choice. `none` says that no reference is given; `auto` reads as a
  voice chosen from the voices.

## Consequences

A caller, an OpenAI client included, reaches VoiceDesign with `"voice": "none"` and the caption in `instructions`. A
program that takes the first voice of a layout 2 file's information as a default gets `none`, and should take an added
voice by its name.
