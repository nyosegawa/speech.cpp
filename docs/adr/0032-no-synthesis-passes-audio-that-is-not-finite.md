# No synthesis passes audio that is not finite

Decided 2026-10-08.

## Context

Each value of a request is checked against its range, and Irodori-TTS's ranges stop at what a float32 holds
(docs/adr/0025). Values within their ranges can still be more than a model computes: v4.1-Small multiplies the
difference of two branches by `cfg_scale_text`, and at 1e38 on Metal, or at the largest float, 3.4e38, on the CPU too,
the velocity overflows and the request returned SPEECH_OK with 24,000 samples of infinities and NaN, which a player
clips to full scale. At 1e38 on the CPU the latent stays finite and the codec's tanh saturates it into noise. Qwen3-TTS
divides logits by its temperature and could overflow the same way. A range per option cannot rule this out, since it
comes of the values together and of the model's activations.

## Decision

- **The audio of every synthesis passes one check on its way to the caller**: RequestRun::audio() in
  src/request.cpp, through which every family passes its audio, refuses a sample that is not finite before the
  callback sees it, so no family returns such audio as speech, now or when added.
- **The request fails with SPEECH_ERROR_OUT_OF_RANGE, naming no option.** The model takes each value but not the
  values together, which is the category's meaning, where SPEECH_ERROR_INVALID_ARGUMENT is the caller's mistake of
  form and SPEECH_ERROR_INTERNAL a defect of the library. No option is named, since the check sees samples and not
  which value caused them; the message says that the speech came out not finite and that the request's scales or
  settings are too large.
- **The request is spent or kept by the rule of every failure**: spent once its work has passed progress or audio to
  the caller, as a guided sampler's has by then, so that the caller makes a new one with smaller values. Audio passed
  before the failure was finite.

The alternatives were turned down:

- A check in each family, such as of Irodori-TTS's latent before the codec. Each family would need its own, and a
  family added later would have none; the boundary covers them all with one line.
- Bounds on each scale small enough that no product overflows. They would depend on the model's activations, which no
  file gives, and would refuse values the runtime runs.

## Consequences

speech.h says it where speech_synthesize() and its callback are described. The worker, the server and the command
line pass the failure on as any error, with no option.
