# A command without a model says what to type rather than choosing one

Decided 2026-10-08.

## Context

With models fetched by name (docs/adr/0033), `speech asr a.wav` could fetch a default model and run, as some tools
do. Which model is right depends on the language: ReazonSpeech for Japanese, Qwen3-ASR for most of 30 languages,
parakeet-tdt-0.6b-v3 for nine European languages Qwen3-ASR does not recognize, and for synthesis Qwen3-TTS's ten
languages or Irodori-TTS's Japanese.

## Decision

Every subcommand that takes a model (`tts`, `asr`, `voice`, `info`, `serve`, `worker`) requires it. A command line
without one, or whose first argument is neither a `.gguf` path nor a name of the catalog while an argument is
missing (`speech asr a.wav`), stops with exit 2 and says what MODEL is, which models of the subcommand's task to start
with for which languages, read from the catalog, an example with the first of them, and that `speech models` lists
the rest. `speech voice` lists the models that take voice files instead. A worker started without a model answers
with `fatal`, as for any usage error.

The alternatives were turned down:

- A default model. It would be wrong for every user of another language, it would fetch a gigabyte the user did not
  ask for, and it could never change: a script that relied on it would get another model's text or voice from one
  release to the next.
- A default chosen by the system's language. The audio's language is not the system's, and the result would differ
  from one machine to another for the same command line.

## Consequences

The first command a user types names a model, as README's examples do, and the message shows the model to start
with for each language when one is forgotten.
