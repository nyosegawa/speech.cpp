# speech serve and speech worker take request options as the defaults of the models they are started with

## Context

A request option's default is the model's, from its file, and a model file holds the official runtime's default:
Irodori-TTS v4.1-Small-MF samples in 4 steps. A program that wants another value for every request, such as 8 steps,
had to send it in each request to `speech serve` or `speech worker`; an OpenAI client sends it through `extra_body` in
every call. `speech tts`,
`speech asr` and `speech vad` already take every option of the C API's vocabulary as a flag.

`speech serve` holds up to three models, one of each task, and several options are taken by more than one task
(`language` by a synthesis and a recognition model). Its page loads other models in place of those given, and sets the
options of each request it sends from a form that leaves out what the user did not change.

## Decision

- **Both take every request option as a flag, as `speech tts` does**, and a flag's value is the default of every
  request that does not set the option, in place of the model's own. A request's own value wins.
- **`speech serve` gives a flag to each model given on its command line that takes the option.** An option that none
  of them takes stops the server before it listens, `unsupported` naming the option; the worker stops with `fatal` the
  same way for an option its model does not take.
- **Each value is checked against each model through a request's setters when the program starts**, so a value out of
  range stops it before it serves rather than failing the first request. Values that contradict each other are refused
  by the requests that run with them, as the library refuses any such request.
- **A model the page loads keeps its own defaults.** The flags belong to the models the command line named.
- **The model information still shows the model's own defaults**: `ready`, `GET /v1/models` and `speech info --json`
  give the same information. The program logs the defaults it was started with on stderr.

The alternatives were turned down:

- Flags with the task's name in front, such as `--tts-steps` and `--asr-language`. No option means one thing for one
  task and another for another, and `speech tts`'s flags would read otherwise than the server's.
- Giving the flags to a model the page loads later. The page sends a request's options from its own form, so a default
  the page cannot show would steer requests its user set up there.
- Rewriting the defaults in the model information. A caller that starts the worker knows the flags it gave, and the
  information would differ from `speech info --json`, which a caller compares it with to learn what a model takes.

## Consequences

A program sets a model's behavior once where it starts the server or the worker, and an OpenAI client needs no
`extra_body`. A boolean option shown unchecked on the page can be true by a flag, since the page shows the model's own
default.
