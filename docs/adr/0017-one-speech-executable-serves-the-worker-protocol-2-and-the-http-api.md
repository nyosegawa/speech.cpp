# One speech executable serves the worker protocol 2 and the HTTP API

Superseded in part by docs/adr/0020: `verbose_json` carries the language the model heard, and a model that gives no times answers it without segments.

Decided 2026-10-06.

## Context

A release carried four executables, `speech-tts`, `speech-asr`, `speech-server` and `speech-worker`, each about 46 MB
with the library and ggml linked in, each reading its command line its own way: `--voice NAME=FILE` registered a voice
while `speech-tts` selected one with `--voice-name`, and the worker read `--seed 12abc` as 12.

The worker's protocol had grown without a number of its own, which docs/adr/0006 turned down because a caller pins a
release. It read one level of JSON and ignored members it did not know. A cancelled request got no answer, and a
cancel that came after a request's end was kept, so a later request with the same id was dropped without one. Between
Irodori-TTS's sampler steps and through a recognition it sent nothing, and ASIST takes 30 s of silence as a hung
worker. A recognition gave text only once its `end` came, so a caller showing live captions had nothing to show while
the speaker talked. Programs other than speech.cpp's worker now speak the protocol: speech-bench's adapters run the
official Irodori-TTS and mlx-audio behind it, each implementing what its callers use.

The HTTP server guessed its status from whether audio had been sent, called every transcription failure a 400, and
took WAV only at the model's rate.

## Decision

- **One executable, `speech`, with subcommands.** One parser reads every command line; every request option is a
  flag of its own name, so a new option is a new flag without the tool changing; `--add-voice` adds a voice and
  `--voice` selects one, as the option does everywhere. A release is one archive per platform and about 140 MB
  smaller. The executable links cpp-httplib for `speech serve`, against docs/adr/0008's rule that the worker stays free
  of it: the header is compiled in and runs only for `serve`, and keeping it out would mean a second executable with
  the copies and the second parser back. The library and `libspeech` stay free of it.
- **The worker speaks protocol 2**, with a number in `ready`, because programs that are not a speech.cpp release now
  speak it and their callers pin none of them. It reads whole JSON and refuses a member it does not know. Every
  request gets exactly one terminal message, `end`, `error` or `cancelled`, so a caller never waits for an answer that
  will not come, and a cancel has no answer of its own and no effect on a later request. Errors carry the library's
  category and the input at fault (docs/adr/0014). Each request reports its seed and stop reason (docs/adr/0016). The
  worker sends progress while a request passes no audio, from the library's own progress, which says that the work
  moves, not only that the process lives. Audio stays base64 of 16-bit PCM.
- **A recognition request can be peeked at** while it collects chunks: a `peek` recognizes the audio received so far
  and answers with a `partial` text, and the request stays open for more chunks and its one terminal message. No
  model speech.cpp runs has a cache-aware streaming encoder, so a peek recognizes the whole audio so far and its cost
  grows with it: parakeet-tdt_ctc-0.6b-ja in F16 on Metal took 0.07 s for 6.4 s of audio and 0.11 s for 10.5 s. The
  C API is unchanged: its caller runs a new
  request on its own buffer. Streaming recognition for a model that has such an encoder would come as added functions
  of a stream object, a minor version, without changing the request.
- **The HTTP server derives its status and its error from the library's category alone**, the same for the same
  mistake whenever it happens, with `param` from the input at fault. `/v1/models` carries the model's information,
  transcription takes WAV at any rate and answers `verbose_json` with segments, and the stop reason reaches the client
  in a `wav` response's headers and in the SSE stream's last event, and in a transcription's headers. The members of
  OpenAI's segment that FastConformer has no value for are left out rather than made up.

Writing the worker and the server settled what the design left open, each so that every request still gets its one
terminal message:

- A recognition request that had its `error` or its `cancelled` while it collected chunks drops its later chunk lines
  up to and including its `transcribe` line, or until a chunk 0 under its id starts a new request. A caller that
  cancels need not send the `transcribe`, and may reuse the id at once.
- A request still collecting chunks when stdin closes gets an `error`, since no `transcribe` can follow.
- `info` and `count_tokens` only read the model's information, which any thread may read while a request runs, so
  they are answered as they arrive rather than in turn: a caller counts the tokens of its next text while the current
  one speaks. `info` therefore gives the voices added so far.
- `add_voice` waits its turn as every request does. A cancel stops it while it waits; once it runs it ends with its
  answer, since the library has no way to stop it, and a voice that was added is reported as added.
- A `synthesize` to a recognition model, and a `chunk`, `transcribe` or `peek` to a synthesis model, are refused by
  the worker for their type, `unsupported` with the option `type`, before any request is made.
- A usage error of `speech worker` is answered with `fatal` on stdout as well as on stderr, so that a caller reading the
  protocol learns why the worker exits.
- The server draws the seed of a speech request that sets none, from the range the library draws from, and sets it on
  the request, since the headers of a `pcm` stream leave before the result that names the seed.

README.md gives the subcommands and their flags, the protocol's messages and the server's mapping of categories to
statuses.

The alternatives were turned down:

- Separate executables, as before. They repeat the same bytes in every release and read their command lines each their
  own way.
- `speech serve` in an executable of its own, to keep cpp-httplib out of the one ASIST runs.
- No protocol number, as docs/adr/0006 decided. A program that speaks the protocol without being a release has no
  release for its callers to pin.
- A heartbeat on a timer. It says the process lives, not that its work moves.
- A stream object in the C API now. Without a model that streams, it would do what a peek does through a second way
  into the library.
- 422 for a value out of range. OpenAI answers 400, and its clients handle 400.

## Consequences

ASIST and speech-bench rewrite their worker clients around one terminal message per request and read the model's
facts from `ready`; ASIST's watchdog allows for one step, which for a long recording's encoder runs for seconds.
Scripts change their commands, `speech-tts` to `speech tts` and `speech-tts make-voice` to `speech voice`. A release's
archives `speech-worker-<tag>-<platform>.zip` and `speech-cpp-tools-<tag>-<platform>.zip` become one,
`speech-<version>-<platform>.zip`, which CI builds and checks on every run, so ASIST downloads and pins that one and
starts `speech worker`. AGENTS.md's rule that the worker never includes cpp-httplib narrows to the library and
`libspeech`.
