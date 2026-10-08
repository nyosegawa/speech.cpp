# The worker speaks JSON Lines with a protocol number of its own and one terminal message per request

## Context

`speech worker` serves one model to a program that starts it and talks to it over its stdin and stdout, as ASIST does.
Programs that are not a release of speech.cpp speak the same protocol: speech-bench's adapters run the official
Irodori-TTS and mlx-audio behind it, each implementing what its callers use, and their callers pin no release of
speech.cpp.

A caller must not wait for an answer that will not come, and must tell a hung worker from a busy one: ASIST takes 30 s
of silence as a hung worker, while between Irodori-TTS's sampler steps and through a recognition no audio flows, and the
encoder of a long recording runs as one step (3.7 s for 311 s of audio with reazonspeech-nemo-v2 in F16 on an Apple M5).

## Decision

- **JSON Lines**: one JSON object per line on stdin and on stdout. stdout carries the protocol and nothing else, since
  `speech` points descriptor 1 at stderr, and every log goes to stderr; a caller treats a line on stdout that is not a
  JSON object as a defect of the worker and fails.
- **The protocol has a number of its own** in `ready`, beside the release and the model's information. It rises when a
  caller must change to keep working; an added member or message does not raise it, since a caller ignores what it
  does not know.
- **The worker reads whole JSON and refuses a member that the message's type does not have**, a misspelled option
  included.
- **Every request gets exactly one terminal message**, `end`, `error` or `cancelled`, and nothing for its id after it.
  A cancel has no answer of its own and no effect on a later request. An error carries the library's category and the
  input at fault; the `end` of a synthesis carries its seed, its samples and its stop reason, and that of a recognition
  its text and stop reason.
- **Progress comes from the library's own progress** while a request passes no audio, at most once a second, so that
  it says that the work moves, not only that the process lives.
- **Audio is base64 of 16-bit PCM** in both directions. A recognition request is its `chunk` lines followed by its
  `transcribe` line, which gives the chunks' rate; the library resamples.

Where a request could otherwise get no answer or two, the worker answers so:

- A recognition request that had its `error` or its `cancelled` while it collected chunks drops its later chunk lines
  up to and including its `transcribe` line, or until a chunk 0 under its id starts a new request. A caller that
  cancels need not send the `transcribe`, and may reuse the id at once.
- A request still collecting chunks when stdin closes gets an `error`, since no `transcribe` can follow.
- `info` and `count_tokens` only read the model's information, which any thread may read while a request runs, so they
  are answered as they arrive rather than in turn: a caller counts the tokens of its next text while the current one
  speaks.
- `add_voice` waits its turn as every request does. A cancel stops it while it waits; once it runs it ends with its
  answer, since the library has no way to stop it.
- A request of the other task is refused by the worker for its type, `unsupported` with the option `type`.
- A usage error of `speech worker` is answered with `fatal` on stdout as well as on stderr, so that a caller reading
  the protocol learns why the worker exits.

The alternatives were turned down:

- A prefix on every line of the protocol, to tell it from what other code prints. Nothing else reaches stdout, and
  every caller would have to learn the prefix.
- The release's number alone, without a number of the protocol's own. A program that speaks the protocol without being
  a release has no release for its callers to pin.
- Ignoring a member the worker does not know. A misspelled option would be dropped without a word.
- A heartbeat on a timer. It says the process lives, not that its work moves.
- A `peek` that recognizes the audio a request has collected so far and answers with interim text, the request staying
  open for more chunks. No caller used it: ASIST shows interim text by transcribing the growing utterance with requests
  of its own, which is what a peek did. Its cost grows with the audio, since a model speech.cpp runs reads the whole
  audio again (0.07 s for 6.4 s and 0.11 s for 10.5 s with parakeet-tdt_ctc-0.6b-ja in F16 on Metal), and transcribing
  speech as it is said takes knowing where each utterance ends, which re-reading a buffer does not give.
- One line with the whole audio of a request. Chunks keep lines small, let a caller send the microphone's audio while
  it records, and mirror what a synthesis worker sends.
- Float samples on the protocol. One form of audio in both directions is simpler for every caller.

## Consequences

A caller allows for the longest single step of its model as silence. docs/worker.md gives the messages, and
`checks/smoke/worker_client.py` is a client that checks every line and one terminal message per request.
