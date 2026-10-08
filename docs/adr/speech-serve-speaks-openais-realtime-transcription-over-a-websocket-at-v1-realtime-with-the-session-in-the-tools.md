# speech serve speaks OpenAI's Realtime transcription over a WebSocket at /v1/realtime, with the session in the tools

## Context

A program that transcribes speech as it is said, such as the page's Live tab or a voice assistant, sends audio while it
records and wants each utterance's text soon after it ends. OpenAI's Realtime API does this over a WebSocket: a
transcription session takes `session.update`, `input_audio_buffer.append`, `.commit` and `.clear`, and answers with
`session.created`, `session.updated`, `input_audio_buffer.committed` and `.cleared`,
`conversation.item.input_audio_transcription.delta`, `.completed` and `.failed`, and `error`. This record follows the
API reference of these events as github.com/openai/openai-openapi gives it at commit 234829e (2026-10-07), the GA
events, whose transcription session takes 16-bit PCM at 24000 Hz alone. openai-python 3.26.0 (2026-10-06) opens the
WebSocket at its base URL followed by `/realtime`, with `model` in the query when it is given, and no other member.
NeMo-Speech.cpp serves its own events at `/v1/audio/transcriptions/realtime`, under OpenAI's names.

cpp-httplib 0.59, which `speech serve` already vendors, has `Server::WebSocket`, upgraded after the server's pre-routing
handler has seen the request. A browser applies no CORS to a WebSocket and sends its page's Origin with it. The library
gives a recognition's text once the recognition has ended. The worker is to carry the same session over JSON Lines.
Turn detection (`server_vad`) needs a detection that takes audio a piece at a time, which the library does not have yet.

## Decision

- **`/v1/realtime` is the address**, the one openai-python opens from a base URL of `http://HOST:PORT/v1`, so that an
  OpenAI client works by changing the address alone. Its query takes `model`, the recognition model's name, and nothing
  else.
- **The session lives in `tools/common`**: it reads and checks each client event, keeps the configuration and the audio
  appended, transcribes each commit on a thread of its own in the order of the commits, and writes the server's events as
  JSON. `tools/server` carries its messages over cpp-httplib's WebSocket and gives it the recognition model held, so that
  the worker can carry the same session.
- **A session is a transcription session of 16-bit PCM at 24000 Hz whose client commits each utterance.** A commit is
  transcribed whole on the recognition model held, in its turn among the HTTP requests on that model, its audio
  resampled by the library as a file's is. Everything else the reference defines is answered with an `error` event that
  names it, never ignored: a conversation session, `turn_detection` (`server_vad` too, until the detection that takes
  audio in pieces is in the library), `noise_reduction`, the G.711 formats and other rates, `include`, `keywords`,
  `delay`, more than one language, and every other client event. A connection is refused before the upgrade, as an HTTP
  error, for another model or another query member.
- **A commit's text goes as one delta, then the completed event**, since the library gives it when the recognition ends,
  and a client that builds the text from the deltas gets it whole. The completed event carries the usage in seconds of audio, the languages
  the model names (Qwen3-ASR) as OpenAI's `gpt-transcribe` gives them, and `stop`, speech.cpp's own, since a text cut at
  the model's limit reads like one that ended.
- **The WebSocket keeps the server's guards**: the Origin rule of every endpoint, which is what keeps other web pages out
  of a WebSocket, and while the server listens on a loopback address the Host rule of the page
  ([the page's record](speech-serve-has-a-page-guarded-by-a-token-a-loopback-host-and-the-origin.md)).
- **A replacement of the recognition model by the page reaches a session as it reaches a request**
  ([the record of a model of each task](speech-serve-holds-a-model-of-each-task-and-replaces-one-only-after-its-requests-end.md)):
  a commit keeps the model it runs on until it ends, a commit while the new model loads fails with `model_loading`, and
  afterwards a session that named no model goes on with the new one while one that named the old fails with
  `model_not_found`.

The alternatives were turned down:

- An address and events of speech.cpp's own, as NeMo-Speech.cpp's `/v1/audio/transcriptions/realtime`. A program written
  for OpenAI's client would need code of its own.
- OpenAI's beta form, `intent=transcription` and `transcription_session.update`. openai-python 3.26.0 sends the GA events.
- Ignoring what speech.cpp does not do. A client that asked for `server_vad` would
  wait for commits that never come, and one that asked for another format would get the text of noise.
- A delta for each segment or token. The library gives them all when the recognition ends, so they would add events
  without arriving sooner.
- The session in `tools/server`. The worker would need a second implementation of the same events.

## Consequences

`server_vad` comes with the detection that takes audio a piece at a time, on the same session. A client that relies on
OpenAI's default turn detection finds `turn_detection` null in `session.created` and commits itself.
