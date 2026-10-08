# speech serve speaks OpenAI's Realtime transcription over a WebSocket at /v1/realtime, with the session in the tools

## Context

A program that transcribes speech as it is said, such as the page's Live tab or a voice assistant, sends audio while it
records and wants each utterance's text soon after it ends. OpenAI's Realtime API does this over a WebSocket: a
transcription session takes `session.update`, `input_audio_buffer.append`, `.commit` and `.clear`, and answers with
`session.created`, `session.updated`, `input_audio_buffer.speech_started`, `.speech_stopped`, `.committed` and
`.cleared`, `conversation.item.input_audio_transcription.delta`, `.completed` and `.failed`, and `error`; with
`turn_detection` `server_vad` the server finds the utterances and commits them itself. This record follows the API
reference of these events as github.com/openai/openai-openapi gives it at commit 234829e (2026-10-07), the GA events,
whose transcription session takes 16-bit PCM at 24000 Hz alone. Its `input_audio_buffer.commit` and `.clear` act on
the buffer with `server_vad` too, and `speech_started` names the item that `speech_stopped` names, "unless the client
manually commits the audio buffer during VAD activation". openai-python 3.26.0 (2026-10-06) opens the WebSocket at its
base URL followed by `/realtime`, with `model` in the query when it is given, and no other member. NeMo-Speech.cpp serves
its own events at `/v1/audio/transcriptions/realtime`, under OpenAI's names.

cpp-httplib 0.59, which `speech serve` already vendors, has `Server::WebSocket`, upgraded after the server's pre-routing
handler has seen the request; its `read()` returns at a timeout the handler sets only between messages, and fails
within a message, so a timeout shorter than a slow client's message resets the connection. A browser applies no CORS to a
WebSocket and sends its page's Origin with it. The library gives a recognition's text once the recognition has ended,
and the assembly of utterances in the tools cuts audio that arrives a piece at a time into utterances, reads each again
while it goes on and recognizes it whole once committed
([its record](live-transcription-cuts-utterances-by-the-detections-regions-and-sends-as-deltas-the-text-two-readings-agree-on.md)).

## Decision

- **`/v1/realtime` is the address**, the one openai-python opens from a base URL of `http://HOST:PORT/v1`, so that an
  OpenAI client works by changing the address alone. Its query takes `model`, the recognition model's name, and nothing
  else.
- **The session lives in `tools/common`**: it reads and checks each client event, keeps the configuration, gives the
  audio to an assembly of utterances, and writes the server's events as JSON, the same events `speech asr --format json`
  writes. `tools/server` carries its messages over cpp-httplib's WebSocket and gives it the models held, so that another
  carrier, such as the worker, can carry the same session.
- **A session is a transcription session of 16-bit PCM at 24000 Hz.** It starts with `turn_detection` null, in which the
  client commits each utterance, the buffer appended since the last commit or clear. Each utterance is recognized whole
  on the recognition model held when it is committed, which it keeps, and in the turn it takes then among the requests
  on that model, so that a request that arrives after it runs after it, its audio resampled by the library as a file's
  is. A turn whose utterance never runs, because the session ended, is given up, so that the requests behind it go on.
- **`server_vad` cuts the utterances by the regions of the server's detection model**, with the options of transcription
  by regions: `threshold`, `prefix_padding_ms` as `speech_pad_ms` and `silence_duration_ms` as
  `min_silence_duration_ms`, defaulting to OpenAI's 0.5, 300 ms and 500 ms, and a longest region of 10 s. A
  `turn_detection` given is the whole of it, a member left out taking its default. Each region gives `speech_started`,
  `speech_stopped` and `committed` of one item, in that order, with `audio_start_ms` and `audio_end_ms` the region's
  bounds from the start of the session's audio, so that the same audio gives the regions and texts of `chunking_strategy`
  with the same values. A server without a detection model answers `server_vad` with an error that says to give
  `speech serve` one.
- **A commit or a clear while `server_vad` runs acts on the buffer**, as the reference has it: a commit recognizes it as
  an utterance, the item `speech_started` named where speech is under way, without `speech_stopped`; a clear drops it;
  and either begins the detection again on the audio that follows. With `server_vad` the buffer keeps only what a region
  may still need, so a commit between utterances recognizes a moment of silence.
- **Deltas carry the text two readings agree on**, for commits and `server_vad` alike: while the buffer grows, or a region
  goes on, the assembly reads it again and sends the beginning that two readings in a row agree on, and the completed
  event carries the recognition of the whole utterance, which replaces them. A reading takes its turn on the recognition
  model when it starts, as a request does: it holds no turn while it waits for audio, and runs after the commits accepted
  before it.
- **A session holds at most 25 MB of audio not yet transcribed**, in its buffer or committed: 546 s of its PCM, the limit
  of a file sent over HTTP. An append past it is an `error` event and adds nothing, so that one connection cannot take
  the server's memory; OpenAI's reference bounds an append at 15 MiB and says nothing of a total. A reading holds no
  audio of its own once the library's request has copied it, and with `server_vad` silence is not kept, so a session
  that listens for hours stays within the bound.
- **A `session.update` changes the members it gives**, as OpenAI's reference has it: a member of the transcription left
  out keeps its value, and one given as null takes it away. A session starts with the model the address names, so that
  naming it there or in `session.update` is the same. Everything else the reference defines is answered with an `error`
  event that names it, never ignored: a conversation session, `semantic_vad`, the members of `server_vad` that steer a
  response (`create_response`, `interrupt_response`, `idle_timeout_ms`), `noise_reduction`, the G.711 formats and
  other rates, `include`, `keywords`, `delay`, more than one language, and every other client event. A connection is
  refused before the upgrade, as an HTTP error, for another model or another query member.
- **The completed event carries** the usage in seconds of audio, the languages the model names (Qwen3-ASR) as OpenAI's
  `gpt-transcribe` gives them, and `stop`, speech.cpp's own, since a text cut at the model's limit reads like one that
  ended.
- **The WebSocket keeps the server's guards**: the Origin rule of every endpoint, which is what keeps other web pages out
  of a WebSocket, and while the server listens on a loopback address the Host rule of the page
  ([the page's record](speech-serve-has-a-page-guarded-by-a-loopback-host-and-the-origin.md)).
- **A replacement by the page reaches a session as it reaches a request**
  ([the record of a model of each task](speech-serve-holds-a-model-of-each-task-and-replaces-one-only-after-its-requests-end.md)):
  an utterance keeps the recognition model it was committed on until it ends, one committed while the new model loads
  fails with `model_loading`, and afterwards a session that named no model goes on with the new one while one that named
  the old, in the address or in `session.update`, fails with `model_not_found`. A session with `server_vad` lets the
  detection model go once the page replaces it, its detection ended and the utterance under way committed, and goes on
  with the new model. A thread of the session's looks every 0.2 s, so that the load does not wait for a silent session,
  while the WebSocket reads each message whole with the transport's own timeout.

The alternatives were turned down:

- An address and events of speech.cpp's own, as NeMo-Speech.cpp's `/v1/audio/transcriptions/realtime`. A program written
  for OpenAI's client would need code of its own.
- OpenAI's beta form, `intent=transcription` and `transcription_session.update`. openai-python 3.26.0 sends the GA events.
- Ignoring what speech.cpp does not do. A client that asked for `semantic_vad` would wait for commits that never come,
  and one that asked for another format would get the text of noise.
- One delta with the whole text once a commit is recognized. No text would come while someone speaks.
- A delta for each segment or token of a recognition. The library gives them all when the recognition ends, so they
  would add events without arriving sooner.
- A buffer of everything appended since the last commit while `server_vad` runs. A session open through minutes of
  silence would reach the bound and refuse audio.
- A detection that keeps its model for the session's life. The page's replacement of the detection model would wait for
  every session with `server_vad` to end.
- The session in `tools/server`. The command line and any other carrier would need a second implementation of the same
  events.

## Consequences

A client that relies on OpenAI's default turn detection finds `turn_detection` null in `session.created` and commits
itself, or asks for `server_vad`.
