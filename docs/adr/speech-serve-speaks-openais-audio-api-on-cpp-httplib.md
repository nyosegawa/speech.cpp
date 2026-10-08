# speech serve speaks OpenAI's audio API on cpp-httplib

## Context

A web app, a script in Python or a shell with curl can use the C API only through a binding, and the worker only by
starting a process and feeding it JSON Lines, while every one of them speaks HTTP. OpenAI's `POST /v1/audio/speech` and
`POST /v1/audio/transcriptions` are the requests most clients of speech already send, and Irodori-TTS-Server serves
Irodori-TTS behind the first. Some of what OpenAI's API asks for speech.cpp does not do: MP3 and the other compressed
formats, in either direction, and usage counted in tokens. The C++ standard library has no sockets, and ggml is the
one dependency of the library.

## Decision

`speech serve` serves the models over HTTP through the C API: `POST /v1/audio/speech`, `POST /v1/audio/transcriptions`,
`GET /v1/models` and `GET /health`. Its HTTP comes from cpp-httplib, one MIT-licensed header vendored in
`vendor/cpp-httplib/` with its license and pinned by the commit that CMakeLists.txt names. It is compiled into `speech`
alone; the library and `libspeech` never include it. cpp-httplib is a single header that needs nothing but the system's
sockets (Winsock on Windows), builds with MSVC, Clang and GCC, streams a response through a chunked content provider
while the handler's thread waits, and serves each connection from a thread pool; llama.cpp's server vendors it the same
way.

Where OpenAI's API leaves room or asks for what speech.cpp cannot do:

- `model` names the model of the endpoint's task, as `/v1/models` lists its id, or is left out. Any other name is a 404
  with `model_not_found`, as OpenAI answers an unknown model, so that a client asking for `tts-1` learns that it is not
  talking to it.
- **A speech request takes every option of the vocabulary as a member of its own name**, OpenAI's `speed` and
  `instructions` among them, with the C API's meaning, and a transcription the options `language`, `prompt` and
  `decoding` as fields of its form. A member speech.cpp does not take is a 400 rather than ignored; a member set to null
  counts as left out. A speech request without `seed` gets one the server draws from the library's range and sets on the
  request, since the headers of a `pcm` stream leave before the result that names the seed; `X-Speech-Seed` returns it,
  so the request can be repeated.
- **Speech is `wav` or `pcm`**, `wav` when left out, where OpenAI's default is `mp3`, which speech.cpp does not encode;
  `mp3`, `opus`, `aac` and `flac` are refused. `pcm` streams as the speech is made; `wav` waits for the whole speech,
  since its header holds the length. `stream_format: "sse"` carries `pcm` as `speech.audio.delta` and
  `speech.audio.done` events, and the done event carries the seed, the number of samples and the stop reason instead of
  the usage in tokens.
- **A transcription takes a WAV file alone**, at any rate, and refuses any other file. It answers `json`, `text` or
  `verbose_json`, which carries the languages the model heard and, from a model that gives times, the segments; the
  members of OpenAI's segment that the model has no value for are left out rather than made up. An upload may be 25 MB,
  OpenAI's limit.
- **`chunking_strategy` takes the form OpenAI's SDKs send**: `"auto"`, or the members of its `server_vad` object as one
  field each, `chunking_strategy[type]` and the rest, as openai-python 3.26.0 sends any object in a form.
  A JSON object in the one field is refused, and so is a member the object does not have.
- **The status and the error come from the library's category alone**, the same for the same mistake whenever it
  happens, with `param` from the input at fault: `invalid_argument`, `unsupported` and `out_of_range` are a 400, every
  other category a 500. The library checks a request before its work starts, so a refused request gets its error
  status before any audio. Once a stream has begun, a failure ends a `pcm` stream without its last chunk and an SSE
  stream with an error event.
- **The stop reason reaches the client** in `X-Speech-Stop` of a `wav` response and of a transcription, and in the SSE
  stream's done event.

A model serves one request at a time, in the order they arrive; a client that goes away while its request waits or
runs cancels it, so that the next request is not held up. The server listens on 127.0.0.1 unless told otherwise and has
no authentication and no TLS.

The alternatives were turned down:

- Writing HTTP by hand on the system's sockets. Request parsing, chunked transfer, keep-alive, timeouts and Winsock's
  differences are a protocol implementation of their own to get right and keep right, for nothing the project is about.
- An HTTP mode of the worker. The worker's protocol is the contract of every program that starts it, and a second
  protocol in the same mode would tie its callers to the needs of every HTTP client.
- A server in Python on a binding. It would need a Python environment where `speech` needs none.
- A larger library (Boost.Beast and Asio, Drogon, Crow on Asio). Each brings a dependency tree or a build system for
  features the server does not use.
- Decoding MP3, FLAC and the other formats OpenAI takes. It would need a decoder as a second dependency besides
  cpp-httplib, or a guess at the format.
- 422 for a value out of range. OpenAI answers 400, and its clients handle 400.

## Consequences

AGENTS.md allows this vendored header for the server and no other dependency for it. Updating cpp-httplib replaces the
header and its license from a newer commit and names that commit in CMakeLists.txt, in one commit. An OpenAI client
that asks for `mp3` or for another model gets an error rather than audio it did not ask for. A server reachable from
other machines goes behind a proxy that adds TLS and authentication. docs/server.md gives the members, the responses and
the mapping of categories to statuses.
