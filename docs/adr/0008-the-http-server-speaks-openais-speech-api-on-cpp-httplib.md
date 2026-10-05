# The HTTP server speaks OpenAI's speech API on cpp-httplib

Superseded in part by docs/adr/0015: the server takes one model file, its codec included.

Decided 2026-10-05.

## Context

speech.cpp reaches a program through the C API or the worker. A web app, a script in Python or a shell with
curl can use neither without a binding or a process to start and feed JSON Lines, while every one of them
speaks HTTP. OpenAI's `POST /v1/audio/speech` is the request most text-to-speech clients already send, and
Irodori-TTS-Server serves Irodori-TTS behind it. The repository's rule was ggml as its one dependency, and the
C++ standard library has no sockets.

## Decision

`speech-server` is a program of its own on the C API, like the worker, and serves one model over HTTP:
`POST /v1/audio/speech` as OpenAI defines it, `GET /v1/models` and `GET /health`. Its HTTP comes from
cpp-httplib, one MIT-licensed header vendored in `vendor/cpp-httplib/` with its license and pinned by the
commit that CMakeLists.txt names. It is the one dependency besides ggml, and only the server includes it:
the library, `libspeech` and the worker stay free of it, and the server is still one file with everything
linked in.

cpp-httplib was chosen because it is a single header that needs nothing but the system's sockets (Winsock
on Windows), builds with MSVC, Clang and GCC, streams a response through a chunked content provider while
the handler's thread waits, and serves each connection from a thread pool. llama.cpp's server vendors it the
same way.

What the server does where OpenAI's API leaves room or asks for what speech.cpp cannot do:

- `model` names the loaded model (`speech_model_name()`, the `id` that `/v1/models` lists) or is left out.
  Any other name is a 404 with `model_not_found`, as OpenAI answers an unknown model, so that a client asking
  for `tts-1` learns that it is not talking to it.
- `response_format` is `wav` or `pcm`, and `wav` when it is left out; OpenAI's default is `mp3`, which
  speech.cpp does not encode. `mp3`, `opus`, `aac` and `flac` are refused. `pcm` streams as it is made;
  `wav` waits for the whole sentence, since its header holds the length.
- `stream_format: "sse"` carries `pcm` only, as `speech.audio.delta` and `speech.audio.done` events.
  OpenAI's done event carries the usage in tokens, which speech.cpp does not count, so it carries none and
  gives the number of samples instead.
- `seed`, `language`, `seconds` and `duration_scale` are speech.cpp's own members, with the C API's meaning.
  A request without `seed` gets a random one, and the response names it in `X-Speech-Seed`, so a request can
  be repeated; the worker's `--seed`, which numbers a session's requests, has nothing to count across
  independent clients.
- A member speech.cpp does not take, `instructions` among them, is refused with a 400 rather than ignored;
  a member set to null counts as left out.
- The library checks a request before its first callback, so an error before then is a 400 with the
  library's message, and one after it a 500. Once a stream has begun, a failure ends a `pcm` stream without
  its last chunk and an SSE stream with an error event.

The model speaks one request at a time, in the order they arrive; the others wait. A client that goes away
while it waits is skipped, and one that goes away while its audio streams cancels its synthesis with
`speech_cancel()`, so that the next request is not held up. The server listens on 127.0.0.1 unless told
otherwise, sends CORS headers only to the origins given with `--cors-origin`, and has no authentication.

The alternatives were turned down:

- Writing HTTP by hand on the system's sockets. Request parsing, chunked transfer, keep-alive, timeouts and
  Winsock's differences are a protocol implementation of their own to get right and keep right, for nothing
  the project is about.
- An HTTP mode in the worker. docs/adr/0005 turned it down: the worker's protocol is ASIST's contract, and
  a second protocol in the same process would tie every new caller's needs to it.
- A server in Python on a binding. It would need a Python environment where the C++ tools need none, and the
  release would no longer be one file per tool.
- A larger library (Boost.Beast and Asio, Drogon, Crow on Asio). Each brings a dependency tree or a build
  system for features the server does not use.

## Consequences

AGENTS.md allows this one vendored header for the server and no other dependency. Updating cpp-httplib
replaces the header and its license from a newer commit and names that commit in CMakeLists.txt, in one
commit. The server answers an OpenAI client that asks for `mp3`, for `instructions` or for another model with
an error rather than audio it did not ask for. It speaks HTTP without TLS; a server reachable from other
machines is put behind a proxy that adds it, and authentication with it.
