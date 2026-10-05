# Every program speaks through one C API

Decided 2026-10-05.

## Context

speech.cpp was written for one caller: ASIST starts `speech-worker` and talks to it in `ASIST_JSON:` lines.
The way into the models, the C++ `Engine` with `make_qwen3_tts()` and `make_irodori_tts()`, lived in the
worker's folder, so no other program could use it. speech.cpp is now to serve anyone building a program
that speaks: an OpenAI-compatible HTTP server, one command-line tool, a Python binding and Linux releases
are to follow, each on the same foundation.

## Decision

`include/speech.h` is one C API, and every program and binding speaks through it, the worker included, so
the API ASIST runs every day is the one others get. It is plain C: opaque handles, UTF-8 strings, errors as
return codes with the message from `speech_last_error()`, every returned pointer owned by the library, and
a version (`SPEECH_API_VERSION`). A model serves one request at a time; `speech_cancel()` stops it from any
thread, and the audio callback stops it from the synthesizing thread.

The library lives in `src/`: the C API, the engine of each family behind it, `src/families/` and
`src/common/`. `tools/` holds the programs, and `checks/` the checks, which are built but not released. CMake
builds the library twice from the same sources, statically into every executable, so that `speech-worker`
stays one self-contained file, and as the shared `libspeech`, which exports the `speech_*` functions alone.

The alternatives were turned down:

- Keeping the C++ `Engine` as the only interface. A C++ interface with `std::function` and `std::string`
  has no stable ABI across compilers and standard libraries, and a Python binding or another language
  would need a C layer anyway.
- A C API alongside an unchanged worker. The worker, the one caller in daily use, would not exercise the
  API, and the two ways in would drift apart.
- The worker gaining an HTTP mode. It would tie every new caller to a process and a protocol, and the
  worker's protocol is ASIST's contract, which other callers' needs should not change.

## Consequences

The worker's behavior and protocol stay as they were: the same audio for the same request and seed, and
the same cancellation. Three fatal messages change their words, since the library writes them: options
that do not fit the model no longer name the worker's flags, and a GGUF of another architecture is one
"speech.cpp does not run". A cancel of the worker takes effect at the same points as before, because the
audio callback is also called without audio between Irodori-TTS's sampler steps. `qwen3-tts` and
`irodori-tts` still call their family directly until a command-line tool on the C API replaces them. A
change to the API that a caller notices raises `SPEECH_API_VERSION`.
