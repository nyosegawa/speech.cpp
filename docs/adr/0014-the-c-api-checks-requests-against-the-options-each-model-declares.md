# The C API checks requests against the options each model declares

Decided 2026-10-06.

## Context

A caller could not learn what a model takes without trying. Speed and length were fields of `speech_request`, and
Qwen3-TTS's context and Irodori-TTS's voices and steps fields of `speech_model_params`, each one family's, with that
family's ranges written in the general header. Nothing could be read before a model was loaded, and loading ran a
synthesis. The callers kept tables of their own, and the tables drifted: speech-bench refuses `durationScale` for
speech.cpp's Irodori-TTS, which has taken it since 0.5.0; ASIST holds Qwen3-TTS's speakers, languages and request
size. The same option was `duration_scale` in the C API and `durationScale` in the worker, which ignored members it
did not know. A language given to a model that does not take one was checked and then not used.

The structs had no size and were returned by value, so every added field broke the ABI, and a ctypes binding with an
older layout read the wrong memory instead of failing. Errors were `SPEECH_ERROR` and a sentence, so the server
guessed its HTTP status from timing, and two failures ended the process: choosing the BLAS device the list offered,
and a GGUF key of the wrong type. Cancellation was per model, and a cancel issued between two requests was lost or
stopped the next one. `SPEECH_API_VERSION` did not count added functions, the shared library had no SOVERSION, the
device "" meant the CPU on a machine without a GPU, and every device call set `GGML_METAL_TENSOR_DISABLE` for the
whole process.

Elsewhere: OpenAI refuses a parameter a model does not take with its name and accepts the neutral value, and once
ignored `speed` on gpt-4o-mini-tts for five weeks of user reports. CrispASR answers an unsupported setter with -2,
which its own Python binding treats as success. audio.cpp declares each family's options with their defaults and
bounds, reads that one declaration to validate and to answer questions, and keeps a major and a minor ABI version
for bindings that bind their imports up front.

## Decision

- **One vocabulary, declared per family.** Every request option is a value of `speech_option` with a fixed
  snake_case name, which the worker, the server and the command line use as it is. Each family declares in one table
  which options it takes, with their types, defaults, ranges or choices, reading the model's own defaults and bounds
  from its file. The setters, the model information, the worker's `ready`, the server's `/v1/models` and `speech info`
  all come from that table, so there is no second copy to drift, and a misspelled option is a compile error in C or
  an unknown name, never an answer of "not supported".
- **Neutral values pass, anything else unsupported is an error.** Speed 1, duration scale 1, language `auto` and no
  timestamps are accepted by every model, so a caller that knows nothing of a model can still send a request; any
  other value of an option a model does not take is refused, with the option named, because the prior art shows
  every silent case reported as a defect.
- **A language steers or is checked.** A model that has one language or finds it itself checks a given language and
  does not use it, and its declaration says so. OpenAI's clients send `language` as a hint, and refusing `ja` for a
  Japanese recognizer would break them.
- **Requests are objects whose setters check each value.** A request is made for one model; a setter refuses a value
  the model does not take as it is set, and the request checks what only the whole request shows before any work. An
  added option never changes the ABI, a binding mirrors no struct, and the mistake fails at the call that made it. A
  request runs once and is cancelled by itself, which removes the race between requests.
- **The information is read without loading**, from the file's metadata, and the same information comes from a
  loaded model as a snapshot with its device, threads and added voices. The library writes it as one JSON object
  that every entry point carries as it is, so the worker's and the server's spellings cannot part again.
- **Errors carry a category and the input at fault**, and no input reaches a ggml abort: the device list leaves out
  accelerators that cannot run a model, and GGUF reads check types (docs/adr/0015). A server can derive its status,
  and ASIST its own message, from the category alone.
- **Loading takes one path and what is the same for every family.** The device is `auto` by default and reported, and
  a device asked for by name or as `gpu` that is not there is an error, never replaced. Threads are a preference
  about how the work runs: they apply to what the model computes on the CPU, and the information reports the number
  in effect, since with `auto` a caller cannot know beforehand where the model will run. The warm-up is a switch, off
  in the library and on in the worker and the server, because a one-shot run gains nothing from computing twice.
  Steps become a request option, voices are added after loading, and the context goes, its limit now in the file.
- **The API has a major and a minor version**, starting at 3.0 after 2, and the shared library's SOVERSION is the
  major, so that a binding that binds its imports up front can tell an older library lacking a function.
- **The process is left as it was found.** Logs go to a callback the host sets. `GGML_METAL_TENSOR_DISABLE`, which
  ggml reads only while it lists its devices and for which it has no API, is set for the library's own first listing
  and then restored, so that a host's own ggml or llama.cpp keeps its tensor API. ggml's Metal backend itself sets
  `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` for the process during that listing, a workaround of its own for long command
  buffers, which the library leaves as ggml sets it.

docs/c-api.md lists the options each family takes, the JSON form of the information and the names of the error
categories.

The alternatives were turned down:

- Structs with a leading size, read by offset, as Piper and the Linux kernel do. A binding still mirrors each
  version's layout, and a wrong value is still found only when the request runs.
- Ignoring an option a model cannot follow. The caller gets audio other than what it asked for and no sign of it.
- Refusing a language on a model that does not steer by it. OpenAI's clients would fail on the language they send as
  a hint.
- Options and capability questions as free strings, as audio.cpp has them. A misspelled question is answered "no".
- A request type per family, as whisper.cpp's `parakeet.h` and sherpa-onnx's sub-structs per model. Most options mean
  the same in every family, and the API would grow with each one.
- One API version that additions do not raise, as before. A binding that binds up front cannot tell a missing
  function.
- Setting `GGML_METAL_TENSOR_DISABLE` for good, as before. It reaches a host's own ggml.

## Consequences

ASIST and speech-bench read voices, languages, ranges and limits from a model's information instead of their tables.
A request that set Irodori-TTS's `seconds` above 30 s with a `speed` that brought it within the bounds, which
docs/adr/0007 accepted, is refused, and so is a duration scale that takes the predicted length outside them. A host
that reads the environment on another thread must not do so while speech.cpp first lists its devices. AGENTS.md's
rule on `SPEECH_API_VERSION` becomes the rule of the two numbers.
