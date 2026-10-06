# The worker speaks plain JSON Lines, and VERSION names the release

Superseded in part by docs/adr/0017: the worker's protocol has a version of its own.

Decided 2026-10-05.

## Context

The worker prefixed every line on stdout with `ASIST_JSON:`. The prefix came from ASIST's earlier Python
worker, whose libraries printed to stdout, so ASIST had to tell its messages from their output. speech.cpp
is C++ and sends ggml's logs to stderr, so the reason was gone, and to any program other than ASIST the
name meant nothing. docs/adr/0005 called the protocol ASIST's contract; speech.cpp now serves other programs
too, and the protocol is theirs as well.

The release's number lived nowhere in the code. Tags were cut by hand, and nothing a caller received said
which speech.cpp it was talking to.

## Decision

The worker speaks plain JSON Lines: stdout carries one JSON object per line and nothing else, and every log
goes to stderr. The worker keeps the stdout it was started with for the protocol and points descriptor 1 at
stderr before anything else runs, so whatever ggml, a system framework or a GPU driver prints to stdout
reaches stderr. A caller treats a line on stdout that is not a JSON object as a defect of the worker and
fails. There is no prefix, no compatibility mode and no option to bring the prefix back.

The release's number is written in one place, the file `VERSION` at the root of the repository. CMake reads
it for `project()` and passes it to the library, `speech_version()` returns it, and the worker's `ready`
message carries it as `version`. Versions follow Semantic Versioning: while they are 0.x, a change a caller
notices and must adapt to (the worker protocol, the C API, the GGUF layout, a voice file's form, a tool's
arguments) raises the minor version, and anything else that is released raises the patch. The release is the
tag `v<VERSION>`, and CI refuses to publish a tag that differs from the file. Removing the prefix breaks the
protocol, so this release is 0.4.0, after 0.3.1.

The alternatives were turned down:

- Keeping the prefix. It guards against output that no longer reaches stdout, and every caller would have to
  learn a name that refers to one of them.
- A protocol version apart from the release's. The worker, the C API and the files change together in one
  release, and a caller pins a release; a second number would have to be raised by the same rule as the
  minor version and would only say less.
- Writing the number by hand in the CMake file, the header, the worker and the workflow. The copies would
  drift, and a release could report a number other than its tag.

## Consequences

ASIST and speech-bench read the prefix and must change to read plain lines before they take 0.4.0; a worker of
0.4.0 is not understood by a caller of 0.3.x, nor the other way round. `SPEECH_API_VERSION` stays 1, since
adding `speech_version()` changes nothing an existing caller uses. `--devices` prints its one JSON object as
before, without the prefix.
