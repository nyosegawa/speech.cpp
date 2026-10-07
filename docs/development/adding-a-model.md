# Adding a model

This page points to the decisions and the folders that a new model or family touches.

## Decisions to read first

- [ADR 0005](../adr/0005-every-program-speaks-through-one-c-api.md): every program reaches a model through the one C API.
- [ADR 0015](../adr/0015-a-model-is-one-gguf-file-with-its-codec-a-layout-version-every-constant-and-ggufs-standard-names.md):
  a model is one GGUF file with its codec, a layout version, every constant, and GGUF's standard names.
- [ADR 0014](../adr/0014-the-c-api-checks-requests-against-the-options-each-model-declares.md): a family declares the
  options it takes, and the C API checks requests against them.
- [ADR 0007](../adr/0007-a-request-sets-speed-and-length-only-where-the-model-can.md): an option only where the model can
  do it.
- [ADR 0004](../adr/0004-the-worker-names-languages-with-bcp-47-tags.md): languages are BCP 47 tags.
- [ADR 0011](../adr/0011-speech-recognition-is-a-task-of-every-entry-point.md): a task is served by every entry point.
- [ADR 0016](../adr/0016-input-audio-is-resampled-and-results-carry-stop-reasons-and-times.md): input audio is resampled,
  and results carry stop reasons and times.
- [ADR 0012](../adr/0012-the-recognizer-decodes-with-the-models-default-decoder.md): decode as the official code does by
  default.
- [ADR 0033](../adr/0033-a-model-is-named-from-a-catalog-each-release-carries-pinned-by-revision-and-sha-256.md): how a
  model is named in the catalog.
- [ADR 0009](../adr/0009-speech-recognition-runs-through-a-fastconformer-port.md) and
  [ADR 0018](../adr/0018-qwen3-asr-runs-in-speech-cpp-checked-against-its-windowed-encoder.md): how earlier ports chose
  their reference.

## Where the code goes

| Folder or file | What it holds |
|---|---|
| `reference/<model>/` | the uv environment that pins the official code, `pins.py`, `convert.py` and `dump.py` |
| `src/families/<family>/` | the architecture, with `layout.cpp`, what its reader takes |
| `src/<family>-engine.cpp` | the engine: the options the family takes and how a request reaches the family |
| `src/speech.cpp` | the table of families, where a new family is one line |
| `src/common/` | what two families use in the same role, moved there when the second one needs it |
| `checks/` | one check per stage, against the dumps |
| `tools/models/catalog.json` | the catalog, which `tools/models/update_catalog.py` pins |
| `docs/models/<family>.md`, `docs/models.md`, `docs/gguf.md` | the pages a user reads |

[checks.md](checks.md) shows how the existing ports are checked.
