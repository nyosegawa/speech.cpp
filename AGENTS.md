# AGENTS.md

## Project

speech.cpp runs speech models in C++ on [ggml](https://github.com/ggml-org/ggml), on macOS arm64 with Metal,
Windows x64 with Vulkan and Linux x64 with Vulkan or the CPU alone, as a library with one C API,
`include/speech.h`, for any program that speaks text or recognizes speech: ASIST's worker, other tools, bindings
and other people's applications. Each model's official
implementation is the reference: every stage of a port is checked against tensors dumped from it.
README.md and `docs/` are the documentation for users; what a developer needs is in the code, this file and
`skills/`, and `docs/adr/` keeps the decisions. Read the relevant implementation and its check before changing behavior.

## Architecture

- `include/speech.h` is the C API and the way into the library for every program and binding: plain C,
  opaque handles, UTF-8 strings, errors as statuses whose category says what failed, with the message from
  `speech_last_error()` and the input at fault from `speech_last_error_option()`, and no C++ exception or type
  crossing it. The API has two versions: a change an existing caller notices raises `SPEECH_API_VERSION_MAJOR`,
  which is the shared library's SOVERSION, and an added function, option or enum value raises
  `SPEECH_API_VERSION_MINOR`.
- `src/speech.cpp`, `src/info.cpp`, `src/request.cpp` and `src/voice.cpp` implement the C API over `src/engine.h`,
  the interface of one family behind it, with one engine per family (`src/<family>-engine.cpp`). An engine declares
  in one table the request options its family takes, with their defaults and ranges read from the model file, and
  turns a request checked against that table into the family's; the setters, the model information and its JSON read
  the table and nothing else. One table in `src/speech.cpp` lists the families: their task (synthesis or
  recognition), the layout their reader takes, which names the `general.architecture` of their files, their engine
  and, for a family that takes voice files, how it makes one. `speech_model_load()` and `speech_model_info_open()` choose the family from
  the table, and adding a family is adding its line.
- `src/families/<family>/` holds the code of one architecture, whichever weights it is given: `qwen3-tts/`
  runs Qwen3-TTS 0.6B and 1.7B. A family reads its model's one GGUF file, its codec included, and turns text into
  audio or audio into text; it knows nothing of the C API, the worker protocol or the command line. Its
  `layout.cpp` says what its reader takes: the architecture, the layout version, every key with its type, and the
  tensors the keys call for.
- `src/common/` holds what two families use in the same role. Code moves there when a second family needs
  it, not before, and never as a framework for families that do not exist yet.
- `tools/` holds `speech`, the one executable for users, with a subcommand per program: `tts`,
  `asr`, `voice`, `info`, `devices` and `quantize` in `tools/cli/` (with `main.cpp`, which dispatches them), `worker` in
  `tools/worker/` and `serve` in `tools/server/`, and what they share in `tools/common/`: the one parser of every
  command line, which makes a flag of each option of the C API's vocabulary, the JSON reader, and the request options
  read from text or JSON and set through the library's setters. Every subcommand reaches the models only through the
  C API. `tools/models/` names models: the catalog of the release (`catalog.json`, which `update_catalog.py` writes
  from Hugging Face and the build compiles in), the cache folder, and the fetching of a named model with the system's
  curl, which `models`, `pull` and `rm` and every subcommand that takes a model share; the library never reaches the
  network. The worker's protocol 2 is JSON Lines, one JSON object per line on stdin and
  stdout, and is the contract of every program that starts it, ASIST among them: stdout carries the protocol and
  nothing else, every log goes to stderr, and every request gets exactly one terminal message. Its `ready` message
  carries the protocol's version, the release and the model's information.
- `checks/` holds one check per ported stage (`*-check.cpp`) that compares the stage with the reference
  dumps, and `speech-api-check`, which runs the C API through the shared library with a synthesis model, Irodori-TTS's
  own rules and voice files in `speech-api-irodori.c`, and, with `transcribe`, with a recognition model
  (`speech-api-recognition.c`), what they share in `speech-api-common.c` and what differs by the operating system in
  `speech-api-platform.c`. Checks reach into
  `src/` for the stage they check; they are built but not released.
- `tools/server/` holds `speech serve`, which serves a synthesis model and a recognition model over HTTP with
  OpenAI's audio API (`POST /v1/audio/speech`, `POST /v1/audio/transcriptions`, `GET /v1/models`, `GET /health`)
  for programs that speak HTTP. Like the worker it reaches the models only through the C API; `openai-api.cpp` reads
  OpenAI's requests and writes its errors, mapped from the library's categories alone, and its stream events,
  `jobs.h` runs one request at a time in arrival order and cancels the request of a client that goes away, and
  `served-models.cpp` holds one model of each task and replaces it once its requests have ended. `page/` is the page
  of `speech serve --open`, plain HTML, CSS and JavaScript modules compiled into `speech`, which `page.cpp` serves
  with the endpoints that fetch and load models; `access.cpp` guards them with the token, the Host and the Origin.
- The smoke scripts in `tools/` drive each entry point as its caller does and fail on a defect:
  `worker_smoke.py` and `worker_recognition_smoke.py` the worker protocol 2 through `worker_client.py`, which checks
  every line and one terminal message per request, `server_smoke.py` every endpoint and the mapping of errors and
  `server_page_smoke.py` the page's endpoints and every guard, both through `server_client.py`, which stops the
  server however the script ends, `speech_cli_smoke.py` the command line against the worker, and `models_smoke.py`
  the naming, fetching and removing of models against a temporary model folder.
- `reference/<model>/` holds, per model, a uv environment that pins the official code, PyTorch and the rest,
  `pins.py`, which pins the checkpoints by revision, the conversion of the official weights to one GGUF file per
  model, and the scripts that run the official implementation to dump reference tensors. Dumps go to
  `reference/<model>/out/`. `reference/resample/` does the same for the resampler of `src/common/`, against
  torchaudio.

Keep these boundaries explicit: code does not reach past its module for an operation that belongs to
another one.

## Code

- C++17 with ggml as a git submodule and no other dependency, except cpp-httplib's single header, vendored in
  `vendor/cpp-httplib/` by commit for `speech serve` and compiled into `speech` with it; the library and `libspeech`
  never include it. `speech` has the library and ggml linked in statically, so that a release is
  one executable beside the shared library `libspeech`, which exports the C API and nothing else, for bindings and
  other programs.
- Anything the C API returns is owned by the library, and the header says for how long. A model serves one
  request at a time, and the header says which functions any thread may call.
- The GGUF layout is this repository's own: one file per model, its codec included, written by
  `reference/<model>/convert.py` and read by the family's `layout.cpp`, with `speech.layout` naming its version.
  The model's identity and languages are in the GGUF specification's own `general.` keys (ggml's `docs/gguf.md`),
  under the names it gives them, and the converter names the file from them by its naming convention; no key of
  speech.cpp's own holds a fact the specification has a key for. Every key is required, but for `general.finetune`
  and `general.version` where a model has none and `general.quantization_version` where it is not quantized, and has
  one type, and the tensors are exactly the ones the keys call for. A change of layout changes the converter and the
  reader in the same commit, and a change a reader of the previous layout cannot read raises `speech.layout`, adds the
  release that reads it to the converter's table and brings the previous layout up in the family's one upgrade
  function. Every model constant lives in the file, from the checkpoint or, where it has none, from the official code,
  and the converter says where.
- Do not add fallback behavior; fail loudly rather than degrade silently. A GGUF without a key or tensor, a key
  of another type, a tensor the keys do not call for, a layout the reader does not know,
  a text longer than the model takes, a WAVE format that is not understood and a device that does not
  start all throw an `Error` of their kind with a message (`src/common/error.h`), naming the input at fault where
  there is one; the C API returns each with its category, which every subcommand passes on: the worker as `error` or
  `fatal`, the server as OpenAI's error by the category alone, and the command line as `speech: <code> (<option>)`. Nothing
  is truncated or moved to another device behind the caller's back, and the library checks what it is given before
  ggml sees it, so that no input makes ggml abort the process.
- Fix a defect where its cause is, in a form in which it cannot happen, rather than with a guard for the
  one case that showed it; the code after the fix reads better than before. When a fix is much larger
  than the defect, or needs a choice only the user can make, stop and ask instead of patching.
- Persist each fact in one authoritative place and derive the rest from it.
- Every download is pinned: a Hugging Face repository by revision, a git dependency by commit, Python
  packages by `uv.lock`, the Vulkan SDK in CI by version.
- Extract code only when it creates a coherent responsibility, a reusable boundary or an independently
  checkable unit. Introduce a shared abstraction only after two current implementations show the same
  responsibility with meaningful variation.
- A file approaching 500 lines calls for a review of its responsibilities; split it when a coherent one
  can be extracted, not to meet a line count.
- On Windows, `speech` reads its command line as UTF-8, sets stdin and stdout to binary, and defines
  `NOMINMAX` before `windows.h`.
- A path is a UTF-8 string from the command line to the file. A C stream opens it with `ggml_fopen()` and a
  C++ stream through `std::filesystem::u8path()`: `fopen()` and a stream opened on a `std::string` read the
  path in the ANSI code page on Windows, so a path with any character outside ASCII is not found.
- A Linux release runs on glibc 2.34 and needs no shared library but glibc's and, in the Vulkan build,
  `libvulkan.so.1`; CI fails a build that needs more. It is built on the oldest Ubuntu GitHub
  hosts, with libstdc++ linked in and ggml's OpenMP off.
- Model weights, reference dumps and audio are never committed; `.gitignore` covers `models/`,
  `reference/*/out/`, `*.gguf` and `*.wav`.

## Comments

- Write comments in English, in full sentences and the present tense. Other languages appear only as
  data, quoted verbatim.
- Say only what the code cannot: the layout of a tensor, a quirk of ggml or of the official implementation
  and the failure it causes, a measured value with its condition and date, a constraint, or why the
  simpler approach was rejected. Do not restate names, narrate steps, or mention history, TODOs, docs,
  tickets or conversations.
- Doc comments (`/** */`) on the declarations in headers and on non-obvious declarations in a file,
  without parameter or return tags. A tool starts with a comment that says what it does and its usage.
- Inside a function, a comment goes on its own line above the code. No trailing comments and no banner
  comments.

## Tests

- The checks are the tests. A check runs one stage on the reference dump's inputs, compares its output
  with the dump, prints the error (SNR, largest difference, argmax agreement), and exits with a failure
  when the error is beyond what the stage's arithmetic explains.
- A failing check points to a defect. A check does not fail on a deliberate change of wording, a name or a
  configured value, and does not only assert that something exists.
- Add or update the check of a stage whose behavior changes, and run it on the CPU in F32 and on the GPU
  before committing. A change to code that runs on the GPU is also run on Vulkan, on the Windows machine.
- A port of a model checks each stage in the order data flows, with the stage's own inputs from the dump,
  so that the first stage that departs is the one reported.
- Checks need weights and dumps that are not in the repository, so CI builds them but does not run them.

## Text

- The output of the tools, README.md, `docs/` and AGENTS.md are English. Japanese and other languages appear only
  as data: texts to speak and model output.
- An error message names what failed and what to do.

## Decisions

`docs/adr/` keeps the decisions the code cannot show, one file each, named by what it decides and without a number;
the file names are the index. Before changing a behavior, list the folder and read the records whose names cover
it; if the change contradicts one, say so to the user first. Before committing, ask whether the work
settled a choice or turned an approach down for good; if so, the record goes into the same commit (`adr` skill). A
record states the decision as it stands, and nothing else names a record: not the docs, this file, a skill or a
comment.

## Workflow

- Build with `cmake -B build && cmake --build build -j`, and run the checks the change touches before
  committing code. A change to the C API or the worker also runs `speech-api-check` and
  `tools/worker_smoke.py` for both synthesis families, and `speech-api-check transcribe` and
  `tools/worker_recognition_smoke.py` for both recognition families; a change to the server runs
  `tools/server_smoke.py` and `tools/server_page_smoke.py`, one to the command line or the parser
  `tools/speech_cli_smoke.py`, with a model of each family, and one to `tools/models/` `tools/models_smoke.py`.
- Never commit on main. Every change reaches main through a pull request, one coherent unit each: a
  model's stage, a fix, a refactor or a documentation change.
- Commit messages and pull request titles are one English sentence in the imperative, without a prefix
  such as `feat:`; the body says what changed, why, and how it was checked.
- The user merges pull requests, with a squash, once CI passes. An agent merges only when told to.
- The release's number is written in `VERSION` and nowhere else; CMake reads it, and `speech_version()` and
  the worker's `ready` report it. Versions follow Semantic Versioning: while they are 0.x, a change a caller
  notices and must adapt to (the worker protocol, the C API, the GGUF layout, a voice file's form, the command
  line's arguments) raises the minor version, and anything else that is released raises the patch. The worker's
  protocol has a version of its own, in `ready`, raised when a caller must change to keep working.
- `VERSION` is raised by a pull request of its own just before a release ("Raise the version to 0.5.0"), to
  the number the changes since the last tag call for; a change does not raise it by itself.
- The tag `v<VERSION>` builds the release in CI, which refuses a tag that differs from `VERSION`. Releases and
  tags are never deleted or moved: callers such as ASIST pin them by SHA-256.
- Converted GGUF files go to Hugging Face only with the user's approval, one repository for each upstream
  repository, named after it with `-GGUF` (sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF for
  Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice). A repository holds one file per model and type, the codec inside it, under
  the name its converter gives it (Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf), and the licenses of what it holds;
  its card says that only speech.cpp reads it. Beside each GGUF file goes the output of `speech info --json` for it,
  under the file's name with `.json` added. A changed file goes up under the same name, with its JSON made again, and
  its card's SHA-256 changes with it.
- The tag's archives, `speech-<VERSION>-<platform>.zip`, are the ones CI builds and checks on every run; a change to
  what they hold changes the packaging steps of `.github/workflows/build.yml` and `docs/install.md`'s Release archives
  together.
- When a change alters what a user does or sees (the C API, the command line's arguments, the worker protocol, the
  HTTP server, the GGUF layout, the models), the same pull request updates the pages that describe it: README.md where
  it shows the change, and the page of `docs/` that covers it. No test checks the documentation against the code; the
  pull request keeps it current.

## Skills

Skills live in `skills/`; `.claude/skills` and `.agents/skills` link to it. Whenever a task matches one, use it; each
holds steps these rules do not repeat.

- `add-model`: porting a model or a family, updating one, or exposing an official option of one.
- `adr`: a decision settled or an approach turned down for good, a change that contradicts a record, or tidying
  `docs/adr/`.
- `docs`: a change that alters what a user does or sees, or writing a page of README.md or `docs/`.
- `pull-request`: starting a change, committing, opening a pull request, its review and CI, merging, cleaning up, and
  taking back a subagent's branch.
- `release`: model files and cards on Hugging Face, the catalog, raising `VERSION`, the tag and checking the release.
- `windows-check`: checking a change on a Windows machine with a Vulkan GPU.
