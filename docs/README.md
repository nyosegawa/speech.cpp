# Documentation

The documentation of speech.cpp. The [README](../README.md) has the shortest path from install to speech; these pages
have the details.

## Use

| Page | What it covers |
|---|---|
| [Install](install.md) | the one-line installers, updating, uninstalling, release archives and requirements |
| [Command line](cli.md) | every subcommand of `speech` and its options, and the exit statuses |
| [Models](models.md) | the catalog, model names, fetching, where models are kept, voices and languages |
| [Server and page](server.md) | `speech serve`: OpenAI's speech and transcription API, errors, the page and its guards |
| [Worker protocol](worker.md) | `speech worker`: the JSON Lines protocol 2 for programs that start speech.cpp |
| [C API](c-api.md) | `speech.h`: a program's shape, errors, options, model information, threads and versions |
| [GGUF files](gguf.md) | converting a model, file names, layouts, and every key of the model and voice files |
| [Build from source](build.md) | building, backends, and the archives CI builds |

## Model families

| Page | Models |
|---|---|
| [Qwen3-TTS](models/qwen3-tts.md) | `qwen3-tts-0.6b`, `qwen3-tts-1.7b` |
| [Irodori-TTS](models/irodori-tts.md) | `irodori-tts-mf`, `irodori-tts` |
| [FastConformer](models/fastconformer.md) | `reazonspeech-v2`, `parakeet-tdt_ctc-0.6b-ja`, `parakeet-tdt-0.6b-v3` |
| [Qwen3-ASR](models/qwen3-asr.md) | `qwen3-asr-0.6b`, `qwen3-asr-1.7b` |

## Development

| Page | What it covers |
|---|---|
| [Checks](development/checks.md) | how every port is checked against its official implementation, with every measurement |
| [Adding a model](development/adding-a-model.md) | the decisions and folders a new model touches |
| [Releasing](development/releasing.md) | version numbers and what the release workflow does |
| [Decisions](adr/) | one record per decision the code cannot show |
