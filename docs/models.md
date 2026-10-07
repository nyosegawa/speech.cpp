# Models

This page lists the models speech.cpp runs, how to name them, where their files are kept, and their voices and
languages.

## The catalog

Each release carries a catalog of models. A command takes a model's name and fetches its file from Hugging Face the
first time.

| Name | Task | Languages | File the name fetches | Family |
|---|---|---|---|---|
| [`qwen3-tts-0.6b`](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF) | synthesis | de en es fr it ja ko pt ru zh | `Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf`, 1.21 GB | [Qwen3-TTS](models/qwen3-tts.md) |
| [`qwen3-tts-1.7b`](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF) | synthesis | the same | `Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf`, 2.29 GB | [Qwen3-TTS](models/qwen3-tts.md) |
| [`irodori-tts-mf`](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF) | synthesis | ja | `Irodori-TTS-848M-MF-v4.1-F16.gguf`, 1.89 GB | [Irodori-TTS](models/irodori-tts.md) |
| [`irodori-tts`](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-GGUF) | synthesis | ja | `Irodori-TTS-841M-v4.1-F16.gguf`, 1.87 GB | [Irodori-TTS](models/irodori-tts.md) |
| [`qwen3-asr-0.6b`](https://huggingface.co/sakasegawa/Qwen3-ASR-0.6B-GGUF) | recognition | 30 languages | `Qwen3-ASR-0.6B-Q8_0.gguf`, 0.84 GB | [Qwen3-ASR](models/qwen3-asr.md) |
| [`qwen3-asr-1.7b`](https://huggingface.co/sakasegawa/Qwen3-ASR-1.7B-GGUF) | recognition | 30 languages | `Qwen3-ASR-1.7B-Q8_0.gguf`, 2.18 GB | [Qwen3-ASR](models/qwen3-asr.md) |
| [`reazonspeech-v2`](https://huggingface.co/sakasegawa/reazonspeech-nemo-v2-GGUF) | recognition | ja | `reazonspeech-nemo-619M-v2-F16.gguf`, 1.24 GB | [FastConformer](models/fastconformer.md) |
| [`parakeet-tdt_ctc-0.6b-ja`](https://huggingface.co/sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF) | recognition | ja | `parakeet-tdt_ctc-0.6B-ja-F16.gguf`, 1.24 GB | [FastConformer](models/fastconformer.md) |
| [`parakeet-tdt-0.6b-v3`](https://huggingface.co/sakasegawa/parakeet-tdt-0.6b-v3-GGUF) | recognition | 25 European languages | `parakeet-tdt-0.6B-v3-F16.gguf`, 1.26 GB | [FastConformer](models/fastconformer.md) |

`speech models` prints the same, with what is fetched. The upstream models are
[Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice),
[Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice),
[Aratako/Irodori-TTS-v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF),
[Aratako/Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small),
[Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B),
[reazon-research/reazonspeech-nemo-v2](https://huggingface.co/reazon-research/reazonspeech-nemo-v2),
[nvidia/parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) and
[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

### Which model to start with

No model is chosen for you, since the right one depends on the language
([ADR 0036](adr/0036-a-command-without-a-model-says-what-to-type-rather-than-choosing-one.md)). The model to start with
is the one for which a first try needs nothing else:

| Task | Model | Languages |
|---|---|---|
| synthesis | `qwen3-tts-0.6b` | de en es fr it ja ko pt ru zh: it speaks with its own named speakers, where Irodori-TTS needs a voice made from a recording |
| recognition | `reazonspeech-v2` | ja: it writes Japanese with punctuation and takes recordings of many minutes whole |
| recognition | `qwen3-asr-0.6b` | the 29 other languages of Qwen3-ASR |
| recognition | `parakeet-tdt-0.6b-v3` | bg et hr lt lv mt sk sl uk, the European languages Qwen3-ASR does not recognize |

A command that names no model stops with exit 2 and says what to type:

```
$ speech asr a.wav
speech: the model is missing: speech asr MODEL [options] AUDIO.wav...
MODEL is a model file (.gguf) or the name of one in `speech models`, which is fetched the first time it is used.
No model is chosen for you; the one to start with depends on the language:
  qwen3-asr-0.6b        ar cs da de el en es fa fi fil fr hi hu id it ko mk ms nl pl pt ro ru sv th tr vi yue zh
  reazonspeech-v2       ja
  parakeet-tdt-0.6b-v3  bg et hr lt lv mt sk sl uk
For example: speech asr qwen3-asr-0.6b [options] AUDIO.wav...
```

## Names

Every subcommand that takes a model (`tts`, `asr`, `voice`, `info`, `serve`, `worker`) takes a model file's path, which
ends in `.gguf`, or `NAME[:TYPE]`:

- NAME is a name of the catalog, or the Hugging Face repository of its converted files.
- TYPE is the file's weight type, `q8_0`, `f16` or `f32`, where the repository holds it. Without it, the name means the
  file in the table above.

```sh
speech asr qwen3-asr-0.6b meeting.wav                       # Qwen3-ASR-0.6B-Q8_0.gguf
speech asr sakasegawa/Qwen3-ASR-0.6B-GGUF meeting.wav       # the same file, named by its repository
speech asr qwen3-asr-0.6b:q8_0 meeting.wav                  # the same file, its type given
speech serve reazonspeech-v2                                # fetched before the server starts
```

A name or a type that the catalog does not hold is refused with exit 2, and the message lists what it holds.

The catalog, `tools/models/catalog.json`, is built into `speech`. It pins each repository at a revision and each file
with its size and SHA-256, so a release fetches the files it was checked with, and a file replaced on Hugging Face
reaches users with the next release
([ADR 0033](adr/0033-a-model-is-named-from-a-catalog-each-release-carries-pinned-by-revision-and-sha-256.md)).

## Fetching

- A fetch runs the system's curl (`System32\curl.exe` on Windows), so the system's proxies, `https_proxy` among them, and
  certificates apply ([ADR 0035](adr/0035-the-command-line-fetches-models-with-the-systems-curl-and-the-library-never-reaches-the-network.md)).
- It writes to `<file>.part` beside where the file goes, checking the size and SHA-256 as the bytes come, and renames the
  file into place once both match the catalog.
- A fetch that stops keeps the part, and the same command resumes it. A part whose bytes cannot be the file's is removed
  with an `io` failure that says so, and the same command fetches the file again.
- Two processes that fetch the same file take turns on `<file>.lock`. The second waits, then loads the file the first
  fetched.
- Progress goes to stderr, so a worker's stdout carries the protocol alone.
- Only the command line and the server fetch. The library and the C API load files and never reach the network.

## Where models are kept

| System | Model folder |
|---|---|
| macOS | `~/Library/Caches/speech.cpp/models` |
| Linux | `$XDG_CACHE_HOME/speech.cpp/models`, or `~/.cache/speech.cpp/models` |
| Windows | `%LOCALAPPDATA%\speech.cpp\models` |

`SPEECH_MODEL_DIR`, an absolute path, replaces the folder. A file goes to `<owner>--<name>/<revision>/<file>` in it, so
releases that pin different revisions never share a path. Nothing is removed automatically
([ADR 0034](adr/0034-fetched-models-live-in-the-systems-cache-folder-and-are-removed-only-on-request.md)):
`speech models` lists the files that no model of this release names as old, and `speech rm --old` removes them.

## Families

A family is the code that runs a model file, its `general.architecture`, which the model information calls
`architecture`. Several model lines can share one: `fastconformer` runs both parakeet lines and ReazonSpeech. A model's
organization and model line are its file's `general.organization` and `general.basename`.

| Family | Organization and model line | Task | Page |
|---|---|---|---|
| `qwen3-tts` | Qwen, Qwen3-TTS-12Hz, 0.6B and 1.7B CustomVoice | synthesis | [models/qwen3-tts.md](models/qwen3-tts.md) |
| `irodori-tts` | Aratako, Irodori-TTS, v4.1-Small-MF and v4.1-Small | synthesis | [models/irodori-tts.md](models/irodori-tts.md) |
| `fastconformer` | nvidia, parakeet-tdt_ctc 0.6b-ja and parakeet-tdt 0.6b-v3; reazon-research, reazonspeech-nemo v2 | recognition | [models/fastconformer.md](models/fastconformer.md) |
| `qwen3-asr` | Qwen, Qwen3-ASR, 0.6B and 1.7B | recognition | [models/qwen3-asr.md](models/qwen3-asr.md) |

## Voices

A synthesis request needs a `voice`.

- **Qwen3-TTS** speaks with the model's nine speakers: `aiden`, `dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`,
  `uncle_fu` and `vivian`. `speech info qwen3-tts-0.6b` lists each with its language, gender and description. `dylan`
  (Beijing) and `eric` (Sichuan) speak a Chinese dialect when the language is `zh` or left to the model.
- **Irodori-TTS** speaks in the voice of a reference recording that you add. A layout 2 file also has the voice `none`,
  with which the model chooses a voice itself ([models/irodori-tts.md](models/irodori-tts.md#voices)). A voice is
  added from a WAVE file, which is encoded each time it is added, or from a voice file that `speech voice` made once.

| Entry point | How to add a voice |
|---|---|
| `speech tts` | `--add-voice NAME=FILE`, repeatable |
| `speech worker` | `--add-voice NAME=FILE`, or the request `add_voice` |
| `speech serve` | `--add-voice NAME=FILE`, or the page's `POST /speech/voices` |
| C API | `speech_voice_add()` |

A voice's name is compared with case, and may be neither a voice of the model nor `none`.

## Languages

- A model's languages are BCP 47 tags: two letters, or three for a language without a two-letter code (`yue`, `fil`).
- A request's `language` is one of them, a region or script of one (`ja-JP`, `zh-Hant`), or `auto`, the default, which
  leaves the choice to the model. Any other language is refused as `out_of_range`.
- Qwen3-TTS and Qwen3-ASR are steered by the language. Irodori-TTS and the FastConformer models only check it against
  their languages and do not use it.
- Qwen3-ASR's results say which language it heard.

## Files on Hugging Face

Each converted repository holds one GGUF file, the codec inside it, and beside it the same name with `.json` added: the
output of `speech info --json` for the file. A program can show a model's identity, voices, languages, options and sizes,
and choose a file, without downloading the model. [gguf.md](gguf.md) describes the files and how to convert them
yourself.
