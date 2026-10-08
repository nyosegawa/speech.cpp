# A model is named from a pinned catalog, and no command chooses one for the user

## Context

A user should not have to find a converted file on Hugging Face and download it by hand. NVIDIA's NeMo-Speech.cpp,
audio.cpp and CrispASR fetch a model the first time it is named, and llama.cpp takes `-hf user/repo:Q4_K_M`. Every
download of speech.cpp is pinned, and a release reads the GGUF layouts it knows, so a file replaced on Hugging Face must
not change what an earlier release loads.

Which model is right depends on the language: ReazonSpeech for Japanese, Qwen3-ASR for most of its 30 languages,
parakeet-tdt-0.6b-v3 for nine European languages Qwen3-ASR does not recognize, and for synthesis Qwen3-TTS's ten
languages or Irodori-TTS's Japanese.

## Decision

- **A model argument is a path or a name.** A path ends in `.gguf`, in any case. Anything else is `NAME[:TYPE]`: NAME is
  a model's short name or the Hugging Face repository of its converted files (`sakasegawa/Qwen3-ASR-0.6B-GGUF`), and TYPE
  the file's weight type in lower case (`q8_0`, `f16`), as llama.cpp's `-hf` takes a quantization. Without a type, a name
  means the file of the type the catalog gives the model. The rule reads the argument alone, not the disk, so the same
  command line means the same thing in every folder.
- **A name is the upstream model's own name in lower case, shortened by what tells nothing apart in the catalog**:
  `qwen3-tts-0.6b` and `qwen3-tts-1.7b` (Qwen3-TTS-12Hz-…-CustomVoice: every Qwen3-TTS the catalog holds has the 12Hz
  codec and is CustomVoice), `irodori-tts-mf` and `irodori-tts` (Irodori-TTS-v4.1-Small-MF and v4.1-Small: one version
  and one size are ported, and MF, the MeanFlow model, is what tells the two apart), `qwen3-asr-0.6b` and
  `qwen3-asr-1.7b`, `reazonspeech-v2` (reazonspeech-nemo-v2, without the toolkit's name), and `parakeet-tdt_ctc-0.6b-ja`
  and `parakeet-tdt-0.6b-v3`, whole, since NVIDIA's names already tell its models apart by their decoder, language and
  version. A name holds no `:` or `/`, which a model argument uses.
- **A released name keeps naming its model.** A later version of a model line gets a name of its own (`irodori-tts-v5`)
  rather than taking over an old one, since a name that moved would change what a script gets.
- **The catalog is one file, `tools/models/catalog.json`, built into `speech`.** For each model it holds what a person
  writes, the name, the repository, the type a name alone means and the languages it is the model to start with, and
  what `tools/models/update_catalog.py` writes from Hugging Face's API: the repository's commit, each GGUF file at it
  with its size and SHA-256, and the task, the languages and whether the model takes voice files, from the
  `speech info --json` output beside each file. A release lists the files it was checked with; a pull request that
  changes files on Hugging Face runs the script, and the diff shows what moved. The script refuses a type the
  repository has no file of, files of one model that disagree, and a language two models of a task are to start with.
- **`speech models` lists the catalog** with what is fetched and, per language, the model to start with;
  `speech models --json` gives every pin, each file's URL and path, and whether it is fetched, so that a program pins
  its files from it. A name or a type the catalog does not hold is a usage error that lists what it holds.
- **No command chooses a model.** Every subcommand that takes a model (`tts`, `asr`, `voice`, `info`, `serve`, `worker`)
  requires it, but for `speech serve --open`, whose page picks the models. A command line without one, or whose first
  argument is neither a `.gguf` path nor a name of the catalog while an argument is missing (`speech asr a.wav`), stops
  with exit 2 and says what MODEL is, which models of the subcommand's task to start with for which languages, read from
  the catalog, an example with the first of them, and that `speech models` lists the rest. `speech voice` lists the
  models that take voice files instead. A worker started without a model answers with `fatal`, as for any usage error.
- **The models to start with**, one per language and task: `qwen3-tts-0.6b` for the ten languages of Qwen3-TTS, Japanese
  among them, which it speaks with its own named speakers; for recognition `reazonspeech-v2` for Japanese, which writes
  punctuation and takes recordings of many minutes whole, `qwen3-asr-0.6b` for the other 29 languages of Qwen3-ASR, and
  `parakeet-tdt-0.6b-v3` for the nine European languages Qwen3-ASR does not recognize.

The alternatives were turned down:

- Fetching a repository's newest files at run time, or a catalog downloaded at run time. Neither is pinned, and a
  release could fetch a layout it does not read.
- The upstream names whole (`qwen3-tts-12hz-0.6b-customvoice`). Their extra parts tell nothing apart among the models a
  release holds, and they are long to type.
- The repository alone, as llama.cpp's `-hf` has it. It is kept as the second form of NAME; a short name is what README
  and the messages can show.
- Telling a name from a path by whether a file of that name exists. The same argument would mean different things in
  different folders.
- A default model. It would be wrong for every user of another language, it would fetch a gigabyte the user did not ask
  for, and it could never change: a script that relied on it would get another model's text or voice from one release
  to the next.
- A default chosen by the system's language. The audio's language is not the system's, and the result would differ from
  one machine to another for the same command line.

## Consequences

A release fetches only the files its catalog lists, and a file replaced on Hugging Face reaches users with the next
release. Adding a model is a line in `catalog.json` and a run of the script. The first command a user types names a
model, as README's examples do. The catalog is in the executable, not in the library: the C API loads files.
