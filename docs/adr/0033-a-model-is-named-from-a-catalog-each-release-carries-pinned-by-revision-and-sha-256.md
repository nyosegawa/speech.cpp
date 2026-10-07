# A model is named from a catalog each release carries, pinned by revision and SHA-256

Decided 2026-10-08.

## Context

Until 0.7.1 a user downloaded a release's archive, found the converted files on Hugging Face from README's tables,
downloaded one by hand and gave `speech` its path. NVIDIA's NeMo-Speech.cpp, audio.cpp and CrispASR fetch a model the
first time it is named, and llama.cpp takes `-hf user/repo:Q4_K_M`. speech.cpp's rule is that every download is pinned,
and a release reads the GGUF layouts it knows: a file replaced on Hugging Face (Irodori-TTS's layout 2 files replace the
layout 1 files at 0.8.0's release) must not change what an earlier release loads.

## Decision

- **A model argument is a path or a name.** A path ends in `.gguf`, in any case, and is used as before. Anything else is
  `NAME[:TYPE]`: NAME is a model's short name or the Hugging Face repository of its converted files
  (`sakasegawa/Qwen3-ASR-0.6B-GGUF`), and TYPE the file's weight type in lower case, `q8_0`, `f16` or `f32`, as
  llama.cpp's `-hf` takes a quantization. Without a type, a name means the file README's table recommends. The rule
  reads the argument alone, not the disk, so the same command line means the same thing in every folder.
- **A name is the upstream model's own name in lower case, shortened by what tells nothing apart in the catalog**:
  `qwen3-tts-0.6b` and `qwen3-tts-1.7b` (Qwen3-TTS-12Hz-…-CustomVoice: every Qwen3-TTS the catalog holds has the 12Hz
  codec and is CustomVoice), `irodori-tts-mf` and `irodori-tts` (Irodori-TTS-v4.1-Small-MF and v4.1-Small: one version
  and one size are ported, and MF, the MeanFlow model, is what tells the two apart), `qwen3-asr-0.6b` and
  `qwen3-asr-1.7b`, `reazonspeech-v2` (reazonspeech-nemo-v2, without the toolkit's name), and `parakeet-tdt_ctc-0.6b-ja`
  and `parakeet-tdt-0.6b-v3`, whole, since NVIDIA's names already tell its models apart by their decoder, language and
  version. A name holds no `:` or `/`, which a model argument uses.
- **A released name keeps naming its model.** A later version of a model line gets a name of its own
  (`irodori-tts-v5`) rather than taking over an old one, for the reason a default model is refused
  (docs/adr/0036): a name that moved would change what a script gets.
- **The catalog is one file, `tools/models/catalog.json`, built into `speech`.** For each model it holds the name, the
  repository and the type a name alone means and the languages it is the model to start with, which a person writes,
  and what `tools/models/update_catalog.py` writes from Hugging Face's API: the repository's commit, each GGUF file at
  it with its size and SHA-256, and the task, the languages and whether the model takes voice files, from the
  `speech info --json` output beside each file. A release lists the files it was checked with; a pull request that
  changes files on Hugging Face runs the script, and the diff shows what moved. The script refuses a type the
  repository has no file of, files of one model that disagree, and a language two models of a task are to start with.
- **`speech models` lists the catalog** with what is fetched and, per language, the model to start with;
  `speech models --json` gives every pin, each file's URL and path, and whether it is fetched, so that a program such
  as ASIST pins its files from it. A name or a type the catalog does not hold is a usage error that lists what it holds.
- **The model to start with** is the one for which a first try needs nothing else: `qwen3-tts-0.6b` for the ten
  languages of Qwen3-TTS, Japanese among them, since it speaks with its own named speakers where Irodori-TTS's layout 1
  files need a voice file made from a recording; for recognition `reazonspeech-v2` for Japanese, which writes
  punctuation and takes recordings of many minutes whole, `qwen3-asr-0.6b` for the other 29 languages of Qwen3-ASR,
  and `parakeet-tdt-0.6b-v3` for the nine European languages Qwen3-ASR does not recognize.

The alternatives were turned down:

- Fetching a repository's newest files at run time, or a catalog downloaded at run time. Neither is pinned, and a
  release could fetch a layout it does not read.
- The upstream names whole (`qwen3-tts-12hz-0.6b-customvoice`). Their extra parts tell nothing apart among the
  models a release holds, and they are long to type.
- The repository alone, as llama.cpp's `-hf` has it. It is kept as the second form of NAME; a short name is what
  README and the messages can show.
- Telling a name from a path by whether a file of that name exists. The same argument would mean different things in
  different folders.

## Consequences

A release fetches only the files its catalog lists, and a file replaced on Hugging Face reaches users with the next
release. Adding a model is a line in `catalog.json` and a run of the script. The catalog is in the executable, not in
the library: the C API still loads files, and the library never reaches the network (docs/adr/0035).
