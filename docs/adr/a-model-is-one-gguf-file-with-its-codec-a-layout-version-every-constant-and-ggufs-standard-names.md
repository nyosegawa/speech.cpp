# A model is one GGUF file with its codec, a layout version, every constant and GGUF's standard names

## Context

Qwen3-TTS and Irodori-TTS each need a codec beside the model, and a pair of files that nothing checks can be the wrong
pair. A reader must tell a file of an earlier form from a broken one, and must not run a file with arithmetic its
converter did not mean: CrispASR makes every key optional with a default, and its `xscaling`, added as "true when
absent", ran published files with the wrong arithmetic for five months. audio.cpp embeds the official configuration
and keeps the defaults of what it leaves out throughout its C++.

Each model has constants that its checkpoint's configuration gives, or where it gives none the official code:
Qwen3-TTS's output rate, control tokens, sampling settings and longest speech, Irodori-TTS's speed range, reference
loudness and tail-cut thresholds, and the like. Qwen3-TTS's checkpoint gives `max_new_tokens` 8192, where the official
package falls back to 2048 for a checkpoint without `generation_config.json`.

The GGUF specification (ggml's `docs/gguf.md`) has standard keys for a model's organization, line, size, fine-tune,
version, source repository, weight type and languages, and a convention for naming a file from them, which other GGUF
files follow. Tools that read GGUF metadata, Hugging Face's file viewer among them, show these keys and read these
names.

## Decision

- **One file per model, codec included.** One download, one path, and no pair to get wrong. Making a voice reads only
  the codec's encoder out of the file.
- **A layout version.** `speech.layout` gives the version of the family's layout: 1 for Qwen3-TTS and Qwen3-ASR, 2 for
  FastConformer and Irodori-TTS. A family's reader brings an earlier layout up to the current one in one function, and
  the rest of the reader knows only the current layout. `speech.requires` names the first release that reads the file,
  by its layout, its weight type and the types of its tensors, and a reader refuses a file of a newer layout, or of a
  type it does not know, naming that release. A file without `speech.layout` is refused. A version tells an old file
  from a broken one, which optional keys cannot.
- **Keys are required and typed, and the tensors follow from them.** A key of the wrong type or a kind the reader does
  not know is a model-file error, and the reader derives the set of tensors from the keys and refuses a tensor too many
  or too few, so that nothing is used because it happens to be there.
- **Every model constant is in the file**, written by the converter from the checkpoint, or from the official code where
  the checkpoint has none, and the converter says where each comes from. Qwen3-TTS's longest speech is the checkpoint's
  8192 frames.
- **The identity is in the GGUF specification's keys, and the file is named by its convention.** The converter writes
  `general.name`, `organization`, `basename`, `size_label`, `finetune` and `version` where the model has them,
  `license`, `source.repo_url`, `file_type`, `quantization_version` in a quantized file, and `languages`, under the
  names the specification gives them, and names the file `<basename>-<size label>-<finetune>-<version>-<type>.gguf`
  (`Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf`). The specification has no key for a revision, so `general.source.url`
  keeps it, `<general.source.repo_url>/tree/<revision>`. A size the model's name does not give is counted from the
  file's tensors as gguf-py counts them (Irodori-TTS's Small is a word where the convention takes a number). A model
  whose name has no fine-tune or version has no such key, as the specification leaves out what a model lacks; these keys
  describe the model and steer nothing, but for Qwen3-TTS's size label ([the record of
  instructions](the-option-instructions-carries-qwen3-tts-instruct-and-irodori-tts-caption.md)). `general.file_type`
  must name the type that holds most of the tensors' bytes.
- **The other model-independent keys are under `speech.`**, the built-in voices with the model card's language, gender
  and description among them, and a family's keys under its architecture's name. A key nothing reads is not written.

How the file holds the languages and how a voice file binds to the codec have records of their own
([languages](languages-are-shortest-iso-639-codes-in-the-file-and-bcp-47-tags-in-requests.md),
[voice files](an-irodori-tts-voice-file-holds-codec-latents-bound-to-the-codec-or-an-embedding-bound-to-its-model.md)).

The alternatives were turned down:

- Two files, the model naming its codec and the reader checking it. It removes the wrong pair but keeps two downloads
  and two paths everywhere.
- Optional keys with defaults in the reader. A key a converter forgot and a file from before the key existed look the
  same, as CrispASR's `xscaling` showed.
- The official configuration embedded in the file. The C++ would still hold the defaults of what it leaves out.
- Keys of speech.cpp's own for the identity, alone or beside the specification's. Tools would not show them, and beside
  the specification's they would hold one fact twice.
- The revision in a key of speech.cpp's own. It would be the one identity key that tools do not show, where the URL of
  the revision is a standard key already.

## Consequences

Each family's converter and `layout.cpp` give every key with its type and the tensors the keys call for, and
`speech info MODEL --meta` prints every key of a file. The model information carries the identity, so that a program
can show it without parsing the file's name. The standard names do not make the files run anywhere but in speech.cpp,
as each repository's card says.
