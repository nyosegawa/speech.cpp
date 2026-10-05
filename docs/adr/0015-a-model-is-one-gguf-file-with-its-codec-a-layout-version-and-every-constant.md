# A model is one GGUF file with its codec, a layout version and every constant

Decided 2026-10-06.

## Context

A Qwen3-TTS or Irodori-TTS model was two GGUF files, the model and its codec, and nothing checked that a pair belonged
together: the model did not name its codec, and the codec's architecture was never read. The Hugging Face repositories
already held the codec beside each model.

Nothing in a file said which form of speech.cpp's layout it had. FastConformer's changed three times, and each time
an older file was refused with "key … is missing". Every key was required, but its type was not checked, and a key of
another type ended the process in a `GGML_ASSERT`. An `irodori.flow` other than `meanflow` meant RF, and Qwen3-TTS's
projection was used when its tensor was there. Keys were written that nothing read, among them
`speech.language_selectable` while each engine hard-coded the same fact, and model constants lived in the C++:
Qwen3-TTS's output rate, newline token, control tokens, sampling settings and 2048 frames, and Irodori-TTS's speed
range, reference loudness and tail-cut thresholds. A voice file named its codec by the codec file's URL, so a codec
stored in another type or uploaded again refused voices of the same latent space, and it had no version of its form.

CrispASR makes every key optional with a default, and its `xscaling`, added as "true when absent", ran published
files with the wrong arithmetic for five months. audio.cpp embeds the official configuration and keeps the defaults of
what it leaves out throughout its C++.

## Decision

- **One file per model, codec included.** One download, one path, and no pair to get wrong; the repositories hold the
  same bytes as before. Making a voice reads only the codec's encoder out of the file.
- **A layout version.** `speech.layout` starts at 1 for every family. A family's reader brings an earlier layout up to
  the current one in one function, and a layout newer than it knows is refused naming the release in
  `speech.requires`. A version tells an old file from a broken one, which optional keys cannot, as ASIST's stored
  files do with their `StoredFormat`. `speech.requires` exists only so that an old reader can name the release.
- **Keys are required and typed, and the tensors follow from them.** A key of the wrong type or a kind the reader does
  not know is a model-file error, and the reader derives the set of tensors from the keys and refuses a tensor too
  many or too few, so that nothing is used because it happens to be there.
- **Every model constant is in the file**, written by the converter from the checkpoint, or from the official code
  where the checkpoint has none. Qwen3-TTS's longest speech becomes the checkpoint's `max_new_tokens`, 8192 frames,
  where the C++ had 2048, the official package's fallback for a checkpoint without `generation_config.json`.
- **Model-independent keys are under `speech.`**, the built-in voices with the model card's language, gender and
  description among them, and a family's keys under its architecture's name. A key nothing reads is not written.
- **A voice file binds to its codec by a hash** of the official codec's tensors, which the converter computes and
  writes into the model file, and carries a layout of its own with the recording's length and rate and the kind of
  device that encoded it. Every conversion of the same codec has the same hash, whatever type it stores, so a voice
  works with v4.1-Small-MF and v4.1-Small in any type, as docs/adr/0002 wants.

README.md lists every key of layout 1, with its type, its meaning and where its value comes from, and the tensors each
family's keys call for.

The alternatives were turned down:

- Two files, the model naming its codec and the reader checking it. It removes the wrong pair but keeps two
  downloads and two paths everywhere.
- Optional keys with defaults in the reader. A key a converter forgot and a file from before the key existed look the
  same, as CrispASR's `xscaling` showed.
- The official configuration embedded in the file. The C++ would still hold the defaults of what it leaves out.
- A hash of the codec's tensors as the file stores them. An F16 and an F32 copy of one codec would refuse each other's
  voices.
- The codec named by its URL, as before. It changes when a repository moves and says nothing of the weights.

## Consequences

Every published file is converted and uploaded again, once. Files and voice files made before 0.7.0 are refused with a
message that says so; ASIST makes its three voice files again from their WAVE files, and speech-bench keys its cache
of voice files by the codec's hash and the voice layout instead of the tool's release.
