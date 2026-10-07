# Irodori-TTS speaks a speaker-inversion embedding as a voice file bound to its model

Decided 2026-10-07.

## Context

The official runtime's request may set `ref_embed`, a `.speaker.safetensors` file of a speaker-inversion embedding:
vectors that its training script learns, through one checkpoint's DiT and duration predictor, to stand for a speaker.
The runtime attends to them as they are, in place of the speaker condition its speaker encoder makes of a reference,
with every token unmasked, and its duration predictor takes the first as the speaker's vector. The file holds the
tensor `speaker_embedding`, float32 [tokens, 768], and records nothing of the checkpoint it was learned against.

A voice file of references binds to the codec's hash, since a latent is the codec's and works with every model of that
codec (docs/adr/0015). v4.1-Small-MF and v4.1-Small share their speaker encoder, bit for bit, but not their DiTs.

## Decision

- **An embedding is a voice file**, made by `speech voice MODEL EMBEDDING.speaker.safetensors VOICE.gguf`, which reads
  the file as the runtime saves it, or by `speech_voice_params_set_embedding()`, which takes the vectors, so that the
  library reads no file format but GGUF and WAVE. The worker, the server and `--add-voice` take it as any voice file.
- **The voice file names the model it was made for by its `general.source.url`**, the repository and revision of the
  checkpoint, and a model with another one refuses it. The F32, F16 and Q8_0 files of one checkpoint share the URL, so
  the binding follows the checkpoint and not the file's type.
- The voice file's layout 2 says what a voice is made of in `irodori-tts-voice.source`, `references` or `embedding`.

The alternatives were turned down:

- No binding, as the runtime has none. A voice learned against v4.1-Small would speak through v4.1-Small-MF's DiT
  without a word, where the caller made a mistake.
- A hash of the DiT's speaker keys and values. It would differ between the types of one checkpoint unless the converter
  computed it from the official weights, a key for one use.
- Reading `.speaker.safetensors` in the library, or taking one in `speech_voice_add()`. The library would read a third
  format with its own JSON header for one family, and a program can hand it the vectors.

## Consequences

The caller says which model an embedding belongs to when it makes the voice file, since the embedding does not say.
The checks run on an embedding made from a reference's speaker condition (`reference/irodori-tts/embedding.py`),
which has the form of a learned one; a learned one is checked the same way once one is at hand.
