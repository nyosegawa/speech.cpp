# An Irodori-TTS voice file holds codec latents bound to the codec, or an embedding bound to its model

## Context

Irodori-TTS speaks in the voice of a reference recording: its codec encodes the recording to a latent, and its speaker
encoder turns the latent into the condition the DiT attends to. The official runtime's request takes several reference
recordings (`ref_wavs`), which it encodes one by one, each brought to `ref_normalize_db` (-16 LUFS by default) or left
as recorded with `None` (scaling down a peak above 1), and joins in order before the speaker encoder, cutting the joined
latent at `max_ref_seconds` (120 s). It may instead take a speaker-inversion embedding (`ref_embed`), a
`.speaker.safetensors` file of vectors that its training script learns through one checkpoint's DiT and duration
predictor, float32 [tokens, 768], which it attends to in place of the speaker condition; the file records nothing of the
checkpoint it was learned against. v4.1-Small-MF and v4.1-Small share their codec and their speaker encoder bit for bit,
but not their DiTs.

The worker and the server add their voices when they start. For bright-young-woman-10s.wav (10.74 s, 48 kHz, 16-bit):

| | WAVE file | Voice file |
|---|---|---|
| Size | 1,031,084 bytes | 34,720 bytes |
| Time to load, Apple M5, Metal | 0.72 s | 0.016 s |
| Time to load, Apple M5, CPU | 5.05 s | 0.028 s |
| Time to load, RTX 2080, Vulkan | 0.43 s | 0.025 s |
| Latent against the official encoder | 99 dB SNR on the CPU, 40 dB on Metal, 33 dB on Vulkan | as made |

A GPU computes the encoder in less precision than the CPU, so the same WAVE file gives a slightly different latent on
each kind of device.

## Decision

- **A voice file is made once** by `speech voice MODEL REF.wav... VOICE.gguf`, or by `speech_voice_make()` and
  `speech_voice_make_from()`, which read only the codec's encoder out of the model file, on the CPU unless told
  otherwise. `--add-voice`, the worker's `add_voice` and `speech_voice_add()` take one file, a voice file or a WAVE
  file.
- **It holds the codec's latent rather than the speaker encoder's output.** The latent depends only on the codec, and
  the speaker encoder takes milliseconds.
- **It binds to the codec by a hash**, `irodori-tts.codec.sha256`, the SHA-256 of the official codec's tensors, which
  the converter computes and writes into the model file. Every conversion of the same codec has the same hash, whatever
  type it stores, so a voice works with v4.1-Small-MF and v4.1-Small in any type, and a model of another codec refuses
  it.
- **Several references and their loudness belong to the voice file**, not to a request. They are encoded as the runtime
  encodes `ref_wavs`, each brought to the model's loudness (`irodori-tts.reference.lufs`), to `--lufs`, or kept with
  `--keep-loudness`; one reference at the model's loudness gives the bytes `speech_voice_make()` gives. Joined
  references longer than the model's `irodori-tts.reference.max_seconds` are refused, naming `references`, where the
  runtime cuts them.
- **An embedding is a voice file too**, made by `speech voice MODEL EMBEDDING.speaker.safetensors VOICE.gguf`, which
  reads the file as the runtime saves it, or by `speech_voice_params_set_embedding()`, which takes the vectors, so that
  the library reads no format but GGUF and WAVE. It names the model it was made for by its `general.source.url`, the
  repository and revision of the checkpoint, and a model with another one refuses it. The F32, F16 and Q8_0 files of one
  checkpoint share the URL, so the binding follows the checkpoint and not the file's type.
- **The voice file has a layout of its own**, now 2: `irodori-tts-voice.source` says whether it holds `references` or an
  `embedding`; a voice of references records each recording's length and rate in the order joined, whether and to
  which loudness they were brought, and the kind of device that encoded them. A voice file of layout 1 reads as one of a
  single recording brought to -16 LUFS, the loudness every model file of layout 1 gives.

The alternatives were turned down:

- The speaker encoder's output in the voice file. It would save the milliseconds the encoder takes and bind the voice to
  a model's weights rather than to the codec alone.
- Several references and the loudness as options of a request, or `add_voice` taking several paths. The worker would
  encode them at every start or with every request, where a voice file encodes them once, and the protocol's voice
  would be a list.
- One function with the references as an array and the loudness as a number, NaN for none. A caller would pass the
  model's -16 to keep the default, and NaN would mean something.
- Cutting joined references at 120 s, as the runtime does. The voice would lose the end of what the caller gave.
- A hash of the codec's tensors as the file stores them. An F16 and an F32 copy of one codec would refuse each other's
  voices.
- The codec named by the URL of the file it came from. It changes when a repository moves, and says nothing of the
  weights.
- An embedding without a binding, as the runtime has none. A voice learned against v4.1-Small would speak through
  v4.1-Small-MF's DiT without a word, where the caller made a mistake.
- A hash of the DiT's speaker keys and values for an embedding. It would differ between the types of one checkpoint
  unless the converter computed it from the official weights, a key for one use.
- Reading `.speaker.safetensors` in the library, or taking one in `speech_voice_add()`. The library would read a third
  format with its own JSON header for one family, and a program can hand it the vectors.

## Consequences

A model with a new codec means making the voices again from their WAVE files. The caller says which model an embedding
belongs to when it makes the voice file, since the embedding does not say. The checks run on an embedding made from a
reference's speaker condition (`reference/irodori-tts/embedding.py`), which has the form of a learned one. A voice file
names 0.8.0 in `speech.requires`, the first release that reads its layout 2.
