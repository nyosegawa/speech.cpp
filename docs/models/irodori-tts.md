# Irodori-TTS

This page describes speech.cpp's port of Aratako's [Irodori-TTS](https://github.com/Aratako/Irodori-TTS), which speaks
Japanese in the voice of a reference recording, a sentence at a time.

| Name | Upstream | File |
|---|---|---|
| `irodori-tts-mf` | [Aratako/Irodori-TTS-v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF) | `Irodori-TTS-848M-MF-v4.1-F16.gguf`, 1.89 GB |
| `irodori-tts` | [Aratako/Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) | `Irodori-TTS-841M-v4.1-F16.gguf`, 1.87 GB |

v4.1-Small-MF samples in 4 MeanFlow steps. v4.1-Small samples in Euler steps (40 by default) with the runtime's
guidance, which makes it slower and lets a request tune more.

```sh
speech voice irodori-tts-mf me.wav me.voice.gguf
speech tts irodori-tts-mf --add-voice me=me.voice.gguf --voice me -o out.wav "明日の東京は晴れです。"
speech tts irodori-tts --add-voice me=me.voice.gguf --voice me --steps 16 --speed 1.2 -o out.wav "明日の東京は晴れです。"
```

The files the names fetch are of layout 1: they speak in a reference's voice and take no instructions. A file of
layout 2, which the converter writes ([gguf.md](../gguf.md#convert-a-model)) and which is named by its larger size
(866M and 859M), adds the voice `none` and `instructions`. A request for either from a layout 1 file is refused, naming
what the file lacks
([ADR 0026](../adr/0026-irodori-tts-layout-2-adds-what-new-requests-need-and-layout-1-reads-as-a-file-without-it.md)).

## What is implemented

A DiT makes the 32-dimensional latent of a 48 kHz codec (Semantic-DACVAE-Japanese-32dim) for a whole sentence, its
length set beforehand by a duration predictor, and the codec decodes it.

- The official text normalization, with NFKC of Unicode 13.0 as the runtime's Python has it. It keeps the 56 emoji the
  model reads as directions (🤭 a giggle, 😮‍💨 a sigh, 👂 a whisper and the rest of the runtime's
  `ALLOWED_ANNOTATION_EMOJIS`).
- The SentencePiece Unigram tokenizer with byte fallback, and ModernBERT-ja with its projector.
- The reference's loudness normalization, the codec encoder, in windows of 100 frames, and voices of several references.
- The speaker encoder, the duration predictor, the DiT and both samplers, with the runtime's guidance modes and scales,
  Sway Sampling, the truncation of the noise, temporal score rescaling and the scaling of the speaker's keys and values.
- Speaking without a reference, the runtime's `no_ref`, as the voice `none`
  ([ADR 0027](../adr/0027-irodori-tts-speaks-without-a-reference-as-its-built-in-voice-none.md)).
- Captions (VoiceDesign), the runtime's `caption`, as the option `instructions`
  ([ADR 0028](../adr/0028-irodori-tts-takes-its-caption-as-the-option-instructions.md)).
- Speaker-inversion embeddings, the runtime's `ref_embed`, as voice files
  ([ADR 0030](../adr/0030-irodori-tts-speaks-a-speaker-inversion-embedding-as-a-voice-file-bound-to-its-model.md)).
- The tail cut where the latent goes flat, and the codec decoder in windows (below).

Not implemented: LoRA adapters and SilentCipher's watermark. The noise comes from speech.cpp's own generator, so a seed
gives other audio than the same seed in the official runtime. A reference at another rate than 48 kHz is resampled with
the library's filter rather than the runtime's torchaudio defaults ([c-api.md](../c-api.md#requests)).

## Voices

A voice is one of these:

- **`none`**, in a layout 2 file: the model chooses the voice, as the runtime's `no_ref` does, and the length is
  predicted with the duration predictor's null speaker. It is listed first among the voices.
- **A reference WAVE file**: at most 120 s, 16-, 24- or 32-bit PCM or 32-bit float at any rate, its channels averaged and
  resampled to 48 kHz. Adding it normalizes its loudness and encodes it, as the runtime does for every request.
- **A voice file**, which `speech voice` (or `speech_voice_make()`) writes once. It holds the references' codec latent
  and the hash of the codec's tensors, so it works with every model file of the same codec, v4.1-Small-MF and
  v4.1-Small in any type, and a model of another codec refuses it
  ([ADR 0002](../adr/0002-asist-carries-irodori-tts-voices-as-voice-files.md),
  [ADR 0029](../adr/0029-irodori-tts-makes-a-voice-file-of-several-references-at-a-loudness-of-its-choice.md)).
- **A voice file of a speaker-inversion embedding**: vectors that the official training learns against one model to
  stand for a speaker. `speech voice` reads the `.speaker.safetensors` file the runtime saves (the tensor
  `speaker_embedding`, float32 [tokens, 768]). The file names the model it was made for, and no other model takes it:
  an embedding made for v4.1-Small is refused by v4.1-Small-MF.

A voice file may join several references, as the runtime's `ref_wavs`, up to 120 s together (3000 frames); the runtime
cuts past that, and speech.cpp refuses. Each reference is brought to the model's loudness, -16 LUFS, or to another
(`--lufs`), or kept as recorded with a peak above 1 scaled down to 1 (`--keep-loudness`, the runtime's
`ref_normalize_db` of `None`). `--add-voice` and the worker's `add_voice` take one file at the model's loudness, so a
voice of several references or another loudness is made into a voice file first.

```sh
speech voice irodori-tts-mf take-1.wav take-2.wav take-3.wav --lufs -23 three-takes.voice.gguf
speech voice Irodori-TTS-859M-v4.1-F16.gguf my-speaker.speaker.safetensors my-speaker.voice.gguf
```

For a 10.7 s reference, the voice file is 35 KB against the WAVE file's 1 MB, and loads in 0.016 s on an Apple M5
(Metal) and 0.025 s on an RTX 2080 (Vulkan), against 0.72 s and 0.43 s to encode the WAVE file. Voice files made before
0.7.0 have no layout and are refused; make them again from their WAVE files.

## Options

| Option | Default | Range and notes |
|---|---|---|
| `voice` | required | `none`, or a voice added since loading |
| `language` | `auto` | `ja`; only checked |
| `seed` | drawn | 0 to 2^53 - 1 |
| `steps` | 4 (MF), 40 (RF) | 1 to 2147483647 |
| `seconds` | none | 0.5 to 30: fixes the length, and the audio is cut at `int(seconds / speed × 48000)` samples; the duration predictor does not run |
| `duration_scale` | 1 | above 0: multiplies the predicted frames |
| `speed` | 1 | 0.25 to 4: divides the length, `seconds / speed` or the prediction times `duration_scale / speed`, as [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server) does with OpenAI's `speed` |
| `instructions` | `""` | layout 2: the voice and the way of speaking in words, at most 512 tokens |
| `keep_tail` | false | keep the whole length, rather than end where the latent goes flat |
| `tail_window_size`, `tail_std_threshold`, `tail_mean_threshold` | 20, 0.05, 0.1 | the speech ends at the first frame from which this many frames have a standard deviation under the threshold and a mean within the other of 0 |

v4.1-Small (RF) also takes the fields of the runtime's `SamplingRequest`
([ADR 0025](../adr/0025-irodori-tts-requests-take-the-runtimes-guidance-schedule-and-tail-settings.md)). v4.1-Small-MF
takes none of them: MeanFlow folded the guidance into its training, and the runtime ignores them for it.

| Option | Default | What it does |
|---|---|---|
| `cfg_scale_text`, `cfg_scale_speaker`, `cfg_scale_instructions` | 3, 5, 3 | the guidance's scales; 0 leaves that condition's branch out |
| `cfg_guidance_mode` | `independent` | `joint` leaves out every condition in one branch at one scale; `alternating` leaves out one condition a step, in turn |
| `cfg_min_t`, `cfg_max_t` | 0.5, 1 | the guidance runs while the step's time lies between them |
| `speaker_uncond_mode` | `mask` | `noise` gives the branch without the speaker noise in place of the speaker condition |
| `sway_coeff` | 0 | bends the linear schedule as Sway Sampling does: below 0 the steps gather near the noise, above 0 near the speech; the runtime's own `sway` uses -1 |
| `truncation_factor` | none | multiplies the starting noise |
| `rescale_k`, `rescale_sigma` | none | temporal score rescaling of each step's velocity; both or neither |
| `speaker_kv_scale` | 1 | above 1, the speech follows the voice more closely |
| `speaker_kv_min_t`, `speaker_kv_max_layers` | 0.9, 12 | the scaling covers the steps that start at or above this time, in the DiT's first layers up to this number |

The numbers are float32, as the runtime's tensors take them, so their ranges stop at the largest float (3.4e38), and an
option above 0 starts at the smallest normal float (1.2e-38). `speech info` shows the bounds.

A request that sets none of these speaks as before they were options, sample for sample. Where the runtime clamps a value
or ignores a setting that another one leaves without effect, speech.cpp refuses the request before any work, naming the
option:

- `seconds` with a `duration_scale` other than 1, and one of `rescale_k` and `rescale_sigma` without the other
  (`invalid_argument`).
- A length outside 0.5 to 30 s: `seconds / speed`, or the prediction times `duration_scale / speed` when either is not 1
  (`out_of_range`, option `seconds`, `duration_scale`, or `speed` when the scale is 1). At 1 and 1 the prediction is kept
  within the bounds, as the runtime keeps it
  ([ADR 0014](../adr/0014-the-c-api-checks-requests-against-the-options-each-model-declares.md)).
- `cfg_min_t` above `cfg_max_t`; the `joint` guidance with scales above 0 that are not equal; a `sway_coeff` that puts two
  steps at the same time.
- A value that another value leaves without effect: the guidance's settings with both scales at 0;
  `speaker_uncond_mode` `noise` with `cfg_scale_speaker` at 0 and a guidance other than `joint`; the tail's settings with
  `keep_tail`; `speaker_kv_min_t` and `speaker_kv_max_layers` with `speaker_kv_scale` at 1; with the voice `none`,
  `cfg_scale_speaker` other than its default and 0, `speaker_uncond_mode` `noise` and `speaker_kv_scale` other than 1;
  and `cfg_scale_instructions` other than its default and 0 without instructions.
- A text of more than 256 tokens, and instructions of more than 512 tokens, `<s>` included (`out_of_range`).

Speech that comes out not finite, as from `cfg_scale_text` at 3.4e38, is never passed on: the request ends with
`out_of_range` naming no option ([ADR 0032](../adr/0032-no-synthesis-passes-audio-that-is-not-finite.md)).

The runtime's request takes more, which speech.cpp does not offer: `num_candidates` and `decode_mode` (make a request per
take, each with its own seed), `context_kv_cache` (it does not change the audio), the deprecated `cfg_scale` (set both
scales), the bounds `min_seconds`, `max_seconds`, `max_ref_seconds`, `max_text_len` and `max_caption_len` (the file gives
them), `ref_embed` in a request (a voice file holds it), `ref_latent` and `ref_latents` (Python pickles; a voice file
holds a latent), and `ref_ensure_max`, which is on, as the runtime's default.

## Instructions

`instructions`, the runtime's caption, describes the voice and the way of speaking, under the name OpenAI's speech API
gives it. With a reference, the speech keeps the reference's voice and follows the description where it can. With the
voice `none`, the description alone chooses the voice:

```sh
speech tts Irodori-TTS-866M-MF-v4.1-F16.gguf --voice none --instructions "低く落ち着いた男性の声で、ゆっくりと読み上げてください。" \
    -o out.wav "明日の東京は晴れです。"
```

The caption loses what Python's `str.strip()` removes at either end, U+3000 among it, and is not normalized as the text
is. A caption that strips to nothing is no caption. A request without instructions computes none of their work.

## Streaming

The sampler makes a sentence's whole latent before any audio. The codec then decodes it in windows and passes each one on:
a first window of 12 frames (0.48 s), then windows of 24 to 48 frames, each the largest the decoder expects to finish
while the listener still has 0.1 s of audio, from its measured speed on the windows before
([ADR 0031](../adr/0031-irodori-tts-sizes-its-decoders-later-windows-by-its-measured-speed.md)). The sizes do not change
the audio: any windows give the samples of decoding the whole latent at once, bit for bit, so a seed repeats its samples.

## Accuracy

Every stage is checked against the official implementation's tensors on an Apple M5
([checks](../development/checks.md#irodori-tts)):

| Check | CPU, F32 | Metal, F32 | Vulkan, F16 model and F32 codec |
|---|---|---|---|
| Normalization and tokens, 164 texts | all equal | all equal | all equal on the 51 texts without emoji |
| Decoded audio | 119 dB SNR | 68 dB | 68 dB |
| Whole synthesis from the official noise | 75 to 110 dB, the official length | 22 to 61 dB, the same length | 58 dB (MF), the same length |

On Metal the audio is the same speech rather than the same waveform: Metal's matrix kernel rounds its inputs to half
precision, and MeanFlow's four large steps carry the difference into the latent.

## Speed

The 20 sentences of speech-bench's `prompts/speak-ja-JP.json` through the worker, in a voice file, one request at a time:

| Model | Device | Median first audio | p90 first audio | Real-time factor | Memory |
|---|---|---|---|---|---|
| v4.1-Small-MF F16, 4 steps | Apple M5, Metal | 0.23 s | 0.51 s | 0.17 | 2.2 GB |
| v4.1-Small-MF Q8_0, 4 steps | Apple M5, Metal | 0.25 s | 0.51 s | 0.18 | 1.5 GB |
| v4.1-Small F16, 16 steps | Apple M5, Metal | 1.12 s | 3.29 s | 0.34 | 2.2 GB |
| v4.1-Small F16, 40 steps | Apple M5, Metal | 2.64 s | 8.04 s | 0.64 | 2.2 GB |
| v4.1-Small-MF F16, 4 steps | RTX 2080, Vulkan | 0.13 s | 0.23 s | 0.10 | 2.1 GB |
| v4.1-Small-MF Q8_0, 4 steps | RTX 2080, Vulkan | 0.13 s | 0.22 s | 0.07 | 1.5 GB |
| v4.1-Small F16, 16 steps | RTX 2080, Vulkan | 0.49 s | 1.13 s | 0.14 | 2.2 GB |

The first audio comes after the text, the whole sampler and the codec's first window, so it grows with the sentence.
Memory is the worker's peak footprint on the M5 and the rise of the GPU's memory on the RTX 2080.
