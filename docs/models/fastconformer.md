# FastConformer

This page describes speech.cpp's port of NVIDIA NeMo's [FastConformer](https://arxiv.org/abs/2305.05084) recognizers:
two parakeet models and ReazonSpeech, which write the text of a recording.

| Name | Upstream | Languages | File |
|---|---|---|---|
| `reazonspeech-v2` | [reazon-research/reazonspeech-nemo-v2](https://huggingface.co/reazon-research/reazonspeech-nemo-v2) | ja, with punctuation | `reazonspeech-nemo-619M-v2-F16.gguf`, 1.24 GB |
| `parakeet-tdt_ctc-0.6b-ja` | [nvidia/parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) | ja | `parakeet-tdt_ctc-0.6B-ja-F16.gguf`, 1.24 GB |
| `parakeet-tdt-0.6b-v3` | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 25 European languages, with punctuation and capitals | `parakeet-tdt-0.6B-v3-F16.gguf`, 1.26 GB |

parakeet-tdt-0.6b-v3's languages are those of its model card: `bg` `cs` `da` `de` `el` `en` `es` `et` `fi` `fr` `hr`
`hu` `it` `lt` `lv` `mt` `nl` `pl` `pt` `ro` `ru` `sk` `sl` `sv` `uk`.

```sh
speech asr parakeet-tdt_ctc-0.6b-ja utterance.wav                  # the text on stdout
speech asr parakeet-tdt-0.6b-v3 --timestamps utterance.wav         # with the times of its segments
speech asr reazonspeech-v2 meeting.wav                             # a recording of minutes, whole
speech asr reazonspeech-v2 --decoding greedy meeting.wav           # the same with greedy decoding
speech worker parakeet-tdt_ctc-0.6b-ja                             # a recognition worker
speech serve parakeet-tdt-0.6b-v3                                  # POST /v1/audio/transcriptions
```

## What is implemented

Each model decodes as NeMo's `transcribe()` decodes it by default
([ADR 0009](../adr/0009-speech-recognition-runs-through-a-fastconformer-port.md),
[ADR 0012](../adr/0012-the-recognizer-decodes-with-the-models-default-decoder.md)).

- The frontend as NeMo runs it in evaluation (pre-emphasis, a centred STFT, the checkpoint's 80 or 128 mel filters, the
  log, and each mel bin normalized over the recording), on the host in double precision.
- The subsampling by 8 and the 24 conformer layers on ggml. The parakeet models attend over the whole recording.
  ReazonSpeech attends locally, as Longformer does in NeMo: over the 128 frames (10.24 s) on either side of each frame,
  and one global token, the first frame.
- The parakeet models' TDT decoder with NeMo's greedy decoding, the model's durations (0 to 4 frames) and at most 10
  tokens on one frame.
- ReazonSpeech's RNN-T decoder with the beam search its checkpoint configures, NeMo's `alsd` with a beam of 4; or, when a
  request sets `decoding` to `greedy`, NeMo's greedy decoding (`greedy_batch`, at most 10 tokens on one frame)
  ([ADR 0021](../adr/0021-an-rnnt-model-decodes-greedily-when-a-request-asks.md)).
- The text written as NeMo's decoding writes it, without the space before each punctuation mark of the vocabulary, and
  the times of the tokens and segments.

Not implemented: the CTC head of parakeet-tdt_ctc-0.6b-ja, which NeMo does not decode with by default and which writes a
different text on some utterances.

## Use

- The models recognize 16 kHz mono audio. The library resamples audio at another rate.
- None of the models has an input for a language: parakeet-v3 finds the language of the audio itself. A request's
  `language` is only checked against the model's languages and changes nothing in the text.
- A recording goes through the encoder whole, however long, as `transcribe()` runs it; speech.cpp does not cut it.
  ReazonSpeech's time and memory grow with the length of the recording. The parakeet models attend over all of it, and
  theirs grow with its square, so a request to them should be one utterance
  ([ADR 0013](../adr/0013-local-attention-recognizes-long-audio-whole-as-transcribe-does.md)). For a long recording,
  use `reazonspeech-v2`.
- The reazonspeech package pads the audio with 0.5 s of silence on either side before it calls `transcribe()`; speech.cpp
  does not, so its text is NeMo's for the audio as given.
- A request with `timestamps` also gets the tokens and segments with their times in seconds, from the frames the
  decoding emitted them on. A segment ends where NeMo ends one, at `.`, `?` or `!` at the end of a word, and, for the
  Japanese models, whose text has no spaces, after `。`, `？`, `！`, `?` and `!` wherever they stand.

| Option | Default | Notes |
|---|---|---|
| `language` | `auto` | one of the model's languages; only checked |
| `timestamps` | false | add the times of the tokens and segments |
| `decoding` | `beam` | reazonspeech-v2 only: `beam` or `greedy`. Greedy computes a fifth of the beam search's graphs and can write another text, often with fewer commas and full stops |

## Accuracy

Every stage is checked against NeMo's tensors on utterances of FLEURS' test split, on an Apple M5
([checks](../development/checks.md#fastconformer)):

| Model | Inputs | Text from the audio, CPU F32, CPU F16 and Metal |
|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | 3 ja_jp utterances, 6 to 26 s | NeMo's text on all three |
| parakeet-tdt-0.6b-v3 | 12 utterances of en_us, de_de, fr_fr and es_419, 6 to 23 s | NeMo's text on all twelve |
| reazonspeech-nemo-v2 | 8 ja_jp utterances and 2 joined inputs of 65 s and 311 s | NeMo's text on all ten, with beam search and with greedy decoding |

The times of tokens and segments are NeMo's exactly on every dump of the three models, from the dump's encoder output.

## Speed

Apple M5, F16 weights on Metal, after loading:

| Model | Audio | Time | Real-time factor |
|---|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | 6.36 s, 10.50 s, 25.50 s | 0.07 s, 0.11 s, 0.28 s | about 0.011 |
| parakeet-tdt-0.6b-v3 | 12 utterances of 5.6 to 23.4 s | 0.08 to 0.32 s | 0.012 |
| reazonspeech-nemo-v2 | 6.36 s, 25.50 s, 64.80 s, 311.22 s | 0.17 s, 0.55 s, 1.4 s, 4.9 s | 0.016 to 0.027 |

reazonspeech-nemo-v2's peak memory on the CPU with F16 weights is 1.35 GB for 6.36 s, 1.55 GB for 64.80 s and 2.35 GB for
311.22 s. parakeet-tdt_ctc-0.6b-ja's is 1.30, 1.49 and 4.14 GB, growing with the square of the length.
