# FastConformer

This page describes speech.cpp's port of NVIDIA NeMo's [FastConformer](https://arxiv.org/abs/2305.05084) recognizers:
two parakeet models, ReazonSpeech and Haas, which write the text of a recording.

| Name | Upstream | Languages | File |
|---|---|---|---|
| `reazonspeech-v2` | [reazon-research/reazonspeech-nemo-v2](https://huggingface.co/reazon-research/reazonspeech-nemo-v2) | ja, with punctuation | `reazonspeech-nemo-619M-v2-Q8_0.gguf`, 0.66 GB |
| `parakeet-tdt_ctc-0.6b-ja` | [nvidia/parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) | ja | `parakeet-tdt_ctc-0.6B-ja-Q8_0.gguf`, 0.66 GB |
| `parakeet-tdt-0.6b-v3` | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 25 European languages, with punctuation and capitals | `parakeet-tdt-0.6B-v3-F16.gguf`, 1.26 GB |
| `haas-ja-v1.0` | [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) | ja, with punctuation | `Haas-616M-ja-v1.0-Q8_0.gguf`, 0.66 GB |

parakeet-tdt-0.6b-v3's languages are those of its model card: `bg` `cs` `da` `de` `el` `en` `es` `et` `fi` `fr` `hr`
`hu` `it` `lt` `lv` `mt` `nl` `pl` `pt` `ro` `ru` `sk` `sl` `sv` `uk`.

```sh
speech asr parakeet-tdt_ctc-0.6b-ja utterance.wav                  # the text on stdout
speech asr haas-ja-v1.0 --timestamps utterance.wav                 # Japanese with punctuation, with times
speech asr parakeet-tdt-0.6b-v3 --timestamps utterance.wav         # with the times of its segments
speech asr reazonspeech-v2 --vad silero-vad meeting.wav            # a recording of minutes, by its regions of speech
speech asr reazonspeech-v2 --decoding greedy meeting.wav           # faster, with greedy decoding
speech worker parakeet-tdt_ctc-0.6b-ja                             # a recognition worker
speech serve parakeet-tdt-0.6b-v3                                  # POST /v1/audio/transcriptions
```

## What it does

- Each model writes the text NeMo's `transcribe()` writes by default: ReazonSpeech with a beam search, the parakeet models
  and Haas with greedy decoding. Haas writes a rare character its vocabulary lacks from its UTF-8 bytes, as its
  tokenizer does.
- A request is recognized whole, however long, but a stretch that holds several sentences loses whole sentences, as in
  NeMo itself: on minutes of Common Voice ja clips joined with pauses, reazonspeech-v2 dropped 522 of 600 and
  parakeet-ja 266. Recognize a long recording by the regions where someone speaks, with `--vad silero-vad` or
  `chunking_strategy` (the [record](../adr/transcription-by-regions-recognizes-each-region-where-someone-speaks-alone-in-the-tools-and-joins-the-texts.md)
  has the measurement). ReazonSpeech's time and memory grow with the length; the parakeet models attend over the whole
  request, so theirs grow with its square, and so do Haas's.
- parakeet-v3 finds the language of the audio itself. None of the models takes a language: a request's `language` is only
  checked against the model's languages.
- With `timestamps`, the result also has the tokens and segments with their times in seconds. A segment ends at a full
  stop, a question mark or an exclamation mark.
- On a GPU, the encoder runs on the GPU, and the decoding runs on the CPU where the file's decoding weights are
  quantized (Q8_0, Q6_K, Q5_K and Q4_K files) and on the GPU where they are F16 or F32: each step of the decoding is a
  small computation, which the CPU finishes before a round trip to the GPU would with quantized weights but not with F16
  ones. `decode_on_cpu` chooses either way per request. When a request first decodes on the CPU, the model copies the
  weights the decoding reads into the CPU's memory and keeps them, 12 MB for the Q8_0 files of reazonspeech-v2 and
  parakeet-ja and 35 MB for the F16 file of parakeet-v3, and the CPU computes with `--threads`. The CPU's arithmetic
  differs from the GPU's, so a long recording can come out a word different where two words score almost the same.

Not supported: the CTC head of parakeet-tdt_ctc-0.6b-ja, which NeMo does not decode with by default, and the custom
vocabulary and `transcribe_longform()` of Haas's Python package; recognize a long recording by its regions instead.

## Options

| Option | Default | Notes |
|---|---|---|
| `language` | `auto` | one of the model's languages; only checked |
| `timestamps` | false | add the times of the tokens and segments |
| `decoding` | `beam` | reazonspeech-v2 only: `beam` or `greedy`. Greedy is faster and can write another text, often with fewer commas and full stops |
| `decode_on_cpu` | true for Q8_0 and K-quant files, false for F16 and F32 | for a model on a GPU, decode on the CPU (true) or on the GPU (false). A model on the CPU decodes there either way |

## Accuracy

Every stage is checked against NeMo's tensors on utterances of FLEURS' test split, on an Apple M5:

| Model | Inputs | Text from the audio, CPU F32, CPU F16 and Metal |
|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | 3 ja_jp utterances, 6 to 26 s | NeMo's text on all three |
| parakeet-tdt-0.6b-v3 | 12 utterances of en_us, de_de, fr_fr and es_419, 6 to 23 s | NeMo's text on all twelve |
| reazonspeech-nemo-v2 | 8 ja_jp utterances and 2 joined inputs of 65 s and 311 s | NeMo's text on all ten, with beam search and with greedy decoding |
| Haas-ja-v1.0 | 5 ja_jp utterances, 6 to 26 s, two of them with characters it writes from their bytes | NeMo's text on all five |

The times of tokens and segments are NeMo's exactly with F32 weights, and with F16 weights decoding on Metal. Decoding on
the CPU with F16 weights, a token can fall one frame (80 ms) from NeMo's where two frames score almost the same: one
token in 2 of reazonspeech-nemo-v2's 20 decodings.

On the 4,483 utterances of Common Voice 8.0's Japanese test set, trimmed to the voice, the CER in Q8_0 is 7.93% for
parakeet-tdt_ctc-0.6b-ja, 12.04% for reazonspeech-nemo-v2 with beam search and 12.47% with greedy decoding (3.01%,
7.12% and 7.66% accepting other spellings of the same words), on the M5 and within 0.01 points on the RTX 2080.
parakeet-tdt-0.6b-v3's WER in F16 on 300 English utterances of FLEURS is 8.76% on the M5 and 8.78% on the RTX 2080.

Haas-ja-v1.0 was trained on Common Voice 26.0, and 1,033 of that test set's clips are in Common Voice 17.0's training
split, so it is compared on the 650 utterances of FLEURS' Japanese test split, which its card says it was not trained
on. In Q8_0 on the M5 (2026-10-30) its CER there is 4.39%, against 5.00% for parakeet-tdt_ctc-0.6b-ja and 7.95% for
reazonspeech-nemo-v2 with beam search (2.86%, 2.82% and 5.70% accepting other spellings of the same words).

## Weight types

Each model comes in five types; `NAME:TYPE` fetches one (`speech asr parakeet-tdt-0.6b-v3:q8_0`). speech-bench, two
rounds, on an Apple M5 with Metal on 2026-10-16 and an RTX 2080 with Vulkan on 2026-10-14. The error is the CER on the first 500
utterances of Common Voice 8.0's Japanese test set, on which the CER is higher than on the whole set, and after the
slash the CER accepting other spellings of the same words; for parakeet-tdt-0.6b-v3 it is the WER on 300 English
utterances of FLEURS, and for haas-ja-v1.0 the CER on the 650 utterances of FLEURS' Japanese test split, measured once
on both machines on 2026-10-30. The error is the M5's, and the RTX 2080's is within 0.05 points of it. The peak memory
is the process's on the M5, whose GPU shares it, and the VRAM the video memory in use on the RTX 2080.

| Model | Type | File | Error | Peak memory, M5 | VRAM, RTX 2080 | RTFx, M5 / RTX 2080 |
|---|---|---|---|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | F16 | 1.24 GB | 9.09% / 2.37% | 1.35 GiB | 1.32 GiB | 105 / 69 |
| parakeet-tdt_ctc-0.6b-ja | Q8_0 | 0.66 GB | 9.14% / 2.35% | 0.78 GiB | 0.74 GiB | 131 / 76 |
| parakeet-tdt_ctc-0.6b-ja | Q6_K | 0.51 GB | 9.08% / 2.48% | 0.63 GiB | 0.60 GiB | 125 / 80 |
| parakeet-tdt_ctc-0.6b-ja | Q5_K | 0.43 GB | 8.79% / 2.35% | 0.55 GiB | 0.52 GiB | 120 / 72 |
| parakeet-tdt_ctc-0.6b-ja | Q4_K | 0.36 GB | 9.26% / 2.48% | 0.47 GiB | 0.44 GiB | 132 / 77 |
| reazonspeech-nemo-v2, beam search | F16 | 1.24 GB | 14.12% / 7.14% | 1.41 GiB | 1.34 GiB | 49 / 28 |
| reazonspeech-nemo-v2, beam search | Q8_0 | 0.66 GB | 14.17% / 7.15% | 0.84 GiB | 0.75 GiB | 75 / 35 |
| reazonspeech-nemo-v2, beam search | Q6_K | 0.51 GB | 13.91% / 6.98% | 0.69 GiB | 0.61 GiB | 74 / 34 |
| reazonspeech-nemo-v2, beam search | Q5_K | 0.43 GB | 13.96% / 7.06% | 0.61 GiB | 0.53 GiB | 70 / 34 |
| reazonspeech-nemo-v2, beam search | Q4_K | 0.36 GB | 14.20% / 7.10% | 0.53 GiB | 0.45 GiB | 75 / 34 |
| parakeet-tdt-0.6b-v3 | F16 | 1.26 GB | 8.76% | 1.40 GiB | 1.45 GiB | 132 / 85 |
| parakeet-tdt-0.6b-v3 | Q8_0 | 0.67 GB | 9.04% | 0.82 GiB | 0.86 GiB | 165 / 101 |
| parakeet-tdt-0.6b-v3 | Q6_K | 0.52 GB | 8.87% | 0.67 GiB | 0.72 GiB | 159 / 103 |
| parakeet-tdt-0.6b-v3 | Q5_K | 0.44 GB | 9.15% | 0.58 GiB | 0.63 GiB | 153 / 102 |
| parakeet-tdt-0.6b-v3 | Q4_K | 0.36 GB | 9.51% | 0.52 GiB | 0.56 GiB | 166 / 102 |
| haas-ja-v1.0 | F16 | 1.23 GB | 4.40% / 2.87% | 1.28 GiB | 1.32 GiB | 121 / 91 |
| haas-ja-v1.0 | Q8_0 | 0.66 GB | 4.39% / 2.86% | 0.75 GiB | 0.78 GiB | 137 / 78 |
| haas-ja-v1.0 | Q6_K | 0.51 GB | 4.35% / 2.83% | 0.61 GiB | 0.65 GiB | 131 / 78 |
| haas-ja-v1.0 | Q5_K | 0.43 GB | 4.47% / 2.87% | 0.53 GiB | 0.57 GiB | 129 / 76 |
| haas-ja-v1.0 | Q4_K | 0.35 GB | 4.42% / 2.84% | 0.46 GiB | 0.50 GiB | 137 / 78 |

In the Q6_K, Q5_K and Q4_K files the prediction network and the joint stay Q8_0. On the RTX 2080, haas-ja-v1.0's F16
file was faster than its quantized files, which decode on the CPU by default; `decode_on_cpu` false decodes them on
the GPU.

## Speed

speech-bench, on the M5 on 2026-10-16 and the RTX 2080 on 2026-10-14, one request at a time, with the files the
names fetch: Q8_0 for parakeet-tdt_ctc-0.6b-ja and reazonspeech-nemo-v2 on the 4,483 utterances of Common Voice 8.0's
Japanese test set (4,479 kept by VAD), one round, F16 for parakeet-tdt-0.6b-v3 on 300 English utterances of FLEURS (299 kept), the medians of three rounds, and Q8_0 for haas-ja-v1.0 on the 650 utterances of FLEURS' Japanese test split, one round on 2026-10-30. The RTFx
is the seconds of audio over the seconds they took, so larger is faster.

| Model | Utterances | p50 latency, M5 / RTX 2080 | p90 latency, M5 / RTX 2080 | RTFx, M5 / RTX 2080 |
|---|---|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | Common Voice ja | 0.027 s / 0.050 s | 0.036 s / 0.076 s | 137 / 76 |
| reazonspeech-nemo-v2, beam search | Common Voice ja | 0.046 s / 0.092 s | 0.076 s / 0.179 s | 76 / 36 |
| reazonspeech-nemo-v2, greedy | Common Voice ja | 0.033 s / 0.043 s | 0.049 s / 0.083 s | 108 / 73 |
| parakeet-tdt-0.6b-v3 | FLEURS en | 0.058 s / 0.095 s | 0.091 s / 0.125 s | 130 / 86 |
| haas-ja-v1.0 | FLEURS ja | 0.063 s / 0.119 s | 0.097 s / 0.160 s | 137 / 78 |

A long recording is recognized whole. ReazonSpeech and parakeet-tdt_ctc-0.6b-ja use global attention, whose memory
grows with the square of the input length. Use `speech asr --vad silero-vad` to transcribe long recordings by region.
