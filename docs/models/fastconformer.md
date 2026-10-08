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
speech asr reazonspeech-v2 --decoding greedy meeting.wav           # faster, with greedy decoding
speech worker parakeet-tdt_ctc-0.6b-ja                             # a recognition worker
speech serve parakeet-tdt-0.6b-v3                                  # POST /v1/audio/transcriptions
```

## What it does

- Each model writes the text NeMo's `transcribe()` writes by default: ReazonSpeech with a beam search, the parakeet models
  with greedy decoding.
- A recording is recognized whole, however long. ReazonSpeech's time and memory grow with the length. The parakeet
  models attend over the whole recording, so theirs grow with its square, and a request to them should be one
  utterance. For a long recording, use `reazonspeech-v2`.
- parakeet-v3 finds the language of the audio itself. None of the models takes a language: a request's `language` is only
  checked against the model's languages.
- With `timestamps`, the result also has the tokens and segments with their times in seconds. A segment ends at a full
  stop, a question mark or an exclamation mark.

Not supported: the CTC head of parakeet-tdt_ctc-0.6b-ja, which NeMo does not decode with by default.

## Options

| Option | Default | Notes |
|---|---|---|
| `language` | `auto` | one of the model's languages; only checked |
| `timestamps` | false | add the times of the tokens and segments |
| `decoding` | `beam` | reazonspeech-v2 only: `beam` or `greedy`. Greedy is faster and can write another text, often with fewer commas and full stops |

## Accuracy

Every stage is checked against NeMo's tensors on utterances of FLEURS' test split, on an Apple M5:

| Model | Inputs | Text from the audio, CPU F32, CPU F16 and Metal |
|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | 3 ja_jp utterances, 6 to 26 s | NeMo's text on all three |
| parakeet-tdt-0.6b-v3 | 12 utterances of en_us, de_de, fr_fr and es_419, 6 to 23 s | NeMo's text on all twelve |
| reazonspeech-nemo-v2 | 8 ja_jp utterances and 2 joined inputs of 65 s and 311 s | NeMo's text on all ten, with beam search and with greedy decoding |

The times of tokens and segments are NeMo's exactly.

## Speed

Apple M5, F16 weights on Metal, after loading:

| Model | Audio | Time | Real-time factor |
|---|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | 6.36 s, 10.50 s, 25.50 s | 0.07 s, 0.11 s, 0.28 s | about 0.011 |
| parakeet-tdt-0.6b-v3 | 12 utterances of 5.6 to 23.4 s | 0.08 to 0.32 s | 0.012 |
| reazonspeech-nemo-v2 | 6.36 s, 25.50 s, 64.80 s, 311.22 s | 0.17 s, 0.55 s, 1.4 s, 4.9 s | 0.016 to 0.027 |

reazonspeech-nemo-v2's peak memory on the CPU with F16 weights is 1.35 GB for 6.36 s, 1.55 GB for 64.80 s and 2.35 GB for
311.22 s. parakeet-tdt_ctc-0.6b-ja's is 1.30, 1.49 and 4.14 GB, growing with the square of the length.
