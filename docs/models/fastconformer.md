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
speech asr reazonspeech-v2 --vad silero-vad meeting.wav            # a recording of minutes, by its regions of speech
speech asr reazonspeech-v2 --decoding greedy meeting.wav           # faster, with greedy decoding
speech worker parakeet-tdt_ctc-0.6b-ja                             # a recognition worker
speech serve parakeet-tdt-0.6b-v3                                  # POST /v1/audio/transcriptions
```

## What it does

- Each model writes the text NeMo's `transcribe()` writes by default: ReazonSpeech with a beam search, the parakeet models
  with greedy decoding.
- A request is recognized whole, however long, but a stretch that holds several sentences loses whole sentences, as in
  NeMo itself: on minutes of Common Voice ja clips joined with pauses, reazonspeech-v2 dropped 522 of 600 and
  parakeet-ja 266. Recognize a long recording by the regions where someone speaks, with `--vad silero-vad` or
  `chunking_strategy` (the [record](../adr/transcription-by-regions-recognizes-each-region-where-someone-speaks-alone-in-the-tools-and-joins-the-texts.md)
  has the measurement). ReazonSpeech's time and memory grow with the length; the parakeet models attend over the whole
  request, so theirs grow with its square.
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

On the 4,483 utterances of Common Voice 8.0's Japanese test set, trimmed to the voice, the CER is 7.88% for
parakeet-tdt_ctc-0.6b-ja, 12.03% for reazonspeech-nemo-v2 with beam search and 12.49% with greedy decoding (2.98%,
7.13% and 7.69% accepting other spellings of the same words). parakeet-tdt-0.6b-v3's WER on 300 English utterances of
FLEURS is 8.76%.

## Speed

speech-bench on 2026-10-10, F16, one warm request at a time. The Japanese timing sample is the first 100 Common Voice
inputs, 99 retained by VAD, repeated three times; English uses 300 FLEURS inputs, 299 retained by VAD, once. These
latencies are measured separately from the complete accuracy sets above. Loading and first-use warmup are excluded.

| Model | Utterances | Median, M5 / RTX 2080 | p90, M5 / RTX 2080 |
|---|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | Common Voice ja | 0.052 s / 0.040 s | 0.073 s / 0.066 s |
| reazonspeech-nemo-v2, beam search | Common Voice ja | 0.085 s / 0.106 s | 0.135 s / 0.191 s |
| reazonspeech-nemo-v2, greedy | Common Voice ja | 0.054 s / 0.047 s | 0.079 s / 0.078 s |
| parakeet-tdt-0.6b-v3 | FLEURS en | 0.093 s / 0.082 s | 0.148 s / 0.111 s |

A long recording is recognized whole. ReazonSpeech and parakeet-tdt_ctc-0.6b-ja use global attention, whose memory
grows with the square of the input length. Use `speech asr --vad silero-vad` to transcribe long recordings by region.
