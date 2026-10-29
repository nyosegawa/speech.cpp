# Nemotron 3.5 Transcribe

This page describes speech.cpp's port of NVIDIA's
[Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B), a speech model of 30B
parameters with 4B active, which writes speech word for word in 40 languages, as they are spoken, and can be told the
names and terms the audio holds.

| Name | Upstream | File |
|---|---|---|
| `nemotron-3.5-transcribe` | [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) | `NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-IQ4_NL.gguf`, 19.0 GB |

```sh
speech asr nemotron-3.5-transcribe meeting.wav
speech asr nemotron-3.5-transcribe --prompt "Nemotron, FLEURS" meeting.wav
speech asr nemotron-3.5-transcribe --vad silero-vad --diarize nemotron-3-diarization meeting.wav
speech serve nemotron-3.5-transcribe silero-vad                  # POST /v1/audio/transcriptions, with "prompt"
```

## What it does

- It writes what the model card's verbatim instruction asks: the words as spoken, fillers and repetitions included,
  with punctuation and capitals, in the language spoken, which may change within a recording. It does not translate.
- It takes up to an hour of audio at once, and writes the text the official implementation, NeMo's
  `SALMAutomodel.generate()`, writes with greedy decoding, after the model card's guardrail against loops: a character
  repeated more than 20 times, or a block repeated 20 times or more, is kept once.
- It writes no language: the result names none.

Not supported: the card's other tasks (speaker tags, translation, timestamps from its CTC head, questions about the
audio and instructions of one's own), and its multi-token prediction, which speeds up decoding on a GPU server.

On near silence the model can write words no one said, as the official implementation does: on a quiet recording of
noise, it wrote `Um, I think I'm gonna go- go ahead and start with the first one.` Give it the regions where someone
speaks (`--vad silero-vad`) for recordings with long silences.

## Options

| Option | Default | Notes |
|---|---|---|
| `prompt` | `""` | names and terms the audio may hold, as the card's contextual transcription takes them: `Nemotron, FLEURS` |
| `language` | `auto` | one of the 40 languages, a region or script of one, or `auto`. It is checked and steers nothing: the model writes the language spoken |
| `experts_on_cpu` | `true` | `false` refuses a request to a model whose GPU leaves some of its experts to the CPU (below) |

The languages are `ar` `bg` `bn` `cs` `da` `de` `el` `en` `es` `et` `fi` `fr` `he` `hi` `hr` `hu` `id` `it` `ja` `ko`
`lt` `lv` `mr` `mt` `nl` `pl` `pt` `ro` `ru` `sk` `sl` `sv` `ta` `te` `th` `tl` `tr` `uk` `vi` `zh`.

## Devices and memory

The model needs its whole file in memory, and from its load on the keys and values of an hour of audio and the
buffers of its largest graphs, about 1.9 GB beyond the weights.

- **A GPU that holds the whole model computes it all**, as an Apple M5 with 32 GB does with the IQ4_NL file: 18.3 GiB
  of Metal's memory at the load, and 18.4 GiB through an hour of audio.
- **A GPU that does not hold it keeps the experts of as many layers as its free memory holds**, beside the rest of the
  model and what the largest request needs, and the CPU computes the experts of the other layers with the load's
  threads (`--threads`). In IQ4_NL the model then needs about 3.6 GB free on the GPU for its weights outside the experts
  (1.7 GB), the caches and the graphs, and each of its 23 layers of experts takes 0.72 GB more on the GPU or in the
  CPU's memory. The load says so:

  ```
  Nemotron 3.5 Transcribe computes the experts of 18 of its 23 layers of experts on the CPU, since Vulkan0 holds those of 5 beside the rest of the model; a GPU with more memory free runs it faster
  ```

- **`experts_on_cpu` false** refuses a request to such a model, for a caller that wants the GPU's speed or nothing.

## Accuracy

Every stage is checked against NeMo's official modules and Transformers' NemotronH, on seven requests made of FLEURS
clips (a Japanese and an English utterance, a meeting of 98 s with five voices, and quiet noise) with the card's
instructions, on an Apple M5:

| Check | F32 on the CPU | F32 on Metal |
|---|---|---|
| The audio's rows the language model takes | 79.2 dB SNR | 52.3 dB SNR |
| The language model's 52 layers, each from its own input | 115.4 dB over the prompt, 97.4 dB decoding | 67.8 dB and 74.0 dB |
| The recognition of the five requests of the verbatim and contextual instructions | the official tokens and text of every one | |

Where two of a layer's experts score within float32's rounding of each other, the port may choose the other, as the
official code may on another machine; the checks leave out such rows, 45 of 76,521 on the CPU and 1,550 on Metal.

## Weight types

The model comes in three types; `nemotron-3.5-transcribe` fetches IQ4_NL, and `NAME:TYPE` another
(`speech asr nemotron-3.5-transcribe:q8_0`). Its language model's rows are not whole blocks of a K-quant, so there are no
Q6_K, Q5_K and Q4_K files, and IQ4_NL, of 4.5 bits a weight, takes their place. speech-bench on 2026-10-29 on an H100
with Vulkan: the first 100 utterances of FLEURS's Japanese and English test sets and the whole of Common Voice 8.0's
Japanese test set, trimmed to the voice, with the CER accepting other spellings of the same words after the slash.

| Type | File | FLEURS ja CER | Common Voice ja CER | FLEURS en WER | The GPU's memory | RTFx, FLEURS ja |
|---|---|---|---|---|---|---|
| F16 | 67.4 GB | 6.05% / 2.64% | 8.85% / 5.06% | 8.73% | 61.6 GiB | 17 |
| Q8_0 | 35.8 GB | 5.86% / 2.64% | 9.01% / 5.17% | 8.64% | 33.4 GiB | 23 |
| IQ4_NL | 19.0 GB | 5.88% / 2.62% | 9.16% / 5.33% | 8.77% | 18.4 GiB | 24 |

On an Apple M5 with Metal, the IQ4_NL file gives 6.11% / 2.82% on FLEURS ja, 11.71% / 5.47% on the first 100 utterances
of Common Voice ja, as on the H100, and 8.68% on FLEURS en.

## Speed

speech-bench on 2026-10-29, one utterance at a time through the worker, IQ4_NL: the first 100 utterances of FLEURS's
Japanese and English test sets and of Common Voice 8.0's Japanese test set, trimmed to the voice. The RTFx is the
seconds of audio over the seconds they took, so larger is faster; the memory is the process's on the M5, whose GPU
shares it, and the GPU's on the H100. A recognition writes a token at a time, so a short utterance with many words takes
longest for its length.

| Device | p50 latency, FLEURS ja / CV ja / FLEURS en | RTFx, FLEURS ja / CV ja / FLEURS en | Memory | Load |
|---|---|---|---|---|
| Apple M5 (32 GB), Metal | 1.29 s / 0.61 s / 0.88 s | 7.0 / 5.4 / 8.5 | 17.2 GiB | 5 to 6 s |
| H100 (80 GB), Vulkan | 0.39 s / 0.22 s / 0.29 s | 24 / 13 / 26 | 18.4 GiB | 7 s |

On an RTX 2080 (8 GB) with Vulkan and an i9-9900K, which leaves the experts of 18 of the 23 layers to the CPU,
`speech asr` wrote 9 s of English in 3.2 s and a 98 s meeting in 46 s, RTFx 2.8 and 2.1, with 6.1 GiB of the GPU's
memory and 13 GB of the CPU's for the experts, after a load of 40 to 48 s.

The first request after a GPU's driver first meets the model, after an install or a driver update, also waits for its
shaders to compile: 5 to 11 s on the H100.
