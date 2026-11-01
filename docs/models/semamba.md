# SEMamba

This page describes speech.cpp's port of NVIDIA NemotronLabs RE-USE, a SEMamba network that removes noise,
reverberation, clipping, band limits and codec damage from recorded speech and gives back the same recording, at the
rate it was given, for people to listen to: a recorded meeting or call, a voice message, or a reference recording for a
cloned voice. In front of a recognizer it lowers a fast recognizer's error a little in loud noise, hardly changes
Qwen3-ASR 1.7B's and raises both on broadcast audio ([below](#recognition-in-noise)), so recognition runs it only when
asked.

| Name | Upstream | Languages | File |
|---|---|---|---|
| `re-use` | [nvidia/NVIDIA-NemotronLabs-RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE), OpenMDW 1.1 | any; it takes none | `NVIDIA-NemotronLabs-RE-USE-10M-F32.gguf`, 38.5 MB |
| `re-use-laughter` | [nvidia/NVIDIA-NemotronLabs-RE-USE-laughter](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE-laughter), OpenMDW 1.1 | any; it takes none | `NVIDIA-NemotronLabs-RE-USE-10M-laughter-F32.gguf`, 38.5 MB |

```sh
speech enhance re-use noisy.wav -o clean.wav
speech enhance re-use --bwe 48000 phone-call.wav -o wide.wav       # resampled to 48 kHz first, so that the model can extend its band
speech enhance re-use-laughter *.wav -o enhanced/
speech asr qwen3-asr-1.7b --enhance re-use crowd.wav                # each file enhanced whole before it is recognized
```

The files are made from the official checkpoints with `reference/semamba/convert.py`, on
[sakasegawa/NVIDIA-NemotronLabs-RE-USE-GGUF](https://huggingface.co/sakasegawa/NVIDIA-NemotronLabs-RE-USE-GGUF) and
[sakasegawa/NVIDIA-NemotronLabs-RE-USE-laughter-GGUF](https://huggingface.co/sakasegawa/NVIDIA-NemotronLabs-RE-USE-laughter-GGUF).
The laughter model keeps laughter that `re-use` removes with the noise; otherwise the two behave alike.

## What it does

- It takes audio at any rate from 8 to 48 kHz as it is and gives back as many samples at the same rate. `--bwe`
  resamples the audio to another rate in that range first, as the official script's `--BWE` does; a rate above the
  audio's lets the model fill in the band the recording lacked.
- It hears the audio in chunks of `chunk_size_in_seconds` that start every `hop_length_portion` of a chunk, the last
  one ending with the audio, and joins them under a Hann window, as the official `inference_chunk.py` does. Audio
  shorter than a chunk is enhanced whole. The enhanced audio comes out a chunk at a time, once no later chunk changes it.
- Each chunk is one STFT, the network and an inverse STFT. The STFT and the inverse run on the CPU, the network on the
  model's device.
- The model takes no language, and its `language` is `auto` alone.
- `speech enhance`, `speech worker`, `speech serve`'s `POST /v1/audio/enhancements` and the C API's `speech_enhance()`
  take it. `speech asr --enhance` and the server's `noise_reduction` enhance each file or utterance whole before it is
  recognized; no recognition runs it unless asked.

## Options

| Option | Default | Notes |
|---|---|---|
| `chunk_size_in_seconds` | 5 | the seconds a chunk holds, above 0 and up to 3600; memory grows with it |
| `hop_length_portion` | 0.5 | above 0 and up to 1: the share of a chunk from one chunk's start to the next |
| `bwe` | none | a rate from 8000 to 48000 Hz to resample the audio to before it is enhanced |

## Accuracy

Checked against RE-USE's official code at commit 31f82f2 on nine inputs: FLEURS speech under babble or white noise at
16 kHz, the same at 8, 22.05, 44.1 and 48 kHz, a clip of 0.03 s, and chunks a quarter apart: every stage of each chunk
from its own dumped input, then each input's audio through the chunks. On an Apple M5 (2026-11-01), as SNR against the
official's:

| Device | Each TFMamba block | The network, each chunk | The enhanced audio |
|---|---|---|---|
| CPU | 125.8 dB | 64.8 to 113.6 dB | 30.1 to 67.7 dB |
| Metal | 124.7 dB | 78.4 to 113.7 dB | 30.1 to 67.4 dB |
| Vulkan through MoltenVK | 124.6 dB | 78.6 to 115.7 dB | 30.1 to 67.4 dB |
| RTX 2080, Vulkan | 109.3 dB | 78.5 to 106.4 dB | 53.8 to 62.7 dB |

The RTX 2080's figures are of four of the inputs, 2026-11-01. On every device the clip of 0.03 s, six frames, comes out
15.8 to 15.9 dB from the official's.

A block's error is the rounding of float. Through 30 blocks of recurrent scans it grows with the length of a chunk, the
most on the 12 s chunk of the input enhanced whole. The audio lies further from the official's than the network's
output does: the phase of each bin is the angle of the phase decoder's two outputs, and where both are near zero a
rounding turns it far, in bins whose magnitude is small. Audio given a piece at a time gives the whole audio's samples
bit for bit on every device.

## Recognition in noise

RE-USE was trained to give speech back for people to listen to: of speech that overlaps it keeps the main speaker and
removes the others. The error rates of a fast recognizer of each language and of `qwen3-asr-1.7b`, each utterance as
recorded and enhanced whole as `speech asr --enhance` enhances it, here by the official code, with speech-bench's
counting, WER in English and CER in Japanese (2026-10-31 to 2026-11-02, speech 0.9.2's files):

| Audio | Fast recognizer | As recorded | Enhanced | `qwen3-asr-1.7b` as recorded | Enhanced |
|---|---|---|---|---|---|
| LibriSpeech test-clean, 2,620 utterances, under pub noise at 10 dB | `parakeet-tdt-0.6b-v3` | 2.82% | 2.51% | 2.28% | 2.30% |
| the same at 5 dB | | 4.11% | 3.82% | 3.07% | 3.28% |
| the same at 0 dB | | 9.13% | 8.18% | 6.36% | 6.79% |
| FLEURS en_us test, the first 300, as recorded | | 5.21% | 4.96% | 3.81% | 4.97% |
| FLEURS ja test, 650 utterances, under a cafeteria, a station or a square of DEMAND at 10 dB | `reazonspeech-v2` | 6.47% | 6.58% | 5.93% | 5.84% |
| the same at 5 dB | | 7.28% | 7.04% | 6.67% | 6.40% |
| the same at 0 dB | | 9.99% | 9.06% | 9.33% | 9.06% |
| FLEURS ja test as recorded | | 6.34% | 6.01% | 5.33% | 5.48% |
| ReazonSpeech test, 5,263 clips of Japanese television | `reazonspeech-v2` | 4.69% | 8.36% | 25.63% | 27.20% |

In loud noise enhancement takes about a tenth of the fast recognizers' errors off, and at 10 dB little or none.
Qwen3-ASR 1.7B makes fewer errors in noise to begin with and gains less: its CER falls by at most 0.27 points under
Japanese noise, and its WER rises by up to 0.43 points under English pub noise. On FLEURS en its rate rises because
after enhancement it writes numbers as words in 44 more of the 300 transcriptions ("six thousand" for "6,000"), which
the counting does not take as the same; over the 244 utterances whose reference holds no digit it falls from 3.55% to
3.18%. On television enhancement nearly doubles ReazonSpeech v2's errors: it removes music and the speech of other
people that the clips hold, and with them, in some clips, a short reply or the voice itself.

speech.cpp's own enhancement of the first 300 utterances of each language at 5 dB gives the official's error within
0.02 points (2.67% against 2.67% in English, 7.08% against 7.10% in Japanese), and chunks of 5 s, as `speech enhance`
takes them by default, give the error of whole utterances within 0.05 points (3.82% at 5 dB in English, 6.99% against
7.04% in Japanese).

## Speed

The RTFx (the seconds of audio over the seconds they took) of `speech enhance` on 60 s of noisy speech, the median of
three runs and one on the CPU, with the peak memory of the process on the Apple M5 and the GPU's memory it adds on the
RTX 2080 (2026-11-01):

| Device | 16 kHz | 48 kHz (`--bwe 48000`) | Memory at 16 kHz |
|---|---|---|---|
| Apple M5, Metal | 2.86 | 0.96 | 177 MiB, and 242 MiB at 48 kHz |
| Apple M5, CPU | 0.17 | not run | 1,090 MiB |
| RTX 2080, Vulkan | 3.29 | 1.15 | 1,011 MiB of VRAM, and 2,200 MiB at 48 kHz |

Each part of the audio is heard twice, since a chunk starts every half chunk, and the network's work grows with the
bins of the STFT, three times as many at 48 kHz as at 16 kHz. Utterances enhanced whole, as `speech asr --enhance`
takes them, are heard once: on the M5 2,641 s of them took 591 s, an RTFx of 4.5.
