# Sortformer

This page describes speech.cpp's port of NVIDIA's Streaming Sortformer, which finds who speaks when in a recording or in
audio as it arrives.

| Model | Upstream | Languages | File |
|---|---|---|---|
| Nemotron 3 Diarization | [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization), OpenMDW 1.1 | any; it takes none | `Nemotron-3-Diarization-99M-F32.gguf`, 397 MB, and F16 (199 MB), Q8_0 (106 MB) |

```sh
speech diarize Nemotron-3-Diarization-99M-F16.gguf meeting.wav          # FILE, START, END and SPEAKER of each segment
speech diarize Nemotron-3-Diarization-99M-F16.gguf --format rttm a.wav b.wav > hyp.rttm
speech diarize Nemotron-3-Diarization-99M-F16.gguf --streaming-mode low_latency meeting.wav
```

The file is made from the official checkpoint with `reference/sortformer/convert.py` and `speech quantize`.

## What it does

- The model gives, every 10 ms, the probability that each of up to 8 speakers speaks, and the segments follow from
  them by the postprocessing of NeMo's `diarize()`. Speakers are numbered from 0 in the order they first speak, and
  keep their number for the whole recording or stream. A number does not name a person.
- Two speakers' segments overlap where they speak at once; one speaker's are apart. Times are seconds from the start of
  the audio, to 10 ms.
- Audio at any rate is resampled to 16 kHz first. The model takes no language, and its `language` is `auto` alone.
- It hears the audio in chunks, each after a speaker cache and a queue of the frames before it, by which it tells the
  speakers it has heard apart. The streaming mode sets how long the chunks are (below).
- Audio that arrives a piece at a time gives the segments of the whole audio through the C API's diarizations
  ([c-api.md](../c-api.md#diarizations)), each once nothing that follows can change it. Each diarization numbers its
  own speakers.
- `speech diarize` and the C API's `speech_diarize()` and diarizations take it; `speech worker` does not.

## Streaming modes

| `streaming_mode` | Waits for | Taken by |
|---|---|---|
| `offline` | 30.4 s | `speech_diarize()` and `speech diarize` by default |
| `low_latency` | 1.04 s | a diarization of audio as it arrives, by default |
| `very_low_latency` | 0.64 s | |
| `ultra_low_latency` | 0.32 s | |

A mode waits for a chunk and the right context after it before it diarizes the chunk. NeMo's settings, in 80 ms frames,
replace a mode's own:

| Option | offline | the others | Notes |
|---|---|---|---|
| `chunk_len` | 340 | 9, 6, 3 | the frames of a chunk |
| `chunk_right_context` | 40 | 4, 2, 1 | the frames after a chunk that it hears |
| `fifo_len` | 40 | 264 | the latest frames kept before they go to the speaker cache |
| `spkcache_len` | 264 | 264 | the frames the speaker cache keeps, 16 or more |
| `spkcache_update_period` | 300 | 222 | the frames the queue gives the cache at once |

## Options

| Option | Default | Notes |
|---|---|---|
| `onset` | 0.5 | from 0 to 1: the probability above which a speaker begins |
| `offset` | 0.5 | from 0 to 1: the probability below which a speaker stops |
| `pad_onset` | 0 | seconds added before each segment, up to 3600, as for the three that follow |
| `pad_offset` | 0 | seconds added after each segment; a speaker's padded segments that meet are joined |
| `min_duration_on` | 0 | seconds below which a segment is dropped |
| `min_duration_off` | 0 | seconds below which a pause between a speaker's segments is filled |

With the defaults a stream gives a segment as soon as the chunk that ends it has been diarized. Padding and the two
durations make it wait as long as they reach.

## Accuracy

Checked against NeMo at commit d32dd55 on three inputs, 109 s in all, of FLEURS clips of five voices with pauses and
overlaps, silence and noise, in each streaming mode: every stage of each step, then each step's speaker cache and
queue, then the probabilities and the segments from the audio, and the segments of six sets of postprocessing options.
On an Apple M5:

| Device | Probabilities, largest difference from the official | Frames of a speaker that differ |
|---|---|---|
| CPU, F32 | 3.2e-5 | none |
| Metal, F32 | 0.12 | 0.41% in `low_latency`, none in the other modes |

On Metal the speaker cache once kept another frame, which then changed what the steps after it heard. Audio given in
pieces of 10 ms or of 150 ms gives the probabilities of the whole audio bit for bit on both devices.

## Speed

98 s of a meeting on an Apple M5's GPU, F32: 0.22 s in `offline`, an RTFx (the seconds of audio over the seconds they
took) of about 450, and 3.9 s in `low_latency`, whose 137 chunks take 28 ms each where audio brings one every 0.72 s.
