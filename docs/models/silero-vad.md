# Silero VAD

This page describes speech.cpp's port of [Silero VAD](https://github.com/snakers4/silero-vad), which finds where
someone speaks in a recording, so that a recognizer can be given one utterance at a time.

| Name | Upstream | Languages | File |
|---|---|---|---|
| `silero-vad` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad), the 16 kHz model of v6.2 | any; it takes none | `silero-vad-309K-v6.2-F32.gguf`, 1.2 MB |

```sh
speech vad silero-vad meeting.wav                                  # FILE, START and END of each region, in seconds
speech vad silero-vad --format json one.wav two.wav > regions.jsonl
speech vad silero-vad --min-silence-duration-ms 1000 --max-speech-duration-s 10 meeting.wav
```

## What it does

- The model gives a speech probability for every 32 ms of audio, each with the 4 ms before it and what it heard before,
  and the regions follow from the probabilities by the rule of the official `get_speech_timestamps()`, with its options.
  speech.cpp gives the official's regions sample for sample.
- Audio at any rate is resampled to 16 kHz first. The times are seconds from the start of the audio.
- A file without speech, or with silence and faint noise alone, has no region.
- The model takes no language, and its `language` is `auto` alone.
- The file is F32 alone: `speech quantize` writes no other type of it.
- `speech worker` and `speech serve` do not take it; `speech vad` and the C API's `speech_detect()` do.

## Options

| Option | Default | Notes |
|---|---|---|
| `threshold` | 0.5 | from 0 to 1: the probability from which a chunk is speech. A region ends below the threshold less 0.15, or 0.01 at least |
| `min_speech_duration_ms` | 250 | a shorter region is dropped |
| `min_silence_duration_ms` | 100 | how long the probability stays below the end's threshold before a region ends |
| `speech_pad_ms` | 30 | added before and after each region; two regions closer than twice it share the silence between them |
| `max_speech_duration_s` | none | the longest a region may be, its padding included. A longer one is cut at its longest silence of more than 98 ms, or where it reaches the limit |

Read speech pauses for breath often, and with the defaults a region is a sentence or less. To give a recognizer a few
sentences at a time, join them with a longer `min_silence_duration_ms` and bound them with `max_speech_duration_s`: on a
minute of read German, a silence of 1000 ms gave 3 regions of 13 to 15 s, and a limit of 10 s cut them into 6 of 5 to
8 s.

## Accuracy

Checked against silero-vad 6.2.3 on six inputs, 122 s in all, of FLEURS and Common Voice clips, silence and noise: every
stage, then the probabilities from the audio, then the regions of seven sets of options, the defaults and 10 s and 3 s
limits among them. On an Apple M5:

| Device | Probabilities, largest difference from the official | Regions |
|---|---|---|
| CPU, F32 | 3.6e-6 | the official's, all 42 |
| Metal | 6.2e-3 | the official's, all 42 |

## Speed

A minute of audio on an Apple M5, after the model's load of 0.04 s: 0.041 s on the CPU and 0.045 s on Metal, a real-time
factor of 0.0007 and 0.0008. The official package takes 0.150 s on the same CPU.
