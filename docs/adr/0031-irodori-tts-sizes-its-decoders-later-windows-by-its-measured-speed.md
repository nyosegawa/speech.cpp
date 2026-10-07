# Irodori-TTS sizes its decoder's later windows by its measured speed

Decided 2026-10-07.

## Context

Irodori-TTS makes a sentence's whole latent, then its codec decodes it in windows and passes each on: 12 frames
(0.48 s) first, then 48 at a time, each with 10 frames on either side decoded and dropped. Through 0.7.1's worker on
speech-bench's 20 sentences, the 48-frame second window came 0.288 s after the first on an Apple M5, with 0.19 s of the
first window's audio left to play, and 0.32 s after it on an RTX 2080. A machine about half as fast as the M5 would run
dry between the first and the second window, every sentence.

A window decodes its frames and 20 more: 12 frames decode 32, 24 decode 44, 48 decode 68. Smaller windows bring audio
sooner and decode more in all.

Qwen3-TTS's codec keeps a cache across chunks, and decoding the same codes in other chunk sizes changes about 98% of
its samples, by up to 1.7e-4 on Metal and 7e-4 on the CPU, so it keeps a fixed schedule of 1, 1, 2 and 4 frames
(docs/adr/0024).

## Decision

- **The first window stays 12 frames**, as early as before.
- **Each later window is the largest from 24 to 48 frames that the decoder expects to finish while the listener still
  has 0.1 s in hand**, the listener taken to play from the first window on and the decoder's time per frame being the
  running estimate of the windows decoded, margins included. Where not even 24 frames finish in time, the window is 24
  when decoding at it still gains on the listener and 48 when decoding is slower than real time, since smaller windows
  would only add margins to decode. The decision is one function without a clock (`next_window()`), which
  `irodori-window-check` checks on simulated machines.
- **48 frames, 0.7.1's window, is never exceeded**, so that a cancel and the silence between a worker's chunks wait no
  longer than they did.
- **The sizes may follow the clock because the audio does not depend on them.** The decoder is convolutions without a
  cache and a margin of 10 frames covers its receptive field of 7.7, so every window pattern gives the samples of one
  decode, bit for bit, which `irodori-codec-check` checks on seven patterns on the CPU and Metal; a seed repeats its
  samples whatever the machine's load.

The alternatives were turned down:

- A fixed later window smaller than 48, such as 24 everywhere. Fast machines would decode 29% more per second of audio
  for nothing and break the stream into twice the chunks.
- A floor of 12. It decodes 88% more per second of audio than 48, which a machine that decodes 48-frame windows barely
  faster than real time cannot afford.
- Adaptive windows for Qwen3-TTS. Its samples depend on the chunking, so a seed would stop repeating its audio.

## Consequences

The windows of one request depend on the machine and its load, the samples do not. The bit-exactness on Vulkan is to be
checked on the Windows machine.
