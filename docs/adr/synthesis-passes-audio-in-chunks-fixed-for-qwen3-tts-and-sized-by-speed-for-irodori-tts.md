# Synthesis passes audio in chunks fixed for Qwen3-TTS and sized by speed for Irodori-TTS

## Context

A player starts on the first audio a synthesis passes and runs dry when the next comes later than the audio it holds
lasts. Measured through the worker on speech-bench's 20 Japanese sentences on 2026-10-07:

- Qwen3-TTS makes its speech a frame (0.08 s) at a time. With the first frame alone and then 4 frames a chunk, the
  second chunk of the 1.7B model arrived 0.096 s after the first on an RTX 2080 and 0.136 s after it on an Apple M5,
  where a frame takes about 24 ms and 34 ms to make, so a player that starts on the first chunk's 0.08 s ran dry for 16
  to 59 ms in every sentence. Its codec carries its state from one call to the next, but its samples still depend, by
  rounding, on how the frames are grouped into calls: decoding one sentence's codes in chunks of 1, 1, 2 and 4 frames
  instead of 1 and 4 changed 98% of its samples, by up to 1.7e-4 on Metal and 7.0e-4 on the CPU with the F16 weights of
  the codec. A call of the codec took about 16 ms on the M5's Metal whether it decoded 1 frame or 4 (a rough measure on
  a shared GPU).
- Irodori-TTS makes a sentence's whole latent, then its codec decodes it in windows and passes each on. With 12 frames
  (0.48 s) first and then 48 at a time, each window with 10 frames on either side decoded and dropped, the second
  window came 0.288 s after the first on the M5 through 0.7.1's worker, with 0.19 s of the first window's audio left to
  play, and 0.32 s after it on the RTX 2080; a machine about half as fast as the M5 would run dry between them in every
  sentence. The decoder is convolutions without a cache whose receptive field reaches 8 frames on either side of a
  frame, so a window that decodes 8 frames on either side of its own and drops them gives the samples of one decode,
  bit for bit, whatever the sizes of the windows: 12 frames decode 28, 48 decode 64, 96 decode 112. On the M5's Metal
  with v4.1-Small-MF's F16 file (2026-10-13) a window takes 4.0 ms a decoded frame at any size from 12 to 256 frames,
  so a 48-frame window spends a quarter of its time on its margins, and the decoder's compute buffer grows by 3.5 MiB
  a decoded frame: 225 MiB for a 48-frame window, 394 MiB for 96, 507 MiB for 128 and 957 MiB for 256.

## Decision

- **Qwen3-TTS passes chunks of 1, 1, 2 and then 4 frames, fixed in advance** (`chunk_frames()`). The first frame alone
  lets the first audio come once one frame is made, the small chunks feed the player while the speech gets ahead of it,
  and 4 frames are the most a chunk holds, so that a cancel and the worker's silence last no longer than 4 frames. The
  same request with the same seed gives the same samples on every run, since the chunks do not depend on time.
- **Irodori-TTS sizes its decoder's later windows by its measured speed, and bounds the wait for one in time.** The
  first window is 12 frames. Each later window is the largest from 24 frames up that the decoder expects to finish
  while the listener still has 0.1 s in hand, the listener taken to play from the first window on and the decoder's
  time per frame being the running estimate of the windows decoded, margins included. A window takes up to 48 frames
  whatever its decoding takes. It takes more only while two things hold. Its decoding is expected to take no longer
  than the second window has, the first window's 0.48 s less the 0.1 s in hand, 0.38 s, so that no later window keeps a
  cancel or the caller waiting longer than the second may on any machine whose stream holds from the start; 0.38 s is
  what a 48-frame window takes on the slowest such machine, 1.5 times slower than the M5. And the listener holds, beyond
  the 0.1 s, the audio of every frame it decodes, margins included, so that it arrives in time even if its decoding
  slows to real time; the second window then stays within 48 frames on every machine, and a later one passes 48 only
  once the listener holds 2.7 s. No window passes 96 frames, twice 48, which bound the decoder's compute buffer: from 96
  to 128 frames a window would save 3.6% of the decoding for 113 MiB more. Where not even 24 frames finish in time, the
  window is 24 when decoding at it still gains on the listener and 48 when decoding is slower than real time, since
  smaller windows would only add margins to decode. The decision is one function without a clock (`next_window()`),
  which `irodori-window-check` checks on simulated machines, among them ones that slow down mid-sentence: on none does
  the listener hold less audio when a window arrives than the least it holds with windows of at most 48 frames. On the
  M5's Metal over speech-bench's 20 sentences with v4.1-Small-MF's F16 file (2026-10-13), windows past 48 frames, of 58
  to 69, came only in sentences longer than 6.2 s; the listener held at least 0.17 s when a window arrived, as with 48
  (0.16 s); the codec decoded 1.6% fewer frames, and a synthesis took 1.4% less time, with the same samples and the same
  first audio. At the M5's speed a sentence of 27 s decodes 8% fewer frames.
- **Irodori-TTS's sizes may follow the clock because its audio does not depend on them**: `irodori-codec-check` checks
  eight patterns of windows of up to 96 frames against one decode, bit for bit, on the CPU, Metal and Vulkan (an RTX
  2080), so a seed repeats its samples whatever the machine's load. Qwen3-TTS's samples depend on the chunking, so its
  sizes stay fixed.

The alternatives were turned down:

- Qwen3-TTS chunks that follow the measured speed. The grouping would change with the machine's load, so the same seed
  would give samples that differ by rounding from run to run, and every extra call costs the codec's 16 ms whatever it
  holds.
- A Qwen3-TTS codec whose samples do not depend on the grouping, with calls of one shape and keys and values kept by
  position. It rewrites the codec's state for a difference far below what anyone hears.
- A first chunk of 2 frames. The first audio would come a frame later.
- Chunks of 1 frame throughout. Each would pay the codec's 16 ms.
- Irodori-TTS's later windows held to 48 frames whatever the machine, which bounds the wait for one in frames. A
  48-frame window spends a quarter of its decoding on margins, and the wait it bounds differs with every machine. A
  prototype that let the windows grow to 96 frames (2026-10-13) made v4.1-Small-MF's codec 15.6% faster and its
  synthesis 12.5% faster on an RTX 2080 with Vulkan, and the codec 13 to 22% faster on the M5's Metal, with the same
  audio and the same first audio.
- Windows past 48 frames whenever their decoding fits within the wait, without asking for audio in hand. On the M5 the
  second window grew to about 70 frames and a synthesis of speech-bench's 20 sentences took 7.0% less time, but the
  second window arrived with 0.05 to 0.1 s of the first window's audio left to play instead of about 0.2 s, while a
  window took up to 1.64 times as long as its estimate on a shared GPU, which would run the listener dry.
- Windows past 48 frames while their expected decoding takes at most a fraction, such as a half or a quarter, of the
  audio in hand beyond the 0.1 s. On the 20 sentences' lengths at the M5's speed they would decode 4.6% fewer frames
  than windows of 48, against 2.3% with the audio of every decoded frame in hand, but a window bets most of what the
  listener holds as soon as it holds a little more than 48 frames need: a machine whose decoding slows to 27 ms a frame
  just after its second window runs dry for 0.3 s where windows of 48 frames keep 0.2 s in hand.
- A higher bound in frames alone, such as 96. A machine half as fast as the M5 would keep a cancel waiting 0.9 s for a
  window, where the bound in time keeps it at 48 frames.
- A wait bounded by what a 48-frame window takes on the M5, 0.26 s. Only a machine faster than the M5 would gain, and
  the second window already may take 0.38 s, so a later one that takes as long adds no wait the stream does not have.
- Windows that grow with the listener's audio in hand and no bound. On a long sentence the M5 would decode the rest
  of it in one window of hundreds of frames, a cancel would wait for seconds, and the compute buffer would grow with
  the sentence.
- A fixed later Irodori-TTS window smaller than 48, such as 24 everywhere. Fast machines would decode 25% more per
  second of audio for nothing and break the stream into twice the chunks.
- A floor of 12 for Irodori-TTS's later windows. It decodes 75% more per second of audio than 48, which a machine that
  decodes 48-frame windows barely faster than real time cannot afford.

## Consequences

`qwen3-tts-codec-check` compares Qwen3-TTS's decode in chunks with the whole decode as an SNR. A machine that makes
frames more slowly than the measured ones can still run a player dry, which then buffers. Irodori-TTS's windows of one
request depend on the machine and its load, and its samples do not. Windows past 48 frames speed up long sentences
alone; a sentence shorter than 6.2 s decodes as it did with 48. On a machine fast enough for them, Irodori-TTS's
decoder holds a compute buffer of up to 394 MiB, where 48 frames held 225 MiB.
