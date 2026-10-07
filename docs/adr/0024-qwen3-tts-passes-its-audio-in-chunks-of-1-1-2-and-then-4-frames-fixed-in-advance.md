# Qwen3-TTS passes its audio in chunks of 1, 1, 2 and then 4 frames, fixed in advance

Decided 2026-10-07.

## Context

Qwen3-TTS makes its speech a frame (0.08 s) at a time, and speech.cpp passed the first frame's audio alone and then 4
frames at a time. Measured with 0.7.1 through the worker on speech-bench's 20 Japanese sentences, the second chunk of
the 1.7B model arrived 0.096 s after the first on an RTX 2080 and 0.136 s after it on an Apple M5, where a frame takes
about 24 ms and 34 ms to make, so a player that starts on the first chunk's 0.08 s ran dry for 16 to 59 ms in every
sentence.

The codec carries its state from one call to the next, but its samples still depend, by rounding, on how the frames
are grouped into calls: ggml's kernels and the codec's window of keys and values change with the frames one call
decodes. Decoding one sentence's codes in chunks of 1, 1, 2 and 4 frames instead of 1 and 4 changed 98 % of its
samples, by up to 1.7e-4 on Metal and 7.0e-4 on the CPU with the F16 weights of the codec. A call of the codec took
about 16 ms on the M5's Metal whether it decoded 1 frame or 4 (a rough measure on a shared GPU).

## Decision

- **A fixed schedule:** the first frame alone, so that the first audio comes as early as before, then chunks of 1 and
  2 frames, and 4 from then on. The small chunks feed the player while the speech gets ahead of it, and 4, the size
  before, stays the most a chunk holds, so that a cancel and the worker's silence last no longer than before.
- **The same request with the same seed gives the same samples** on every run, as before, since the chunks do not
  depend on time. The samples differ from 0.7.1's by the rounding of the new grouping alone.
- **One rule in one place**, `chunk_frames()`, whose comment keeps the measurements.

The alternatives were turned down:

- Chunks that follow the measured speed: one is sent as soon as waiting for another frame would let the listener's
  buffer, estimated from the audio sent and the time since the first chunk, fall below a margin. The grouping would
  change with the machine's load, so the same seed would give samples that differ by rounding from run to run, and
  every extra call costs the codec's 16 ms whatever it holds.
- A codec whose samples do not depend on the grouping, with calls of one shape and keys and values kept by position. It
  rewrites the codec's state for a difference far below what anyone hears.
- A first chunk of 2 frames. The first audio would come a frame later.
- Chunks of 1 frame throughout. Each would pay the codec's 16 ms.

## Consequences

A request gives samples that differ from 0.7.1's by rounding. A machine that makes frames more slowly than the
measured ones can still run a player dry, which then buffers as before. README.md gives the decode in chunks against
the whole decode as an SNR, where it said the samples were the same.
