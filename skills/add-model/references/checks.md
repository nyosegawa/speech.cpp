# Stage checks and their bounds

Each ported stage has a check in `checks/<family>/` (`<family>-<stage>-check.cpp`) that starts from the dump's own
inputs, so that a difference points to the stage that made it, not to everything before it.

## What a check compares

| Output | How | Bound |
|---|---|---|
| tokens, text, lengths in frames | exactly | equal |
| tensors | SNR in dB against the official, and the largest absolute difference | per stage and backend |
| audio | SNR against the official audio; for a chunked or windowed decoder, bit for bit against one whole decode where the design claims it | per backend |

- CPU F32 is the strict reference: most stages land 80 to 120 dB from the official. A stage far below that on the
  CPU is a defect until shown otherwise.
- Metal and Vulkan round matrix products' inputs to half precision; their bounds are lower, set from measurement.
- An autoregressive decoder compares greedy output token by token with teacher forcing, and reports the first row
  where the official top two candidates are closer than the backend's error, if any.

## Loosening a bound

Lower a bound for one case only with evidence that the gap is rounding and not a defect, written beside the bound:

1. **Each step from the dump's inputs** stays close (for example a DiT step at 42 dB or more while the whole synthesis
   is at 8.7 dB): the gap accumulates, no stage is wrong.
2. **The trend follows the cause:** the gap grows with what amplifies rounding (a guidance scale, a scaling of keys),
   measured at several values.
3. **The rounding reproduces it:** rounding only the suspected input on the CPU gives the same gap.
4. **The meaning is kept:** a recognizer writes the same text for the official audio and for the backend's.

Never lower a bound for every case to let one pass, and never change a bound to make a quantized file pass: record its
numbers in the pull request's description instead.

## Bit-for-bit claims

- A request without new options gives the previous release's float samples: `checks/compare/same_audio.py <old lib>
  <model> <new lib> <model> <prompts.json> --voice … --seeds …`, on Metal and on the CPU, and on Vulkan before the
  release.
- A decoder that may be cut into windows of any size gives the samples of one whole decode, for several forced
  patterns, on every backend. A decoder that carries state between chunks (Qwen3-TTS's codec) does not, and keeps a
  fixed schedule.
- The same seed gives the same samples on the same device.
