# Live transcription gives each recognition the reading before as a draft, which Qwen3-ASR checks several tokens at a time

## Context

Live transcription recognizes an utterance again from its start each time audio comes (every 0.2 s of it while the
model keeps up), and once more whole when the utterance ends, so most of its work writes again the text the reading
before wrote. Qwen3-ASR writes a token a step, and its decoding is most of a recognition: 104 of 178 ms for 6.4 s of
Japanese with the 0.6B model in Q8_0 on an Apple M5. A step reads every weight once whatever the rows, so on a GPU a
step of five rows costs little more than one: a Q8_0 product of 4,096 × 14,336 took the same time for one to five
columns on the M5, and Qwen3-ASR's head 441 µs for one row and 626 µs for four on an RTX 2080 (2026-10-15).

A step that computes several rows rounds differently from a step of each, on every backend: Metal multiplies one column
with one kernel and two to eight with another, Vulkan on NVIDIA quantizes the activations of products of two to eight
columns to 8 bits and not those of one, and the CPU's products and attention take other paths for several rows. Where
two tokens score almost the same, the two can choose different ones.

## Decision

- **A recognition takes a request option `draft`, the text it is likely to write,** and Qwen3-ASR, which takes it,
  checks it: a step feeds the token written and the draft's next four tokens, picks on the device the token after each,
  keeps the draft's tokens up to the first its picks disagree with and the pick after them, and rewinds the cache past
  the rest. Where the text has gone astray of the draft, it takes up the draft again after the last two tokens it wrote,
  nearest where it was. The option steers nothing, and the text is the one written without it but at near ties.
- **Live transcription gives each reading the text of the reading before, and the recognition of the whole utterance
  its last reading**, where the model takes the option. A one-off recognition has no draft.
- **A draft may change a token where two score almost the same, and the documentation says so.** On the checks' dumps
  and readings simulated from them every 0.2 s (2026-10-15): with the 0.6B and 1.7B models in Q8_0, no reading of four
  recordings changed on the M5, 3 of 135 and 7 of 135 changed on an RTX 2080 with Vulkan, and 1 of 52 changed on the
  M5's CPU, each at the end of a reading cut in the middle of a word or between a comma and a full stop; no recognition
  of a whole recording changed. The decoder's check fails where a draft changes a token whose margin is wider than
  the difference between the two steps' logits of the two tokens, which points to a defect of the drafting.

Measured on the same recordings, the drafts of live transcription took the whole recording's recognition from 225 to
124 ms (Japanese), 241 to 125 ms (English), 216 to 121 ms (Chinese) and 353 to 186 ms (German) with the 0.6B model on
the M5, 494 to 270, 529 to 259 and 862 to 416 ms with the 1.7B, and 93 to 56, 106 to 60, 106 to 58 and 170 to 85 ms
with the 0.6B on the RTX 2080; 86 to 93% of the drafted tokens were kept.

The alternatives were turned down:

- The model keeping the tokens it wrote last as the next recognition's draft, with no option. Requests would hold state
  between them, and a server's sessions on one model would draft from one another's text.
- Computing every step of the decoding with the same number of rows, padded, so that a draft changes nothing. Every
  recognition's text would change from today's at near ties, and a step without a draft would cost about a tenth more.
- Quantizing the activations of Vulkan's one-column Q8_0 products to 8 bits as those of several columns are, so that the
  two round alike on NVIDIA. It changes every model's output on Vulkan by the quantization of its activations.
- No drafts on Vulkan. Live transcription there would stay 1.7 to 2 times slower to its final text.

## Consequences

The option is part of the C API from 3.3; a model that does not take it refuses it, so the tools set it only where the
model takes it.
