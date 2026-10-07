# Irodori-TTS takes its caption as the option instructions

Decided 2026-10-07.

## Context

Irodori-TTS v4.1 is trained with a caption, a sentence that describes the voice and the way of speaking, which the
official runtime's request takes as `caption` and docs/models/irodori-tts.md calls VoiceDesign without a reference. The
runtime strips it with Python's `str.strip()` and nothing else, tokenizes it with the text's tokenizer after `<s>`, runs
it through the ModernBERT-ja it shares with the text and the caption's own projector and norm, lets the DiT attend to it
after the speaker, gives the duration predictor the mean of its tokens instead of the null caption, and for RF adds a
branch without it to the guidance at `cfg_scale_caption` (3). A caption that strips to nothing is masked whole, which
computes the same as none. The runtime cuts a caption past 512 tokens.

OpenAI's speech API has a member `instructions` that describes the voice and the way of speaking in words, which
`speech serve` refused as a member it did not have. Qwen3-TTS 1.7B's instruct is the same kind of input.

ASIST and speech-bench measure every request without a caption, and their measurements hold only while such a request
does the work it did before.

## Decision

- **The caption is the option `instructions`**, a string whose neutral value `""` every model takes, declared by a
  file that holds the caption's encoder; RF's scale is `cfg_scale_instructions`, its default the file's
  `irodori-tts.sampler.cfg_caption`. OpenAI's SDK sends `instructions=` to `speech serve` unchanged, and another
  family's instruct takes the same option. The file's keys keep the runtime's name, `irodori-tts.caption.*`, since
  they are facts of the model.
- **A request without a caption computes nothing of it.** The caption's encoder does not run, the DiT gets no caption
  keys, and RF's batch has no caption branch: the graph is the one a file without the caption builds, so the first
  audio and the real-time factor of such a request are what they were. A caption that strips to nothing is no caption.
- **A caption past `irodori-tts.caption.max_tokens` is refused** as `out_of_range`, naming `instructions`, where the
  runtime cuts it, as speech.cpp refuses a text past its bound (docs/adr/0014).
- **`cfg_scale_instructions` without a caption is refused** but at its default and 0, as other settings without effect
  are (docs/adr/0025).

The alternatives were turned down:

- The option `caption`, the runtime's name. An OpenAI client would send it in `extra_body` and have its own
  `instructions` refused, and the scale would read `cfg_scale_caption`, against one name for one input across families.
- Computing the caption always and masking it when there is none, as the runtime does. Every request would pay for a
  ModernBERT pass and the DiT's extra keys for nothing.
- Cutting a long caption as the runtime does. The caller would get speech that follows part of what it asked.

## Consequences

A caption costs, by the estimate made before it was built, about 10 ms of ModernBERT and under 5 ms of the DiT's keys
over MeanFlow's four steps on an Apple M5 (4 to 6% of the median first audio of 0.23 s), and with RF a fourth branch
at each guided step, which raises the DiT's work by a quarter while t ≥ 0.5 and the sampling time by an estimated 10 to
25%; `checks/irodori-caption-timing.py` measures it. A layout 1 file refuses instructions naming the caption's encoder
it lacks (docs/adr/0026).
