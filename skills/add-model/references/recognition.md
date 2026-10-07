# Recognition models

## Decoding

- Decode as the official default does (`transcribe()`'s defaults, the checkpoint's decoding config); another decoding is
  a request option (`decoding`), never a silent change.
- Build the prompt exactly as the official code does: the system turn, the language prefix, the padding, the number of
  feature frames. A missing system turn or one extra frame changes the text (llama.cpp's Qwen3-ASR did both).
- Keep the official parse of the output: what it strips, how it splits a language label from the text.

## Languages

- Languages are BCP 47 tags of the model's own languages (`general.languages`). A request may force one or leave it to
  the model; a model that names the language it heard returns it in the result.
- A language a model only checks and does not use is documented as such.

## Long audio

Cut long audio where and how the official code cuts it, and merge the parts as it merges them. A model whose attention
covers the whole input (local attention) recognizes it whole, as the official `transcribe()` does.

## Results

- The text, the languages, why it stopped (complete, a length limit, the model's limit), and times per segment or token
  where the model gives them.
- Audio at any sample rate is resampled to the model's; the resampler is checked against torchaudio.

## Streaming

A model trained for streaming (cache-aware) carries its caches between chunks; check every chunk's stage and the
carried caches bit for bit against the official streaming code. An offline model is not streamed by cutting it into
overlapping windows, which changes its text; the worker's `peek` gives interim text instead.
