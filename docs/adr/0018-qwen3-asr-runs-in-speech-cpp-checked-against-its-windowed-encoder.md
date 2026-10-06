# Qwen3-ASR runs in speech.cpp, checked against its windowed encoder

Decided 2026-10-06.

## Context

docs/adr/0009 left Qwen3-ASR (Qwen/Qwen3-ASR-0.6B and 1.7B: a Whisper-style encoder, a projector and a Qwen3 decoder)
in llama.cpp, which already ran Qwen3's decoder with its cache and quantization. ASIST runs llama-server for Qwen3-ASR
alone, and with it a second runtime, a second GGUF layout, a port, a key, a watcher and a context of 4096 tokens that
fails after about 270 s of audio. llama.cpp b11246, which ASIST pins, also differs from the official implementation:
its prompt has no system turn, its log-mel has one frame more, and its last 1 s chunk gives 13 audio tokens from zero
padding where the official gives only the valid ones. speech.cpp has meanwhile come to run Qwen3's decoder itself:
the Qwen3-TTS talker is the same block, with exactly the shapes of Qwen3-ASR's decoders at both sizes, a cache that
grows with the request, and Q8_0 weights.

The official code has two encoders. The model was trained to attend within windows of 1 to 8 s, and runs with windows
of 104 tokens (8 s). qwen-asr 0.0.6, Qwen's package, computes those windows but hands them only to FlashAttention 2, so
on the CPU, and on CUDA without flash-attn, every frame attends to the whole utterance. Its vLLM backend, transformers
5.18's own implementation and llama.cpp attend within the windows on every path. Measured on the CPU in float32 on
2026-10-06 (`reference/qwen3-asr/`: qwen-asr 0.0.6 with transformers 4.57.6 against transformers 5.18.0, torch 2.10.0)
on FLEURS utterances, both sizes and four requests each (with and without a forced language and a prompt):

| Audio | Encoder output, windowed against whole | Requests whose text differs |
|---|---|---|
| one window: 5.76, 6.36 and 6.66 s | equal, bit for bit | 0 of 24 |
| two windows: 8.64, 10.50 and 11.16 s | 9.1 to 10.5 dB SNR | 0 of 24 |
| three or four windows: 23.64 and 25.50 s | 7.3 to 8.6 dB | 14 of 16 |

Where the texts differ, the windowed one is as close to FLEURS' transcription or closer on the 0.6B model
(ノーザンアリゾナ大学 where the whole attention writes 農山アリゾナ大学), and neither is the closer on the 1.7B.

## Decision

- **A family of its own.** speech.cpp runs Qwen3-ASR 0.6B and 1.7B as the family `qwen3-asr`, one GGUF file per model
  in this repository's layout, each stage checked against tensors dumped from the official model. The Qwen3 decoder
  and the byte-level BPE tokenizer move to `src/common/` for Qwen3-TTS and Qwen3-ASR together.
- **The windowed encoder is the reference.** The dumps come from transformers 5.18's implementation on the CPU in
  float32 with the pinned checkpoints' weights. What transformers does not do comes from qwen-asr's own code: the
  normalization of the audio, the prompt with its system turn, the forced language as the prefill
  `language <Name><asr_text>`, the parse of the output with its repetition fix, and the split of long audio. qwen-asr
  as released runs beside it in an environment of its own, which checks that its processor gives the dumps' features,
  prompts and decoded text, and records its own text.
- **Greedy decoding to at most 4096 new tokens**, stopping at 151643 or 151645, the default of the model's
  `generate()` and of qwen-asr's vLLM backend; a request that reaches the limit reports it.
- **Audio over 1200 s is split as qwen-asr splits it**: at 1200 s, moved to the quietest 100 ms within 5 s on either
  side, each part recognized alone and the texts joined without a separator. Where the split falls depends on the
  audio alone, so the dumps check it on the 0.6B model only; the 1.7B model in float32 needs about 17 GB and over an
  hour for 1338 s on an Apple M5.
- **A `prompt` option** carries qwen-asr's context into the system turn, from the family's first layout on.
- **No timestamps for now.** They need Qwen3-ForcedAligner-0.6B, a second model, and for Japanese a port of nagisa's
  word segmenter.

The alternatives were turned down:

- Keeping Qwen3-ASR in llama.cpp. ASIST's local speech would keep two runtimes and two layouts, and a prompt and audio
  tokens that differ from the official.
- qwen-asr as it runs on the CPU as the reference. Its attention over the whole utterance comes from a fallback that
  drops the windows it computes; every other path of the official code, and the training, uses them.
- 512 new tokens, the default of qwen-asr's transformers backend and of the `-hf` checkpoints. At the 2.6 to 3.0 tokens
  per second of speech measured on FLEURS, it cuts a recording after about three minutes, where 4096 cover the 1200 s
  the model takes.
- Refusing audio over 1200 s. qwen-asr's `transcribe()` splits it.

## Consequences

speech.cpp owns a Qwen3 decoder with prefills of up to 15,000 positions, which the talker's attention, holding every
score at once, cannot take, so the shared decoder prefills in blocks. The text equals transformers' on the CPU, which
for audio over 8 s is not qwen-asr's on the CPU. Audio shorter than one chunk of 1 s is padded to the chunk, as
transformers pads it, where qwen-asr alone runs it unpadded: on a cut of 0.9 s the encoder's output differs at 11 to
12 dB, and the 0.6B model's text with it. A model that loops stops at the limit and says so: on the first 1203 s of
1338 s joined from FLEURS, the 0.6B model repeats three sentences from the first minute on until it reaches 4096
tokens. ASIST can
replace llama-server once the port's speed is measured against llama.cpp's. Another model of the same architecture
needs a pin and a conversion, and timestamps a decision of their own.
