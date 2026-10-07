# Qwen3-TTS

This page describes speech.cpp's port of Qwen's [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
CustomVoice models, which speak ten languages with nine named speakers and pass their audio as they make it.

| Name | Upstream | File |
|---|---|---|
| `qwen3-tts-0.6b` | [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | `Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf`, 1.21 GB |
| `qwen3-tts-1.7b` | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | `Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf`, 2.29 GB |

```sh
speech tts qwen3-tts-0.6b --voice ono_anna --language ja -o out.wav "明日の東京は晴れです。"
speech tts qwen3-tts-1.7b --voice ryan --instructions "Speak slowly and softly." -o out.wav "Good night."
speech worker qwen3-tts-0.6b         # behind the worker protocol
speech serve qwen3-tts-0.6b          # behind OpenAI's speech API
```

## What is implemented

- The talker, a Qwen3 decoder, predicts the first code of each frame of 0.08 s, and the code predictor predicts the
  frame's other 15 codes.
- The 12 Hz codec decoder turns the codes into 24 kHz audio, carrying each stage's state from one call to the next.
- The Qwen2 byte-level BPE tokenizer and the CustomVoice prompt, with the 1.7B model's instruction before it as the
  official `generate_custom_voice()` puts its `instruct`. Text is brought to NFC before it is tokenized.
- Sampling as transformers' `generate()` does it, with the official defaults unless a request sets them.
- Audio passes in chunks of 1, 1, 2 and then 4 frames: the first frame alone, so that audio starts once one frame is
  made, and small chunks next, so that a player that starts on the first chunk stays fed
  ([ADR 0024](../adr/0024-qwen3-tts-passes-its-audio-in-chunks-of-1-1-2-and-then-4-frames-fixed-in-advance.md)).
  The chunks add no artifacts at their boundaries, and the same request with the same seed gives the same samples.
- The talker's cache grows with the speech, and it reads its prompt 512 rows at a time, so the memory of a long text
  grows with its length rather than its square: a prompt of 5137 rows takes 0.18 GB to compute, where reading it whole
  took 1.8 GB.

Not implemented: voice cloning, VoiceDesign, the codec encoder and the speaker encoder.

## Options

| Option | Default | Range and notes |
|---|---|---|
| `voice` | required | one of the nine speakers ([models.md](../models.md#voices)) |
| `language` | `auto` | `de` `en` `es` `fr` `it` `ja` `ko` `pt` `ru` `zh`; it goes into the prompt and steers the model |
| `seed` | drawn | 0 to 2^53 - 1 |
| `max_seconds` | none | above 0, up to the model's limit of 8192 frames (655.36 s) |
| `do_sample` | true | draw each token, or take the most likely one |
| `top_k` | 50 | 0 (every token) to 2147483647: the most likely tokens a draw keeps |
| `top_p` | 1 | 0 to 1: the smallest set of tokens whose probability reaches it |
| `temperature` | 0.9 | above 0: what the logits are divided by |
| `repetition_penalty` | 1.05 | above 0: divides the positive logit of a token already taken, and multiplies a negative one |
| `code_predictor_do_sample`, `code_predictor_top_k`, `code_predictor_top_p`, `code_predictor_temperature` | true, 50, 1, 0.9 | the same for the code predictor, which the official package sets with `subtalker_dosample` and the rest |
| `instructions` | `""` | 1.7B only: how to speak, in words (`怒った口調で`, "Speak slowly and softly.") |

The sampling options are transformers' names for the talker, and the code predictor's under its own name
([ADR 0022](../adr/0022-qwen3-tts-requests-set-sampling-under-transformers-names.md)). The defaults are the official
ones, which the model file holds. The code predictor's repetition penalty is no option: the official package sets none,
and the file's value, 1, applies.

The 0.6B model takes no instruction, and its information lists no such option: the official `generate_custom_voice()`
drops the instruction for that size, which the file tells by its size label
([ADR 0023](../adr/0023-qwen3-tts-follows-an-instruction-where-the-official-code-does-told-by-the-size-label.md)).

What a request is refused for:

- `speed` or `duration_scale` other than 1, and `seconds`: the model has no control of its rate or length
  (`unsupported`; [ADR 0007](../adr/0007-a-request-sets-speed-and-length-only-where-the-model-can.md)).
- `top_k`, `top_p` or `temperature` set while `do_sample` is false, which transformers leaves unused
  (`invalid_argument`); the same for the code predictor's.
- A text of more than 24565 tokens, what the talker's 32768 positions leave beside the longest speech and the 11 rows of
  its prompt (`out_of_range`, option `text`). An instruction's tokens, and 5 more for its turn, count against the same
  24565 (option `instructions`).
- A `temperature` or `repetition_penalty` that rounds to 0 or to infinity as a float, below about 1e-45 or above about
  3.4e38 (`out_of_range`). One so far from 1 that it pushes a logit beyond a float's range, as 1e-40 does, ends the
  request with `out_of_range` while it runs, where the official package fails with an error of PyTorch's.

A request stops at its `max_seconds` or at the model's 8192 frames, the checkpoint's `max_new_tokens`, as the official
implementation stops, and the result says which
([ADR 0016](../adr/0016-input-audio-is-resampled-and-results-carry-stop-reasons-and-times.md)).

## Accuracy

Every stage is checked against the official implementation's tensors ([checks](../development/checks.md#qwen3-tts)):

| Check | Result |
|---|---|
| Codec decoder, CPU, F32 | 114 dB SNR from the official decoder |
| Codec decoder in a synthesis's chunks against the whole utterance | 132 dB SNR in F32 on the CPU, 64 dB with a Q8_0 file on Metal |
| Talker and code predictor, F32, CPU and Metal | the official argmax on every frame; greedy decoding gives the official frames |
| Tokenizer | the official tokens on 27 texts, and the official text of 1191 sequences of ids |

In the smaller types that `speech quantize` makes, Q6_K, Q5_K and Q4_K, the talker's logits part further from the
official's, and in Q4_K the 0.6B model spoke two of 20 sentences as other words, where F32 did so with one
([checks](../development/checks.md#lower-bit-widths)).

## Speed

Q8_0, Japanese sentences, after the shaders are compiled:

| Model | Device | First audio | Real-time factor | VRAM |
|---|---|---|---|---|
| 0.6B | Apple M5, Metal | 0.04 s | 0.31 | |
| 1.7B | Apple M5, Metal | 0.07 s | 0.43 | |
| 0.6B | RTX 2080, Vulkan | 0.07 s | 0.31 | 1.6 GB |
| 1.7B | RTX 2080, Vulkan | 0.08 s | 0.36 | 2.7 GB |
