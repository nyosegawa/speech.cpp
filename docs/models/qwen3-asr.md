# Qwen3-ASR

This page describes speech.cpp's port of Qwen's [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), which writes
the text of speech in 30 languages, finds the language itself, and can be told the names and terms the audio holds.

| Name | Upstream | File |
|---|---|---|
| `qwen3-asr-0.6b` | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | `Qwen3-ASR-0.6B-Q8_0.gguf`, 0.84 GB |
| `qwen3-asr-1.7b` | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | `Qwen3-ASR-1.7B-Q8_0.gguf`, 2.18 GB |

```sh
speech asr qwen3-asr-1.7b utterance.wav                          # the language left to the model
speech asr qwen3-asr-1.7b --language ja utterance.wav            # Japanese, forced
speech asr qwen3-asr-1.7b --prompt "Claude Code、CI、渋谷" meeting.wav
speech worker qwen3-asr-1.7b
speech serve qwen3-asr-0.6b                                      # POST /v1/audio/transcriptions, with "prompt"
```

## What is implemented

The reference is transformers 5.18's own implementation with its windowed encoder, and qwen-asr 0.0.6's code for what
transformers leaves out ([ADR 0018](../adr/0018-qwen3-asr-runs-in-speech-cpp-checked-against-its-windowed-encoder.md)).

- The frontend: qwen-asr's normalization of the audio, an utterance under 0.5 s padded with zeros, and Whisper's log-mel
  as transformers computes it, on the host in double precision.
- The encoder: convolutions over each chunk of 1 s, and 18 (0.6B) or 24 (1.7B) layers that attend within windows of 8 s,
  then the projector to the decoder.
- The prompt as qwen-asr writes it, with the request's `prompt` in the system turn and, for a forced language, the
  prefill that names it.
- The Qwen3 decoder it shares with Qwen3-TTS's talker, with a key/value cache in F16 that grows with the request, the
  prompt read 512 rows at a time, and flash attention on a GPU whose backend computes it
  ([ADR 0019](../adr/0019-the-qwen3-decoder-attends-with-flash-attention-where-a-gpu-computes-it.md)).
- Greedy decoding until an end token or 4096 new tokens, the limit of the model's `generate()`.
- The output parsed as qwen-asr's `parse_asr_output()` parses it, with its repetition fix and the language the model
  wrote before its text ([ADR 0020](../adr/0020-recognition-results-carry-the-languages-qwen3-asr-writes.md)).
- Audio over 1200 s cut as qwen-asr cuts it (below).

Not implemented: timestamps, which need a second model, Qwen3-ForcedAligner-0.6B (a request takes `timestamps` only as
false); and qwen-asr's streaming.

## Options

| Option | Default | Notes |
|---|---|---|
| `language` | `auto` | one of the 30 languages, which then steers the model, or `auto`, which lets it find the language |
| `prompt` | `""` | what the model is told of the audio before it hears it: the names and terms it may hold. The model was trained to use it as background, not to follow it as instructions |

The languages are `ar` `cs` `da` `de` `el` `en` `es` `fa` `fi` `fil` `fr` `hi` `hu` `id` `it` `ja` `ko` `mk` `ms` `nl`
`pl` `pt` `ro` `ru` `sv` `th` `tr` `vi` `yue` `zh`, and 22 Chinese dialects under `zh` or `auto`. Cantonese and Filipino
have no two-letter code and take three letters (`yue`, `fil`).

The result gives the language the model heard as its tag (`speech_result_language()`, the worker's and
`speech asr --format json`'s `languages`, the server's `verbose_json` `language`), or the forced language. It gives
none for audio without speech, where the model writes `language None`, or where it writes nothing after a forced
language, and none for a name that is not one of the 30, which a warning in the log reports.

A `prompt` is refused (`out_of_range`) when its tokens, with those of the longest part of the audio (at most 1200 s, 13
tokens a second) and the 4096 the model may write, are more than the decoder's 65536 positions.

## Long audio

The model takes at most 1200 s at once. Longer audio is cut as qwen-asr's `transcribe()` cuts it: at 1200 s from the last
cut, moved to the quietest 0.1 s within 5 s on either side and to its quietest sample, with a part shorter than 0.5 s
padded with zeros. Each part is recognized alone, and the texts are joined without a separator. The languages are merged
as qwen-asr's `merge_languages()` merges them, in the order of the audio, one for each run of parts in the same language
and none for a part without one, so a recording heard in Japanese and then in English gives `ja` and `en`. The memory
is that of the longest part.

A part that reaches the 4096 tokens stops there, and the request says `model_limit` with the text of every part. A
recording of over a few minutes may reach them: on the first 1203 s of FLEURS ja_jp joined, the 0.6B model repeats three
sentences until the limit, as the official implementation does. To get the whole text of a long recording, cut it at its
pauses into pieces of a few minutes.

## Accuracy

Every stage is checked against transformers' tensors on 10 inputs from FLEURS' test split (ja_jp, en_us, cmn_hans_cn
and de_de, and two near-silent cuts), each with four requests, on an Apple M5
([checks](../development/checks.md#qwen3-asr)):

| Weights | Text from the audio, 0.6B and 1.7B |
|---|---|
| F32 on the CPU, F32 or F16 on Metal | the official text on all 80 requests |
| Q8_0 on the CPU | the official text on 37 and 35 of 40 |
| Q8_0 on Metal | the official text on 37 and 40 of 40 |

Where a Q8_0 text differs, the official choice between two tokens was within the arithmetic's error of a tie.

## Speed

Apple M5, Q8_0 weights on Metal, after loading, the language left to the model:

| Audio | 0.6B | 1.7B |
|---|---|---|
| ja_jp, 6.36 s | 0.19 s | 0.43 s |
| ja_jp, 10.50 s | 0.28 s | 0.64 s |
| de_de, 11.16 s | 0.34 s | 0.78 s |
| en_us, 23.64 s | 0.63 s | 1.41 s |
| ja_jp, 25.50 s | 0.87 s | 1.98 s |

The decoding runs as fast as llama.cpp's on the same machine. The 1338.42 s input takes 120 s with the 0.6B model, nearly
all of it the first part's 4096 tokens, with a peak memory footprint of 2.03 GB.
