# Haru

This page describes speech.cpp's port of higurashi-ai's [Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B), a model
of 4B parameters that holds a spoken conversation, listening and speaking at the same time, in nine languages.

| Name | Upstream | File |
|---|---|---|
| `haru-4b` | [higurashi-ai/Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B) | `Haru-4B-F16.gguf`, 10.56 GB |

Haru-4B is a pretrained base model, released for research under the Apache License 2.0. It holds a natural
conversation, but it is not a reliable source of facts, does not follow instructions as an assistant does, and breaks
easily: it may answer in another language, say the same thing again or fall silent. It can speak in the voice of a
recording; only use voices you have the right to use.

```sh
speech talk haru-4b --live                                   # through the microphone and the speakers; use headphones
speech talk haru-4b --live --language ja --name ハル --persona "落ち着いた話し方をする大学生。"
speech talk haru-4b --add-voice me=me.wav --voice me -o reply.wav question.wav
speech serve haru-4b --open                                  # the page's Talk tab, and /v1/dialogue
```

## Weight types

`haru-4b` fetches F16, and `NAME:TYPE` another (`speech talk haru-4b:q8_0`). The table compares the first frame of a
conversation of the dumps with the official implementation in float32 (the SNR of the language model's last hidden
state, of the logits that decide whether the model speaks, and of the audio latent), on an Apple M5 with Metal, beside
the official MLX implementation of the same frame, whose app starts in bfloat16. The step is the median time of one frame
on the M5 with its screen locked, and the peak memory that of `speech talk` with keys and values for 60 minutes
([Speed](#speed)).

| Type | File | Hidden state | Speak or not | Latent | Step p50, M5 | Peak memory, M5 |
|---|---|---|---|---|---|---|
| F16 | 10.56 GB | 66.5 dB | 74.8 dB | 57.4 dB | 89 ms | 10.84 GB |
| Q8_0 | 6.62 GB | 40.1 dB | 40.7 dB | 28.0 dB | 60 ms | 6.90 GB |
| Q6_K | 5.61 GB | 26.7 dB | 17.3 dB | 17.7 dB | 54 ms | 5.89 GB |
| Q5_K | 5.05 GB | 9.0 dB | -4.0 dB | 9.0 dB | 49 ms | 5.33 GB |
| Q4_K | 4.52 GB | 7.4 dB | -3.6 dB | 2.7 dB | 46 ms | 4.80 GB |
| official MLX, float32 | | 65.9 dB | 77.7 dB | 54.8 dB | | |
| official MLX, bfloat16 | | 41.4 dB | 53.4 dB | 29.9 dB | 86 ms | |
| official MLX, 8-bit | | 33.2 dB | 22.5 dB | 23.2 dB | 56 ms | |

F16 lies as close to the official float32 as the official MLX code in float32 does, and Q8_0 between the official app's
8-bit and bfloat16 models. Q6_K and below lie further than the official 8-bit model, and Q4_K's text breaks: with
`speak_first` and no card it writes runs of punctuation and stray characters where F16 and Q8_0 speak sentences.

- **On the CPU use Q8_0.** F16 runs at RTFx 0.6 on an Apple M5's CPU and 0.21 on an i9-9900K, too slow to keep up,
  and on an ARM CPU, such as a Mac's, its products are added in half precision, which overflows on a conversation with
  a card and a voice and stops it with `device`. The CPU also rounds Q8_0's inputs to 8 bits, which
  takes it further from the official model than on a GPU (20.3 dB of the hidden state).
- **On a GPU of 8 GB** a conversation of 60 minutes holds 1.7 GB of keys and values beside the weights: with nothing
  else on the GPU, Q8_0 needs `--max-seconds 600` or less and Q4_K holds the full hour, and other programs' memory
  leaves less ([A conversation](#a-conversation)).

## A conversation

- Every 133 ms, 3,200 samples at 24 kHz, the model hears a frame and gives a frame of its own audio with the text it
  began to say in it, one frame ahead of what it hears. The start runs the prompt (the card and the voice) and gives
  the first frame.
- `speech talk --live` and the page's Talk tab play the frames as they come. Use headphones: with speakers the model
  hears itself. A conversation that falls a second behind stops with `device`: the device computes the frames slower
  than they last, and a smaller type or a faster device keeps up. On a Mac whose screen is busy drawing, F16's steps
  take most of a frame where Q8_0's keep well within it ([Speed](#speed)).
- On a WAVE file, `speech talk` writes the model's frames end to end, one frame for the start and one for each whole
  frame of the file, so that the reply lines up with the file in time.
- `speak_first` makes the model open the conversation instead of waiting to be spoken to.
- A conversation lasts `max_seconds` at most, 60 minutes unless given, and holds the memory of its keys and values for
  that length from its start: about 1.7 GB for 60 minutes and 0.3 GB for 10. One that reaches it stops with
  `out_of_range` naming `max_seconds`.
- On a GPU whose memory other programs share, as a desktop with a browser and chat apps open does on Windows, keep the
  model and its keys and values well within what they leave free. On an RTX 2080 (8 GB) whose other programs held
  3.2 GB, Q8_0 with 600 s (6.8 GB) was late on 68 of the 143 steps of the first conversation after its load, by up to
  0.8 s, as the system moved memory out of the GPU, and on none of the next two; Q4_K with 120 s (4.3 GB) was late on
  one. A live conversation stops a second behind, so on such a GPU Q4_K and a shorter `max_seconds` keep it going.

## The character card

The card tells the model who it is and whom it talks with. Its fields are request options, written into the card in
the conversation's `language`, which a card needs: a field set with `language` left `auto` is refused.

| Option | What it sets |
|---|---|
| `name` | the model's name |
| `persona` | who the model is and how it speaks |
| `gender` | `female` or `male` |
| `backchannel` | how often the model gives backchannels: `low`, `mid` or `high` |
| `other_name` | the name of the person it talks with |
| `relationship` | how the two stand to each other |
| `situation` | where and why the two talk |

A conversation with no field set has no card, and the model speaks as it will, often in a language of its own choosing.

## Voices

The model speaks in the voice of a recording of one speaker: at least 3 seconds of sound, its silence at both ends cut,
and its last 3 minutes kept.

```sh
speech talk haru-4b --add-voice me=me.wav --voice me --live           # the recording encoded when it is added
speech voice haru-4b me.wav me.voice.gguf                             # or made into a voice file once
speech talk haru-4b --add-voice me=me.voice.gguf --voice me --live
```

A voice file holds the recording's codec latents and works with every type of the model; another model's voice file is
refused. Each conversation draws its own latents from them, as the official session does, so the same voice and seed
give the same conversation and another seed a slightly different voice. Without a voice the model speaks in one of its
own. The page of `speech serve` makes a voice from a recording dropped on it or recorded in the browser.

## Options

| Option | Default | Notes |
|---|---|---|
| `voice` | none | a voice added from a recording or a voice file |
| `language` | `auto` | one of the nine languages, the card's language; a card needs one |
| `seed` | drawn | 0 to 2^53 - 1 |
| `max_seconds` | 3600 | the longest the conversation lasts, above 0 to 3600 |
| `temperature`, `top_k`, `top_p` | 0.7, 25, 1 | the draw of the text |
| `guidance_scale` | 1.5 | how strongly the audio follows the model's hidden state, 1 to 2 |
| `eop_bias` | 0 | added to the logit with which the model starts to speak: above 0 it starts sooner, below 0 later |
| `speak_first` | false | the model opens the conversation |
| `name`, `persona`, `gender`, `backchannel`, `other_name`, `relationship`, `situation` | none | the card ([above](#the-character-card)) |

The defaults are those of the official session. The official session's DRY penalty on repeated text is always applied
and is not an option.

## Accuracy

Every stage is checked against the official implementation at commit 7fe6b76 on two conversations dumped from it, one in
which the model listens to a question in Japanese and one in which it speaks first:

| Check | CPU, F32 | Metal (M5) | Vulkan (RTX 2080) |
|---|---|---|---|
| Codec, each stage | 90.9 to 136.5 dB | 47.6 to 90.8 dB | 47.3 to 75.0 dB |
| Language model, each layer from its own input | 109.6 dB | 63.5 dB | 63.5 dB |
| A whole conversation replayed | the official text on every frame | the official text on every frame | not run: the F32 file needs more than 8 GB |

On a GPU a conversation's audio departs from the official one's after a few frames from rounding alone, since the model
draws each frame from what it said before.

## Speed

`measure/haru-timing` with the 19.1 s recording of a question and a voice of 16 s, keys and values for 60 minutes unless
noted, steps computed one after another, the medians of three runs, 2026-11-15, with the M5's screen locked so that
nothing else drew on its GPU:

| Device | Type | Step p50 | Step p99 | RTFx |
|---|---|---|---|---|
| Apple M5, Metal | F16 | 89 ms | 92 ms | 1.50 |
| Apple M5, Metal | Q8_0 | 60 ms | 64 ms | 2.20 |
| Apple M5, Metal | Q6_K | 54 ms | 57 ms | 2.49 |
| Apple M5, Metal | Q5_K | 49 ms | 51 ms | 2.71 |
| Apple M5, Metal | Q4_K | 46 ms | 47 ms | 2.91 |
| Apple M5, CPU | Q8_0 | 118 ms | 122 ms | 1.12 |
| NVIDIA GeForce RTX 2080, Vulkan | Q8_0, 600 s | 34 ms | 35 ms | 3.94 |
| NVIDIA GeForce RTX 2080, Vulkan | Q4_K | 26 ms | 28 ms | 5.07 |

A step is the user's frame encoded, the language model's row with the head, and the model's frame decoded; a
conversation keeps up while its steps take less than a frame, 133 ms. A screen that draws shares the GPU: on the same
M5 with its screen on and the desktop drawing, five runs took 92 to 108 ms a step in F16 and 65 to 81 ms in Q8_0, and
the official MLX session 89 to 109 ms in bfloat16 and 58 to 83 ms in 8-bit. With the screen locked, the official
session took 86 ms in bfloat16 and 56 ms in 8-bit over 428 steps.
