# Checks

This page says how every port is checked against its official implementation, and gives the measurements: each stage
check's results, the evidence for each bound, and the timings behind the numbers on the model pages.

## How a port is checked

- `reference/<model>/` pins the official implementation in a uv environment, and `pins.py` pins the checkpoints by
  revision. Its `dump.py` runs the official implementation, on the CPU in float32, and saves the tensors of every stage
  to `reference/<model>/out/`.
- A check (`checks/*-check.cpp`, built as `build/*-check`) runs one stage on the dump's inputs, compares its output with
  the dump, prints the error (SNR, largest difference, argmax agreement), and fails when the error is beyond what the
  stage's arithmetic explains. The stages are checked in the order data flows, each with its own inputs from the dump,
  so that the first stage that departs is the one reported.
- `speech-api-check` runs the C API through the shared library with a synthesis model, and `speech-api-check
  transcribe` with a recognition model in F32 or F16 and the dumps of `reference/fastconformer/` or
  `reference/qwen3-asr/`, whose texts it compares byte for byte.
- The smoke scripts in `tools/` drive each entry point as its caller does: `worker_smoke.py` and
  `worker_recognition_smoke.py` the worker, `server_smoke.py` and `server_page_smoke.py` the server and its page,
  `speech_cli_smoke.py` the command line, and `models_smoke.py` the naming, fetching and removing of models. AGENTS.md
  says which to run for which change.
- The checks need weights and dumps that are not in the repository, so CI builds them but does not run them.

The measurements below are on an Apple M5 unless they say otherwise; the Vulkan ones are on an RTX 2080.

## Qwen3-TTS

`reference/qwen3-tts/dump.py` runs the official implementation with greedy decoding and saves the tensors
of every stage, with an instruction for the 1.7B model where `--instruct` gives one; the check tools compare against
them:

```sh
cd reference/qwen3-tts
uv run python dump.py <Qwen3-TTS 1.7B checkpoint dir> out/1.7b-ja-weather ono_anna japanese "明日の東京は晴れで、最高気温は二十四度の予報です。"
uv run python dump.py <Qwen3-TTS 1.7B checkpoint dir> out/1.7b-ja-weather-instruct ono_anna japanese \
    "明日の東京は晴れで、最高気温は二十四度の予報です。" --instruct "怒った口調で話してください。"
uv run python dump.py <Qwen3-TTS 1.7B checkpoint dir> out/1.7b-ja-weather-instruct-quoted ono_anna japanese \
    "明日の東京は晴れで、最高気温は二十四度の予報です。" --instruct $'Speak "softly" \\ slowly,\nas in a whisper.'
cd ../..
build/talker-check <model.gguf, F32> reference/qwen3-tts/out/1.7b-ja-weather-instruct cpu
build/codec-check <model.gguf> reference/qwen3-tts/out/1.7b-ja-weather cpu
```

| Check | Result |
|---|---|
| Codec decoder, whole utterance, CPU, F32 (`codec-check`) | 114 dB SNR against the official decoder |
| Codec decoder in a synthesis's chunks of 1, 1, 2 and 4 frames, and one frame at a time, against whole, F32 | 132.5 and 132.7 dB SNR on the CPU; 62.9 and 62.8 dB on Metal, where the whole decode is 62.8 dB from the official one |
| The same with the F16 codec weights of a Q8_0 file | 59.8 and 60.1 dB on the CPU, where the whole decode is 55.8 dB from the official one; 64.3 dB on Metal |
| Codec decoder on Metal | error at -63 dB of the voice |
| Talker and code predictor, F32, teacher forcing (`talker-check`), CPU and Metal | argmax matches on every frame; greedy decode gives the same 54 frames |
| The same with the instruction 怒った口調で話してください。 (`talker-check`, out/1.7b-ja-weather-instruct), CPU and Metal | the official processor's 13 tokens; the prompt's 44 rows within 1.2e-6 (CPU) and 7.8e-5 (Metal); argmax matches but at one of 855 rows of the code predictor, where the reference's two best codes are 6.7e-4 apart, and greedy decode follows the reference's 57 frames to that row |
| The same with an instruction of quotes, a backslash and a line break (out/1.7b-ja-weather-instruct-quoted), CPU and Metal | the official processor's 18 tokens; the prompt's 49 rows within 1.2e-6 (CPU) and 7.7e-5 (Metal); greedy decode gives the reference's 73 frames on the CPU, and on Metal follows them to a row of the code predictor where the reference's two best codes are 1.2e-3 apart |
| Sampling against transformers' logits processors as the official `generate()` builds them (`sampler-check`) | on 13 cases of the 1.7B dump's logits, 55 rows of the talker or 300 of the code predictor each, with the official settings and others: the same tokens kept in every row and their probabilities within 6e-15; the same greedy pick in every row; the settings of both files equal to `generation_config.json`'s; a repetition penalty of 1e-40, which takes the logits of the tokens taken before beyond a float, refused as `out_of_range` by a greedy pick and by a draw |
| Tokenizer (`tokenizer-check`) | encodes 27 texts, 8 of which NFC changes, and decodes 1191 sequences of ids as the model's `tokenizer.json` does |
| Talker's prompt in blocks of 512 against one block, F32 (`qwen3-decoder-check`) | the same keys, values and logits on the CPU (5137 rows) and on Metal with flash attention (3665 rows); with the two products forced on Metal, 2.3e-3 at most for a cached row and 1.5e-3 for the logits, with the same argmax (3665 rows) |

`reference/qwen3-tts/sampling_cases.py` runs transformers' own logits processors, as `generate()` builds them for the
talker and for the code predictor, on the logits of a dump; `sampler-check` reads the model file for the official
settings and the codes a frame may not take. A draw is not compared, since `torch.multinomial` draws with PyTorch's own
generator; the sampler draws from the same tokens and probabilities with the request's seed:

```sh
cd reference/qwen3-tts
uv run python sampling_cases.py <Qwen3-TTS 1.7B checkpoint dir> out/1.7b-ja-weather out/sampling
cd ../..
build/sampler-check <model.gguf> reference/qwen3-tts/out/sampling
```

The tokenizer follows the pre-tokenizer of the `tokenizer.json` that ships with the model. The official
package loads it through transformers 4.57.3 with `fix_mistral_regex=True`, which swaps in Mistral's
pattern; the two differ on Latin words in mixed case, contractions and `/`. Before it splits a text, the tokenizer
brings it to NFC, as the normalizer of `tokenizer.json` does ([Unicode normalization](#unicode-normalization), below).
`reference/qwen3-tts/tokenizer_cases.py` writes the cases from the model's own tokenizer:

```sh
cd reference/qwen3-tts
uv run python tokenizer_cases.py <Qwen3-TTS checkpoint dir> out/tokenizer-cases.tsv out/decode-cases.tsv
cd ../..
build/tokenizer-check <model.gguf> reference/qwen3-tts/out/tokenizer-cases.tsv reference/qwen3-tts/out/decode-cases.tsv
```

The codec carries each stage's causal state from one chunk to the next, so the samples of a synthesis's chunks differ
from those of decoding the whole utterance at once by rounding alone.

## Irodori-TTS

`reference/irodori-tts/dump.py` runs the official implementation on the CPU in float32 with fixed noise and
saves every stage; the check tools compare each stage, given the dump's own inputs, with it. Apple M5:

| Check | CPU, F32 | Metal, F32 | Vulkan, F16 model and F32 codec |
|---|---|---|---|
| Normalization and tokenizer, 164 texts with the 56 direction emoji (`irodori-text-check`) | all equal | all equal | all equal on the 51 texts without them |
| Text condition (`irodori-text-check`) | 118 to 123 dB SNR | 65 to 123 dB | 65 to 74 dB |
| Reference latent (`irodori-codec-check`) | 99 dB | 40 dB | 33 dB |
| Speaker condition (`irodori-condition-check`) | 111 dB | 51 dB | 51 dB |
| Length, predicted, scaled, fixed and at a speed (`irodori-condition-check`) | the official frames on every dump | the same | the same on the dumps without options |
| DiT steps, MF and RF (`irodori-dit-check`) | 95 dB or more | 46 dB or more | 63 dB or more (MF) |
| Sampled latent, MF / RF 40 steps | 86 to 122 dB / 109 to 111 dB | 36 to 67 dB / 59 to 67 dB | 66 dB (MF, 27 frames) |
| Decoded audio (`irodori-codec-check`) | 119 dB | 68 dB | 68 dB |
| Decoding in windows against at once, seven patterns of window sizes for the decoder | equal | equal | 89 dB (encoder), equal (decoder, 0.7.1's windows) |
| Whole synthesis from the dump's noise (`irodori-synthesis-check`) | 75 to 110 dB, the same length | 22 to 61 dB, the same length | 58 dB (MF, 27 frames), the same length |

The Metal and Vulkan columns were measured on an Apple M5 and an RTX 2080 with driver 591.86.

Metal's matrix kernel rounds both its inputs to half precision (`kernel_mul_mm_f32_f32` keeps its tiles as
`half`), which is the gap between the CPU and Metal; MeanFlow's four large steps carry it into the latent,
so on Metal the audio is the same speech rather than the same waveform. Vulkan accumulates in float32 as
the port asks. The codec on a GPU is checked against the CPU as well: on Metal its error lies 68 dB below the
voice and on Vulkan 68 dB, and the quietest tenth of the 20 ms frames stays as quiet as on the CPU (-78 and
-76 dBFS against -76). audio.cpp v0.8.2's Irodori-TTS adds a distorted copy of the voice 14 dB below it on
Metal and raises the quiet parts to -60 dBFS.

A reference at another rate, made from the dumps' 48 kHz reference with torchaudio's `kaiser_best`, gives on the CPU
the latent the official codec makes of the same audio resampled to 48 kHz by torchaudio with `kaiser_best`, to 99 dB.
That latent lies 30.6 dB from the 48 kHz reference's at 44.1 kHz and 10.7 dB at 24 kHz, which has nothing above
11.4 kHz; the official runtime, which resamples with torchaudio's defaults, is 29.5 and 12.5 dB from its own.

The audio may end before the length where the latent goes flat, as in the runtime. The frames are the
official runtime's for every combination the dumps cover (`irodori-condition-check`, `irodori-synthesis-check`).

Of the types the converter writes, Qwen3-ASR 1.7B transcribed the 20 sentences of the speed table
([models/irodori-tts.md](../models/irodori-tts.md#speed)) with 2.99% CER in F32 and in F16, and 3.81% in Q8_0, which
garbled one phrase.

For the 10.7 s reference bright-young-woman-10s.wav, the voice file is 35 KB against the WAVE file's 1 MB,
and loads in 0.016 s on an Apple M5 (Metal) and 0.025 s on an RTX 2080 (Vulkan), against 0.72 s and 0.43 s
to encode the WAVE file (5.05 s on the M5's CPU). Made on the CPU, `speech voice`'s default device, its latent is the
official encoder's to 99 dB SNR; on Metal it is 40 dB and on Vulkan 33 dB (Irodori-TTS, Accuracy, above). ASIST
carries voice files ([ADR 0002](../adr/0002-asist-carries-irodori-tts-voices-as-voice-files.md)).

### Streaming

Through 0.7.1's worker on speech-bench's 20 sentences, an Apple M5 sent the 48-frame second window 0.288 s after the
first, 0.19 s before the first window's audio ran out, and an RTX 2080 0.32 s after; a machine half as fast as the M5
would have run dry there, and now sends 24 frames second. 24 frames decode 44 with their margins, 29% more decoding per
second of audio than 48 frames (68), where 12 frames would decode 32, 88% more.

The window sizes follow the clock, and the audio does not: the decoder is convolutions without a cache, and the 10
frames on either side cover its receptive field of 7.7, so decoding in windows of any sizes gives the samples of
decoding the whole latent at once, bit for bit (`irodori-codec-check` decodes seven patterns of sizes against one
decode on the CPU and Metal), and a seed repeats its samples. Qwen3-TTS's codec, which keeps a cache, gives other
samples in other chunkings and keeps its fixed schedule.

### Speed

The speed table is on [models/irodori-tts.md](../models/irodori-tts.md#speed), its memory with the F32 codec.
audio.cpp v0.8.2 took 1.22 s (M5) and 0.80 s (RTX 2080) to the median first audio with v4.1-Small at 16 steps,
answering with the whole sentence.

That table's requests have no instructions, and a layout 2 file runs them as a layout 1 file does. Instructions add the
caption's pass through ModernBERT-ja beside the text's and its keys to every block of the DiT, and with RF a branch to
the batch of each guided step. `checks/irodori-caption-timing.py` times the same sentences without instructions and
with a caption through the worker:

```sh
python3 checks/irodori-caption-timing.py build/speech Irodori-TTS-866M-MF-v4.1-F16.gguf prompts/speak-ja-JP.json \
    bright=bright-young-woman-10s.voice.gguf [--steps 16] [--caption TEXT]...
```

## FastConformer

`reference/fastconformer/dump.py` runs the official model on the CPU in float32 and saves every stage; the
checks compare each stage, given the dump's own inputs, with it:

```sh
cd reference/fastconformer
uv run python dump.py parakeet-tdt_ctc-0.6b-ja out <16 kHz mono WAVE files>
uv run python dump.py parakeet-tdt-0.6b-v3 out <16 kHz mono WAVE files>
uv run python dump.py reazonspeech-nemo-v2 out <16 kHz mono WAVE files>
uv run python dump.py --times reazonspeech-nemo-v2 out   # NeMo's times alone, added to the dumps already there
uv run python dump.py --greedy reazonspeech-nemo-v2 out  # NeMo's greedy decoding, added to the dumps already there
cd ../..
build/fastconformer-frontend-check <model.gguf> reference/fastconformer/out
build/fastconformer-encoder-check <model.gguf> reference/fastconformer/out [gpu|cpu|device name]
build/fastconformer-transducer-check <model.gguf> reference/fastconformer/out [gpu|cpu|device name]
build/fastconformer-times-check <model.gguf> reference/fastconformer/out [gpu|cpu|device name]
```

`dump.py` writes the dumps of each model to `out/<model>/<file name>/`, and each check reads those of the model
it is given, by the GGUF file's `general.name`. `--greedy` adds to each of reazonspeech-nemo-v2's a folder `greedy/`
with the tokens, the text and the times of NeMo's greedy decoding of the dump's encoder output, as
`change_decoding_strategy()` sets it up for the strategy `greedy_batch` with the checkpoint's other settings, and
checks that the strategy `greedy`, which goes frame by frame, gives the same tokens on the same frames;
`fastconformer-transducer-check`, `fastconformer-times-check` and `speech-api-check transcribe` check the option
`decoding`'s `greedy` against it.

On three utterances of FLEURS ja_jp's test split (12677001980660723842, 6.36 s; 13903496305700695803, 10.50 s;
2630315561484880103, 25.50 s), on an Apple M5:

| Check | CPU, F32 | CPU, F16 | Metal, F32 or F16 |
|---|---|---|---|
| Features (`fastconformer-frontend-check`) | 117 to 127 dB SNR | the same | the same (on the host) |
| Subsampling (`fastconformer-encoder-check`) | 123 dB | 60 to 61 dB | 68 to 70 dB |
| Encoder output, after 24 layers | 114 to 118 dB | 55 to 56 dB | 63 to 64 dB |
| Prediction network on the dump's labels (`fastconformer-transducer-check`) | 130 to 133 dB | 57 to 60 dB | 132 to 134 dB with F32, 66 to 70 dB with F16 |
| Joint log-probabilities on the dump's frames and prediction outputs | 140 dB | 76 dB | 85 to 87 dB |
| Greedy tokens and text from the dump's encoder output | equal | equal | equal |
| Text from the audio, every stage ours | equal on all three | equal on all three | equal on all three |

For parakeet-tdt-0.6b-v3, on twelve utterances of FLEURS' test split, three each of en_us, de_de, fr_fr and
es_419 (5.64 to 23.40 s, one of each language over 20 s), on an Apple M5:

| Check | CPU, F32 | CPU, F16 | Metal, F32 or F16 |
|---|---|---|---|
| Features | 94 to 125 dB SNR | the same | the same (on the host) |
| Subsampling | 122 dB | 59 to 61 dB | 69 to 70 dB |
| Encoder output, after 24 layers | 106 to 114 dB | 31 to 55 dB | 52 to 61 dB |
| Prediction network on the dump's labels | 130 to 133 dB | 56 to 60 dB | 130 to 133 dB with F32, 66 to 69 dB with F16 |
| Joint log-probabilities on the dump's frames and prediction outputs | 141 to 142 dB | 77 dB | 88 to 89 dB |
| Greedy tokens and text from the dump's encoder output | equal | equal | equal |
| Text from the audio, every stage ours | equal on all twelve | equal on all twelve | equal on all twelve |

The lowest features, 94 dB on fr_fr 10043298898524273336, are the dump's own float32: the same steps in float64 in
PyTorch differ from it as much. parakeet-v3's conformer layers carry values of 250 to 500, where parakeet-ja's stay
near 100, so half precision costs more: most on the CPU with F16 weights, whose dot product on ARM also sums in half
precision, down to 31 dB on de_de 10229344228128634115, which still gives NeMo's text.

Metal gives the same numbers with F32 and F16 weights for the encoder and the joint, since its matrix kernel
rounds both its inputs to half precision either way; the prediction network multiplies a single vector, which
Metal does in float32.

For reazonspeech-nemo-v2, on eight utterances of FLEURS ja_jp's test split (the three above, 6183819757443715774,
8.94 s; 14931648021649736041, 10.86 s; 16124271561776664380, 14.52 s; 17416616907086415885, 17.88 s;
9518252661993015549, 28.20 s) and two long inputs, the first 4 and the first 21 utterances of the split in the order
of its `test.tsv` joined end to end (64.80 s and 311.22 s), on an Apple M5:

| Check | CPU, F32 | CPU, F16 | Metal, F32 | Metal, F16 |
|---|---|---|---|---|
| Features | 113 to 128 dB SNR | the same | the same (on the host) | the same |
| Subsampling | 122 to 123 dB | 61 to 62 dB | 70 dB | 70 dB |
| Encoder output, after 24 layers of local attention | 104 to 119 dB, 95.9 dB on the 311 s input | 46 to 56 dB, 29.7 dB on the 311 s input | 55 to 71 dB, 37.7 dB on the 65 s input | the same |
| Prediction network on the dump's labels, along the beam's hypotheses | 125 to 128 dB | 61 to 64 dB | 125 to 128 dB | 69 to 73 dB |
| Joint log-probabilities on the dump's frames and prediction outputs | 124 to 126 dB | 71 to 72 dB | 83 to 88 dB | 83 to 88 dB |
| Beam search's tokens and text from the dump's encoder output | equal | equal | equal | equal |
| Greedy decoding's tokens and text from the dump's encoder output | equal | equal | equal | equal |
| Text from the audio, every stage ours | equal on all ten | equal on all ten | equal on all ten | equal on all ten |
| Text from the audio with greedy decoding | equal on all ten | equal on all ten | equal on all ten | equal on all ten |

The dumps record every evaluation of the beam search, 464 to 2,234 joint evaluations an utterance and 18,272 for the
311 s input, and the check replays the search on the dump's encoder output. In half precision the error of the
encoder grows in a few near-silent frames between the joined utterances, from 70 dB at the first layer to 42 dB at
the thirteenth in the 65 s input, and the text is NeMo's all the same; on the CPU in float32 a local attention whose
band missed one frame on one side would give 27 to 50 dB, far below what the arithmetic explains.
`fastconformer-encoder-check` therefore asks for 90 dB on the CPU with F32 weights and 25 dB where half precision
enters.

The 25 dumps' audio at 44.1 and 48 kHz, made with torchaudio's `kaiser_best`, gives with F32 weights on the CPU and on
Metal the text NeMo gives for the same file brought to 16 kHz by torchaudio with `kaiser_best`, on all 50. That text is
the dump's own on 21 of 25 at either rate; NeMo's `transcribe()` of the files, which resamples them with soxr, gives
the dump's text on 22 and speech.cpp's on 22. The others differ in a hyphen, a comma, a full stop and a few words of
the 311 s input: making the files and resampling them back takes away the band above 7.6 kHz, which the 16 kHz audio
has, and soxr takes away a slightly different one.

`dump.py` also saves the times NeMo's `transcribe(timestamps=True)` gives: the frame each token was emitted on and,
for TDT, the duration predicted with it, and the spans of the tokens and of the segments in frames and in seconds.
`fastconformer-times-check` decodes the dump's encoder output and compares the frames, the durations, the tokens'
spans in frames and in seconds and the segments NeMo's separators give, at the ends of words as NeMo ends them, with
them: on every dump of the three models, on the CPU with F32 weights and on Metal with F16, all are NeMo's exactly. It
does the same for reazonspeech-nemo-v2's greedy decoding against the times `dump.py --greedy` saves, for which NeMo
records the frame each token was emitted on: on the CPU with F32 weights and on Metal with F32 and F16, the frames, the
spans and the segments of all ten are NeMo's exactly from the dump's encoder output, and the frames are NeMo's from
the audio as well.
For the Japanese models, whose text has no spaces and so no end of a word before its last token, it also cuts
segments as the recognizer does with their files' `fastconformer.segment.breaks`, after `。`, `？`, `！`, `?` and `!`
wherever they stand, and checks that each segment ends there, at a
separator that ends a word, or with the last token. NeMo's beam search records with each token
the step of its search, the frame plus the tokens before it, so that its times for ReazonSpeech run past the end of
the audio, to 386.16 s for the 311.22 s input; speech.cpp keeps the frame, and the check compares the step it gives.
From the audio, every stage ours, the frames are NeMo's but for one token of ReazonSpeech's 9518252661993015549 on
Metal with F16, one frame early: the two alignments of its tokens differ by 0.007 in log-probability on the CPU in
float32, and half precision reverses them. None of the checkpoints sets NeMo's separators, which are `.`, `!` and `?`
by default, and the vocabularies of the Japanese models hold `。` and the ASCII `?` and `!`, not `！` or `？`.

### Speed

`speech asr` on an Apple M5 with F16 weights on Metal, after loading: 0.07 s for the 6.36 s utterance, 0.11 s for
the 10.50 s one and 0.28 s for the 25.50 s one. Of the last, the encoder takes 0.20 to 0.26 s, the TDT decoding
0.07 to 0.08 s (107 steps of the prediction network, about 0.6 ms each on the GPU) and the frontend 13 to 15 ms. On
the CPU with F32 weights and ggml's default four threads it takes 6.3 s, 0.2 to 0.3 s of it the decoding.

parakeet-tdt-0.6b-v3 on the same M5 with F16 weights on Metal, after loading: 0.08 s for 5.64 s of audio, 0.12 to
0.17 s for 10.20 to 12.84 s and 0.24 to 0.32 s for 20.76 to 23.40 s, a real-time factor of 0.012 over the twelve
utterances. Of the 23.40 s one, the encoder takes 0.20 s, the TDT decoding 0.08 s (128 steps of the prediction
network, and a joint over 8,198 outputs where parakeet-ja's has 3,078) and the frontend 16 ms. On the CPU with F32
weights it takes 9.4 s.

reazonspeech-nemo-v2 on the same M5 with F16 weights on Metal, after loading: 0.17 s for the 6.36 s utterance, 0.27 s
for the 10.50 s one, 0.55 s for the 25.50 s one and 0.63 s for the 28.20 s one, 1.4 s for the 64.80 s input and
4.9 s for the 311.22 s one, a real-time factor of 0.016 to 0.027. The beam search takes about half of it: each of its
steps, one graph of the new predictions and the joint of up to 4 hypotheses on the GPU, takes about 0.7 ms, and the
311 s input takes some 4,860 steps (timed apart in `fastconformer-transducer-check`: the encoder 3.7 s, the beam
search 3.3 s, the frontend 0.14 s). The process's peak memory footprint on the CPU with F16 weights, which
holds every buffer, is 1.35 GB for 6.36 s, 1.55 GB for 64.80 s and 2.35 GB for 311.22 s, growing with the length;
parakeet-tdt_ctc-0.6b-ja's is 1.30, 1.49 and 4.14 GB, growing with its square, and the 311 s input takes it 62 s on
the CPU where ReazonSpeech takes 26 s.

With `decoding` set to `greedy`, reazonspeech-nemo-v2 computes one graph for each token and a few for the runs of
blanks, against one for each step of the beam search: 19 against 160 for the 6.36 s utterance, 107 against 638 for
the 25.50 s one, 233 against 1,620 for the 64.80 s input and 1,008 against 4,864 for the 311.22 s one, 1,658 against
9,427 for the ten inputs of the checks. A graph runs the prediction for the last token and the joint at 8 frames, or
up to 64 after blanks, where NeMo's own greedy decoding evaluates the joint at one frame at a time, 4,822 times for the
311.22 s input. On the same M5 with F16 weights on Metal on 2026-10-07, while other work loaded the machine so that the
times are rough, `fastconformer-transducer-check` timed the decoding of the 311.22 s input at 1.2 s against 3.1 s and
of the 25.50 s utterance at 0.09 s against 0.55 s.

## Qwen3-ASR

`reference/qwen3-asr/dump.py` runs transformers 5.18's Qwen3-ASR on the CPU in float32 with qwen-asr's prompt and
parse, and saves every stage; the checks compare each stage, given the dump's own inputs, with it:

```sh
cd reference/qwen3-asr
uv run python dump.py Qwen3-ASR-0.6B out
uv run python dump.py Qwen3-ASR-1.7B out
uv run python parse_cases.py out    # outputs with qwen-asr's language and text of each, for qwen3-asr-decoder-check
uv run python split_cases.py out    # synthetic audio with qwen-asr's split of each, for qwen3-asr-split-check
# texts and ids with the checkpoint's own tokenizer, for tokenizer-check; the folder is the one pins.py downloads to
uv run python ../qwen3-tts/tokenizer_cases.py <checkpoint dir> out/tokenizer-cases.tsv out/decode-cases.tsv
cd ../..
build/tokenizer-check <model.gguf> reference/qwen3-asr/out/tokenizer-cases.tsv reference/qwen3-asr/out/decode-cases.tsv
build/qwen3-asr-frontend-check <model.gguf> reference/qwen3-asr/out
build/qwen3-asr-encoder-check <model.gguf> reference/qwen3-asr/out [gpu|cpu|device name]
build/qwen3-asr-decoder-check <model.gguf> reference/qwen3-asr/out [gpu|cpu|device name] [flash|products]
build/qwen3-asr-split-check <model.gguf> reference/qwen3-asr/out [gpu|cpu|device name]
```

The inputs are utterances of FLEURS' test split: ja_jp 12677001980660723842 (6.36 s), 13903496305700695803
(10.50 s) and 2630315561484880103 (25.50 s), en_us 10197164397713068203 (5.76 s) and 2880067776280655708 (23.64 s),
cmn_hans_cn 12933878060487367144 (6.66 s) and 14716260585206763911 (8.64 s), de_de 10009182821551087671 (11.16 s),
and two cuts of the first, 0.9 s from 0.6 s on and its first 0.6 s, near silence; each with four requests: the
language left to the model, forced, and both with a prompt that names the utterance's terms. A 1338.42 s input, the
first 100 utterances of ja_jp joined, is dumped to qwen-asr's split and each part's ids and text with the 0.6B model.
On an Apple M5, for the 0.6B and the 1.7B model:

| Check | CPU, F32 | Metal, F32 or F16 | CPU, Q8_0 | Metal, Q8_0 |
|---|---|---|---|---|
| Tokenizer, 29 texts, 10 of which NFC changes and 2 with added tokens, and 1263 sequences of ids (`tokenizer-check`) | equal | equal | equal | equal |
| Features (`qwen3-asr-frontend-check`) | 123.5 to 143.6 dB SNR | the same (on the host) | the same | the same |
| Projector output from the dump's features (`qwen3-asr-encoder-check`) | 85.6 to 111.0 dB, 94.4 to 112.5 dB | 44.4 to 70.6 dB, 50.2 to 68.6 dB with F32; 49.8 to 66.7 dB, 50.7 to 64.8 dB with F16 | 20.2 to 33.2 dB, 24.3 to 33.0 dB | 20.4 to 38.2 dB, 17.1 to 38.3 dB |
| Prompt ids (`qwen3-asr-decoder-check`) | the dump's, all 80 | the same | the same | the same |
| Logits of the prompt's last four rows from the dump's input | 97.4 to 115.1 dB, 96.1 to 114.8 dB | 51.2 to 68.5 dB, 50.1 to 73.2 dB | 19.3 to 33.9 dB, 18.2 to 32.4 dB | 23.1 to 39.5 dB, 21.8 to 37.0 dB |
| Argmax teacher-forced on the dump's ids (1176 and 1184 steps) | every step | every step | all but 8 and 1 | all but 5 and 1 |
| Greedy ids from the dump's projector output, every later stage ours | the dump's on all 80 requests | the same | the dump's on 37 and 38 of 40 | on 37 and 39 of 40 |
| Text from the audio, every stage ours | the dump's on all 80 | the same | on 37 and 35 of 40 | on 37 and 40 of 40 |
| Decoding of the dump's ids, parse of its raw text, 262 cases of the parse, each with its language | equal | equal | equal | equal |

The prompt's logits come from an F32 cache, which leaves the arithmetic of the weights alone; the recognizer's own
F16 cache puts them 42 to 56 dB from transformers' on the CPU in F32, where every text is the dump's as well. With F16
weights on the CPU, whose dot product sums in half precision, the 0.6B model's logits are 37.3 to 53.4 dB and every
text is the dump's. Where a greedy choice differs from the dump's, the check takes it for the arithmetic's only when
the dump's margin between the two tokens is within our error on their two logits, and so it was each time, at margins
of 0.03 to 1.39: the near-silent input with its language forced, where the model writes another filler; the 25.50 s
input, in a name's middle dot (グレン・クッシング for グレンクッシング), a comma, 熱気道 for 熱挙動 and 安定 for 判定; and
the spaces around a Latin name in the 8.64 s one.

The language the result gives, from the dump's raw text and from the audio with every stage ours, is the one qwen-asr
parsed for every dumped request, the forced requests and the 0.6B model's `language None` for the near-silent input
included (the 1.7B model hears Chinese in it, 嗯。, without a prompt): on all 80 on Metal in F32 and in Q8_0, where it
is the dump's also for the texts that differ, and on the 0.6B model's 40 on the CPU in F32. The 262 cases of
`parse_cases.py` hold what the dumps do not: names in other cases and in the code points outside ASCII that Python's
case mappings turn into ASCII, names that are none of the model's (8, which give none where qwen-asr passes the name
on), `None` in other cases and places, a language line after another line, and the line breaks of `str.splitlines()`.

The Metal column is the decoder's flash attention. With the two products forced (`products`), as a GPU without ggml's
flash attention runs them, the prompt's logits lie 47.9 to 71.3 dB and 48.8 to 66.5 dB from transformers' in F32 and
23.2 to 39.2 dB and 21.9 to 37.0 dB in Q8_0, the teacher-forced and greedy rows are as above, and every text is as
above but one: the 1.7B model in Q8_0 writes は、日本語の語源である。 for the near-silent input with its language forced,
at the step where its greedy decoding from the projector output already takes the other token.

The split is qwen-asr's on five synthetic inputs of 1200 to 3700 s (`split_cases.py`: noise with quiet stretches,
speech-like bursts, silence, a last part of 0.2 s that is padded, and audio of exactly 1200 s, which is not split)
and on the 1338.42 s input, cut at 1202.97 s. There, with the 0.6B model in Q8_0 on Metal, the first part's prompt
and its 4096 ids are the dump's, the model repeating three sentences until the limit, and the second part's prompt is
the dump's and its first 137 ids of 376. With the 0.6B model in F32 on the CPU, every id of both parts, 4096 and 376,
is the dump's, and so is the joined text, as on Metal in F32. Both parts write Japanese, and the result's language is
the dump's, `ja`, on Metal in F32 and Q8_0.

### Speed

On an Apple M5 with Q8_0 weights on Metal, after loading, the language left to the model and, in parentheses, forced:
the median of five rounds (three with the 1.7B model) of `build/qwen3-asr-timing`, alternated with
`checks/llama-server-timing.py`, which asks llama.cpp b11246's `llama-server` (the release ASIST bundled before it
recognized speech with speech.cpp) with ggml-org's Qwen3-ASR GGUF files in Q8_0 for the same audio as ASIST asked it, a
16-bit WAV with the prompt cache off:

```sh
build/qwen3-asr-timing <model.gguf> gpu 1 <dump folder>...
python3 checks/llama-server-timing.py <llama-server> <model.gguf> <mmproj.gguf> MTL0 1 <dump folder>...
```

| Audio | 0.6B, speech.cpp | 0.6B, llama.cpp | 1.7B, speech.cpp | 1.7B, llama.cpp |
|---|---|---|---|---|
| ja_jp, 6.36 s | 0.19 s (0.17 s) | 0.17 s (0.16 s) | 0.43 s (0.38 s) | 0.47 s (0.44 s) |
| ja_jp, 10.50 s | 0.28 s (0.25 s) | 0.25 s (0.23 s) | 0.64 s (0.59 s) | 0.58 s (0.58 s) |
| de_de, 11.16 s | 0.34 s (0.32 s) | 0.31 s (0.30 s) | 0.78 s (0.74 s) | 0.74 s (0.71 s) |
| en_us, 23.64 s | 0.63 s (0.59 s) | 0.52 s (0.51 s) | 1.41 s (1.36 s) | 1.25 s (1.20 s) |
| ja_jp, 25.50 s | 0.87 s (0.80 s) | 0.78 s (0.75 s) | 1.98 s (1.89 s) | 1.81 s (1.76 s) |

The decoding is as fast as llama.cpp's: 146 to 158 tokens fed back a second with the 0.6B model against llama.cpp's
147 to 157, and 58 to 63 with the 1.7B against 58 to 62, the two within the few percent by which one round differs
from the next. What remains is the encoder and the prompt: of the 25.50 s input, the 0.6B model's encoder takes
0.133 s and the prefill of its 347 rows 0.110 s, where llama.cpp takes 0.169 s for both, and the 1.7B model's 0.163 s
and 0.305 s, where llama.cpp takes 0.295 s. llama.cpp computes their matrix products through Metal 4's tensor API,
which speech.cpp leaves off for a defect of ggml's kernel ([ADR 0003](../adr/0003-metal-runs-without-the-tensor-api.md)); without it llama.cpp took 0.250 s and
0.523 s for them. llama.cpp's prompt has no system turn, its log-mel one frame more and its last chunk the tokens of
its padding, so its prompt differs from the official one, and it writes another text than the official on 6 of the
20 requests with the 0.6B model and 4 with the 1.7B: with the 1.7B model 軍港や湖ではカマザタ寿司もヨット for the official
群島や湖では必ずしもヨット on the 6.36 s utterance, and with the 0.6B model 光も for 日陰も on the 10.50 s one.

The 1338.42 s input takes 120 s with the 0.6B model, nearly all of it its first part: an encoder of 5.3 s, a prefill
of 15,654 rows in 15 s, and 4096 tokens in 94 s, 43 a second, as each reads a cache of up to 19,750 positions. The
process's peak memory footprint is 2.03 GB: the memory is that of the longest part.

## Resampling

The library resamples the audio it is given, a recording to recognize and the reference recording of an Irodori-TTS
voice, to the model's rate (`speech_model_info_sample_rate()`), and never the audio it makes. It uses the method of
torchaudio's `functional.resample()` with the parameters torchaudio's documentation gives for librosa's `kaiser_best`:
a rational polyphase windowed sinc with 64 zero crossings on each side, cut off at 0.9475937167399596 of the lower
rate's Nyquist frequency, under a Kaiser window of beta 14.769656459379492, computed in double precision. From 48 to
16 kHz it passes everything up to 0.9 of the lower Nyquist frequency within 0.022 dB and keeps everything from 1.05
of it at -146 dB or below; torchaudio's defaults would lose 2.4 dB at 0.9 and fold a tone at 1.1 back into the band
at -14 dB. Its output is torchaudio's with these parameters, with the same length and to double precision: 301 dB SNR
or more on chirps and noise at eight pairs of rates. torchaudio rounds the window's beta and the output's length
through float32, and the library does the same. Audio at the model's rate passes unchanged, so the checks against the
dumps are unaffected. Two rates whose ratio in lowest terms has a term above 4096 (44101 and 16000 Hz) are refused
with an error.

```sh
cd reference/resample
uv run python dump.py out
cd ../..
build/resample-check reference/resample/out
```

The official implementations resample each in their own way, NeMo's `transcribe()` with librosa's soxr and the
Irodori-TTS runtime with torchaudio's defaults, so audio at another rate gives a text or a voice slightly different
from theirs.

## Unicode normalization

The library normalizes text as each model's reference does, with the tables of the version of Unicode the reference
has: Irodori-TTS takes NFKC with those of Unicode 13.0, as its runtime's Python 3.10 does, and Qwen3-TTS and Qwen3-ASR
bring the text they tokenize to NFC with those of Unicode 9.0, as the normalizer of their tokenizers does in the
tokenizers library, whose unicode-normalization-alignments crate has 9.0. Text that is not in NFC, such as Japanese
copied from a macOS file name, whose voiced marks stand apart, gets the tokens of its NFC. The two versions differ only
on characters assigned after 9.0, and one set of tables holds both. `unicode-check` compares the library's NFC and
NFKC in both versions with the tokenizers library's and Python 3.10's on 3626 texts, which hold every code point that
a normalization changes or orders, every mark beside one of each combining class, every pair of a canonical
decomposition with and without a mark between them, every Hangul syllable and jamo, and random sequences of them; all
are equal.

```sh
cd reference/unicode
uv run python gen_unicode.py ../../src/common/unicode-data.inc
uv run python normalization_cases.py out
cd ../..
build/unicode-check reference/unicode/out/normalization-cases.tsv
```

## The server against the worker

A `pcm` response of `speech serve` sends its first bytes with the model's first chunk. On an Apple M5 under heavy load
from other work, the same requests alternated between the server and the worker gave a median first byte of 0.106 s
against the worker's 0.102 s for Qwen3-TTS 0.6B Q8_0, and 0.64 s against 0.60 s for Irodori-TTS v4.1-Small-MF F16, and
the same audio, byte for byte, for the same seed.
