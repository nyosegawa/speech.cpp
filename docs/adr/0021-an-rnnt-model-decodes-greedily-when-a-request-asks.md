# An RNN-T model decodes greedily when a request asks, its beam search staying the default

Supersedes docs/adr/0012 in part: reazonspeech-nemo-v2 also runs NeMo's greedy decoding, when a request asks for it.

Proposed 2026-10-07. The comparison under Consequences decides whether the option is released.

## Context

docs/adr/0012 decodes a model with the one decoder NeMo's `transcribe()` runs for it by default, and with that one
alone. For reazonspeech-nemo-v2 that is the beam search its checkpoint configures, NeMo's alignment-length synchronous
search with a beam of 4. Each step of the search is one graph on the device, the new outputs of the prediction network
and the joint of up to 4 hypotheses, and the steps number about the frames and the labels together: 4,864 graphs for
the 311 s input of the checks, where the encoder takes one. On an Apple M5 with F16 weights on Metal, the beam search
took 3.3 s of the 4.9 s that input took (2026-10-06). A caller that waits for the text once a person stops speaking,
as ASIST does, waits for those steps.

The checkpoint also configures greedy decoding (`decoding.greedy.max_symbols` 10), which NeMo runs when the decoding's
strategy is `greedy_batch`, its default strategy, by label looping (`GreedyBatchedRNNTInfer`), or `greedy`, frame by
frame (`GreedyRNNTInfer`); the two emit the same tokens on the same frames on all ten inputs of the checks. Its text
differs from the beam search's on five of the ten: the greedy texts lack some of the commas and full stops, one has
サウスカロライナ where the beam search writes サウスカロロナイナ, and the 311 s input has a few other words.

## Decision

- **Greedy decoding is a request option, the beam search the default.** The C API's vocabulary gains `decoding`, a
  string whose choices a model declares: reazonspeech-nemo-v2 takes `beam`, the default, and `greedy`. A request that
  does not set it decodes as before, with the same text, tokens and times. The choices name the decodings, not NeMo's
  strategies, two of which, `greedy_batch` and `greedy`, are one decoding.
- **Only a model with a choice declares it.** The parakeet models have one decoding, greedy TDT, and do not take the
  option, as a model that cannot change its speed does not take `speed`. The option has no neutral value, since each
  model's default is its own.
- **One graph per token.** The prediction network's output changes only when a token is emitted, so one graph computes
  the prediction for the last token and the joint at a run of frames from the current one, and the host takes the
  first frame whose label is a token, the first of equal values as torch's argmax takes it. A run of 8 frames follows
  each token, and a run twice as long a run of blanks, up to 64 frames: the 311 s input takes 1,008 graphs for its
  931 tokens, where NeMo evaluates the joint 4,822 times.
- **The limit is in the file, and the files already published stay.** The most tokens greedy decoding emits on one
  frame, the checkpoint's 10, is a model constant (docs/adr/0015): fastconformer's layout 2 adds
  `fastconformer.decoder.rnnt.max_symbols`. The reader brings a file of layout 1 up to layout 2 as it reads it, giving
  an RNN-T file the limit of 10: layout 1's converter pinned one RNN-T checkpoint, reazonspeech-nemo-v2, and the
  upgrade refuses an RNN-T file of layout 1 under another `general.name`. The files of layout 1 on Hugging Face, which
  ASIST and speech-bench pin, are read as they are and recognize as before, and take the option too; nobody downloads
  a model again.

Measured on an Apple M5 on 2026-10-07 while other work loaded the machine, so that the times are rough and the graphs
the measure: the decoding of the 311 s input took 1,008 graphs against 4,864 and about 1.2 s against 3.1 s with F16
weights on Metal, and of the 25.5 s FLEURS utterance 107 graphs against 638 and about 0.09 s against 0.55 s; the ten
inputs of the checks take 1,658 graphs against 9,427.

The alternatives were turned down:

- The beam search alone, as docs/adr/0012 had it. Greedy decoding takes a fifth of the graphs, which a caller that
  waits for the text can prefer to the commas.
- Greedy decoding as the default. `transcribe()` runs the beam search, whose text has the commas and full stops, and
  a caller that compares speech.cpp with NeMo or the model card compares it with the beam search.
- Greedy decoding frame by frame, one graph per evaluation of the joint, as `GreedyRNNTInfer` runs it: 4,822 graphs for
  the 311 s input, as many as the beam search's steps.
- The argmax on the device with `ggml_argmax`. ggml's CPU backend keeps the last of equal values where torch keeps the
  first, so the blank, the last output, would win a tie that NeMo gives to a token.
- A beam size as the option, greedy being a beam of 1. NeMo's beam search with a beam of 1 runs a greedy search of its
  own, not the decoding of the strategy `greedy_batch`, and beams of 2 and 3 have no dump to check them against.
- A boolean `greedy`. It leaves no room for another decoding, and its false would misdescribe the parakeet models.
- The parakeet models declaring `decoding` with `greedy` its one choice. It steers nothing; the language is declared
  where a model only checks it because OpenAI's clients send it as a hint, and no client sends a decoding.
- The limit of 10 in the C++, NeMo's default as well as the checkpoint's. It is the checkpoint's setting, which another
  RNN-T checkpoint can set otherwise.
- The new key optional, read as 10 where a file lacks it. A file whose converter forgot the key and a file of layout 1
  would look the same, which docs/adr/0015 turned down.

## Consequences

ASIST and speech-bench find the option in a model's information and choose the decoding per request, through every
entry point: the worker's `transcribe` and `peek`, the server's transcription form field `decoding` and
`speech asr --decoding`. The C API is version 3.1. A file converted from now on is of layout 2, which 0.7.x refuses,
naming 0.8.0; the reazonspeech-nemo-v2 file in its Hugging Face repository stays of layout 1, and the `speech info
--json` beside it lists the option once it is made again with this release. reference/fastconformer/dump.py `--greedy`
dumps NeMo's greedy decoding beside the beam search's, and the transducer, times and C API checks compare `greedy`
with it.

Whether the option is released waits for one comparison, on an idle machine: the 4,483 clips of Common Voice 8.0
Japanese's test split through speech-bench, the CER and the wait of reazonspeech-nemo-v2 with each decoding, in F16 on
Metal. If greedy decoding does not make recognition clearly faster, the option goes and docs/adr/0012 stands as it was.
