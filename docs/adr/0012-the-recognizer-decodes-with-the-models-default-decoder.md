# The recognizer decodes with the model's default decoder

Decided 2026-10-06.

## Context

docs/adr/0009 began the FastConformer port with parakeet-tdt_ctc-0.6b-ja's CTC head, the shortest path from audio
to text that could be checked end to end. The checkpoint is a hybrid: one encoder with two heads, a CTC head and a
TDT decoder (a prediction network over the previous tokens and a joint that gives a token and a duration). NeMo's
`transcribe()` decodes with TDT unless told otherwise, and the model card's results are TDT's. The two do not always
agree: on one of the three FLEURS utterances the dumps cover, CTC writes ブラックスタフ where TDT writes ブラックスタッフ.
A caller who compares speech.cpp with NeMo, or with the model card, compares it with TDT.

TDT is also the decoder of parakeet-tdt-0.6b-v3, which has no CTC head at all, and ReazonSpeech's RNN-T has the same
prediction network and joint without the durations.

## Decision

A model is decoded with the one decoder its official implementation uses by default, and with that one alone. For
parakeet-tdt_ctc-0.6b-ja that is greedy TDT as the checkpoint configures it (`greedy_batch`, durations 0 to 4,
at most 10 tokens on one frame), ported from NeMo's label-looping decoder and checked step by step against it.

The CTC head is dropped from the runtime, the C++ and the GGUF conversion. There is no option to choose CTC and no
fallback to it.

The alternatives were turned down:

- Keeping CTC as an option beside TDT. Two decoders double the checks, the GGUF and the surface a caller has to
  understand, for a head NeMo itself does not use by default and that other FastConformer models lack.
- Keeping CTC as the default for its speed. On Metal TDT adds about 0.02 s to a 25.5 s utterance in `speech-asr`
  (0.28 s against 0.25 to 0.27 s), and the text it gives is the one NeMo and the model card give.

## Consequences

The recognizer's text equals what NeMo's `transcribe()` returns. GGUF files converted before carry the CTC head and
no prediction network or joint, and are refused when loaded; they are converted again. The prediction network and
the joint are written without durations in them, so that an RNN-T decoder can use them; the RNN-T decoding itself
waits for a model that needs it.
