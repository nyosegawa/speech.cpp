# Local attention recognizes long audio whole, as transcribe() does

## Context

reazon-research's reazonspeech-nemo-v2 is a FastConformer model whose card says it transcribes Japanese audio of up to
several hours. Its encoder attends locally, as Longformer does: NeMo's `rel_pos_local_attn` with 128 frames of 80 ms on
either side of each frame and one global token, the first frame, which every frame attends to and which attends to
every frame.

NeMo's `transcribe()` runs a file through the encoder in one piece, however long; it does not cut it. Measured with NeMo
3.0.0 on the CPU on 2026-10-06, a 311 s input built from FLEURS utterances runs whole and gives the same text as the
stages run by hand. The reazonspeech package that the card recommends pads the audio with 0.5 s of silence on either
side and then calls the same `transcribe()` on the whole of it. NeMo computes the local attention in chunks, so its time
and memory grow with the length of the audio, where the attention of the parakeet models over the whole utterance grows
with its square.

## Decision

- **The whole audio, at once.** A recognition request runs the whole of its audio through the encoder in one graph, as
  `transcribe()` does. speech.cpp does not cut long audio into segments, and does not pad it as the reazonspeech package
  does; a caller that wants either does it before the request.
- **Local attention in blocks.** `rel_pos_local_attn` runs as NeMo computes it, a band of 2w + 1 frames per query plus
  the global tokens, rather than as attention over the whole utterance with a mask: the queries go in blocks of w
  frames, each against a window of 3w keys. Its scores are NeMo's term by term, the global token counted twice where it
  is also within the band, and a global token's own output is its attention over every frame without the positional
  term, as `RelPositionMultiHeadAttentionLongformer` computes them.

The alternatives were turned down:

- Masking the attention over the whole utterance to the band. It gives the same numbers, but each of its score tensors
  takes T² × 8 heads in float32: 0.48 GB for 311 s, and 65 GB for an hour.
- Cutting long audio into segments with overlaps, as some recognizers do. It changes the text where the cuts fall, no
  NeMo dump checks it, and `transcribe()` does not do it.
- Padding with the reazonspeech package's 0.5 s of silence. It is that package's choice, not NeMo's, and a caller
  comparing with NeMo would get a different text.

## Consequences

ReazonSpeech recognizes minutes of audio in one request, with memory that grows linearly with its length. The parakeet
models attend over the whole utterance, and their memory grows with its square. The encoder runs as one graph, so a
cancel takes effect before or after it, which on long audio runs for seconds, or between the decoder's frames. The
server takes a file of at most 25 MB, about 13 minutes of 16-bit audio at 16 kHz.
