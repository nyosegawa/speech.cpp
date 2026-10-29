# Nemotron 3.5 Transcribe times its words by NeMo's CTC alignment with its defaults, and fails where NeMo drops words

## Context

Nemotron 3.5 Transcribe writes text and no times. Its checkpoint holds a CTC head, its SentencePiece vocabulary and an
`alignment_config` whose `parameters` hold tuned values (`speaker_logprob_weight` 0.6057, `max_word_frames` 19 and
others). NVIDIA's vLLM plugin and NeMo Speech's `generate(generate_timestamps=True)` align a finished transcript with
the audio by `MultiSpeakerSOTWordTimestampAligner` (NeMo Speech 040574f7): the head over the ASR encoder's frames in
bfloat16, each `<spk:N>` tag's words as a stream matched to a column of the speaker branch, a banded search per
stream, and the words' times refined by the speaker branch's 10 ms labels. Neither reads `alignment_config`: the plugin
reads a top-level `ctc_timestamps.speaker_logprob_weight` the checkpoint lacks, and the encoder's own alignment options
are empty, so both run the aligner's defaults.

The aligner gives words in transcripts with spaces, and the letters, marks and numbers of words in Japanese and Chinese.
It raises, and the plugin returns no words, for a character its vocabulary cannot spell (鬱, 叱, full-width digits and
letters), a stream with more tokens than frames, and a word its path skips. A stream whose speaker it finds no frames
for, as a tag the speaker branch never hears, gets no path, and the plugin drops that stream's words and returns the
rest with no error. Thai's units are pythainlp's `newmm` words, a dictionary segmentation.

## Decision

- **NeMo's alignment, every stage as it computes it, with the aligner's default settings**, which layout 2 holds under
  `nemotron-transcribe.align.*` beside the checkpoint's tuned values, which nothing reads. A transcript is the text the
  request returns, after the card's guardrail.
- **A request that cannot be timed returns its text and no times, and says why**: NeMo's failures, a stream without a
  path, whose words NeMo drops in silence, and Thai, whose segmentation this release does not run. Its text is the
  recognition's all the same; the reason is a message, not a category.
- **Times are those of the plugin's public result**, rounded to the millisecond, and tokens keep the order of the text,
  where the plugin sorts its words by time.
- **The speaker branch's 10 ms probabilities give the labels**, from the windows the encoder already runs, so timing
  costs the head and the search alone.

The alternatives were turned down:

- The tuned `alignment_config` values. No official path uses them, so the times would match neither vLLM nor NeMo.
- Dropping a stream's words as NeMo does. A result with some words timed and others missing reads as complete.
- Times for the rest of a transcript beside a failed part. The tokens' texts would no longer join into the text.
- Spelling an unknown character by a neighbour, or Thai by spaces. Each would give times the official alignment never
  gives.

## Consequences

`nemotron-transcribe-alignment-check` compares every stage with NeMo's on dumps, the failures included. On the dumps of
reference/nemotron-transcribe, the audio through this port's perception module, head and alignment times each of 2,573
words as NeMo does on the CPU, and 10 move by a frame on an Apple M5's Metal (2026-10-29). A transcript with a character
outside the head's vocabulary, which the model writes rarely in Japanese, has no times.
