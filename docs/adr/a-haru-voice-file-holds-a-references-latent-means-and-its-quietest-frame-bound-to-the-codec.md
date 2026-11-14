# A Haru voice file holds a reference's latent means and its quietest frame, bound to the codec

## Context

Haru speaks in the voice of a reference, 3 s to 3 min of one speaker. The official session trims the reference's
leading and trailing silence, keeps its last 180 s, encodes it with the codec and, at every start, draws each latent
about its mean with noise from the session's generator; a conversation the model opens puts the reference's quietest
whole frame, the speaker's noise floor, on the rows before its first words. Encoding takes about 0.56 s for 10 s of
reference on an Apple M5's Metal (F32, 2026-11-14), some 10 s for three minutes.

## Decision

- **A voice file holds the latent means** of the trimmed reference, `[latent, frames]`, encoded whole from the codec's
  zero state, **the index of the quietest whole frame** (-1 where there is none), and the length and rate of the
  recording it was made of, under `general.architecture` `haru-voice`.
- **It is bound to the codec** by `haru-voice.codec_sha256`, the model file's `haru.codec.sha256`, a hash of the
  checkpoint's codec tensors; a voice file of another codec is refused, naming it, with how to make it again.
- **A conversation draws the latents at its start** from the means, as the official session does, so that one voice file
  gives each conversation its own draw.
- **A WAVE file is a voice too** (`speech_voice_add()`), trimmed and encoded when it is added; `speech_voice_make()`
  writes the voice file of one recording.

The alternatives were turned down:

- The drawn latents in the file. Every conversation with the voice would share one draw, where the official session
  draws at each start.
- The recording in the file, encoded at each start. The encoding would delay every conversation's first frame by the
  reference's length.
- Several recordings or speakers in one voice. The official session takes one reference of one speaker.
