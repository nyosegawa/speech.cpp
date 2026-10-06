# Input audio is resampled, and results carry stop reasons and times

Decided 2026-10-06.

## Context

docs/adr/0011 refused audio at any rate but the model's, so every caller resampled for itself, and the server could
not take the 44.1 or 48 kHz files OpenAI's clients send; Irodori-TTS's references had to be at 48 kHz. The official
implementations resample, each its own way: Irodori-TTS's runtime with `torchaudio.functional.resample()` at its
defaults, NeMo's `transcribe()` with librosa's soxr. Measured with torchaudio 2.10.0 from 48 to 16 kHz on 2026-10-06,
torchaudio's defaults lose 2.4 dB at 0.9 of the lower Nyquist frequency and fold a tone at 1.1 of it back into the
band at −14 dB.

Qwen3-TTS decides frame by frame when its speech ends, and on a short interjection it goes on talking: ASIST measured
"あー。" between 0.6 and 6.5 s in eight generations with `ono_anna` (2026-09-21). speech.cpp cut it at 2048 frames and
returned success, so a caller could not tell a cut sentence from a finished one, and ASIST cancels and generates again
by a length of its own. The worker's seed belonged to the process and was never reported.

A recognition returned one string, although its decoders know on which encoder frame each token was emitted. NeMo's
`transcribe(timestamps=True)` groups tokens into words at spaces and ends a segment at a word ending in `.`, `!` or
`?`, so a Japanese recording, which has no spaces, is one segment however long.

## Decision

- **The library resamples what it is given** (recognition audio and voice references) to the model's rate, and never
  what it makes. The method is torchaudio's rational polyphase windowed sinc with the parameters torchaudio documents
  as librosa's `kaiser_best` (a Kaiser window of beta 14.769656459379492, 64 zero crossings, a cutoff at
  0.9475937167399596 of the lower Nyquist frequency), in double precision except where torchaudio rounds to float32:
  beta, the window's peak i0(beta) and the output length. Measured as above, it passes the band to 0.9 of the lower
  Nyquist frequency within 0.022 dB and keeps everything from 1.05 of it at −146 dB or below, and anyone
  with torchaudio can reproduce its output, which the check compares with. Audio at the model's rate passes unchanged,
  so the checks against the dumps are unaffected.
- **A synthesis reports its seed, its length and why it stopped**: complete, at the request's `max_seconds`, at the
  model's limit, or cancelled. Qwen3-TTS takes `max_seconds`, and its limit is the 8192 frames of its file
  (docs/adr/0015); its cache grows with the request so that a sentence costs the memory of its own length.
  Irodori-TTS does not take `max_seconds`, since it fixes the length before it makes the speech. A request without a
  seed gets one drawn from 0 to 2^53 − 1, the integers a JSON reader in JavaScript holds exactly, so that a reported
  seed repeats the audio.
- **A recognition returns its text, and on request its tokens and segments with times.** A token's time is the
  encoder frame its decoder emitted it on, and for TDT the frames its predicted duration covers, as NeMo computes
  them, a punctuation mark taking the end of the token before it. For the beam search NeMo records its search step,
  which is the frame plus the number of tokens before it and runs past the end of the audio (386 s for a 311 s input);
  speech.cpp gives the frame. Segments follow NeMo's rule for the separators of the model's file (the checkpoint's, or
  NeMo's default `.`, `?`, `!`): a segment ends after a word whose last mark is one, words being split at spaces. A
  model whose languages are written without spaces also lists breaks (`。`, `？`, `！`, `?`, `!` for the Japanese
  models), which end a segment after any token that ends in one, since NeMo's word rule never cuts such text.

The alternatives were turned down:

- Refusing other rates, as docs/adr/0011 decided. Every caller and every OpenAI client resamples with a tool of its own.
- torchaudio's defaults, which would give Irodori-TTS's references the official runtime's latent exactly, at the
  aliasing measured above, which recognition would hear as well.
- A port of soxr, which NeMo's files go through: more code to port and check for a quality the Kaiser filter reaches.
- Keeping the 2048 frames. They are not the checkpoint's, and a long text the model would finish was cut.
- `max_seconds` for Irodori-TTS. A cap would cut a latent made to be longer; `seconds` asks for the length itself.
- A cap kept by the caller, as ASIST's. It cancels, cannot tell its stop from the model's, and wastes the frames after
  it.
- NeMo's segments as they are. A Japanese recording of minutes would be one segment.
- One list of marks that end a segment after any token. It cuts Japanese, but also European text inside numbers and
  abbreviations ("1.000", "z.B.") where NeMo does not.

## Consequences

Audio at another rate gives a text or a latent slightly different from the official implementation's, whose resampler
is not this one. ASIST sends its plausible length as `max_seconds` instead of cancelling, and a Qwen3-TTS request
without it can run to 655 s where it stopped at 164 s. The beam search keeps each hypothesis's frames.
