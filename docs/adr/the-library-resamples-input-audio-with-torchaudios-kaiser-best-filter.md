# The library resamples input audio with torchaudio's kaiser_best filter

## Context

Audio comes in at whatever rate it was recorded: OpenAI's clients send files of 44.1 or 48 kHz, a microphone gives its
own rate, and a reference recording for Irodori-TTS may be at any rate. The recognizers take 16 kHz and Irodori-TTS's
codec 48 kHz. The official implementations resample, each its own way: Irodori-TTS's runtime with
`torchaudio.functional.resample()` at its defaults, NeMo's `transcribe()` with librosa's soxr. Measured with torchaudio
2.10.0 from 48 to 16 kHz on 2026-10-06, torchaudio's defaults lose 2.4 dB at 0.9 of the lower Nyquist frequency and
fold a tone at 1.1 of it back into the band at −14 dB.

## Decision

The library resamples what it is given, a recognition's audio and a voice's references, to the model's rate, and never
what it makes. The method is torchaudio's rational polyphase windowed sinc with the parameters torchaudio documents as
librosa's `kaiser_best` (a Kaiser window of beta 14.769656459379492, 64 zero crossings, a cutoff at 0.9475937167399596
of the lower Nyquist frequency), in double precision except where torchaudio rounds to float32: beta, the window's peak
i0(beta) and the output length. Measured as above, it passes the band to 0.9 of the lower Nyquist frequency within 0.022
dB and keeps everything from 1.05 of it at −146 dB or below, and anyone with torchaudio can reproduce its output, which
`resample-check` compares with. Audio at the model's rate passes unchanged, so the checks against the dumps are
unaffected.

The alternatives were turned down:

- Refusing audio at any rate but the model's. Every caller and every OpenAI client would resample with a tool of its
  own.
- torchaudio's defaults, which would give Irodori-TTS's references the official runtime's latent exactly, at the
  aliasing measured above, which recognition would hear as well.
- A port of soxr, which NeMo's files go through: more code to port and check for a quality the Kaiser filter reaches.

## Consequences

Audio at another rate gives a text or a latent slightly different from the official implementation's, whose resampler is
not this one. Two rates whose ratio in lowest terms has a term above 4096 are refused, which no two common rates of
recorded audio have.
