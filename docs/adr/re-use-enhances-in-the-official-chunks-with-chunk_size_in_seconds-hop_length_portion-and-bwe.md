# RE-USE enhances in the official chunks, with chunk_size_in_seconds, hop_length_portion and bwe

## Context

NVIDIA NemotronLabs RE-USE (9.6M parameters, OpenMDW-1.1) is a SEMamba network: an STFT whose sizes scale with the
sample rate (an FFT of 320 samples and a hop of 40 at its reference of 8 kHz, so that it has 200 frames a second at
any rate), a dense convolutional encoder that quarters the frames and halves the bins, 30 blocks of bidirectional
Mamba-1 over time and then over frequency, and two decoders, of the magnitude and of the phase, before an inverse STFT.
Its model card lists inputs from 8 to 48 kHz. NVIDIA-NemotronLabs-RE-USE-laughter is the same network with weights that
keep laughter, under the same license.

The official code runs on CUDA alone, through mamba-ssm's kernels. `inference.py` enhances a file at once, and
`inference_chunk.py` in chunks of `chunk_size_in_seconds` (5 s in its script) that start every `hop_length_portion` of a
chunk (0.5), the last one aligned with the end of the file, joined by a Hann window and divided by the sum of the
windows; a file shorter than a chunk is one chunk, which gives what `inference.py` gives. Both zero every frame of whose
bins more than half come out of the magnitude decoder at zero or below, which the code says removes a "strange sweep
artifact". `--BWE` resamples the input with librosa's `kaiser_best` before enhancing it. The activations of one chunk
grow with the chunk's length and with the rate: at 48 kHz a chunk of 5 s has 1003 frames of 963 bins in 64 channels at
the encoder's input.

## Decision

- **Every request runs as `inference_chunk.py` does**, with the options `chunk_size_in_seconds` (a number, 5 by
  default) and `hop_length_portion` (a number above 0 and at most 1, 0.5 by default), under the official names and with
  the defaults of its script, which the file holds. A request whose audio is shorter than a chunk gets what
  `inference.py` gives, and a caller that wants a whole file at once sets a chunk at least as long. The frames are zeroed
  as the official code zeroes them, in every chunk.
- **`bwe` is the rate to resample to**, an integer from 8000 to 48000 Hz under the official flag's name: the audio is
  resampled to it with the library's resampler, the one every task's input goes through, enhanced at that rate and
  passed at it. Without it the audio is enhanced at its own rate, which must lie in the same range. The model's
  information has the sample rate 0, and the range of `bwe` is the range of rates it takes.
- **The STFT, the phase's angle and the inverse STFT run on the host** in float32, before and after each chunk's
  network, which runs on the model's device whole: they are the official code's preprocessing and postprocessing, and
  ggml has no atan2. The bidirectional Mamba-1 runs on every device, through the ggml fork's selective scan on Vulkan.
- **Both checkpoints are models of the family**, `re-use` and `re-use-laughter` in the catalog, each one file with every
  tensor of its checkpoint.

The alternatives were turned down:

- The whole audio at once by default, as `inference.py` and the evaluation of the ReazonSpeech test set ran. The
  activations grow with the audio's length, past what a GPU holds for a recording of minutes at 48 kHz, and the first
  audio would wait for the last.
- Names of speech.cpp's own for the options, such as `chunk_seconds` or `sample_rate`. The official names are the ones
  a user of RE-USE knows, as Silero VAD's options keep theirs, and `sample_rate` is already the member of the worker's
  requests that gives the rate of the audio sent.
- librosa's `kaiser_best` for `bwe`. The library has one resampler for every input, checked against torchaudio, and a
  second filter would serve one option of one model.
- A sample rate of 48 kHz in the information, to which every input is resampled. Audio at 16 kHz would gain a band it
  never had, and the model's own rates would not be used.

## Consequences

A request at 48 kHz computes about three times what one at 16 kHz does, and its chunk's activations are three times
larger. A caller that streams the audio out gets its first audio one chunk after the request starts.
