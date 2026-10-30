# Haas runs through the FastConformer port with its KDA prediction network, checked against fla on a CUDA GPU

## Context

Third Intelligence's Haas-ja-v1.0 is a Japanese recognizer built as NVIDIA's parakeet-tdt is: a FastConformer encoder
of 24 layers, greedy TDT with durations 0 to 4, and a NeMo checkpoint (`EncDecRNNTBPEModel`). Its prediction network is
not NeMo's LSTM but Haas-ASR's `KDATransducerDecoder`: two pre-norm residual blocks of flash-linear-attention's
`KimiDeltaAttention` (Kimi Delta Attention) and a final layer norm. Its SentencePiece vocabulary of 3,072 pieces has byte
fallback, so a character outside it is written as the byte pieces of its UTF-8.

fla's layer decodes with Triton kernels alone (its short convolution's `causal_conv1d_update`, `fused_recurrent_kda` and
`FusedRMSNormGated`), which need a CUDA GPU: the model card supports Linux with an NVIDIA GPU alone, and its accuracy and
speed were measured in bfloat16. Haas-ASR pins NeMo below 2.8 and fla at 0.5.2, which it refuses to run without, where
speech.cpp's other FastConformer checkpoints are dumped with NeMo 3.0.0. Haas-ASR also offers `transcribe_longform()`,
which cuts a recording at quiet points found by volume.

## Decision

- **A model of the family `fastconformer`.** Layout 3 adds `fastconformer.decoder.prediction`, `lstm` or `kda`, with the
  KDA layers' tensors and constants, and `fastconformer.tokenizer.byte_pieces`, the piece of each byte. The reader brings
  every earlier file up with LSTM layers and no byte pieces, so that no file of the catalog changes. A KDA layer's step is
  ggml's own operations, its recurrence `ggml_gated_delta_net()` with a gate per key channel, as fla computes it in float32:
  q and k divided by the square root of their squares' sum plus the kernel's 1e-6, the gate's softplus, and an output RMS
  norm gated by a sigmoid.
- **The reference is NeMo 2.7.3 with Haas-ASR at its pinned commit, on the CPU in float32, but for the prediction network,
  which runs fla's kernels on a CUDA GPU in float32 without TF32.** Its inputs move to the GPU and its outputs and state
  back, so that the encoder, the joint and the decoding are computed and recorded as for every other FastConformer
  checkpoint. The environment of `reference/fastconformer/haas/` pins Haas-ASR's tested versions apart from the family's
  own.
- **Byte pieces decode as SentencePiece decodes them**: a character's text goes to the piece of its last byte, and a
  byte that begins no well-formed character is U+FFFD. The text is NeMo's. NeMo's timestamps decode each byte piece
  alone, so a character of byte pieces stands in NeMo's words and segments as one U+FFFD per byte; speech.cpp's segments
  hold the character, so that the segments joined are the text.
- **`transcribe_longform()` is not offered.** A long recording is recognized by the regions where someone speaks, as for
  the other models (`speech asr --vad`, `chunking_strategy`), and a request is recognized whole as `transcribe()` does.

The alternatives were turned down:

- A family of its own. Its frontend, encoder, joint, TDT decoding, times and detokenizer are the FastConformer port's;
  only the prediction network's step differs, behind the step the decodings already take.
- The whole reference on the GPU. NeMo's joint returns its log-softmax on the CPU alone, CUDA graphs would run the
  label-looping decoding where nothing records its steps, and the encoder's dumps would no longer come from the CPU's
  float32 as the other checkpoints' do.
- fla's naive PyTorch functions on the CPU in place of its kernels. They are not the code the model runs, and fla's
  layer calls the kernels directly.
- bfloat16 dumps, the model card's precision. The checks compare float32 stages, and the F32 file holds the checkpoint's
  bfloat16 weights exactly.
- `ggml_l2_norm()` for q and k. It divides by the norm bounded below by its epsilon, where fla adds 1e-6 to the sum of
  squares.

## Consequences

The reference of Haas runs on Linux with a CUDA GPU alone, and its dumps' prediction outputs come from the GPU's float32
arithmetic. A request to Haas is recognized whole: its attention over the whole request grows with the square of its
length, as parakeet's does.
