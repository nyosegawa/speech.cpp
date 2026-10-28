# Nemotron 3.5 Transcribe runs in speech.cpp, checked against NeMo Speech's modules with the Transformers language model on the CPU

## Context

Nemotron 3.5 Transcribe (nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B) is a speech LLM of 34B parameters, 4.3B active:
a parallel expert encoder, NeMo's Transformer ASR encoder beside Nemotron 3 Diarization, whose speakers it adds to the
ASR frames; a projection to Nemotron 3.5 Lightning 30B-A3B, a hybrid of 23 Mamba-2, 6 attention and 23 MoE layers with
two layers of multi-token prediction; and a CTC head that aligns the text to the audio for word timestamps. Its
checkpoint is one safetensors file of 67.5 GB in BF16. NVIDIA serves it with vLLM 0.32 and its NeMo Speech plugin on
one 80 GB GPU; NeMo Speech's own SALMAutomodel builds the language model through NeMo Automodel. NVIDIA's NeMo-Speech.cpp
and community conversions for MLX also run it.

Automodel's Mamba layers run mamba_ssm's Triton kernels and nothing else, so the official entry point cannot run on a
CPU, and a GPU in float32 needs about 140 GB. At the pinned commit of NeMo Speech, from_pretrained() also stops on the
checkpoint's CTC head: read_ctc_timestamp_config() takes no I64 tensor, and the head holds its upsampling factor as
one.

## Decision

- **A family of its own**, `nemotron-transcribe`, one GGUF file per model in this repository's layout. Nemotron 3
  Diarization's network and speaker cache, NeMo's Transformer encoder and the key-value cache of a stack of attention
  layers are in `src/common/` for the families that run them.
- **The reference is NeMo Speech's modules on the CPU in float32, with Transformers' NemotronH as the language model.**
  The dumps come from SALMAutomodel as from_pretrained() builds it, its tokenizer, prompt format, perception module with
  the parallel expert encoder in its windowed inference and generate(), on Modal's CPU with 180 GiB, but for the
  language model: Transformers 5.18's NemotronHForCausalLM, whose PyTorch paths run anywhere, from the checkpoint's
  tensors renamed by Automodel's own state dict adapter, and greedy decoding in the dump's own loop, since Transformers
  refuses generate() from embeddings for NemotronH. Every tensor of the checkpoint is loaded, but for the MTP layers',
  which generation does not use, and the CTC head's, which only timestamps read.
- **The Mamba layers' step size is limited as the checkpoint says, (0, inf), not as Transformers limits it.**
  Transformers 5.18's NemotronHMamba2Mixer clamps the step size from below at the configuration's time_step_min, 0.001,
  where vLLM 0.32's Mamba-2 layer and Automodel's take the checkpoint's time_step_limit, (0.0, inf), and clamp nothing.
  The clamp moves each Mamba layer's output 30 dB away from the unclamped computation, and on a quiet input changes the
  transcript: from an empty one with the clamp to a sentence nobody spoke without it. The reference sets each mixer's
  limit to the checkpoint's and refuses a checkpoint whose limit is another.
- **The prompt is the vLLM plugin's**: the user's text, one space and the audio locator, which the nemotron3p5 format
  and the tokenizer's chat template render alike, both checked by the converter.
- **The file keeps every part of the checkpoint an inference reads, the MTP layers and the CTC head among them**, so
  that speculative decoding and timestamps need no new file. It leaves out the speaker branch's own preprocessor, which
  equals the perception module's that both branches take, and Nemotron 3 Diarization's hidden_to_spks head, which no
  inference applies.

The alternatives were turned down:

- The official entry point on a GPU in float32. It needs a GPU of more than 140 GB, and its Mamba kernels compute in
  TF32, which would bound the checks far below what the port's float32 computes.
- Running the language model through llama.cpp, as NeMo-Speech.cpp does. A program that speaks and recognizes would run
  two runtimes with two file layouts.
- Leaving the MTP layers and the CTC head out. Each would cost a new layout and a new download once it is used.

## Consequences

The reference's language model is Transformers', not the Automodel or vLLM implementation NVIDIA trains and serves, so
the transcripts of the official vLLM serving are compared with the dumps' in speech-bench: the dumps' greedy texts are
what vLLM gives at temperature 0 only if the implementations agree. The F32 file is about 136 GB, larger than any GPU
speech.cpp runs on, so the language model's checks load one layer at a time.
