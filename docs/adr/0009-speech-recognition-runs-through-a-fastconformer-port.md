# Speech recognition runs through a FastConformer port

Decided 2026-10-05.

## Context

speech.cpp is to recognize speech as well as speak, for the same callers and through the same C API. The
recognizers in view are NVIDIA NeMo's parakeet-tdt-0.6b-v3 (English, French, German, Italian, Spanish and
Portuguese among its languages), parakeet-tdt_ctc-0.6b-ja (Japanese) and reazon-research's
reazonspeech-nemo-v2 (Japanese), and Qwen3-ASR. The three NeMo models are one architecture, FastConformer: a
log-mel frontend, a subsampling by 8 with depthwise convolutions, and conformer layers with relative
positional attention. They differ in their heads and a few options: TDT for parakeet-v3, TDT and a CTC head
for parakeet-ja, RNN-T and local attention for ReazonSpeech, and the number of mel bins. Qwen3-ASR is an
audio encoder in front of a Qwen3 language model, which decodes text token by token with a cache.

## Decision

speech.cpp ports FastConformer once, as the family `src/families/fastconformer/`, on ggml, and runs the three
NeMo models through it, each stage checked against tensors dumped from NeMo itself
(`reference/fastconformer/`). The port begins with parakeet-tdt_ctc-0.6b-ja and its CTC head, the shortest path
from audio to text that can be checked end to end; TDT decoding, parakeet-tdt-0.6b-v3 and ReazonSpeech's
local attention and RNN-T head follow on the same encoder.

Qwen3-ASR stays outside speech.cpp, in llama.cpp, which already runs Qwen3's decoder with its cache, sampling
and quantization; a port here would repeat that work.

The alternatives were turned down:

- Running parakeet through CrispASR or audio.cpp. Each is a runtime of its own with its own copy of ggml, so
  the worker and libspeech would carry two ggml builds and two ways of choosing a device, and their ports are
  not checked stage by stage against NeMo as this repository's are. audio.cpp v0.8.2's Irodori-TTS adds a
  distorted copy of the voice on Metal, a defect such checks catch (README.md, Irodori-TTS's accuracy).
- Running the models as ONNX through sherpa-onnx. It brings ONNX Runtime as a second dependency beside ggml,
  runs exported weights rather than NeMo's, and reaches the GPU through ONNX Runtime's providers rather than
  the Metal and Vulkan builds every release already has.

## Consequences

Recognition costs one family, whose encoder all three models share; the C API, the worker and the server gain
a recognition entry point in a later change. The GGUF layout of `reference/fastconformer/convert.py` grows
with each head. A model outside FastConformer and Qwen3-ASR needs a decision of its own.
