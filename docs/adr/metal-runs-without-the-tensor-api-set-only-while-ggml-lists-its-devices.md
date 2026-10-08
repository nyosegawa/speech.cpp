# Metal runs without the tensor API, set only while ggml lists its devices

## Context

ggml v0.25.3 runs Metal's matrix products through Metal 4's tensor API on the M5, M6, A19 and A20, and through its older
simdgroup kernel on earlier chips. On an Apple M5, the tensor-API `kernel_mul_mm` writes past its output when the
output's column count is 64 modulo 128, whatever the weights' type. In a graph the stray writes land in the next tensor
of the allocator's buffer, so the result is garbage that changes from run to run. One of 64 Irodori-TTS samples hit it:
its third codec window spanned exactly 64 frames, and the second half of the sentence came out as noise. Any graph can
hit it, whether through frames, text tokens or a Qwen3-TTS prompt.

ggml's own `test-backend-ops` has no case of that shape and passes; two added cases fail with a sentinel mismatch and
pass with `GGML_METAL_TENSOR_DISABLE=1`. A two-line change to the kernel's store also makes them pass (issue #13).
ggml-org/llama.cpp master at 0c1e570 has the same kernel, and the defect has not been reported there.

On the M5, over the 20 sentences of speech-bench:

| | Tensor API | Without it |
|---|---|---|
| Irodori-TTS MF F16, median first audio | 0.19 s | 0.23 s |
| Irodori-TTS F16 at 16 steps, median first audio | 0.85 s | 1.12 s |
| Qwen3-TTS 0.6B Q8_0, median first audio | 0.045 s | 0.043 s |
| Irodori-TTS codec decoder against the official, F32 | 47.6 dB | 68.0 dB |

ggml has no API for the setting: it reads `GGML_METAL_TENSOR_DISABLE` from the environment only while it first lists its
devices. A host may run a ggml or llama.cpp of its own in the same process. ggml's Metal backend itself sets
`AGX_RELAX_CDM_CTXSTORE_TIMEOUT` to 1 for the process during that listing, a workaround of its own for long command
buffers.

## Decision

On macOS, the library sets `GGML_METAL_TENSOR_DISABLE` for its first listing of ggml's devices (`start_ggml()`) and then
puts back the value the host had, so Metal runs the simdgroup kernel on every chip and a host's own ggml keeps its
tensor API. `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` is left as ggml sets it.

The alternatives were turned down:

- Carrying the kernel fix, which would mean a fork of ggml as the submodule, against keeping ggml as the one pinned
  dependency.
- Avoiding the shape, for instance by changing the codec's windows, which would leave the DiT, the text encoder and
  Qwen3-TTS exposed.
- Setting the variable for the whole process. It reaches a host's own ggml.

## Consequences

On the M5, Irodori-TTS takes a fifth to a third longer to its first audio, still under audio.cpp's 1.22 s at 16 steps;
Qwen3-TTS keeps its speed. Metal comes closer to the CPU in every check. Setting the environment is not safe while
another thread reads it, so a host must not read the environment on another thread while speech.cpp first lists its
devices, which `speech.h` says. When a ggml release passes the two cases of issue #13 with the tensor API, the setting
goes and the M5 numbers in docs/models/ are measured again.
