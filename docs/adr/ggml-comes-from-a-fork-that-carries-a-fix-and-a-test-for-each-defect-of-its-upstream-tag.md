# ggml comes from a fork that carries a fix and a test for each defect of its upstream tag

## Context

speech.cpp builds ggml from a git submodule. ggml-org/ggml v0.25.3 gives speech.cpp's models wrong results on two
supported platforms:

- **Metal's tensor API.** On the M5, M6, A19 and A20, Metal runs matrix products through Metal 4's tensor API by
  default. On an Apple M5 its `kernel_mul_mm` stores its tile past the output when 64 columns remain, so an output of 64
  or 192 columns corrupts the next tensor of its buffer: one of 64 Irodori-TTS samples decoded a codec window of 64
  frames to noise. Under the relaxed precision it asks for, it also computes less precisely than the simdgroup kernel
  that the earlier chips run: Irodori-TTS v4.1-Small's synthesis lay 2.3 to 48.3 dB from the official implementation's
  against 8.7 to 62.5 dB, and reazonspeech-nemo-v2's beam search of a 311 s input changed its text. Without relaxed
  precision it is as precise as the simdgroup kernel but slower, by speech-bench on the M5 (2026-10-12): median first
  audio 0.351 s against 0.251 s for Irodori-TTS v4.1-Small-MF, 1.762 s against 1.239 s for v4.1-Small at 16 steps and
  0.072 s against 0.039 s for Qwen3-TTS 0.6B, and median recognition 0.150 s against 0.123 s for Qwen3-ASR 0.6B.
- **Vulkan's q8_1 activations.** Vulkan quantizes a matrix product's activations to q8_1 for its integer dot products,
  and q8_1 carries the sum of each block of 32 activations in half precision. The products of Q4_0, Q4_1, Q5_0, Q5_1,
  Q4_K, Q5_K, Q2_0 and IQ1_S read that sum, and Qwen3-TTS's activations sum past 65504 in a block, so its Q4_K and Q5_K
  files gave no finite logit on an RTX 2080, where the CPU and Metal spoke.

ggml reads no setting that avoids either through its API, only environment variables: `GGML_METAL_TENSOR_DISABLE`,
which it reads only while it lists its devices, so that a library sets it in its host's process, unsafely while another
thread reads the environment; and `GGML_VK_DISABLE_MMVQ`, which it reads when it first creates each Vulkan device, and
which also moves Q8_0's batched products off their kernels. ggml-org/ggml master on 2026-10-12 has the same code.

## Decision

- **The submodule is the branch `speech.cpp` of github.com/nyosegawa/ggml**, pinned by commit: an upstream release tag,
  then one commit of `test-backend-ops` cases that show the defects, then one commit for each fix, and nothing else.
- **Metal keeps the tensor API off by default on every chip**, as ggml already does on the chips before the M5, and runs
  it only when `GGML_METAL_TENSOR_ENABLE` asks for it, so that every chip runs the simdgroup kernel.
- **Vulkan keeps the types whose q8_1 dot products read the block sums off q8_1 activations**, through the
  dequantizing kernels, which then accumulate in F32 as the q8_1 kernels do, whatever precision the product asks for:
  in half precision, which they take by default, a product past 65504 is clipped. Q8_0, Q6_K and the other types keep
  the q8_1 path.
- **Each fix comes with what shows its defect, and goes when its upstream no longer has it:**

  | Fix | Shown by | Dropped when |
  |---|---|---|
  | Metal's tensor API off | `test-backend-ops -o MUL_MAT`: F16 products of 64 and 192 columns, with `GGML_METAL_TENSOR_ENABLE` on an M5, M6, A19 or A20; and the stage checks of speech.cpp on Metal | the tag's tensor API passes those cases and every stage check, and is no slower in speech-bench |
  | Vulkan's q8_1 block sums | `test-backend-ops -o MUL_MAT -p large_b`: Q4_0, Q5_0, Q4_K and Q5_K products of activations from 2048 to 4096, and of activations from 512 to 1024 with weights from 0.5 to 1, on Vulkan | the tag passes those cases |

  Moving ggml to a newer tag starts the branch again from that tag: the commit of cases first, on which they run on the
  devices above, then each fix the table still calls for.

The alternatives were turned down:

- The environment variables above, set by the library. They change the environment of a host's process, and Vulkan's
  would also move Q8_0 off its kernels.
- Fixing the tensor API's store and running it without relaxed precision. Every speech model is slower to its first
  result on the M5 with it than with the simdgroup kernel.
- Avoiding the shapes and the activations in speech.cpp, such as changing the codec's windows or keeping Qwen3-TTS
  from K-quants on Vulkan. Every graph can meet the shapes, and every model with large activations the sums.
- Waiting for ggml-org/ggml to fix them. The models give wrong results on supported platforms until then.

## Consequences

Metal computes on the M5 as it did when the library set `GGML_METAL_TENSOR_DISABLE`, and the library no longer touches
the environment. A ggml update runs the cases on the M5 and on the Windows machine's Vulkan GPU, from the Windows
build of CI with checks, before the submodule moves.
