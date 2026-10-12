# A FastConformer model on a GPU decodes on the CPU from a copy of the decoding's weights, unless a request keeps it on the GPU

## Context

A FastConformer recognition runs the encoder as one graph and then decodes: each step of greedy TDT, greedy RNN-T or
the beam search is one small graph of the prediction network and the joint, a few products of a matrix of at most 640
by 8,198 with a vector or four. The beam search computes one graph per step of its search, 638 for 25.5 s of audio;
greedy decoding about one per token. On a GPU each graph is a round trip of about 0.3 ms on Metal and 0.6 to 0.7 ms on
Vulkan, which is as long as the CPU takes to compute it or longer.

The weights the decoding reads, the prediction network and the joint but for its projection of the encoder, are 2 to 3%
of a file: 11.5 MB of reazonspeech-nemo-v2's Q8_0 file, 11.6 MB of parakeet-tdt_ctc-0.6b-ja's Q8_0 file and 35.0 MB of
parakeet-tdt-0.6b-v3's F16 file, the files the names fetch.

ggml's CPU backend, built without OpenMP as every release is, starts and joins its threads for every graph unless it is
given a pool of its own.

## Decision

- **Decoding on the CPU is part of the model's specification.** A FastConformer model loaded on a GPU also loads the
  tensors its decodings read into the CPU's memory, once, beside the copy on the GPU, and decodes there by default. A
  model on the CPU keeps no second copy.
- **The request option `decode_on_cpu`, true by default, lets a caller keep the decoding on the GPU.** Every
  FastConformer model declares it, so that a request does not depend on the device the model was loaded on; on a model
  on the CPU both values decode there. The copy on the GPU stays loaded for it.
- **The CPU decodes with the load's threads, in a pool that waits between graphs.** The model's information reports the
  threads of a FastConformer model on a GPU, which computes part of each request on the CPU.

Measured on an Apple M5 with Metal on 2026-10-13, the median time of a recognition of 4 utterances of 6 to 28 s, 3 runs
in each of 3 rounds that alternate the builds, summed over the utterances; the copy in the file's type and, for
comparison, converted to F32:

| Model, file, decoding | Decoding on Metal | On the CPU | On the CPU from an F32 copy |
|---|---|---|---|
| reazonspeech-nemo-v2, Q8_0, beam search | 1,262 ms | 1,001 ms | 1,641 ms |
| reazonspeech-nemo-v2, F16, beam search | 1,409 ms | 1,312 ms | 1,839 ms |
| reazonspeech-nemo-v2, Q8_0, greedy | 714 ms | 707 ms | 855 ms |
| parakeet-tdt_ctc-0.6b-ja, Q8_0 | 635 ms | 585 ms | 678 ms |
| parakeet-tdt-0.6b-v3, F16 | 583 ms | 555 ms | 678 ms |

On Windows with an RTX 2080 on Vulkan and an i9-9900K, the total of 100 utterances in 3 alternating rounds against
0.8.4, which decodes on the GPU: reazonspeech-nemo-v2 in Q8_0 13.93 s against 11.56 s, parakeet-tdt_ctc-0.6b-ja in Q8_0
4.65 s against 3.97 s, and parakeet-tdt-0.6b-v3 in F16 8.78 s against 10.38 s, slower.

A step's time follows the bytes of weights it reads, about 4 products of 2,560 by 640, one of 640 by 640 and the joint's
output layer: on the M5 with 4 threads, parakeet-tdt-0.6b-v3's step took 595 us in F32 (46.6 MB), 209 us in F16 and
116 us in Q8_0, measured with ggml's CPU backend alone. The same day, with reazonspeech-nemo-v2's beam search on 3 of
the utterances above, the CPU took 1,071 ms with 1 thread, 950 ms with 4 threads started for each graph and 856 ms with
a pool of 4, where Metal took 1,091 ms.

The alternatives were turned down:

- A second copy of the whole model on the CPU. It doubles the memory of the weights, 0.66 GB more for a Q8_0 file and
  1.26 GB for an F16 one, where the decoding reads 2 to 3% of them.
- The CPU reading the GPU's own buffers. It works only where the GPU's memory is the host's, as Metal's is on Apple
  silicon; a discrete GPU on Vulkan has no buffer the CPU can read in place.
- Decoding on the GPU alone. Every step pays a round trip that costs as much as the step's arithmetic or more.
- ggml's threads started for each graph. The pool took a further 10% off the beam search.
- A copy converted to F32, whose arithmetic would match the GPU's more closely. It gives main's text on every input
  above and the 311 s one, with no token a frame from NeMo's, but reads twice the bytes of F16 and four times those of
  Q8_0 at each step, and was slower than decoding on Metal for every model, by 7 to 30%, and holds 43 to 70 MB.

## Consequences

The decoding computes in the CPU's arithmetic, which differs from the GPU's: the CPU multiplies Q8_0 weights by inputs
it rounds to 8 bits and, on ARM, sums products of F16 weights in half precision. Where two choices score almost the
same, the text or a token's frame can differ from the GPU's. On the checks' dumps the texts are NeMo's either way; with
F16 weights one token in 2 of reazonspeech-nemo-v2's 20 decodings falls one frame from NeMo's, and with the Q8_0 file
the 311 s input comes out one word different, once nearer NeMo's text and once further, with each decoding. The beam
search on that input then runs to its last step, 7,782 where Metal stops at 4,889, and keeps 0.2 GB more of finished
hypotheses.

A FastConformer model on a GPU takes 6 to 50 MB more of the host's memory at its peak for the files the names fetch,
and its recognitions occupy the CPU's threads while they decode. Where the CPU computes a step more slowly than the GPU's
round trip, as the i9-9900K does with parakeet-tdt-0.6b-v3's F16 weights, decoding on the CPU is slower. A caller that
wants the CPU free, the GPU's arithmetic or the GPU's speed there sets `decode_on_cpu` to false.
