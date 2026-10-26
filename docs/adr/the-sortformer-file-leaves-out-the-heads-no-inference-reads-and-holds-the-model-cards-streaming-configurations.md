# The Sortformer file leaves out the heads no inference reads and holds the model card's streaming configurations

## Context

Nemotron 3 Diarization (nvidia/Nemotron-3-Diarization, OpenMDW 1.1) is a Streaming Sortformer: NeMo's
SortformerEncLabelModel with a 31-layer Transformer encoder whose attention rotates queries and keys (RoPE), a feature
stacking of eight 10 ms frames into each 80 ms frame, a convolution that upsamples the encoder's frames to 10 ms, and a
speaker cache and a FIFO queue that carry pre-encoded frames from chunk to chunk. nemo-toolkit 3.0.0 cannot run it: the
upsampling (`high_resolution`) and the learnable silence embedding came to NeMo's main branch after it.

Its .nemo holds 99.3 M parameters in BF16, the mel filterbank and the window included, and a configuration whose
streaming settings are those of training (chunks of 264 frames, no FIFO queue). The four configurations of the model
card's table, offline with chunks of 27.2 s and three that stream with 0.32 to 1.04 s of latency, are written in the
repository's config.json and processor_config.json, the configuration of its transformers port, and in the card. Two
heads of the checkpoint have no part in inference: `hidden_to_spks`, which SortformerModules builds without ever applying
it, and the activity head, which classifies each frame as silence, one speaker or overlap for an auxiliary loss of
training and which forward_streaming_step() returns only with `return_logits`.

## Decision

- **reference/sortformer pins NeMo's main branch by commit**, and the checkpoint and the two configuration files by the
  repository's revision, size and SHA-256.
- **The file holds the model card's four configurations by name**: `offline`, as the card's quick start runs diarize(),
  and `low_latency`, `very_low_latency` and `ultra_low_latency`, the names the transformers processor gives its
  streaming modes, each with the five settings the card names (`spkcache_len`, `fifo_len`, `chunk_len`,
  `chunk_right_context`, `spkcache_update_period`), and the mode a stream takes by default, processor_config.json's
  `low_latency`. The settings of the cache's compression come from the checkpoint's configuration and the
  postprocessing's defaults from NeMo's PostProcessingParams.
- **The two heads no inference reads are left out**, 4K parameters. Every other tensor is kept, the fused projection of
  the queries, keys and values as three matrices.
- **F32 holds the BF16 values exactly.** The head after the upsampling, 37K parameters that give the probabilities,
  stays in F32 in a file of every weight type.

The alternatives were turned down:

- nemo-toolkit 3.0.0 from PyPI, as reference/fastconformer pins it. It builds the model without its upsampling and its
  silence embedding and gives other probabilities.
- The streaming settings of the .nemo's configuration. They are training's, and none of the card's configurations.
- The activity head kept for a later use. Its classes say less than the speakers' probabilities, from which silence and
  overlap follow, and keeping it would hold tensors no request reads.

## Consequences

A file of another layout is needed before speech.cpp reads the activity head. A model whose configurations differ, such
as diar_streaming_sortformer_4spk-v2.1, writes its own in its file; one with a FastConformer encoder is not this
family's encoder.
