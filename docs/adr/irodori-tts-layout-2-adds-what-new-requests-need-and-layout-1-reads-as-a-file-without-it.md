# Irodori-TTS's layout 2 adds what new requests need, and layout 1 reads as a file without it

## Context

Speaking without a reference and captions (VoiceDesign) need weights that layout 1 of Irodori-TTS's file leaves out: the
duration predictor's null speaker, which the official runtime's `no_ref` predicts the length with, and the caption's
projector, its norm and the DiT's caption keys and values, which a request without a caption masks. Both official
checkpoints hold all of them (v4.1-Small-MF and v4.1-Small are trained with `use_caption_condition`), 17,436,672 values
in all. The runtime's scaling of the speaker's keys and values (`speaker_kv_scale`) ends at a time the runtime's code
fixes at 0.9, which layout 1 holds in no key.

A family's reader brings an earlier layout up to the current one in one function, `Layout::upgrade`, which sets the keys
the earlier layout lacks. Layout 1 files of Irodori-TTS, 1.9 GB each, are on users' machines, ASIST's among them, and
their voice files are of the same codec.

GGUF's naming convention counts the size label from the file's tensors, so the added values move v4.1-Small-MF from 848M
to 866M and v4.1-Small from 841M to 859M.

## Decision

- **Layout 2** adds the keys and tensors these requests need, each called for by a key that says whether the file has
  them: `irodori-tts.duration.null_speaker` for `duration.null_speaker`, and `irodori-tts.caption_condition` for the
  caption's projector and norm, the DiT's `attn_k_caption` and `attn_v_caption`, `irodori-tts.caption.max_tokens` and,
  for RF, `irodori-tts.sampler.cfg_caption`. `irodori-tts.caption.dim` gives the width of the duration predictor's
  caption in every file, and RF's `irodori-tts.sampler.speaker_kv_min_t` the runtime's 0.9. Its `speech.requires` is
  0.8.0.
- **A layout 2 file is named by the convention**, `Irodori-TTS-866M-MF-v4.1-F16.gguf` and
  `Irodori-TTS-859M-v4.1-F16.gguf`, so that the name tells the two layouts apart and the convention has no exception.
- **Irodori-TTS's `Layout::upgrade` sets the keys a layout 1 file lacks, with the values layout 1 means**: it holds no
  null speaker and no caption's encoder, its duration predictor's caption is as wide as the text condition, and its RF
  model ends the scaling at 0.9. The tensors still follow from the keys alone, and the rest of the reader knows only the
  current layout. The model information's `layout` stays the file's own, and its metadata lists the keys the upgrade
  sets.
- **A request for what a file lacks is refused naming the file's lack**, not as an option the model does not take, so
  that its owner knows that converting again, or downloading the layout 2 file, gives it.

The alternatives were turned down:

- Refusing layout 1 files. Every user of one would download 1.9 GB again, also for requests they never make.
- The new tensors required in layout 2 without a key. An upgraded layout 1 file could not say it lacks them, and the
  reader would test for the tensor, where the tensors follow from the keys alone.
- A second file beside the model for the new weights. A model is one file.
- The old names kept for layout 2 files. Two files of one name would hold different layouts, and the size label would
  be the one label not counted from the file.

## Consequences

The files are 35 MB larger in F16, and a request without a caption leaves the caption's weights in memory without
computing with them. The catalog names the layout 2 files.
