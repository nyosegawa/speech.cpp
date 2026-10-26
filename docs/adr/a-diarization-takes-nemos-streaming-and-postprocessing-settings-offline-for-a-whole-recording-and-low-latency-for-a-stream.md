# A diarization takes NeMo's streaming and postprocessing settings, offline for a whole recording and low latency for a stream

## Context

NeMo configures Streaming Sortformer's streaming with five settings of SortformerModules, in 80 ms frames:
`spkcache_len`, `fifo_len`, `chunk_len`, `chunk_right_context` and `spkcache_update_period`. The model card of
Nemotron 3 Diarization defines streaming by these five and gives four configurations: offline (27.2 s chunks, 30.4 s
of latency), which its quick start's diarize() runs, and three that stream, low latency (1.04 s), very low (0.64 s) and
ultra-low (0.32 s). Its transformers port runs the offline configuration when it is not streaming and names the others
`low_latency`, `very_low_latency` and `ultra_low_latency`, streaming in `low_latency` by default. The offline
configuration sets `spkcache_update_period` (300) below `chunk_len` (340), which streaming_update() takes as a chunk,
as NeMo's _check_streaming_parameters() warns. SortformerModules also has `chunk_left_context`, which neither the
checkpoint's configuration nor any of the card's sets. diarize()'s other settings are PostProcessingParams (`onset`,
`offset`, `pad_onset`, `pad_offset`, `min_duration_on`, `min_duration_off`), and DiarizeConfig's
`max_num_of_spks`, which only sizes a dataset's targets, and `session_len_sec`, which cuts the audio.

The C API's option table declares one default for each option, read by the setters, the information and its JSON.

## Decision

- **`streaming_mode` names one of the configurations the file holds**: `offline`, `low_latency`, `very_low_latency` or
  `ultra_low_latency`. It has no default in the table, as the seed has none: speech_diarize() takes `offline`, and a
  diarization of audio given a piece at a time the file's mode for streams, `low_latency`, as transformers runs them.
- **The five settings are options under NeMo's names**, each replacing the mode's value, with SortformerModules'
  minimums (a cache of a silence frame and one more for each speaker, chunks and update periods of a frame), no default,
  and a largest value of 2^20 frames. An update period outside a chunk and the queue with a chunk is taken as
  streaming_update() takes it, not refused.
- **The postprocessing takes PostProcessingParams' names and defaults**, from the file, the thresholds from 0 to 1 and
  the pads and durations from 0 to 3600 s, within which the segments' float32 times keep their 10 ms.
- **Not offered**: `chunk_left_context`, which no configuration of the card or the checkpoint gives; `max_num_of_spks`
  and `session_len_sec`, which change nothing a request computes or which a caller does by giving less audio.

The alternatives were turned down:

- One default of the five settings for both calls, the low latency's or the offline one's. A whole recording would take
  18 times as long (98 s of a meeting in 3.9 s against 0.22 s, F32 on an Apple M5's GPU, 2026-10-26) at the card's
  higher error (13.18% against 12.73% on DIHARD III), or a stream would wait 30 s for its first segment.
- The configurations alone, without the five settings. A caller could not set a latency between the card's, which NeMo
  and the card describe by the five.
- The five settings alone, without the configurations' names. A caller would repeat the card's table to ask for one.
- Refusing an update period that NeMo warns of. The card's own offline configuration has one.

## Consequences

The information of a diarization model shows no default of the streaming settings; a caller that wants a
configuration's values reads its metadata (`sortformer.streaming.*`) or sets its name.
