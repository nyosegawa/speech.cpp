# Irodori-TTS requests take the runtime's guidance, schedule and tail settings

Decided 2026-10-07.

## Context

A request to Irodori-TTS took a voice, a seed, the speed, the length and the steps. The official runtime's request
(`SamplingRequest` in `irodori_tts/inference_runtime.py` at 89f9d8f) takes 41 fields, and those that need no other
weights change only the sampler and the cut at the tail: the scales, the mode and the time range of RF's guidance, what
the branch without the speaker attends to, Sway Sampling, the truncation of the noise, the temporal score rescaling,
and the cut's window and thresholds. The runtime drops every RF setting for a MeanFlow checkpoint, some with a message
and some without a word (truncation, rescaling, the schedule, the time range). It ignores a setting that another one
leaves without effect, clamps a length outside its bounds, and cuts a text or a reference past its bounds.

The other fields return several takes from one batch (`num_candidates`, `decode_mode`), decide how the runtime computes
without changing the audio (`context_kv_cache`), repeat another field (`cfg_scale`, deprecated), are bounds of the
model (`min_seconds`, `max_seconds`, `max_ref_seconds`, `max_text_len`, `max_caption_len`), read Python's pickle
(`ref_latent`, `ref_latents`), or need weights or a reference of another kind (captions, no reference, several
references, speaker-inversion embeddings, LoRA adapters).

ASIST and speech-bench measure Irodori-TTS through speech.cpp, and their measurements hold only while a request that
sets none of the new fields gives the same audio as before.

## Decision

- **The fields that need no other weights are options of the C API's vocabulary, by the runtime's names**:
  `cfg_scale_text`, `cfg_scale_speaker`, `cfg_guidance_mode`, `cfg_min_t`, `cfg_max_t`, `truncation_factor`,
  `rescale_k`, `rescale_sigma`, `speaker_uncond_mode`, `sway_coeff`, `keep_tail`, `tail_window_size`,
  `tail_std_threshold` and `tail_mean_threshold`. Two differ from the runtime's: `t_schedule_mode` and `sway_coeff`
  are one option, since Sway Sampling at 0 is the linear schedule, bit for bit; and `trim_tail`, true by default, is
  `keep_tail`, false by default, since a boolean flag on the command line can only set true.
- **An RF file alone declares the guidance and the schedule.** A MeanFlow file does not take them, so a request that
  sets one is `unsupported` there, as any option a model does not take; the cut at the tail is both models'.
- **Defaults come from the file**, where the checkpoint's own (`irodori-tts.sampler.*`, `irodori-tts.tail.*`); the
  guidance mode and the speaker mode default to the runtime's default path, which the port already ran. A default the
  file holds in float32 shows as its shortest decimal (0.05), and the request rounds it back.
- **What the runtime would ignore or cannot run is refused before any work**, naming the option: one of the rescaling's
  two settings alone, `cfg_min_t` above `cfg_max_t`, the joint guidance with two scales that differ, a schedule that
  stops, and a value other than the default that another value leaves without effect (the guidance's mode, range and
  speaker mode with both scales at 0, speaker noise without a branch without the speaker, the tail's settings with
  `keep_tail`). This follows docs/adr/0007 and 0014: a caller that asks for something gets it or an error.
- **The other fields are not offered.** A caller wanting several takes makes several requests with seeds of its own:
  each is a take of the same distribution, though not the runtime's candidates, as no seed of speech.cpp's generator
  gives the runtime's noise. The bounds stay the model's, from its file, past which speech.cpp refuses (docs/adr/0014).
- **A request that sets none of the new options gives the audio it gave before, sample for sample**, on every device:
  each default builds the same graph and the same order of float32 operations as before, which `tools/same_audio.py`
  shows by comparing the callback's samples of two builds bit for bit. The noise of `speaker_uncond_mode` `noise` comes
  after the latent's from the request's seed, so that the latent's noise stays the seed's.
- The options belong to the C API's minor version 3.1.

The alternatives were turned down:

- `t_schedule_mode` as an option of its own beside `sway_coeff`. Linear and Sway at 0 are the same schedule, and a
  coefficient without the mode, or the mode with 0, would be one more combination to refuse.
- `trim_tail`, true by default. The command line could not turn it off.
- Accepting RF's options on a MeanFlow file and ignoring them, as the runtime does. The caller gets audio other than it
  asked for and no sign of it.
- Ignoring a setting that has no effect, as the runtime does. The same, for a request that combines them.
- `num_candidates` as several streams of one request. Every entry point gives one stream of audio a request.

## Consequences

The worker, the server and `speech tts` take the new options by their names without a change of their own, and the
model information lists them with their defaults and ranges. Each option's effect is checked against dumps of the
official runtime made with it (`reference/irodori-tts/dump.py` writes the options into `meta.json`, and
`irodori-dit-check` and `irodori-synthesis-check` read them). The default thresholds of the cut never cut v4.1's latent,
whose silence has a standard deviation near 0.7, so the cut only acts with thresholds a request raises.
