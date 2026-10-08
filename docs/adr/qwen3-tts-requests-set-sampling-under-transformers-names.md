# Qwen3-TTS requests set the sampling under transformers' names

## Context

Qwen3-TTS samples twice per frame of its speech: the talker draws the frame's first code, and the code predictor the
other 15. The official `generate_custom_voice()` lets a caller set both through transformers' `generate()`: `do_sample`,
`top_k`, `top_p`, `temperature` and `repetition_penalty` for the talker, and `subtalker_dosample`, `subtalker_top_k`,
`subtalker_top_p` and `subtalker_temperature` for the code predictor, whose repetition penalty no argument sets. Each
takes `generation_config.json`'s value when the caller sets none, and the model file holds those values.

transformers refuses a temperature or a repetition penalty that is not above 0, a top-k below 0 and a top-p outside 0 to
1, and uses top-k, top-p and the temperature only when `do_sample` is true, warning when they are set without it. It
divides float32 logits by the temperature and the penalty, so a value far enough from 1 gives an infinite logit, and
`torch.multinomial` then fails. Its draw comes from PyTorch's own generator, which nothing outside PyTorch reproduces.

## Decision

- **Nine options under the official names.** The talker's settings are `do_sample`, `top_k`, `top_p`, `temperature` and
  `repetition_penalty`, transformers' names, which every caller of `generate()` knows. The code predictor's are
  `code_predictor_do_sample`, `code_predictor_top_k`, `code_predictor_top_p` and `code_predictor_temperature`:
  `subtalker` is an argument prefix that nothing else names, while the code predictor is the module's name in the
  official code and in the file's keys, and docs/models/qwen3-tts.md says what it predicts. Its repetition penalty stays
  the file's value, 1, as the official package sets none, and applies to the codes of the frame made so far, as the code
  predictor's `generate()` sees them.
- **The defaults are the file's, and none is neutral.** A request that sets none of the options samples with the
  checkpoint's settings. The information shows each default as the decimal the checkpoint writes (0.9), which converts
  back to the file's float. A model that does not sample tokens refuses any value of them, as it refuses any option it
  cannot follow.
- **The ranges are transformers'.** A temperature or penalty that the float the sampler computes with rounds to 0 or to
  infinity is refused before any work, since the sampler divides by it, and one that pushes a logit beyond a float while
  the speech is made ends the request with `out_of_range`, naming the option, where the official package fails with
  PyTorch's error.
- **What transformers leaves unused is refused.** `top_k`, `top_p` or `temperature` set while their stack's `do_sample`
  is false is `invalid_argument` when the request runs, as a request is refused for any setting an official
  implementation would ignore.
- **The sampler is checked against transformers' processors**, which `_get_logits_processor()` itself builds, on the
  logits of a dump: the tokens kept and their probabilities, and the greedy pick. The draws are not compared.

The alternatives were turned down:

- `greedy` in place of `do_sample`. It reads more plainly but inverts the official sense, so a caller moving from the
  official package would translate every setting.
- `subtalker_*`, the official arguments' names. A reader has to know the official code to understand them.
- `residual_*` or `acoustic_*`, after what the codes after the first are. The official code names neither, and the
  file's keys would then use another name than the options.
- Neutral values, such as a temperature of 1. A model that does not sample tokens would accept a temperature and do
  nothing with it, and 1 is not even Qwen3-TTS's default.
- Computing the temperature and the penalty in double, which cannot overflow. The arithmetic would no longer be the
  official float32's, and every request would sample from slightly other probabilities than the checkpoint's settings
  give.

## Consequences

The worker, the server and the command line take the options as members and flags of their names without changes of
their own; `--do-sample=false` turns the talker's sampling off. A request that sets every option at the default the
information shows gives the same audio as one that sets none, which speech-api-check checks.
