# GGUF files

This page describes speech.cpp's model files and voice files, and how to convert a model, for people who convert
models or read the files with other tools.

Every model is one GGUF file, its codec included
([ADR 0015](adr/0015-a-model-is-one-gguf-file-with-its-codec-a-layout-version-every-constant-and-ggufs-standard-names.md)).
The layout is speech.cpp's own, and only speech.cpp runs these files. The model's identity and languages are in the GGUF
specification's own keys (ggml's `docs/gguf.md`, "Standardized key-value pairs"), under the names it gives them, so that
tools that read GGUF metadata show them. The rest is under `speech.` and the family's architecture.
`speech info MODEL --meta` prints every key of a file.

## Convert a model

`reference/<model>/convert.py` writes a model's F32 file from the official checkpoint, which `reference/<model>/pins.py`
pins by revision, in a uv environment that pins the official code. `speech quantize` makes every other weight type of
that file ([cli.md](cli.md#speech-quantize)), each tensor in the type its family's layout gives it
([Weight types](#weight-types); [ADR 0040](adr/0040-speech-quantize-makes-every-weight-type-from-the-f32-file-as-the-layout-tables-store-each-tensor.md)).

```sh
cd reference/qwen3-tts
uv run python convert.py 0.6b ../../models       # Qwen3-TTS-12Hz-0.6B-CustomVoice-F32.gguf, 4.1 GB
uv run python convert.py 1.7b ../../models       # Qwen3-TTS-12Hz-1.7B-CustomVoice-F32.gguf, 8.1 GB

cd ../irodori-tts
uv run python convert.py mf ../../models         # Irodori-TTS-866M-MF-v4.1-F32.gguf, 3.5 GB
uv run python convert.py rf ../../models         # Irodori-TTS-859M-v4.1-F32.gguf, 3.4 GB

cd ../fastconformer
uv run python convert.py parakeet-tdt_ctc-0.6b-ja ../../models   # parakeet-tdt_ctc-0.6B-ja-F32.gguf, 2.5 GB
uv run python convert.py parakeet-tdt-0.6b-v3 ../../models       # parakeet-tdt-0.6B-v3-F32.gguf, 2.5 GB
uv run python convert.py reazonspeech-nemo-v2 ../../models       # reazonspeech-nemo-619M-v2-F32.gguf, 2.5 GB

cd ../qwen3-asr
uv run python convert.py Qwen3-ASR-0.6B ../../models   # Qwen3-ASR-0.6B-F32.gguf, 3.14 GB
uv run python convert.py Qwen3-ASR-1.7B ../../models   # Qwen3-ASR-1.7B-F32.gguf, 8.16 GB

cd ../..
build/speech quantize models/Qwen3-TTS-12Hz-0.6B-CustomVoice-F32.gguf models --type q8_0   # ...-Q8_0.gguf
build/speech quantize models/Irodori-TTS-866M-MF-v4.1-F32.gguf models --type f16           # ...-F16.gguf
```

Of the F32 file a released file was converted from, `speech quantize` writes the released F16 or Q8_0 file byte for
byte, which `tools/quantize_compare.py` checks tensor by tensor and key by key. The files of each type, in GB:

| Model | F16 | Q8_0 | Q6_K | Q5_K | Q4_K | Notes |
|---|---|---|---|---|---|---|
| Qwen3-TTS 0.6B | 2.06 | 1.21 | 0.99 | 0.87 | 0.76 | the codec's large weights are F16 in every type but F32 |
| Qwen3-TTS 1.7B | 4.08 | 2.29 | 1.82 | 1.57 | 1.33 | |
| Irodori-TTS v4.1-Small-MF | 1.92 | 1.21 | 1.04 | 0.95 | 0.86 | the codec stays F32 in every type. The converter writes layout 2; the files the catalog's names fetch are of layout 1 |
| Irodori-TTS v4.1-Small | 1.91 | 1.20 | 1.03 | 0.94 | 0.86 | |
| parakeet-tdt_ctc-0.6b-ja, parakeet-tdt-0.6b-v3, reazonspeech-nemo-v2 | 1.24 to 1.26 | 0.66 to 0.67 | 0.51 to 0.52 | 0.43 to 0.44 | 0.36 | `reference/fastconformer/` pins NeMo 3.0.0 and PyTorch 2.10.0. The converter refuses a checkpoint with an option the C++ does not run (another subsampling, attention or decoding, a prompt, a language tag to strip, a tokenizer piece it cannot write) rather than write a file that would recognize differently from NeMo |
| Qwen3-ASR 0.6B | 1.57 | 0.84 | 0.68 | 0.59 | 0.51 | `reference/qwen3-asr/` pins transformers 5.18.0, qwen-asr 0.0.6 and PyTorch 2.10.0. The token embeddings, which are also the output matrix, are stored once, where the checkpoint holds them twice |
| Qwen3-ASR 1.7B | 4.08 | 2.18 | 1.68 | 1.41 | 1.16 | |

## Weight types

A file's weight type is the type that holds most of its tensors' bytes, which `general.file_type` names. Each family's
`layout.cpp` gives every tensor the types it takes in a file of each weight type, in order: the file holds it in the
first whose blocks its rows are whole blocks of. The reader takes a tensor in any of the types its layout lists for it,
whatever the file's weight type, and `speech quantize` writes the one the order gives.

| Tensors | F32 file | F16 file | Q8_0 file | Q6_K, Q5_K or Q4_K file |
|---|---|---|---|---|
| The matrices and embeddings that only `ggml_mul_mat()` and `ggml_get_rows()` read: every family's linear layers, Qwen3-TTS's talker and code predictor with their embeddings and heads, Irodori-TTS's text encoder, speaker encoder, duration predictor and DiT, FastConformer's subsampling output, encoder, prediction network and joint, Qwen3-ASR's encoder, projector and decoder with its token embeddings | F32 | F16 | Q8_0, or F16 where a row is not whole blocks of 32 values | the type, or Q8_0 where a row is not whole blocks of 256 values, and F16 where it is not whole blocks of 32 either |
| Qwen3-TTS's codec's convolutions and matrices, and Qwen3-ASR's convolution kernels, which `ggml_im2col()` reads in F16 or F32 alone | F32 | F16 | F16 | F16 |
| The norms, biases, codebooks, Irodori-TTS's codec, its DiT's input projection and its duration predictor's output, FastConformer's convolution kernels and the rest | F32 | F32 | F32 | F32 |

The rows that are not whole blocks: Qwen3-ASR 0.6B's encoder and projector, 896 wide, and FastConformer's prediction
network and joint, 640 wide, take Q8_0 in a K-quant file; so do Irodori-TTS's AdaLN, of rank 192, its DiT's
feed-forward output, rows of 3680, and its speaker encoder's input, rows of 128, while the speaker encoder's
feed-forward output, rows of 1996, takes F16 in a Q8_0 file as well.

`general.file_type` is 0 for F32, 1 for F16, 7 for Q8_0, 18 for Q6_K (`MOSTLY_Q6_K`), 16 for Q5_K (`MOSTLY_Q5_K_S`) and
14 for Q4_K (`MOSTLY_Q4_K_S`), the one of gguf-py's two values for each of these two that names no mix of wider types.

`speech quantize` writes in `speech.requires` the latest of the F32 file's own, the first release that reads the
`general.file_type`, and the first release whose reader of the family takes each tensor in its type, which each family's
`layout.cpp` gives with its storages. A Q6_K, Q5_K or Q4_K file names 0.8.0, which releases before refuse as a
`general.file_type` they do not know, and so does a FastConformer file in Q8_0, whose matrices 0.7's reader took in F16
and F32 alone. The F16 and Q8_0 files of the other families keep the F32 file's own: 0.7.0 for layout 1, which 0.7.0 and
0.7.1 read, and 0.8.0 for Irodori-TTS's layout 2. `tools/quantize_releases.py` checks each against `speech` of earlier
releases built from their tags.

## File names

A file is named under GGUF's naming convention from its general keys:
`<basename>-<size label>-<finetune>-<version>-<type>.gguf`, without the parts the model has none of, as gguf-py's
`naming_convention()` writes it. Where the model's name gives no size, the size label is the parameters of the file's
tensors, counted and rounded as gguf-py counts them. Irodori-TTS's names give Small, a word, so its size labels are
counted, with the codec: 848M for v4.1-Small-MF, whose DiT has MeanFlow's 7.2M parameters more, and 841M for v4.1-Small
in layout 1, and 866M and 859M in layout 2, whose files hold the null speaker and the caption's encoder.
reazonspeech-nemo-v2's name gives no size, and its label is 619M.

The converted files go to Hugging Face in one repository for each upstream repository, named after it with `-GGUF`
([models.md](models.md#the-catalog)). Beside each GGUF file goes the output of `speech info --json` for it, under the
file's name with `.json` added. A program that pinned that JSON can compare it with the worker's `ready`, which carries
the same object with the device, the threads and the voices added.

## Layouts

A file says the version of its family's layout in `speech.layout`, and in `speech.requires` the first release whose
reader takes it:

| Family | Layout | `speech.requires` |
|---|---|---|
| fastconformer | 2 | 0.8.0 |
| irodori-tts | 2 | 0.8.0 |
| qwen3-tts | 1 | 0.7.0 |
| qwen3-asr | 1 | 0.7.0 |

A reader brings a file of an earlier layout of its family up to its own as it reads it, so the files of layout 1 on
Hugging Face recognize and speak as they did:

- fastconformer's layout 1 lacks `fastconformer.decoder.rnnt.max_symbols`. An RNN-T file of layout 1 gets
  reazonspeech-nemo-v2's 10, the one RNN-T checkpoint that layout 1's converter took; an RNN-T file of layout 1 with
  another `general.name` is refused.
- irodori-tts's layout 1 gets the keys that say it holds neither the null speaker nor the caption's encoder
  ([irodori-tts](#irodori-tts)).

The model information lists such a key among the file's metadata, beside the file's own `speech.layout`. A newer layout,
a weight type the reader does not know and a tensor in a type its layout does not list are refused with a message that
names `speech.requires` where it names a later release. A file without `speech.layout`, converted for a release before
0.7.0, is refused as such; convert it again. Such files remain in the history of the Hugging Face repositories.

## What a reader checks

- Every key below is required in its family's layout, unless the table says when it is present, and has exactly the
  type listed. A key that is missing or of another type is refused with a message that names it, and so is a string
  that names a kind other than the ones listed.
- The tensors are exactly the ones the keys call for, each of the shape the keys give it and of a type its layout
  lists ([Weight types](#weight-types)). A tensor that is missing, one not called for, and one of another shape or type are refused before any weight
  is loaded, naming the tensor and the shape or type expected and found.
- A width that no key gives, listed with each family's tensors, is taken from one tensor, and every other tensor of that
  width is checked against it.
- A key that sizes the model is refused when it is 0, and so is a value the model cannot run with: heads that do not
  divide their width or are of an odd width, which RoPE cannot turn in pairs, an id outside the vocabulary or table it
  indexes, or a stride the codec does not take.
- A key whose meaning the tables leave empty means what the official configuration's field of the same name means.

## Keys of every model file

| Key | Type | Meaning | Source |
|---|---|---|---|
| `general.architecture` | string | the family, the code that runs the file: `qwen3-tts`, `irodori-tts`, `fastconformer` or `qwen3-asr` | the converter |
| `general.name` | string | the model's name | the pinned repository's name (`Qwen3-TTS-12Hz-0.6B-CustomVoice`, `Irodori-TTS-v4.1-Small-MF`, `parakeet-tdt-0.6b-v3`, `Qwen3-ASR-1.7B`) |
| `general.organization` | string | the organization that publishes the model | the pinned repository's namespace (`Qwen`, `Aratako`, `nvidia`, `reazon-research`) |
| `general.basename` | string | the model line, which the file's name begins with | the repository's name through the converter's table of names (`Qwen3-TTS-12Hz`, `Irodori-TTS`, `parakeet-tdt_ctc`, `parakeet-tdt`, `reazonspeech-nemo`, `Qwen3-ASR`) |
| `general.size_label` | string | the number of parameters with its scale, B or M | the repository's name (`0.6B`, `1.7B`), or, where it gives none, the parameters of the file's tensors as gguf-py's `size_label()` rounds them (`866M`, `859M`, `619M`; Irodori-TTS's layout 1 files `848M` and `841M`) |
| `general.finetune` | string | what the model was trained toward beyond its line; present where its name gives it | the repository's name (`CustomVoice`, `MF`, `ja`) |
| `general.version` | string | present where the model's name gives one | the repository's name (`v4.1`, `v3`, `v2`) |
| `general.license` | string | SPDX expression | the model card |
| `general.source.repo_url` | string | the repository converted, `https://huggingface.co/<repository>` | the pin |
| `general.source.url` | string | the revision converted: `<general.source.repo_url>/tree/<revision>`, since the specification has no key of its own for a revision | the pin |
| `general.file_type` | u32 | the type that holds most of the tensors' bytes, as gguf-py's `LlamaFileType` numbers it: 0 (F32), 1 (F16), 7 (Q8_0), 18 (Q6_K), 16 (Q5_K) or 14 (Q4_K); a file whose tensors say otherwise is refused | 0 from the converter; `speech quantize --type` |
| `general.quantization_version` | u32 | the version of ggml's quantized blocks (2); present when the file holds a quantized tensor | ggml's `GGML_QNT_VERSION`, through `speech quantize` (gguf-py's `GGML_QUANT_VERSION` in the files the converters quantized before 0.8.0) |
| `general.languages` | [string] | each language's shortest ISO 639 code, sorted, which requests and the model information give as BCP 47 tags: two letters, or three for a language that has no two-letter code (`yue`, `fil`), where the GGUF specification asks for two letters | Qwen3-TTS: the names of `codec_language_id` through the converter's table of codes, dialects left out; Qwen3-ASR: the tags of transformers' `LANGUAGE_CODE_TO_NAME`; the others: the model card |
| `speech.layout` | u32 | the version of the family's layout: 2 for fastconformer and irodori-tts, 1 for the others | the converter |
| `speech.requires` | string | the first release whose reader takes this file: `0.8.0` for layout 2, for Q6_K, Q5_K and Q4_K, and for FastConformer's Q8_0, `0.7.0` for the other files of layout 1 | the converter's table of layouts, and `speech quantize`, which writes the later of the F32 file's and the weight type's |
| `speech.task` | string | `synthesis` or `recognition`; must be the family's | the converter |
| `speech.sample_rate` | u32 | the rate of the audio made or recognized | Qwen3-TTS: `speech_tokenizer/config.json` `output_sample_rate`; Irodori-TTS: the DACVAE's `sample_rate`; FastConformer: the featurizer's `sample_rate`; Qwen3-ASR: qwen-asr's `SAMPLE_RATE`, the feature extractor's rate |
| `speech.language_use` | string | `steers` or `checked`; must be what the family does | `steers` for qwen3-tts and qwen3-asr, `checked` for the others |
| `speech.voices` | [string] | the built-in voices' names; present for qwen3-tts alone | `talker_config.spk_id`'s names, sorted |
| `speech.voice_languages` | [string] | each voice's language, aligned with `speech.voices` | the model card's "Native Language" column through the table of codes (Dylan's and Eric's dialects are `zh`) |
| `speech.voice_genders` | [string] | `female` or `male`, aligned | the model card's "Voice Description" column |
| `speech.voice_descriptions` | [string] | the description, aligned | the model card's "Voice Description" column |

## qwen3-tts

From the checkpoint's `config.json` (`talker_config`, its `code_predictor_config`, and the top level),
`generation_config.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json` and `speech_tokenizer/config.json`
(`decoder_config`), and the official code at the commit the converter's environment pins (QwenLM/Qwen3-TTS@022e286).

| Key | Type | Meaning | Source |
|---|---|---|---|
| `qwen3-tts.talker.hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim` | u32 | | `talker_config` |
| `qwen3-tts.talker.rms_norm_eps`, `rope_theta` | f32 | | `talker_config` |
| `qwen3-tts.talker.vocab_size` | u32 | the codec ids, audio codes and control tokens | `talker_config.vocab_size` |
| `qwen3-tts.talker.num_code_groups` | u32 | codes per frame | `talker_config.num_code_groups` |
| `qwen3-tts.talker.max_position_embeddings` | u32 | the positions of the talker's cache, prompt and frames together | `talker_config.max_position_embeddings` |
| `qwen3-tts.talker.codec_bos_id`, `codec_eos_token_id` (the end of speech), `codec_pad_id`, `codec_think_id`, `codec_nothink_id`, `codec_think_bos_id`, `codec_think_eos_id` | u32 | | `talker_config` |
| `qwen3-tts.talker.suppressed_tokens` | u32 | the ids at the end of the vocabulary that sampling never picks, the end of speech excepted (1024) | the official `generate()`'s `suppress_tokens`, `range(vocab_size - 1024, vocab_size)` |
| `qwen3-tts.code_predictor.hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size` | u32 | | `code_predictor_config` |
| `qwen3-tts.code_predictor.rms_norm_eps`, `rope_theta` | f32 | | `code_predictor_config` |
| `qwen3-tts.text.tts_bos_token_id`, `tts_eos_token_id`, `tts_pad_token_id`, `im_start_token_id`, `im_end_token_id`, `assistant_token_id` | u32 | | `config.json` |
| `qwen3-tts.text.newline_token_id` | u32 | the token of "\n" in the chat template (198) | `vocab.json`'s id of `Ċ`, the byte-level form of "\n" |
| `qwen3-tts.language_ids` | [i32] | the codec id of each of `general.languages`, aligned | `talker_config.codec_language_id` |
| `qwen3-tts.speaker_ids` | [i32] | the codec id of each of `speech.voices`, aligned | `talker_config.spk_id` |
| `qwen3-tts.dialect_ids` | [i32] | the codec id of the dialect each voice speaks, or −1, aligned | `talker_config.spk_is_dialect` through `codec_language_id` |
| `qwen3-tts.dialect_language` | string | the language in which, as with `auto`, a voice with a dialect speaks it (`zh`) | the official prompt's `language.lower() in ["chinese", "auto"]` through the table of codes |
| `qwen3-tts.generation.min_frames` | u32 | frames before the end of speech may be sampled (2) | the official `generate()`'s `min_new_tokens` |
| `qwen3-tts.generation.max_frames` | u32 | the model's limit in frames (8192) | `generation_config.json` `max_new_tokens` |
| `qwen3-tts.generation.talker.do_sample` | bool | the default of the option `do_sample` | `generation_config.json` `do_sample` |
| `qwen3-tts.generation.talker.temperature`, `top_p`, `repetition_penalty` | f32 | the defaults of the options of the same names | `generation_config.json` |
| `qwen3-tts.generation.talker.top_k` | u32 | the default of the option `top_k` | `generation_config.json` |
| `qwen3-tts.generation.code_predictor.do_sample` | bool | the default of the option `code_predictor_do_sample` | `subtalker_dosample` |
| `qwen3-tts.generation.code_predictor.temperature`, `top_p` | f32 | the defaults of `code_predictor_temperature` and `code_predictor_top_p` | `subtalker_temperature`, `subtalker_top_p` |
| `qwen3-tts.generation.code_predictor.top_k` | u32 | the default of `code_predictor_top_k` | `subtalker_top_k` |
| `qwen3-tts.generation.code_predictor.repetition_penalty` | f32 | the code predictor's repetition penalty on the codes of the frame it has made, which no option sets | `code_predictor_config.repetition_penalty`, which the code predictor's `generate()` uses since the official code passes none |
| `qwen3-tts.tokenizer.tokens` | [string] | the BPE vocabulary in id order | `vocab.json` and `tokenizer_config.json` `added_tokens_decoder` |
| `qwen3-tts.tokenizer.merges` | [string] | merges in rank order | `merges.txt` |
| `qwen3-tts.codec.num_quantizers` | u32 | must equal `talker.num_code_groups` | `decoder_config.num_quantizers` |
| `qwen3-tts.codec.latent_dim`, `codebook_dim`, `hidden_size`, `num_attention_heads`, `head_dim`, `num_hidden_layers`, `sliding_window` | u32 | | `decoder_config` |
| `qwen3-tts.codec.num_key_value_heads` | u32 | must equal the heads, the one form the C++ runs | `decoder_config.num_key_value_heads` |
| `qwen3-tts.codec.rms_norm_eps`, `rope_theta` | f32 | | `decoder_config` |
| `qwen3-tts.codec.upsample_rates` | [i32] | the decoder blocks' strides | `decoder_config.upsample_rates` |
| `qwen3-tts.codec.upsampling_ratios` | [i32] | the upsampling stages' strides | `decoder_config.upsampling_ratios` |

A frame lasts the product of `upsample_rates` and `upsampling_ratios` (1920) samples at `speech.sample_rate`, 0.08 s.
The text a request takes at most is `talker.max_position_embeddings` − `generation.max_frames` − 11, the rows its
prompt adds to the text (24565 tokens for both sizes).
`general.size_label` is the checkpoint's `tts_model_size` under the specification's name, `0b6` as 0.6B and `1b7` as
1.7B, which the converter maps one to one; a file of 0.6B takes no instruction (`instructions`), as the official
`generate_custom_voice()` drops it for that size.

Tensors, with T = `talker.num_hidden_layers`, C = `code_predictor.num_hidden_layers`, G = `talker.num_code_groups`,
Q = `codec.num_quantizers`, L = `codec.num_hidden_layers`, U = the length of `codec.upsampling_ratios`, B = the
length of `codec.upsample_rates`:

- `talker.text_embd`, `talker.text_proj.fc1.{weight,bias}`, `talker.text_proj.fc2.{weight,bias}`, `talker.codec_embd`,
  `talker.codec_head`, `talker.norm`;
- `talker.blk.{0..T-1}.` and `cp.blk.{0..C-1}.` each with `attn_norm`, `ffn_norm`, `attn_q`, `attn_k`, `attn_v`,
  `attn_o`, `attn_q_norm`, `attn_k_norm`, `ffn_gate`, `ffn_up`, `ffn_down`;
- `cp.norm`, `cp.codec_embd.{0..G-2}`, `cp.head.{0..G-2}`, and `cp.in_proj.{weight,bias}` when
  `code_predictor.hidden_size` differs from `talker.hidden_size` (the 1.7B model), as the official model makes
  `small_to_mtp_projection` a Linear exactly then;
- `codec.vq.first.codebook.0`, `codec.vq.first.out_proj`, `codec.vq.rest.codebook.{0..Q-2}`, `codec.vq.rest.out_proj`;
- `codec.pre_conv.{weight,bias}`, `codec.tf.in_proj.{weight,bias}`, `codec.tf.out_proj.{weight,bias}`, `codec.tf.norm`,
  and `codec.tf.blk.{0..L-1}.` with `attn_norm`, `ffn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_o`, `attn_scale`,
  `ffn_gate`, `ffn_up`, `ffn_down`, `ffn_scale`;
- `codec.up.{0..U-1}.` with `tconv.{weight,bias}`, `dwconv.{weight,bias}`, `norm.{weight,bias}`, `pw1.{weight,bias}`,
  `pw2.{weight,bias}`, `gamma`;
- `codec.dec.in_conv.{weight,bias}`, `codec.dec.blk.{0..B-1}.` with `snake.{alpha,inv_beta}`, `tconv.{weight,bias}`
  and `res.{0,1,2}.` with `snake1.{alpha,inv_beta}`, `conv1.{weight,bias}`, `snake2.{alpha,inv_beta}`,
  `conv2.{weight,bias}`; `codec.dec.out_snake.{alpha,inv_beta}`, `codec.dec.out_conv.{weight,bias}`.

The three residual units per decoder block (dilations 1, 3 and 9) are the architecture's, as the official module
builds them, and stay in the C++, and so do the widths the official decoder fixes in its code rather than in its
configuration: 3 for `pre_conv`, 1 for a residual unit's second convolution, 7 for every other convolution, and the
fourfold width of a ConvNeXt block. The widths that no key gives are the text embedding's width and rows
(`text_hidden_size` and `text_vocab_size`, from `talker.text_embd`, whose rows must cover `tokenizer.tokens`), the
codebooks' entries (`codebook_size`, from `codec.vq.first.codebook.0`, which must cover the talker's and the code
predictor's codes), the decoder's width (`decoder_dim`, from `codec.dec.in_conv.weight`) and the codec transformer's
feed-forward width (its `intermediate_size`, from `codec.tf.blk.0.ffn_gate`).

## irodori-tts

From the checkpoint's `model.safetensors` metadata (`config_json`, `text_encoder_config_json`), its `tokenizer/` and
model card, the DACVAE codec (Aratako/Semantic-DACVAE-Japanese-32dim at its pin), the official runtime at the commit
the converter's environment pins (`SamplingRequest`'s defaults in `irodori_tts/inference_runtime.py`) and
Irodori-TTS-Server@61012c760f22f7b4a6c21c5c5f8f9e148120b6f9 for the speed.

| Key | Type | Meaning | Source |
|---|---|---|---|
| `irodori-tts.flow` | string | `meanflow` or `rf_velocity` | `config_json` `flow_parameterization` (`rf_velocity` when absent, the config's default) |
| `irodori-tts.latent_dim` | u32 | the codec latent's channels, the DiT's and the speaker encoder's input | `config_json` `latent_dim`; the converter checks the codec's `codebook_dim` equals it |
| `irodori-tts.norm_eps` | f32 | | `config_json` `norm_eps` |
| `irodori-tts.rope_theta` | f32 | the RoPE base of the speaker encoder and the DiT (10000) | the official `precompute_freqs_cis()` default, which both keep |
| `irodori-tts.text.hidden_size`, `num_heads`, `num_layers` | u32 | ModernBERT-ja | `text_encoder_config_json` `hidden_size`, `num_attention_heads`, `num_hidden_layers` |
| `irodori-tts.text.norm_eps` | f32 | | `norm_eps` |
| `irodori-tts.text.window` | u32 | the tokens a local layer reaches on either side | `local_attention / 2` |
| `irodori-tts.text.layer_global` | [i32] | 1 for a layer with global attention | `layer_types`, `full_attention` |
| `irodori-tts.text.rope_theta_global`, `rope_theta_local` | f32 | | `rope_parameters.full_attention.rope_theta`, `rope_parameters.sliding_attention.rope_theta` |
| `irodori-tts.text.dim` | u32 | the text condition's channels | `config_json` `text_dim` |
| `irodori-tts.text.max_tokens` | u32 | the longest text (256) | `config_json` `max_text_len` |
| `irodori-tts.speaker.dim`, `num_layers`, `num_heads`, `patch_size` | u32 | | `config_json` `speaker_dim`, `speaker_layers`, `speaker_heads`, `speaker_patch_size` |
| `irodori-tts.duration.num_layers` | u32 | | `duration_layers` |
| `irodori-tts.duration.null_speaker` | bool | whether the file holds the duration predictor's null speaker, which the voice `none` speaks with without a reference | true when the checkpoint holds `duration_predictor.null_speaker` |
| `irodori-tts.caption.dim` | u32 | the caption condition's channels, which the duration predictor's caption takes | `config_json` `caption_dim`, or `text_dim` where it is absent, as `ModelConfig.caption_dim_resolved` gives it |
| `irodori-tts.caption_condition` | bool | whether the file holds the caption's encoder: its projector and norm and the DiT's caption keys and values | `config_json` `use_caption_condition`; the converter checks that the caption shares the text's tokenizer, `<s>` and ModernBERT, and that the duration predictor takes it with `adarn_zero` and `masked_mean` |
| `irodori-tts.caption.max_tokens` | u32 | the longest caption (512), `<s>` included; present when `caption_condition` is true | `config_json` `max_caption_len` |
| `irodori-tts.dit.dim`, `num_layers`, `num_heads`, `timestep_dim` | u32 | | `model_dim`, `num_layers`, `num_heads`, `timestep_embed_dim` |
| `irodori-tts.sampler.default_steps` | u32 | the default of the sampler's steps (4 or 40) | the runtime's `num_steps` default, 4 for MeanFlow and 40 otherwise |
| `irodori-tts.sampler.cfg_text` | f32 | present when `flow` is `rf_velocity` (3.0) | `SamplingRequest.cfg_scale_text` |
| `irodori-tts.sampler.cfg_speaker` | f32 | the same (5.0) | `SamplingRequest.cfg_scale_speaker` |
| `irodori-tts.sampler.cfg_min_t`, `cfg_max_t` | f32 | the same (0.5 and 1.0) | `SamplingRequest.cfg_min_t`, `cfg_max_t` |
| `irodori-tts.sampler.cfg_caption` | f32 | present when `flow` is `rf_velocity` and `caption_condition` is true (3.0) | `SamplingRequest.cfg_scale_caption` |
| `irodori-tts.sampler.speaker_kv_min_t` | f32 | present when `flow` is `rf_velocity`: the time below which a request's `speaker_kv_scale` stops (0.9) | the runtime's `synthesize()`, which takes 0.9 when a request names none; the converter checks its source says so |
| `irodori-tts.length.min_seconds`, `max_seconds` | f32 | the shortest and the longest speech (0.5 and 30) | `SamplingRequest.min_seconds`, `max_seconds` |
| `irodori-tts.length.min_speed`, `max_speed` | f32 | the speed's bounds (0.25 and 4) | Irodori-TTS-Server's speed bounds, OpenAI's |
| `irodori-tts.reference.max_seconds` | f32 | the longest reference recording (120) | `config_json` `ref_max_seconds` |
| `irodori-tts.reference.lufs` | f32 | the loudness a reference is brought to (−16) | `SamplingRequest.ref_normalize_db` |
| `irodori-tts.tail.window` | u32 | the frames over which the latent's tail is found flat (20) | `SamplingRequest.tail_window_size` |
| `irodori-tts.tail.std_threshold`, `mean_threshold` | f32 | (0.05 and 0.1) | `SamplingRequest.tail_std_threshold`, `tail_mean_threshold` |
| `irodori-tts.tokenizer.tokens` | [string] | the Unigram pieces | `tokenizer/tokenizer.json` `model.vocab` |
| `irodori-tts.tokenizer.scores` | [f64] | their scores | the same |
| `irodori-tts.tokenizer.added_ids` | [i32] | | `added_tokens` |
| `irodori-tts.tokenizer.bos_id` | u32 | | `tokenizer_config.json` `bos_token` |
| `irodori-tts.tokenizer.unknown_id` | u32 | | `model.unk_id` |
| `irodori-tts.codec.hop_length` | u32 | samples per latent frame | the DACVAE's `hop_length`; the converter checks it is the product of the encoder's rates |
| `irodori-tts.codec.encoder_rates`, `decoder_rates` | [i32] | | the DACVAE's `encoder_rates`, `decoder_rates` |
| `irodori-tts.codec.sha256` | string | the codec's identity, which voice files carry | the SHA-256 the converter computes over the official codec's tensors (below) |

The codec's hash is SHA-256 over the tensors of `weights.pth` as `DACVAE.load()` gives them, before the weight
normalization is folded: for each tensor in ascending order of its name, the name in UTF-8, a 0 byte, the number of
dimensions as a little-endian u32, each dimension as a little-endian u64, and the values as little-endian float32 in
row-major order. Every conversion of the same official codec carries the same hash, whatever type it stores:
Semantic-DACVAE-Japanese-32dim at its pin is `67cb1a241c3b75d7c5f46c966fca711e7962422da98f7f32ee2863d39dbe794d`.

Tensors, with X = `text.num_layers`, S = `speaker.num_layers`, D = `duration.num_layers`, N = `dit.num_layers`, E and
F the lengths of `codec.encoder_rates` and `codec.decoder_rates`:

- `text.embd`, `text.embd_norm`, `text.blk.{0..X-1}.` with `attn_q`, `attn_k`, `attn_v`, `attn_out`, `ffn_norm`,
  `ffn_act`, `ffn_gate`, `ffn_down`, and `attn_norm` for every layer but the first, which ModernBERT does not
  normalize; `text.final_norm`, `text.proj.{weight,bias}`, `text.proj.res_norm`, `text.proj.res_up.{weight,bias}`,
  `text.proj.res_down.{weight,bias}`, `text.norm`;
- when `caption_condition` is true, the caption's projector of ModernBERT's output and its norm, as the text's:
  `caption.proj.{weight,bias}`, `caption.proj.res_norm`, `caption.proj.res_up.{weight,bias}`,
  `caption.proj.res_down.{weight,bias}`, `caption.norm`;
- `speaker.in_proj.{weight,bias}`, `speaker.blk.{0..S-1}.` with `attn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_o`,
  `attn_gate`, `q_norm`, `k_norm`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down`; `speaker.norm`;
- `duration.in_proj.{weight,bias}`, `duration.blk.{0..D-1}.` with `norm`, `mod.{weight,bias}`,
  `caption_mod.{weight,bias}`, `ffn_gate`, `ffn_up`, `ffn_down`; `duration.out_norm`, `duration.out_proj.{weight,bias}`,
  `duration.null_caption`, and `duration.null_speaker` when `duration.null_speaker` is true;
- `dit.cond.{0,1,2}`, and `dit.delta_cond.{0,1,2}` when `flow` is `meanflow`; `dit.in_proj.{weight,bias}`,
  `dit.blk.{0..N-1}.` with `attn_q`, `attn_k`, `attn_v`, `attn_o`, `attn_gate`, `attn_k_text`, `attn_v_text`,
  `attn_k_speaker`, `attn_v_speaker`, `attn_k_caption` and `attn_v_caption` when `caption_condition` is true,
  `q_norm`, `k_norm`, `ffn_gate`, `ffn_up`, `ffn_down`, and for `attn_ada` and `ffn_ada` each of `shift`, `scale`,
  `gate` with `down`, `up.weight`, `up.bias`; `dit.out_norm`, `dit.out_proj.{weight,bias}`;
- `codec.enc.conv_in.{weight,bias}`, `codec.enc.blk.{0..E-1}.` with `res.{0,1,2}.` (`snake1.{alpha,inv_alpha}`,
  `conv1.{weight,bias}`, `snake2.{alpha,inv_alpha}`, `conv2.{weight,bias}`), `snake.{alpha,inv_alpha}`,
  `down.first`, `down.second`, `down.bias`; `codec.enc.snake.{alpha,inv_alpha}`, `codec.enc.conv_out.{weight,bias}`,
  `codec.bottleneck.mean.{weight,bias}`;
- `codec.dec.in_proj.{weight,bias}`, `codec.dec.conv_in.{weight,bias}`, `codec.dec.blk.{0..F-1}.` with
  `snake.{alpha,inv_alpha}`, `up.{weight,bias}` and `res.{0,1,2}.` as in the encoder;
  `codec.dec.out_snake.{alpha,inv_alpha}`, `codec.dec.conv_out.{weight,bias}`.

The widths that no key gives are the feed-forward widths of ModernBERT (from `text.blk.0.ffn_act`), of its projectors
(`text.proj.res_up.weight`, `caption.proj.res_up.weight`), of the speaker encoder (`speaker.blk.0.ffn_gate`) and of the DiT (`dit.blk.0.ffn_gate`),
the duration predictor's width (`duration.in_proj.weight`), the rank of the DiT's AdaLN
(`dit.blk.0.attn_ada.shift.down`), and the codec's first, latent and decoder widths (`codec.enc.conv_in.weight`,
`codec.enc.conv_out.weight`, `codec.dec.conv_in.weight`). The codec's strides are even, and each of
`codec.encoder_rates` and `codec.decoder_rates` multiplies to `codec.hop_length`.

A file of layout 1, which releases from 0.7.0 wrote, is read as one of layout 2 with `duration.null_speaker` and
`caption_condition` false, `caption.dim` equal to `text.dim`, the width layout 1 gave the duration predictor's caption,
and, for RF, `sampler.speaker_kv_min_t` 0.9, the runtime's at the commit every layout 1 file was converted from. Its
tensors are then the ones layout 1 held. The model information shows its own layout, and its metadata, in the
information and in `speech info --meta`, lists the keys the upgrade sets.

Constants that stay in the C++, since they are not the model's: the decoder's first window of 12 frames and the bounds
of the later ones, 24 and 48 frames with 0.1 s in hand, which are how speech.cpp streams a latent the official runtime
decodes whole; the encoder's and decoder's margins of 8 and 10
frames, which follow from the codec's architecture; SentencePiece's penalty for an unknown piece; and the widths
DACVAE fixes in its code rather than in its configuration, 7 for the first and the last convolution and a residual
unit's first, 3 for the encoder's last, and 1 for a residual unit's second and the decoder's input projection.

## fastconformer

From the `.nemo` checkpoint as NeMo 3.0.0 restores it (`model.cfg`, the featurizer, the encoder, the decoding and the
SentencePiece model), NeMo's own constants where it has them, and the model card for the languages and the license.

| Key | Type | Meaning | Source |
|---|---|---|---|
| `fastconformer.frontend.n_fft` | u32 | | featurizer `n_fft` |
| `fastconformer.frontend.hop_length` | u32 | samples per mel frame | featurizer `hop_length` |
| `fastconformer.frontend.n_mels` | u32 | the mel bins, also the subsampling's input width | featurizer `nfilt` |
| `fastconformer.frontend.preemphasis` | f32 | | featurizer `preemph` |
| `fastconformer.frontend.log_guard` | f32 | | featurizer `log_zero_guard_value` |
| `fastconformer.frontend.std_guard` | f32 | (1e-5) | `CONSTANT` in NeMo's `features.py` |
| `fastconformer.encoder.d_model`, `num_layers`, `num_heads`, `conv_kernel` | u32 | | `cfg.encoder.d_model`, `n_layers`, `n_heads`, `conv_kernel_size` |
| `fastconformer.encoder.subsampling_factor` | u32 | a power of two | `cfg.encoder.subsampling_factor` |
| `fastconformer.encoder.norm_eps` | f32 | | the layers' `norm_out.eps` |
| `fastconformer.encoder.pos_base` | f32 | (10000) | `INF_VAL` of NeMo's `multi_head_attention.py` |
| `fastconformer.encoder.xscale` | f32 | | `encoder.xscale`, or 1 when it is None |
| `fastconformer.encoder.ff_factor` | f32 | | the layers' `fc_factor` |
| `fastconformer.encoder.use_bias` | bool | whether the linear layers and pointwise convolutions have biases | the layers' `use_bias` |
| `fastconformer.encoder.attention` | string | `rel_pos` or `rel_pos_local_attn` | `encoder.self_attention_model` |
| `fastconformer.encoder.attention_context` | u32 | present for `rel_pos_local_attn`: frames on either side | `encoder.att_context_size[0]` |
| `fastconformer.encoder.global_tokens` | u32 | present for `rel_pos_local_attn` | `encoder.global_tokens` |
| `fastconformer.decoder.kind` | string | `tdt` or `rnnt` | the decoding `transcribe()` runs (`GreedyBatchedTDTInfer` or `BeamRNNTInfer`) |
| `fastconformer.decoder.blank_id` | u32 | | `decoding.blank_id` |
| `fastconformer.decoder.prediction_layers` | u32 | the prediction network's LSTM layers | `decoder.pred_rnn_layers` |
| `fastconformer.decoder.tdt.durations` | [i32] | present for `tdt` | `decoding.cfg.durations` |
| `fastconformer.decoder.tdt.max_symbols` | u32 | present for `tdt`: tokens on one frame at most | the decoding's `max_symbols` |
| `fastconformer.decoder.rnnt.beam_size` | u32 | present for `rnnt` | the beam search's `beam_size` |
| `fastconformer.decoder.rnnt.score_norm` | bool | present for `rnnt` | `score_norm` |
| `fastconformer.decoder.rnnt.max_target_ratio` | f32 | present for `rnnt` | `alsd_max_target_length` |
| `fastconformer.decoder.rnnt.max_symbols` | u32 | present for `rnnt`: tokens on one frame at most in the greedy decoding a request may choose (`decoding`); added in layout 2, and given as 10 to a file of layout 1 | the decoding configuration's `greedy.max_symbols`, or `greedy.max_symbols_per_step` where it has none, as NeMo's `RNNTDecoding` reads it |
| `fastconformer.segment.separators` | [string] | the marks that end a segment where a word ends, as NeMo ends one | the decoding's `segment_seperators`, or NeMo's default `.`, `?`, `!` when the checkpoint sets none (none of the three does) |
| `fastconformer.segment.breaks` | [string] | the marks that end a segment wherever they stand: `。`, `？`, `！`, `?`, `!` for a model whose languages are written without spaces, which NeMo's word ends never cut, and none otherwise | speech.cpp's own, from the model's languages |
| `fastconformer.tokenizer.tokens` | [string] | the SentencePiece pieces in id order | the tokenizer's model proto |
| `fastconformer.tokenizer.unknown_id` | u32 | | `unk_id()` |
| `fastconformer.tokenizer.unknown_surface` | string | | `trainer_spec.unk_surface` |
| `fastconformer.tokenizer.strip_leading_space` | bool | whether the first "▁" of the text is dropped | `normalizer_spec.add_dummy_prefix or remove_extra_whitespaces` |
| `fastconformer.tokenizer.punctuation` | [string] | the marks before which the decoding removes a space | `decoding.supported_punctuation` |

An encoder frame lasts `frontend.hop_length × encoder.subsampling_factor / speech.sample_rate` (160 × 8 / 16000 =
0.08 s).

Tensors, with L = `encoder.num_layers`, K = log2(`encoder.subsampling_factor`) and P = `decoder.prediction_layers`:

- `frontend.window`, `frontend.filterbank`;
- `sub.conv.0.{weight,bias}`, `sub.conv.{1..K-1}.` with `dw.{weight,bias}` and `pw.{weight,bias}`,
  `sub.out.{weight,bias}`;
- `blk.{0..L-1}.` with `ff1_norm.{weight,bias}`, `ff2_norm.{weight,bias}`, `attn_norm.{weight,bias}`,
  `conv_norm.{weight,bias}`, `out_norm.{weight,bias}`, `attn_pos.weight`, `attn_pos_bias_u`, `attn_pos_bias_v`,
  `conv_dw.{weight,bias}`, and `ff1_up`, `ff1_down`, `ff2_up`, `ff2_down`, `attn_q`, `attn_k`, `attn_v`, `attn_out`,
  `conv_pw1_a`, `conv_pw1_gate` and `conv_pw2`, each a `.weight` and, when `encoder.use_bias` is true, a `.bias`;
- `pred.embed.weight`, `pred.lstm.{0..P-1}.` with `ih.weight`, `hh.weight`, `bias`;
- `joint.enc.{weight,bias}`, `joint.pred.{weight,bias}`, `joint.out.{weight,bias}`; `joint.out` has
  `decoder.blank_id` + 1 outputs, and as many more as `decoder.tdt.durations` has entries for `tdt`.

The widths that no key gives are the window's length (from `frontend.window`, at most `frontend.n_fft`), the
subsampling's channels (`sub.conv.0.weight`), the feed-forward width (`blk.0.ff1_up.weight`), and the widths of the
prediction network (`pred.embed.weight`) and the joint (`joint.enc.weight`). The subsampling's convolutions are 3 wide,
NeMo's default, which the C++ pads by 1 on either side; `encoder.d_model` is even, and `decoder.tdt.durations` is
not empty.

## qwen3-asr

From the checkpoint's `config.json` (`thinker_config`, its `audio_config` and `text_config`), `preprocessor_config.json`,
`generation_config.json`, `chat_template.json`, `vocab.json`, `merges.txt` and `tokenizer_config.json`, transformers
5.18's Qwen3-ASR (the feature extractor, the encoder and its language tags) and qwen-asr 0.0.6's `inference/utils.py`
(the limits of the audio, the forced language and the parse of the output), both pinned by the converter's `uv.lock`
([ADR 0018](adr/0018-qwen3-asr-runs-in-speech-cpp-checked-against-its-windowed-encoder.md)).

| Key | Type | Meaning | Source |
|---|---|---|---|
| `qwen3-asr.language_names` | [string] | the name the forced language's prefill writes for each language of `general.languages`, aligned with it (`Cantonese` for `yue`), and the one the model writes when the language is left to it, which the result gives as its tag; ASCII, with no capital but the first letter, the form in which the parse compares a name the model writes | transformers' `LANGUAGE_CODE_TO_NAME`, the names of qwen-asr's `SUPPORTED_LANGUAGES` |
| `qwen3-asr.frontend.n_fft`, `hop_length`, `n_mels` | u32 | | `preprocessor_config.json` `n_fft`, `hop_length`, `feature_size` |
| `qwen3-asr.frontend.log_floor`, `dynamic_range`, `log_offset`, `log_divisor` | f32 | the log10's guard, the range kept below the utterance's maximum, and the shift and scale after it (1e-10, 8, 4, 4) | the feature extractor's code |
| `qwen3-asr.audio.min_samples` | u32 | an utterance shorter is padded with zeros to it (8000, 0.5 s) | qwen-asr's `MIN_ASR_INPUT_SECONDS`, the extractor's `min_length` |
| `qwen3-asr.audio.max_samples` | u32 | longer audio is split (19,200,000, 1200 s) | qwen-asr's `MAX_ASR_INPUT_SECONDS` |
| `qwen3-asr.audio.split_search_samples`, `split_window_samples` | u32 | how far on either side of a cut the split looks for the quietest window, and the window (80,000 and 1,600: 5 s and 0.1 s) | `split_audio_into_chunks()`'s `search_expand_sec` and `min_window_ms` |
| `qwen3-asr.encoder.d_model`, `num_layers`, `num_heads`, `ffn_dim` | u32 | | `audio_config` `d_model`, `encoder_layers`, `encoder_attention_heads`, `encoder_ffn_dim` |
| `qwen3-asr.encoder.chunk_frames` | u32 | the frames of a chunk the convolutions take at once (100) | 2 × `n_window` |
| `qwen3-asr.encoder.window_frames` | u32 | the frames of a window the layers attend within (800) | `n_window_infer` |
| `qwen3-asr.encoder.norm_eps` | f32 | the LayerNorms' epsilon (1e-5) | the encoder's LayerNorms |
| `qwen3-asr.encoder.max_timescale` | f32 | the positions' sinusoids (10000) | `SinusoidsPositionEmbedding` |
| `qwen3-asr.decoder.hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `vocab_size`, `max_position_embeddings` | u32 | | `text_config` |
| `qwen3-asr.decoder.rms_norm_eps`, `rope_theta` | f32 | | `text_config` |
| `qwen3-asr.prompt.before_context`, `before_audio`, `after_audio` | string | the chat template's text before the context, between it and the audio's tokens, and after them | `chat_template.json` filled as qwen-asr fills it, split at the context and the audio token |
| `qwen3-asr.prompt.audio_token` | string | the added token whose rows the projector's output replaces (`<\|audio_pad\|>`) | the processor's `audio_token` |
| `qwen3-asr.prompt.language_prefix`, `asr_text` | string | the forced language's prefill around the name (`language `, `<asr_text>`), and around the name the model writes in its output, after which its text begins; the prefix in ASCII | qwen-asr's `_LANG_PREFIX` and `_ASR_TEXT_TAG` |
| `qwen3-asr.output.repetition_threshold`, `repetition_max_period` | u32 | the repetition fix of the parse: runs and patterns repeated past the threshold kept once (20, 20) | `detect_and_fix_repetitions()` |
| `qwen3-asr.generation.eos_ids` | [i32] | the tokens that end the decoding | `generation_config.json` `eos_token_id` |
| `qwen3-asr.generation.max_new_tokens` | u32 | the most tokens a recognition writes (4096) | the model's `generate()` and qwen-asr's vLLM backend ([ADR 0018](adr/0018-qwen3-asr-runs-in-speech-cpp-checked-against-its-windowed-encoder.md)) |
| `qwen3-asr.tokenizer.tokens`, `merges` | [string] | the BPE tokens in id order, the added ones included, and the merges | `vocab.json`, `tokenizer_config.json`'s `added_tokens_decoder`, `merges.txt` |
| `qwen3-asr.tokenizer.added_ids`, `special_ids` | [i32] | the added tokens, at which a text is split before the BPE, and the special ones among them, which decoding drops | `added_tokens_decoder` |

Tensors, with E = `encoder.num_layers` and D = `decoder.num_hidden_layers`:

- `frontend.window`, `frontend.filterbank`;
- `enc.conv.{1,2,3}.{weight,bias}`, `enc.conv_out.weight`;
- `enc.blk.{0..E-1}.` with `attn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_out`, `ffn_norm`, `ffn_up` and `ffn_down`, each a
  `.weight` and a `.bias`; `enc.norm.{weight,bias}`;
- `proj.{1,2}.{weight,bias}`, the projector;
- `dec.token_embd`, which is also the output matrix, the checkpoint's being tied to it;
- `dec.blk.{0..D-1}.` with `attn_norm`, `ffn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_o`, `attn_q_norm`, `attn_k_norm`,
  `ffn_gate`, `ffn_up` and `ffn_down`; `dec.norm`.

The width no key gives is the convolutions' channels, from `enc.conv.1.weight` (480). The convolutions are 3 × 3 with
a stride of 2 and a padding of 1, which the official module fixes in its code and the converter checks. The parse's
mark of audio without speech, `language none` in any case before `<asr_text>`, is the text of qwen-asr's parse, which
the converter finds in its code, and no key holds it.

## Voice files

A voice file of Irodori-TTS is a GGUF file of its own. Every voice file this release makes, of one reference at the
model's loudness too, is of layout 2:

| Key | Type | Meaning |
|---|---|---|
| `general.architecture` | string | `irodori-tts-voice` |
| `speech.layout` | u32 | 2 |
| `speech.requires` | string | `0.8.0` |
| `irodori-tts-voice.source` | string | `references` or `embedding`: what the voice is made of |
| `irodori-tts-voice.codec_sha256` | string | the `irodori-tts.codec.sha256` of the model that encoded it; a model with another hash refuses it. Present for `references`, as the keys below to `device_kind` |
| `irodori-tts-voice.references.seconds` | [f32] | each reference recording's length, in the order they are joined |
| `irodori-tts-voice.references.sample_rates` | [i32] | each recording's rate, before it was resampled, aligned |
| `irodori-tts-voice.normalized` | bool | whether each recording was brought to a loudness before it was encoded |
| `irodori-tts-voice.lufs` | f32 | that loudness in LUFS; present when `normalized` is true |
| `irodori-tts-voice.device_kind` | string | `cpu`, `gpu` or `igpu`: the kind of device that encoded it |
| `irodori-tts-voice.model` | string | the `general.source.url` of the model the embedding was made for; present for `embedding`. Another model refuses it |

and one tensor: for `references`, `latent`, F32 with ne = [`latent_dim`, frames], the recordings' latents joined,
`latent_dim` being the model's; for `embedding`, `speaker`, F32 with ne = [`speaker.dim`, tokens]. A voice file of
another codec or another model is refused as the caller's mistake before its tensor is checked. A voice file
of layout 1, which releases from 0.7.0 wrote with one reference brought to the model's -16 LUFS, is read as one of
layout 2 that says so, and releases before 0.8.0 refuse layout 2 naming 0.8.0. Voice files made before 0.7.0 have no
`speech.layout` and are refused; they are made again from their WAVE files.
