"""Runs Qwen3-ASR on the CPU in float32 and saves the tensors the C++ port is checked against.

usage: uv run python dump.py <model> <out dir> [input...]

<model> is Qwen3-ASR-0.6B or Qwen3-ASR-1.7B, and each input, a name from inputs.py (when none is given, every input
inputs.py gives the model), goes to <out dir>/<model>/<input>/.

The model is transformers 5.18's own Qwen3ASRForConditionalGeneration with the pinned checkpoint's configuration and
weights. Its encoder attends within windows of 104 tokens (8 s) on every attention path, as qwen-asr's vLLM backend and
its FlashAttention 2 path do, where qwen-asr 0.0.6 run on the CPU attends over the whole utterance, a fallback that
drops the windows the model was trained with. What transformers does not do comes from qwen-asr's own code, as its
transcribe() runs it: the normalization of the audio, the split of audio over 1200 s at its quietest point within 5 s,
the prompt with its system turn and the forced language, and the parse of the output with its repetition fix. Decoding
is greedy with the checkpoint's generation_config.json and at most 4096 new tokens, the default of the model's
generate() and of qwen-asr's vLLM backend.

Each input is run once without a language or a prompt, once with its language forced, and both again with its
prompt (qwen-asr's context, which goes into the system turn): the variants auto, forced, auto-prompt and
forced-prompt. Integers are int32, the rest float32. The stages of the audio, in the order data flows:

  audio            the samples after qwen-asr's normalization, [N]
  features         the log-mel features of the valid frames, ⌊N / 160⌋ of them, [frames, mels]
  conv1, conv2, conv3
                   the output of each convolution after its GELU, for each chunk of 100 frames, the last one padded
                   with zeros to 100 frames, [chunks, channels, mels / 2^k, 50, 25 or 13]. transformers pads an
                   utterance shorter than one chunk as well, where qwen-asr runs it unpadded and its encoder gives
                   another output (inputs.py's cut).
  encoder_input    conv_out on conv3 plus the sinusoid of each token's position within its chunk, the tokens of the
                   padding dropped, [tokens, d_model]
  windows          where the attention's windows start and end in tokens, [windows + 1]: rows windows[i] to
                   windows[i + 1] of encoder_input, encoder_layers and encoder_output are window i
  encoder_layers   the output of each layer, [layers, tokens, d_model]
  encoder_output   after ln_post, [tokens, d_model]
  audio_embeds     the projector's output, which replaces the prompt's audio tokens, [tokens, hidden]

and under <variant>/, the decoder's:

  prompt_ids       the prompt's token ids, the audio tokens expanded, the forced language's prefill included, [n]
  embeds           the decoder's input: the token embeddings with audio_embeds in the rows of the audio tokens, [n, hidden]
  prefill_hidden   the decoder's last hidden state, after its final norm, for every row of the prompt, [n, hidden]
  prefill_logits   the logits of the prompt's last 4 rows, [4, vocabulary]; the last gives the first new token
  ids              the generated ids, the stop token included when one ended the decoding, [m]
  step_top_ids     the 8 highest logits of each step and their ids, [m, 8], to tell how close the greedy choice was
  step_top_logits
  raw.txt          the ids decoded without special tokens, as qwen-asr decodes them
  text.txt         the text qwen-asr's parse_asr_output() gives
  meta.json        the request, the stop and both texts

meta.json beside the stages holds the versions, the pins, the FLEURS utterances and the lengths. An input that
qwen-asr splits (over 1200 s) is dumped only to audio, split (where each part starts and ends in samples) and, under
auto/part<i>/, each part's prompt_ids, ids, raw.txt and text.txt, with the joined text in auto/text.txt.
"""

import argparse
import copy
import importlib.metadata
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import (AutoTokenizer, GenerationConfig, Qwen3ASRConfig, Qwen3ASRFeatureExtractor,
                          Qwen3ASRForConditionalGeneration)
from transformers.models.qwen3_asr.modeling_qwen3_asr import _get_feat_extract_output_lengths

from inputs import INPUTS, SAMPLE_RATE, read
from official_utils import qwen_asr_utils
from pins import FLEURS, MODELS, snapshot

MAX_NEW_TOKENS = 4096
VARIANTS = {"auto": (False, False), "forced": (True, False), "auto-prompt": (False, True), "forced-prompt": (True, True)}
TOP = 8
PREFILL_LOGIT_ROWS = 4
VERSIONS = {name: importlib.metadata.version(name) for name in ("transformers", "qwen-asr", "torch")}


utils = qwen_asr_utils()

# The checkpoint names its weights as qwen-asr's model does, under `thinker.`; transformers' -hf repositories hold the
# same tensors under the names of transformers' implementation.
RENAMES = (
    ("thinker.audio_tower.proj1.", "model.multi_modal_projector.linear_1."),
    ("thinker.audio_tower.proj2.", "model.multi_modal_projector.linear_2."),
    ("thinker.audio_tower.", "model.audio_tower."),
    ("thinker.model.", "model.language_model."),
    ("thinker.lm_head.", "lm_head."),
)
AUDIO_KEYS = ("num_mel_bins", "encoder_layers", "encoder_attention_heads", "encoder_ffn_dim", "d_model",
              "activation_function", "scale_embedding", "n_window", "n_window_infer", "output_dim", "downsample_hidden_size")
TEXT_KEYS = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
             "num_key_value_heads", "head_dim", "hidden_act", "max_position_embeddings", "rms_norm_eps", "attention_bias",
             "tie_word_embeddings")


def rename(key):
    for old, new in RENAMES:
        if key.startswith(old):
            return new + key[len(old):]
    raise SystemExit(f"the checkpoint holds {key}, which no name of transformers' implementation matches")


def load(pin, folder):
    """transformers' model with the checkpoint's configuration and weights, its tokenizer, chat template, generation
    configuration and feature extractor."""
    thinker = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))["thinker_config"]
    audio, text = thinker["audio_config"], thinker["text_config"]
    # transformers' text model runs plain RoPE where qwen-asr's runs interleaved M-RoPE; with the same position in all
    # three sections, as get_rope_index() gives every token of an audio-and-text prompt, the two are the same.
    assert sum(text["rope_scaling"]["mrope_section"]) * 2 == text["head_dim"]
    config = Qwen3ASRConfig(
        audio_config={"model_type": "qwen3_asr_encoder", **{key: audio[key] for key in AUDIO_KEYS},
                      "max_position_embeddings": audio["max_source_positions"]},
        text_config={"model_type": "qwen3", **{key: text[key] for key in TEXT_KEYS},
                     "rope_parameters": {"rope_type": "default", "rope_theta": text["rope_theta"]}},
        audio_token_id=thinker["audio_token_id"],
        tie_word_embeddings=text["tie_word_embeddings"],
    )
    model = Qwen3ASRForConditionalGeneration(config).float().eval()
    state = {}
    for name in pin["weights"]:
        state.update((rename(key), value) for key, value in load_file(os.path.join(folder, name)).items())
    # The checkpoint stores the tied output matrix twice; load_state_dict() copies the bfloat16 tensors into the
    # float32 parameters.
    assert torch.equal(state["lm_head.weight"], state["model.language_model.embed_tokens.weight"])
    model.load_state_dict(state, strict=True)
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight

    preprocessor = json.load(open(os.path.join(folder, "preprocessor_config.json"), encoding="utf-8"))
    assert preprocessor["feature_extractor_type"] == "WhisperFeatureExtractor"
    extractor = Qwen3ASRFeatureExtractor(
        feature_size=preprocessor["feature_size"], sampling_rate=SAMPLE_RATE, hop_length=preprocessor["hop_length"],
        n_fft=preprocessor["n_fft"], padding_value=preprocessor["padding_value"], dither=preprocessor["dither"],
        n_window=audio["n_window"])
    tokenizer = AutoTokenizer.from_pretrained(folder)
    template = json.load(open(os.path.join(folder, "chat_template.json"), encoding="utf-8"))["chat_template"]
    generation = GenerationConfig.from_pretrained(folder)
    assert not generation.do_sample
    return model, extractor, tokenizer, template, generation


def prompt_text(tokenizer, template, context, language):
    """The prompt as qwen-asr's _build_messages() and _build_text_prompt() write it (inference/qwen3_asr.py:448-465):
    the chat template with a system turn holding the context, empty when there is none, the user's audio, the generation
    prompt, and a forced language as the prefill "language <Name><asr_text>"."""
    messages = [{"role": "system", "content": context or ""}, {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
    text = tokenizer.apply_chat_template(messages, chat_template=template, add_generation_prompt=True, tokenize=False)
    return text + f"language {language}<asr_text>" if language else text


def features_of(extractor, samples):
    """transformers' log-mel features of one utterance, padded to whole chunks of 100 frames, and their mask."""
    out = extractor(samples, sampling_rate=SAMPLE_RATE, padding=True, truncation=False, return_attention_mask=True,
                    return_tensors="pt")
    return out["input_features"], out["attention_mask"]


def audio_stages(model, features, mask):
    """Runs the encoder and the projector once with every stage recorded."""
    tower = model.model.audio_tower
    saved = {"conv": [], "layers": []}
    # Each convolution's output goes through F.gelu() in the encoder's forward; the hook applies the same.
    hooks = [conv.register_forward_hook(lambda m, a, o: saved["conv"].append(F.gelu(o)))
             for conv in (tower.conv2d1, tower.conv2d2, tower.conv2d3)]
    hooks.append(tower.layers[0].register_forward_pre_hook(lambda m, a: saved.update(encoder_input=a[0], windows=a[1])))
    hooks += [layer.register_forward_hook(lambda m, a, o: saved["layers"].append(o[0])) for layer in tower.layers]
    hooks.append(tower.ln_post.register_forward_hook(lambda m, a, o: saved.update(encoder_output=o)))
    audio_embeds = model.model.get_audio_features(features, mask).pooler_output
    for hook in hooks:
        hook.remove()
    stages = {f"conv{k + 1}": conv for k, conv in enumerate(saved["conv"])}
    stages.update(encoder_input=saved["encoder_input"], windows=saved["windows"], encoder_layers=torch.stack(saved["layers"]),
                  encoder_output=saved["encoder_output"], audio_embeds=audio_embeds)
    return stages


def decode(model, tokenizer, template, generation, features, mask, audio_tokens, context, language, record):
    """Runs one request through generate() as qwen-asr's transformers backend does, returning what it saves."""
    text = prompt_text(tokenizer, template, context, language)
    pad = tokenizer.audio_token
    # qwen-asr's processor expands the first audio token to one per encoder output (Qwen3ASRProcessor.__call__).
    ids = torch.tensor([tokenizer(text.replace(pad, pad * audio_tokens, 1))["input_ids"]])
    n = ids.shape[1]
    captured = {}

    def before_decoder(module, args, kwargs):
        captured.setdefault("embeds", kwargs["inputs_embeds"][0].clone())

    def after_decoder(module, args, kwargs, output):
        captured.setdefault("hidden", output.last_hidden_state[0].clone())

    def after_projector(module, args, output):
        captured.setdefault("audio_embeds", output.clone())

    hooks = []
    if record:
        decoder = model.model.language_model
        hooks = [decoder.register_forward_pre_hook(before_decoder, with_kwargs=True),
                 decoder.register_forward_hook(after_decoder, with_kwargs=True),
                 model.model.multi_modal_projector.register_forward_hook(after_projector)]
    config = copy.deepcopy(generation)
    config.update(max_new_tokens=MAX_NEW_TOKENS, output_logits=record, return_dict_in_generate=True)
    started = time.time()
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), input_features=features,
                         input_features_mask=mask, generation_config=config)
    seconds = time.time() - started
    for hook in hooks:
        hook.remove()
    generated = out.sequences[0, n:]
    stop_id = int(generated[-1]) if int(generated[-1]) in generation.eos_token_id else None
    assert stop_id is not None or generated.shape[0] == MAX_NEW_TOKENS
    raw = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    parsed_language, parsed_text = utils.parse_asr_output(raw, user_language=language)
    result = {
        "prompt_ids": ids[0], "ids": generated, "raw": raw, "text": parsed_text,
        "meta": {"language": language, "prompt": context or "", "prompt_tokens": n, "audio_tokens": audio_tokens,
                 "audio_start": int((ids[0] == tokenizer.convert_tokens_to_ids(pad)).nonzero()[0]),
                 "generated": int(generated.shape[0]), "stop": "eos" if stop_id is not None else "max_new_tokens",
                 "stop_id": stop_id, "seconds": round(seconds, 2), "raw": raw, "parsed_language": parsed_language,
                 "text": parsed_text},
    }
    if record:
        logits = torch.cat(out.logits)
        prefill_logits = model.lm_head(captured["hidden"][-PREFILL_LOGIT_ROWS:])
        torch.testing.assert_close(prefill_logits[-1], logits[0], rtol=0, atol=1e-4)
        top = logits.topk(TOP, dim=-1)
        result.update(embeds=captured["embeds"], prefill_hidden=captured["hidden"], prefill_logits=prefill_logits,
                      step_top_ids=top.indices, step_top_logits=top.values, audio_embeds=captured["audio_embeds"])
    return result


def save(folder, arrays, texts=None, meta=None):
    os.makedirs(folder, exist_ok=True)
    for key, value in arrays.items():
        array = value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        array = array.astype(np.int32 if array.dtype.kind in "iu" else np.float32)
        np.save(os.path.join(folder, f"{key}.npy"), np.ascontiguousarray(array))
    for key, value in (texts or {}).items():
        with open(os.path.join(folder, f"{key}.txt"), "w", encoding="utf-8", newline="") as f:
            f.write(value)
    if meta is not None:
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)


def dump_input(name, out):
    spec = INPUTS[name]
    samples, rows = read(name)
    # transcribe() normalizes each input (resampling, a peak above 1 brought to 1) and splits it into parts of at most
    # 1200 s.
    [audio] = utils.normalize_audios([(samples, SAMPLE_RATE)])
    parts = utils.split_audio_into_chunks(wav=audio, sr=SAMPLE_RATE, max_chunk_sec=utils.MAX_ASR_INPUT_SECONDS)
    language = utils.normalize_language_name(spec["language"])
    utils.validate_language(language)
    meta = {
        **VERSIONS, "model": {key: pin[key] for key in ("repository", "revision")}, "fleurs": FLEURS,
        "fleurs_config": spec["config"], "fleurs_utterances": [{"file": row[1], "transcription": row[2]} for row in rows],
        "cut": {"start": spec["start"], "seconds": spec["seconds"]} if "start" in spec else None,
        "input": name, "samples": int(audio.shape[0]), "seconds": audio.shape[0] / SAMPLE_RATE,
        "language": language, "prompt": spec["prompt"], "max_new_tokens": MAX_NEW_TOKENS,
        "eos": list(generation.eos_token_id), "attention": model.config._attn_implementation,
    }
    started = time.time()
    if len(parts) > 1:
        starts = [round(offset * SAMPLE_RATE) for _, offset in parts]
        split = starts + [int(audio.shape[0])]
        assert all(starts[k] + parts[k][0].shape[0] == split[k + 1] for k in range(len(parts) - 1))
        meta["split"] = split
        # qwen-asr pads a part shorter than 0.5 s with zeros to 0.5 s.
        meta["part_samples"] = [int(part.shape[0]) for part, _ in parts]
        save(out, {"audio": audio, "split": np.array(split, dtype=np.int64)})
        texts, languages = [], []
        for k, (part, _) in enumerate(parts):
            features, mask = features_of(extractor, part)
            audio_tokens = int(_get_feat_extract_output_lengths(mask.sum(-1), extractor.n_window)[0])
            result = decode(model, tokenizer, template, generation, features, mask, audio_tokens, None, None, record=False)
            save(os.path.join(out, "auto", f"part{k}"), {"prompt_ids": result["prompt_ids"], "ids": result["ids"]},
                 {"raw": result["raw"], "text": result["text"]}, result["meta"])
            texts.append(result["text"])
            languages.append(result["meta"]["parsed_language"])
            print(json.dumps({"part": k, **result["meta"]}, ensure_ascii=False), flush=True)
        # transcribe() joins the parts' texts without a separator and merges their languages.
        joined = {"text": "".join(texts), "language": utils.merge_languages(languages)}
        save(os.path.join(out, "auto"), {}, {"text": joined["text"]}, joined)
        meta["variants"] = {"auto": joined}
    else:
        [(part, _)] = parts
        features, mask = features_of(extractor, part)
        frames = int(mask.sum())
        stages = audio_stages(model, features, mask)
        tokens = stages["encoder_output"].shape[0]
        assert tokens == int(_get_feat_extract_output_lengths(mask.sum(-1), extractor.n_window)[0])
        save(out, {"audio": part, "features": features[0, :, :frames].T, **stages})
        windows = stages["windows"].tolist()
        meta.update(frames=frames, chunks=stages["conv1"].shape[0], tokens=tokens,
                    windows=[[a, b] for a, b in zip(windows[:-1], windows[1:])], variants={})
        for variant, (forced, prompted) in VARIANTS.items():
            result = decode(model, tokenizer, template, generation, features, mask, tokens,
                            spec["prompt"] if prompted else None, language if forced else None, record=True)
            assert torch.equal(result.pop("audio_embeds"), stages["audio_embeds"])
            save(os.path.join(out, variant),
                 {key: result[key] for key in ("prompt_ids", "embeds", "prefill_hidden", "prefill_logits", "ids",
                                               "step_top_ids", "step_top_logits")},
                 {"raw": result["raw"], "text": result["text"]}, result["meta"])
            meta["variants"][variant] = {key: result["meta"][key] for key in ("generated", "stop", "parsed_language", "text")}
            print(json.dumps({"variant": variant, **result["meta"]}, ensure_ascii=False), flush=True)
    meta["seconds_to_dump"] = round(time.time() - started, 1)
    save(out, {}, meta=meta)
    print(json.dumps({"input": name, "seconds_to_dump": meta["seconds_to_dump"]}), flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
parser.add_argument("inputs", nargs="*", metavar="input")
args = parser.parse_args()
if unknown := sorted(set(args.inputs) - set(INPUTS)):
    parser.error(f"inputs.py has no input {', '.join(unknown)}; it has {', '.join(INPUTS)}")

torch.set_grad_enabled(False)
pin = MODELS[args.model]
model, extractor, tokenizer, template, generation = load(pin, snapshot(pin))
for name in args.inputs or [name for name, spec in INPUTS.items() if args.model in spec.get("models", MODELS)]:
    dump_input(name, os.path.join(args.out_dir, args.model, name))
