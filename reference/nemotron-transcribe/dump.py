"""Runs Nemotron 3.5 Transcribe's official modules on the CPU in float32 and saves the tensors the C++ port is checked
against.

usage: uv run python dump.py <out dir> <case>... [--data <folder>] [--max-new-tokens N] [--layers N]

A case is <input>-<prompt>, an input of inputs.py read from the clips under <folder> (by default ~/speech-bench-data)
and a prompt of PROMPTS, and goes to <out dir>/<name>/<case>/, <name> being the model's. SALMAutomodel.generate() runs
the request as NVIDIA's vLLM plugin renders it (nemo/collections/speechlm2/vllm/salm/runtime_compat.py): the prompt's
text, one space and the audio locator as the user's turn, without a system prompt, formatted by the checkpoint's
nemotron3p5 format with reasoning off, which is asserted to give the ids of the tokenizer's own chat template; the audio
through the perception module's windowed inference, which generation always takes; and greedy decoding until the end of
the turn. The language model is Transformers' (official.py), and the decoding is the loop below in place of
Transformers' generate(), which NemotronH does not run from embeddings. Wrappers around the official calls record what
each took and gave. --layers keeps only the first N layers of the language model, to try the script on less memory; its
text means nothing.

audio.npy            the samples, [N]
features.npy         the preprocessor's log-mel features, which the encoder normalizes itself, [T, 128], T = N // 160 + 1,
                     the last zero
asr/<k>/             the ASR encoder on window k, as the official code encodes it (meta.json gives its frames):
  input.npy            the window's features, normalized per feature over the whole input, [W, 128]
  pre_encode.npy       the stacked frames' projection, before the embedding norm, [n, 1280]
  layers.npy           the embedding norm's and each layer's output, [33, n, 1280]
  output.npy           the final norm's output, [n, 1280]
diar_steps/<k>.npy   the speaker branch's probabilities from streaming step k, as forward_streaming_step() gave them
diar.npy             the speaker probabilities the encoder fuses, aligned to its frames, [F, 8]
asr.npy              the ASR encoder's kept frames of every window, [F, 1280]
fused.npy            the parallel expert encoder's output, [F, 1280]
audio_embeds.npy     the projection to the language model's width, [F, 2688], of which the language model takes the
                     first meta.json's audio_frames, the frames the encoder gives the input's length
prompt_ids.npy       the prompt's token ids, the audio locator's once, [P]
llm/
  input_embeds.npy     the language model's input, the audio's rows in place of the locator, [P + F - 1, 2688]
  prefill.npy          each layer's output over the input, [52, P + F - 1, 2688]
  router_logits.npy    each MoE layer's router logits over the input, before the sigmoid, [23, P + F - 1, 128]
  norm.npy             the final norm's output of the input's last row, [2688]
  logits.npy           the logits of each of the first steps, the first from the input's last row, [S0, vocab]
  decode.npy           each layer's output for each of the first decoded tokens, [S1, 52, 2688]
  top.npy, top_ids.npy the 16 largest logits of every step and their ids, [S, 16]
  tokens.npy           the generated ids, the end of turn included when it was reached, [S]
meta.json            the versions, the pin, the request, the windows, the lengths, the text and why decoding stopped
"""

import argparse
import importlib.metadata
import json
import os
import time

import numpy as np
import torch
import transformers
from transformers import GenerationConfig

import inputs
import official
from pins import AUTOMODEL_COMMIT, MODELS, NEMO, NEMO_COMMIT, snapshot

PROMPTS = {
    # The model card's recommended prompt for verbatim transcription.
    "verbatim": "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. "
    "Formatting: punctuation and capitalization. Output: transcript only.",
    # The model card's quick start.
    "plain": "Transcribe the following audio.",
    # The model card's contextual transcription, with terms the meeting holds.
    "context": "Transcribe with punctuation and capitalization. Context terms: FLEURS, Nemotron.",
    # The model card's prompt for who said what, with speaker tags.
    "speakers": "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal "
    "fillers. Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the "
    "transcription.",
}
# How many first steps keep their whole logits and every layer's output.
LOGIT_STEPS = 4
DECODE_STEPS = 3

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("cases", nargs="+")
parser.add_argument("--data", default=os.path.expanduser("~/speech-bench-data"))
parser.add_argument("--max-new-tokens", type=int, default=1024)
parser.add_argument("--layers", type=int)
args = parser.parse_args()
for case in args.cases:
    name, _, prompt = case.partition("-")
    if name not in inputs.INPUTS or prompt not in PROMPTS:
        raise SystemExit(f"{case} is not <input>-<prompt>; inputs: {', '.join(inputs.INPUTS)}, prompts: {', '.join(PROMPTS)}")

torch.set_grad_enabled(False)
MODEL = "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
PIN = MODELS[MODEL]
started = time.time()
model = official.load(snapshot(PIN), args.layers)
print(f"loaded in {time.time() - started:.0f} s", flush=True)
llm = model.llm
pee = model.perception.encoder
tokenizer = model.tokenizer


def save(folder, name, array):
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, name + ".npy"), np.ascontiguousarray(array))


class Record:
    """What the wrapped calls of one request took and gave."""

    def __init__(self):
        self.mel = None
        self.windows = None
        self.cores = None
        self.window_calls = []
        self.layer_outputs = []
        self.diar_steps = []
        self.fuse = None
        self.proj = None
        self.router = []
        self.blocks = []


rec = Record()


def hook(store):
    def run(module, inputs_, output):
        store(module, inputs_, output)
    return run


model.perception.preprocessor.register_forward_hook(hook(lambda m, i, o: setattr(rec, "mel", (o[0][0].T.clone(), int(o[1][0])))))
encoder = pee.asr_encoder
encoder.embed_norm.register_forward_hook(hook(lambda m, i, o: rec.layer_outputs.append(("pre_encode", i[0].clone(), o.clone()))))
for index, layer in enumerate(encoder.layers):
    layer.register_forward_hook(hook(lambda m, i, o, index=index: rec.layer_outputs.append((index, None, o.clone()))))
encoder.final_norm.register_forward_hook(hook(lambda m, i, o: rec.layer_outputs.append(("final", None, o.clone()))))
model.perception.proj.register_forward_hook(hook(lambda m, i, o: setattr(rec, "proj", o[0].clone())))
model.perception.register_forward_hook(hook(lambda m, i, o: setattr(rec, "audio_frames", int(o[1][0]))))
moe_layers = [i for i, kind in enumerate(llm.config.layers_block_type) if kind == "moe"]
for i in moe_layers:
    llm.model.layers[i].mixer.gate.register_forward_hook(hook(lambda m, i_, o: rec.router.append(o[0].clone())))
for i, block in enumerate(llm.model.layers):
    block.register_forward_hook(hook(lambda m, i_, o, i=i: rec.blocks.append((i, o[0].clone()))))

encode_windows = pee._encode_asr_windows


def wrapped_encode_windows(asr_signal, asr_length, windows):
    rec.windows = [tuple(int(v) for v in w) for w in windows]
    rec.asr_input = asr_signal[0].T.clone()
    cores = encode_windows(asr_signal, asr_length, windows)
    rec.cores = [c[0].T.clone() for c in cores]
    return cores


window_forward = pee._asr_window_forward


def wrapped_window_forward(rows, lengths):
    first = len(rec.layer_outputs)
    out = window_forward(rows, lengths)
    rec.window_calls.append((rows.clone(), lengths.clone(), out.clone(), rec.layer_outputs[first:]))
    return out


diar_step = pee._streaming_diar_step


def wrapped_diar_step(chunk, chunk_length, streaming_state, empty_preds, **kwargs):
    state, preds = diar_step(chunk, chunk_length, streaming_state, empty_preds, **kwargs)
    rec.diar_steps.append({"preds": preds[0].clone(), "frames": int(chunk.shape[1]), "length": int(chunk_length[0]),
                           **{k: int(v) if isinstance(v, (int, bool)) else v for k, v in kwargs.items()}})
    return state, preds


fuse = pee._fuse_diar_and_asr


def wrapped_fuse(asr_encoded, spk_targets, **kwargs):
    out = fuse(asr_encoded, spk_targets, **kwargs)
    rec.fuse = (asr_encoded[0].T.clone(), spk_targets[0].clone(), out[0].T.clone())
    return out


pee._encode_asr_windows = wrapped_encode_windows
pee._asr_window_forward = wrapped_window_forward
pee._streaming_diar_step = wrapped_diar_step
pee._fuse_diar_and_asr = wrapped_fuse


def greedy(inputs_embeds, attention_mask, generation_config, max_new_tokens):
    """Greedy decoding of one request from its embeddings, recording each step; the ids it generated, [1, S]."""
    if inputs_embeds.shape[0] != 1 or not bool(attention_mask.all()):
        raise SystemExit("the dump decodes one request without padding")
    eos = generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    rec.input_embeds = inputs_embeds[0].clone()
    rec.blocks.clear()
    rec.router.clear()
    out = llm(inputs_embeds=inputs_embeds, use_cache=True)
    rec.prefill = [o for _, o in sorted(rec.blocks, key=lambda b: b[0])]
    rec.prefill_router = list(rec.router)
    rec.norm = llm.model.norm_f(rec.prefill[-1][-1:])[0].clone()
    cache, logits = out.past_key_values, out.logits[0, -1]
    tokens, rec.logits, rec.decode, rec.top, rec.top_ids = [], [], [], [], []
    rec.stop = "max_new_tokens"
    while len(tokens) < max_new_tokens:
        if len(rec.logits) < LOGIT_STEPS:
            rec.logits.append(logits.clone())
        top = torch.topk(logits, 16)
        rec.top.append(top.values.clone())
        rec.top_ids.append(top.indices.clone())
        token = int(torch.argmax(logits))
        tokens.append(token)
        if token in eos:
            rec.stop = "eos"
            break
        rec.blocks.clear()
        out = llm(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
        if len(rec.decode) < DECODE_STEPS:
            rec.decode.append(torch.stack([o[-1] for _, o in sorted(rec.blocks, key=lambda b: b[0])]))
        cache, logits = out.past_key_values, out.logits[0, -1]
    return torch.tensor([tokens])


llm.generate = lambda inputs_embeds, attention_mask, generation_config, max_new_tokens, **_: greedy(
    inputs_embeds, attention_mask, generation_config, max_new_tokens)

for case in args.cases:
    input_name, _, prompt_name = case.partition("-")
    out = os.path.join(args.out_dir, MODEL, case)
    rec.__init__()
    audio = inputs.make(args.data, input_name)
    text = f"{PROMPTS[prompt_name]} {model.audio_locator_tag}"
    turns = [{"role": "user", "content": text}]
    formatter_ids = None
    from nemo.collections.common.prompts import PromptFormatter

    formatter = PromptFormatter.resolve(model.cfg.prompt_format)(tokenizer)
    formatter_ids = formatter.encode_dialog(turns=[dict(t) for t in turns], enable_thinking=False)["input_ids"].tolist()
    template_ids = tokenizer.tokenizer.apply_chat_template(turns, add_generation_prompt=True, enable_thinking=False,
                                                           tokenize=True, return_dict=False)
    if hasattr(template_ids, "input_ids"):
        template_ids = template_ids["input_ids"]
    if formatter_ids != list(template_ids):
        raise SystemExit(f"the nemotron3p5 format gives {formatter_ids}, the tokenizer's chat template {list(template_ids)}")
    started = time.time()
    answer = model.generate(
        prompts=[[dict(t) for t in turns]],
        audios=torch.from_numpy(audio)[None],
        audio_lens=torch.tensor([len(audio)]),
        generation_config=GenerationConfig(bos_token_id=model.text_bos_id, eos_token_id=model.text_eos_id,
                                           pad_token_id=model.text_pad_id),
        enable_thinking=False,
        max_new_tokens=args.max_new_tokens,
    )
    seconds = time.time() - started
    tokens = answer[0].tolist()
    body = tokens[:-1] if rec.stop == "eos" else tokens
    transcript = tokenizer.tokenizer.decode(body, skip_special_tokens=False)

    save(out, "audio", audio)
    save(out, "features", rec.mel[0].numpy())
    calls = []
    for rows, lengths, encoded, layer_outputs in rec.window_calls:
        pre = [o for o in layer_outputs if o[0] == "pre_encode"]
        layers = [o for o in layer_outputs if isinstance(o[0], int)]
        final = [o for o in layer_outputs if o[0] == "final"]
        if len(pre) != 1 or len(final) != 1 or [o[0] for o in layers] != list(range(len(encoder.layers))):
            raise SystemExit("an ASR window call did not run the encoder's stages once each")
        calls.append({"rows": rows, "lengths": lengths, "pre": pre[0][1], "embed": pre[0][2],
                      "layers": [o[2] for o in layers], "final": final[0][2], "encoded": encoded})
    # A call encodes the windows of one width; its rows are those windows in order, the official grouping.
    by_width = {}
    for index, (_, _, cs, ce) in enumerate(rec.windows):
        by_width.setdefault(ce - cs, []).append(index)
    order = [i for width in by_width for i in by_width[width]]
    if len(calls) != len(by_width) or sum(int(c["rows"].shape[0]) for c in calls) != len(rec.windows):
        raise SystemExit(f"{len(calls)} ASR window calls for {len(rec.windows)} windows of {len(by_width)} widths")
    window_meta, flat = [], [(c, r) for c in calls for r in range(int(c["rows"].shape[0]))]
    for (call, row), index in zip(flat, order):
        start, end, cs, ce = rec.windows[index]
        n = int(call["layers"][0].shape[1])
        if not torch.equal(call["rows"][row].T, rec.asr_input[cs:ce]):
            raise SystemExit(f"window {index}'s rows are not the normalized features from {cs} to {ce}")
        folder = os.path.join(out, "asr", str(index))
        save(folder, "input", call["rows"][row].T.numpy())
        save(folder, "pre_encode", call["pre"][row].numpy())
        save(folder, "layers", torch.stack([call["embed"][row]] + [o[row] for o in call["layers"]]).numpy())
        save(folder, "output", call["final"][row].numpy())
        if not torch.equal(call["encoded"][row].T, call["final"][row]):
            raise SystemExit(f"window {index}'s output is not its final norm's")
        left_drop = (start - cs) // pee.subsampling_factor
        if not torch.equal(rec.cores[index], call["final"][row][left_drop:left_drop + rec.cores[index].shape[0]]):
            raise SystemExit(f"window {index}'s kept frames are not its output from frame {left_drop}")
        window_meta.append({"start": start, "end": end, "context_start": cs, "context_end": ce, "frames": n,
                            "length": int(call["lengths"][row]), "left_drop": left_drop,
                            "kept": int(rec.cores[index].shape[0])})
    asr, diar, fused = rec.fuse
    if not torch.equal(asr, torch.cat(rec.cores)):
        raise SystemExit("the encoder fused other frames than the windows kept")
    save(out, "asr", asr.numpy())
    save(out, "diar", diar.numpy())
    save(out, "fused", fused.numpy())
    for k, step in enumerate(rec.diar_steps):
        save(os.path.join(out, "diar_steps"), str(k), step.pop("preds").numpy())
    save(out, "audio_embeds", rec.proj.numpy())
    if not torch.equal(model.perception.proj(fused[None])[0], rec.proj):
        raise SystemExit("the projection's output is not the projection of the fused frames")
    save(out, "prompt_ids", np.array(formatter_ids, dtype=np.int32))
    llm_dir = os.path.join(out, "llm")
    save(llm_dir, "input_embeds", rec.input_embeds.numpy())
    locator = formatter_ids.index(model.audio_locator_tag_id)
    frames = rec.audio_frames
    if frames > len(rec.proj) or not torch.equal(rec.input_embeds[locator:locator + frames], rec.proj[:frames]):
        raise SystemExit(f"the audio's rows of the language model's input are not the projection's first {frames} of "
                         f"{len(rec.proj)} rows")
    if len(rec.input_embeds) != len(formatter_ids) - 1 + frames:
        raise SystemExit(f"the language model's input has {len(rec.input_embeds)} rows, not the prompt's and {frames} of audio")
    save(llm_dir, "prefill", torch.stack(rec.prefill).numpy())
    save(llm_dir, "router_logits", torch.stack(rec.prefill_router).numpy())
    save(llm_dir, "norm", rec.norm.numpy())
    save(llm_dir, "logits", torch.stack(rec.logits).numpy())
    if rec.decode:
        save(llm_dir, "decode", torch.stack(rec.decode).numpy())
    save(llm_dir, "top", torch.stack(rec.top).numpy())
    save(llm_dir, "top_ids", torch.stack(rec.top_ids).numpy().astype(np.int32))
    save(llm_dir, "tokens", np.array(tokens, dtype=np.int32))
    meta = {
        "model": MODEL,
        "nemo_toolkit": NEMO,
        "nemo_commit": NEMO_COMMIT,
        "automodel_commit": AUTOMODEL_COMMIT,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "pin": {k: PIN[k] for k in ("repository", "revision")},
        "language_model_layers": len(llm.model.layers),
        "input": {"name": input_name, **inputs.INPUTS[input_name]},
        "prompt": {"name": prompt_name, "text": PROMPTS[prompt_name], "user": text,
                   "rendered": tokenizer.tokenizer.decode(formatter_ids, skip_special_tokens=False)},
        "samples": len(audio),
        "mel_frames": rec.mel[1],
        "windows": window_meta,
        "online_inference": {"length": pee.online_inference_length, "left_context": pee.chunk_left_context,
                             "right_context": pee.chunk_right_context, "diar_fifo_len": pee.diar_fifo_len,
                             "diar_spkcache_len": pee.diar_spkcache_len,
                             "diar_spkcache_update_period": pee.diar_spkcache_update_period,
                             "speaker_activity_threshold": pee.speaker_activity_threshold,
                             "spk_kernel_scale": pee.spk_kernel_scale,
                             "diar_async_streaming": bool(pee.diarization_model.async_streaming),
                             "diar_high_resolution": bool(pee.diarization_model.high_resolution),
                             "diar_output_subsampling_factor": int(pee.diarization_model.output_subsampling_factor)},
        "diar_steps": rec.diar_steps,
        "frames": int(rec.proj.shape[0]),
        "audio_frames": rec.audio_frames,
        "locator": locator,
        "eos": sorted(set(model.text_eos_id if isinstance(model.text_eos_id, list) else [model.text_eos_id])),
        "tokens": tokens,
        "stop": rec.stop,
        "text": transcript,
        "seconds": round(seconds, 1),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"{case}: {len(tokens)} tokens ({rec.stop}) in {seconds:.0f} s: {transcript}", flush=True)
