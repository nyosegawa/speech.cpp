"""Runs the official Qwen3-TTS CustomVoice with greedy decoding and saves the tensors the C++ port is checked against.

usage: uv run python dump.py <model dir> <out dir> <speaker> <language> <text> [max frames] [--instruct TEXT]

--instruct gives the 1.7B model an instruction of how to speak, as generate_custom_voice() does; the 0.6B model takes
none, and generate_custom_voice() drops it there.

Writes <out dir>/*.npy and meta.json:
  input_ids          the tokenized `<|im_start|>assistant\\n{text}<|im_end|>\\n<|im_start|>assistant\\n`
  instruct_ids       with --instruct, the tokenized `<|im_start|>user\\n{instruction}<|im_end|>\\n`
  prefill_embeds     what the talker receives before the first frame, the instruction's rows first, [T, hidden]
  trailing_text      the text embeddings added to each generated frame, [N, hidden]
  step_embeds        the talker input of each generated frame, [frames, hidden]
  talker_logits      the talker logits of each frame, [frames + 1, vocab] (row 0 is the prefill)
  talker_hidden      the talker's last hidden state of each frame, [frames + 1, hidden]
  cp_logits          the code predictor logits of the first frames, [frames, 15, vocab]
  codes              the generated codes, [frames, 16]
  codec_*            the codec decoder stages for `codes`, and wav, [samples]
"""

import argparse
import json
import os

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel

parser = argparse.ArgumentParser()
for name in ["model_dir", "out_dir", "speaker", "language", "text"]:
    parser.add_argument(name)
parser.add_argument("max_frames", type=int, nargs="?", default=400)
parser.add_argument("--instruct")
args = parser.parse_args()
model_dir, out_dir, speaker, language, text, max_frames = (args.model_dir, args.out_dir, args.speaker, args.language, args.text,
                                                           args.max_frames)
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(0)

tts = Qwen3TTSModel.from_pretrained(model_dir, dtype=torch.float32, device_map="cpu")
model = tts.model
talker = model.talker

saved = {"step_embeds": [], "talker_logits": [], "talker_hidden": [], "cp_logits": [], "cp_calls": 0}


def talker_inner_pre(module, args, kwargs):
    emb = kwargs["inputs_embeds"][0].detach().float().numpy()
    if emb.shape[0] > 1:
        saved["prefill_embeds"] = emb
    else:
        saved["step_embeds"].append(emb[0])


def talker_post(module, args, kwargs, output):
    saved["talker_logits"].append(output.logits[0, -1].detach().float().numpy())
    saved["talker_hidden"].append(output.past_hidden[0, -1].detach().float().numpy())
    if "trailing_text" not in saved and kwargs.get("trailing_text_hidden") is not None:
        saved["trailing_text"] = kwargs["trailing_text_hidden"][0].detach().float().numpy()


cp_frame = []


def cp_post(module, args, kwargs, output):
    cp_frame.append(output.logits[0, -1].detach().float().numpy())
    if len(cp_frame) == model.config.talker_config.num_code_groups - 1:
        saved["cp_logits"].append(np.stack(cp_frame))
        cp_frame.clear()


talker.model.register_forward_pre_hook(talker_inner_pre, with_kwargs=True)
talker.register_forward_hook(talker_post, with_kwargs=True)
talker.code_predictor.register_forward_hook(cp_post, with_kwargs=True)

prompt = tts._build_assistant_text(text)
input_ids = tts._tokenize_texts([prompt])[0]
if args.instruct is not None and model.tts_model_size in "0b6":
    raise SystemExit("the 0.6B model takes no instruction; generate_custom_voice() drops it")
instruct_ids = None if args.instruct is None else tts._tokenize_texts([tts._build_instruct_text(args.instruct)])[0]

with torch.no_grad():
    codes_list, _ = model.generate(
        input_ids=[input_ids],
        instruct_ids=[instruct_ids],
        languages=[language],
        speakers=[speaker],
        non_streaming_mode=True,
        max_new_tokens=max_frames,
        do_sample=False,
        subtalker_dosample=False,
        repetition_penalty=1.05,
    )
codes = codes_list[0]

np.save(f"{out_dir}/input_ids.npy", input_ids[0].numpy().astype(np.int32))
if instruct_ids is not None:
    np.save(f"{out_dir}/instruct_ids.npy", instruct_ids[0].numpy().astype(np.int32))
np.save(f"{out_dir}/prefill_embeds.npy", saved["prefill_embeds"])
np.save(f"{out_dir}/trailing_text.npy", saved["trailing_text"])
np.save(f"{out_dir}/step_embeds.npy", np.stack(saved["step_embeds"]))
np.save(f"{out_dir}/talker_logits.npy", np.stack(saved["talker_logits"]))
np.save(f"{out_dir}/talker_hidden.npy", np.stack(saved["talker_hidden"]))
np.save(f"{out_dir}/cp_logits.npy", np.stack(saved["cp_logits"]))
np.save(f"{out_dir}/codes.npy", codes.numpy().astype(np.int32))

# The codec decoder, stage by stage, on the generated codes.
decoder = model.speech_tokenizer.model.decoder
with torch.no_grad():
    c = codes.T.unsqueeze(0)
    stages = {}
    h = decoder.quantizer.decode(c)
    stages["codec_quantized"] = h
    h = decoder.pre_conv(h)
    stages["codec_pre_conv"] = h
    h = decoder.pre_transformer(inputs_embeds=h.transpose(1, 2)).last_hidden_state.permute(0, 2, 1)
    stages["codec_transformer"] = h
    for i, blocks in enumerate(decoder.upsample):
        for block in blocks:
            h = block(h)
        stages[f"codec_upsample{i}"] = h
    for i, block in enumerate(decoder.decoder):
        h = block(h)
        stages[f"codec_decoder{i}"] = h
    wav = h.clamp(min=-1, max=1)
    reference_wav = decoder(c)
    assert torch.equal(wav, reference_wav), "the staged decode differs from decoder(codes)"
for name, t in stages.items():
    np.save(f"{out_dir}/{name}.npy", t[0].float().numpy())
np.save(f"{out_dir}/wav.npy", wav[0, 0].float().numpy())

import soundfile as sf  # noqa: E402

sf.write(f"{out_dir}/wav.wav", wav[0, 0].float().numpy(), 24000)
json.dump({"model_dir": os.path.basename(os.path.normpath(model_dir)), "speaker": speaker, "language": language,
           "text": text, "prompt": prompt, "frames": int(codes.shape[0]),
           **({} if args.instruct is None else {"instruct": args.instruct}),
           "stages": list(stages.keys())},
          open(f"{out_dir}/meta.json", "w"), ensure_ascii=False, indent=1)
print("frames", codes.shape[0], "samples", wav.shape[-1], "->", out_dir)
