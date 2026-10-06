"""Runs Qwen's qwen-asr 0.0.6 as it is released, on the CPU in float32, on the dumps of ../dump.py, and records what it
gives beside them.

usage: uv run python compare.py <model> <out dir> [input...]

On the CPU, qwen-asr's encoder attends over the whole utterance, where the reference of the dumps attends within
windows of 104 tokens (8 s) as qwen-asr's FlashAttention 2 path and vLLM backend do (docs/adr/0018); for audio of one
window the two attend alike, and audio shorter than one chunk of 1 s differs in its padding (../dump.py). For each dump
under <out dir>/<model>/ (all of them when no input is given)
this asserts that qwen-asr's own processor gives the dump's features, the prompt ids of each variant of the request,
the split of audio over 1200 s and the decoding of the dump's ids to raw.txt, and writes qwen-asr.json: the SNR of the
dump's audio_embeds against qwen-asr's encoder and projector output, and for each variant the language and text
qwen-asr's transcribe() gives, and whether that text is the dump's.
"""

import argparse
import importlib.metadata
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import MAX_ASR_INPUT_SECONDS, SAMPLE_RATE, split_audio_into_chunks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pins import MODELS, snapshot  # noqa: E402

MAX_NEW_TOKENS = 4096


def load(folder):
    """qwen-asr's model as Qwen3ASRModel.from_pretrained() makes it, with at most 4096 new tokens as dump.py decodes.

    from_pretrained() loads the processor with fix_mistral_regex=True, which in transformers 4.57.6 replaces the
    tokenizer's pre-tokenizer with Mistral's regex when the checkpoint is a local folder and leaves it when the
    checkpoint is named by its Hub id, as qwen-asr's README does. The pinned checkpoint is a local folder, so the
    processor is loaded without the flag, which gives the tokenizer of the Hub id; the two split mixed-case words such
    as "iPhone" differently, which only a prompt can contain."""
    model = AutoModel.from_pretrained(folder, dtype=torch.float32, device_map="cpu")
    processor = AutoProcessor.from_pretrained(folder)
    return Qwen3ASRModel(backend="transformers", model=model, processor=processor, max_new_tokens=MAX_NEW_TOKENS)


def snr(reference, x):
    """The SNR of `x` against `reference` in dB, None where the two are equal."""
    reference, x = reference.double(), x.double()
    error = ((reference - x) ** 2).sum()
    return float(10 * torch.log10((reference ** 2).sum() / error)) if error > 0 else None


def read_text(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def compare(asr, dump):
    meta = json.load(open(os.path.join(dump, "meta.json"), encoding="utf-8"))
    audio = np.load(os.path.join(dump, "audio.npy"))
    record = {"qwen_asr": importlib.metadata.version("qwen-asr"), "transformers": importlib.metadata.version("transformers"),
              "torch": importlib.metadata.version("torch"), "max_new_tokens": MAX_NEW_TOKENS, "variants": {}}
    started = time.time()
    if "split" in meta:
        parts = split_audio_into_chunks(wav=audio, sr=SAMPLE_RATE, max_chunk_sec=MAX_ASR_INPUT_SECONDS)
        starts = [round(offset * SAMPLE_RATE) for _, offset in parts]
        assert starts + [audio.shape[0]] == meta["split"], f"qwen-asr splits {dump} elsewhere"
        variants = {"auto": (None, None)}
    else:
        variants = {}
        for variant in ("auto", "forced", "auto-prompt", "forced-prompt"):
            v = json.load(open(os.path.join(dump, variant, "meta.json"), encoding="utf-8"))
            variants[variant] = (v["prompt"] or None, v["language"])
        inputs = asr.processor(text=[asr._build_text_prompt(context="", force_language=None)], audio=[audio],
                               return_tensors="pt", padding=True)
        frames = int(inputs["feature_attention_mask"].sum())
        features = torch.from_numpy(np.load(os.path.join(dump, "features.npy")))
        ours = inputs["input_features"][0, :, :frames].T
        assert ours.shape == features.shape and torch.allclose(ours, features, rtol=0, atol=1e-5), \
            f"qwen-asr's features differ from {dump}'s by {float((ours - features).abs().max())}"
        record["features_max_difference"] = float((ours - features).abs().max())
        thinker = asr.model.thinker
        embeds = thinker.get_audio_features(inputs["input_features"], feature_attention_mask=inputs["feature_attention_mask"])
        record["audio_embeds_snr_db"] = snr(torch.from_numpy(np.load(os.path.join(dump, "audio_embeds.npy"))), embeds)
        for variant, (context, language) in variants.items():
            prompt = asr._build_text_prompt(context=context or "", force_language=language)
            ids = asr.processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)["input_ids"][0]
            assert ids.tolist() == np.load(os.path.join(dump, variant, "prompt_ids.npy")).tolist(), \
                f"qwen-asr's prompt ids differ from {dump}/{variant}'s"
            generated = torch.from_numpy(np.load(os.path.join(dump, variant, "ids.npy"))).long()
            raw = asr.processor.batch_decode(generated[None], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            assert raw == read_text(os.path.join(dump, variant, "raw.txt")), f"qwen-asr decodes {dump}/{variant}'s ids otherwise"
    for variant, (context, language) in variants.items():
        [result] = asr.transcribe(audio=(audio, SAMPLE_RATE), context=context or "", language=language)
        text = read_text(os.path.join(dump, variant, "text.txt"))
        record["variants"][variant] = {"language": result.language, "text": result.text, "equal": result.text == text}
        print(json.dumps({"dump": os.path.basename(dump), "variant": variant, **record["variants"][variant],
                          "dump_text": text}, ensure_ascii=False), flush=True)
    record["seconds"] = round(time.time() - started, 1)
    with open(os.path.join(dump, "qwen-asr.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    print(json.dumps({"dump": os.path.basename(dump), **{k: v for k, v in record.items() if k != "variants"}}), flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
parser.add_argument("inputs", nargs="*", metavar="input")
args = parser.parse_args()

torch.set_grad_enabled(False)
root = os.path.join(args.out_dir, args.model)
dumps = args.inputs or sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "meta.json")))
if not dumps:
    parser.error(f"no dump of ../dump.py is under {root}")
asr = load(snapshot(MODELS[args.model]))
for name in dumps:
    compare(asr, os.path.join(root, name))
