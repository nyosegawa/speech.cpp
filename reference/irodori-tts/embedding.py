"""Writes a speaker-inversion embedding, the form the official runtime's ref_embed reads, for the checks of speech.cpp's
voices of embeddings.

usage: uv run python embedding.py <mf|rf> <reference.wav> <tokens> <out.speaker.safetensors>

The embedding is the model's speaker condition of the reference, after its norm and without the mean token the model
prepends, cut to its first `tokens` tokens, and saved with the official save_speaker_inversion_safetensors(). The
official training learns an embedding of the same form against one model; this one is made without training, so that
the runtime and the port can be compared on it, and its first token, which the duration predictor takes as the
speaker's, is a patch of the reference rather than a mean.
"""

import os
import sys

import torch

from irodori_tts import inference_runtime as ir
from irodori_tts.model import patch_sequence_with_mask
from irodori_tts.speaker_inversion import save_speaker_inversion_safetensors
from pins import CODEC, MODELS, snapshot

name, reference, tokens, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
torch.set_grad_enabled(False)
ir.SilentCipherWatermarker._load_backend = staticmethod(lambda **_: None)
runtime = ir.InferenceRuntime.from_key(
    ir.RuntimeKey(
        checkpoint=os.path.join(snapshot(MODELS[name]), "model.safetensors"),
        model_device="cpu",
        codec_repo=os.path.join(snapshot(CODEC), "weights.pth"),
        codec_device="cpu",
    )
)
model = runtime.model
wav, rate = ir._load_audio(reference)
latent = runtime.codec.encode_waveform(wav.unsqueeze(0), sample_rate=int(rate), normalize_db=-16.0, ensure_max=True).cpu()
patched, mask = patch_sequence_with_mask(latent, torch.ones(latent.shape[:2], dtype=torch.bool), model.cfg.speaker_patch_size)
state = model.speaker_norm(model.speaker_encoder(patched, mask))[0]
assert tokens <= state.shape[0], f"the reference gives {state.shape[0]} tokens"
save_speaker_inversion_safetensors(out, {"speaker_embedding": state[:tokens]})
print("wrote", out, "with", tokens, "tokens of", state.shape[1])
