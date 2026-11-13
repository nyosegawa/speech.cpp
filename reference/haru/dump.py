"""Runs the official Haru conversation on the CPU in float32 and saves the tensors the C++ port is checked against.

usage: uv run python dump.py OUT --data DIR [--frames N] [--seed S] [--speak-first] [--guidance G]

It talks to the model as haru.session.DuplexSession does in a live call: the official Japanese example card rendered with
the official template, a FLEURS clip as the voice reference (prepare_reference() trims its silence), then a second
FLEURS clip as the user's speech, a frame of 3200 samples at a time, followed by silence for the model to answer, N
frames in all. DIR holds FLEURS as speech-bench keeps it (fleurs/<revision>/<config>/test/<id>.wav), each clip checked
against its SHA-256. Hooks on the official modules save what they computed; where a finer stage is recomputed here from
official modules, the dump asserts that it equals the official result. The model takes about 21 GB of memory in
float32. It writes OUT/*.npy and meta.json:

  reference_audio      the voice reference at 24 kHz after prepare_reference(), [samples]
  codec_encoder_N      the codec encoder's output after stage N for the reference, [time, channels], N < 7, then
                       codec_encoder_head, the latents' means, [frames, 64]
  reference_means      the reference's latent means, [frames, 64]
  reference_scale      the one normal draw that scales the reference's noise, [1]
  reference_noise      the normal draws of the reference's noise, [frames, 64]
  reference_latents    the reference's latents as the prompt takes them, [frames, 64]
  silence_latent       the latent of digital silence, the last of 16 frames of zeros, [64]
  prompt_tokens        the prompt's tokens: the card's, PAD on the reference's frames but the last, EOP on it, [rows]
  prompt_model         the model's latents on the prompt's rows, zero where it has none, [rows, 64]
  prompt_user          the user's latents on the prompt's rows, zero where they have none, [rows, 64]
  prompt_model_mask    the rows that carry the model's audio, [rows]
  prompt_user_mask     the rows that carry the user's audio, [rows]
  user_audio           the user's side, N - 1 frames, [samples]
  user_means           the latent means of the user's frames, encoded a frame at a time, [N - 1, 64]
  user_scale           the normal draw that scales each user frame's noise, [N - 1]
  user_noise           the normal draws of each user frame's noise, [N - 1, 64]
  user_latents         the user's latents as the language model takes them, [N - 1, 64]
  embeddings           the language model's input rows: the prompt's, then each step's, [rows + N - 1, 2560]
  lm_layers            the residual stream after each layer, [32, rows + N - 1, 2560]
  lm_mixer             the gated attention's or the Gated DeltaNet's output of each layer, [32, rows + N - 1, 2560]
  lm_hidden            the final norm's output, [rows + N - 1, 2560]
  delta_state          each Gated DeltaNet layer's recurrent state after the prompt, [24, 32, 128, 128]
  conv_state           each Gated DeltaNet layer's convolution state after the prompt, [24, 8192, 4]
  logits               the text logits of each frame's hidden state, from the prompt's last row and each step, [N, vocab]
  silent_logits        the float32 logits of PAD and EOP that replace those of `logits`, [N, 2]
  dry                  the DRY penalties subtracted before each draw: frame, token, penalty, [count, 3]
  sampled_logits       the logits each token is drawn from, after DRY, the float32 logits, the EOP bias and the bans, [N, vocab]
  tokens               the drawn text tokens, [N]
  head_noise           the head's noise, [N, 64]
  head_cond            the head's condition after its SiLU: the projected hidden state and the embeddings of t, the
                       interval, h and the guidance, [N, 2560]
  head_layers          the head's stream after each of its 4 layers, [N, 4, 2560]
  head_latents         the head's normalized latents, [N, 64]
  latents              the latents the codec decodes, [N, 64]
  codec_audio          the codec's output for each frame's latent, decoded as it is drawn, [N x 3200]
  audio                the model's audio as the session passes it: codec_audio clamped to [-1, 1], [N x 3200]
  codec_decoder_N      the codec decoder's output after stage N for the N latents decoded whole, [time, channels],
                       N < 7, then codec_decoder_head, the audio decoded whole, [N x 3200]
"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.initialization import no_init_weights

import haru.models.vibevoice_codec as vibevoice_codec
import haru.session as haru_session
from haru.audio import SAMPLE_RATE, load_audio, prepare_reference
from haru.config import HaruConfig
from haru.models.haru import HaruModel
from haru.prompt import card_templates, render_card
from haru.sampling import DryState
from tokenizers import Tokenizer

import pins

FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
# The voice reference (10.0 s) and the user's speech (6.4 s), both from FLEURS' Japanese test split.
REFERENCE = ("ja_jp/9306707928443027514", "b51d392ba3d1cbd5687496b8b25cef78a1a1ab339b0eedf1bfe20cc33334effb")
USER = ("ja_jp/12677001980660723842", "c89ea3a0fac8b19b553723693770c6a00130965955cb04b98e252ecca857ed2e")


def array(tensor):
    """A copy of `tensor` as float32 numpy, not sharing the tensor's memory."""
    return tensor.detach().to("cpu", torch.float32, copy=True).numpy()


def fleurs(data, pin):
    name, sha256 = pin
    config, ident = name.split("/")
    path = os.path.join(data, "fleurs", FLEURS_REVISION, config, "test", ident + ".wav")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{path} has the SHA-256 {digest}, not the pinned {sha256}")
    return path


def load_float32(folder):
    """The official model in float32 on the CPU, each tensor converted as it is read, so that the bfloat16 checkpoint and
    the float32 model are never in memory whole at once, as load_model(dtype=float32) would have them."""
    config = HaruConfig.from_file(os.path.join(folder, "config.json"))
    with no_init_weights():
        model = HaruModel(config)
    state = {}
    with safe_open(os.path.join(folder, "model.safetensors"), "pt") as f:
        for name in f.keys():
            state[name] = f.get_tensor(name).to(torch.float32)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.tokenizer = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
    return model.eval().requires_grad_(False)


parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--data", required=True)
parser.add_argument("--frames", type=int, default=80)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--speak-first", action="store_true")
parser.add_argument("--guidance", type=float, default=None)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(0)

folder = pins.snapshot()
model = load_float32(folder)
cfg = model.config
frame = cfg.frame_samples
saved = {}


def save(name, value):
    saved[name] = np.ascontiguousarray(value)


card = json.load(open(os.path.join(os.path.dirname(haru_session.__file__), "assets", "cards", "ja.json"), encoding="utf-8"))["chat"]
system_prompt = render_card(card, card_templates()["templates"]["ja"])
reference = prepare_reference(load_audio(fleurs(args.data, REFERENCE)))
user_clip = load_audio(fleurs(args.data, USER))
steps = args.frames - 1
user = np.zeros(steps * frame, dtype=np.float32)
user[: min(len(user_clip), len(user))] = user_clip[: len(user)]
save("reference_audio", reference)
save("user_audio", user)
config = haru_session.DuplexConfig(seed=args.seed, max_frames=args.frames + 1, codec_backend="torch",
                                   speak_first=args.speak_first,
                                   **({} if args.guidance is None else {"guidance_scale": args.guidance}))
session = haru_session.DuplexSession(model, system_prompt, reference, config)

# The codec: the encoder's stages for the reference, and the draws of every latent's noise.
encoder, decoder = model.codec.encoder, model.codec.decoder


def stage_ends(stack, depths):
    """The index in `stack.sequence()` of each stage's last module, and of the head."""
    ends, at = [], 0
    for depth in depths:
        at += 1 + depth
        ends.append(at - 1)
    return ends + [at]


capture = {"encoder": None, "decoder": None}
for side, stack, depths in (("encoder", encoder, cfg.codec.encoder_depths), ("decoder", decoder, cfg.codec.decoder_depths)):
    ends = stage_ends(stack, depths)
    for index, module in enumerate(stack.sequence()):
        if index in ends:
            label = "head" if index == ends[-1] else str(ends.index(index))

            def hook(m, inputs, output, side=side, label=label):
                # The first encode while capturing is the reference's; the silence latent may be encoded after it.
                if capture[side] is not None and label not in capture[side]:
                    capture[side][label] = array(output[0][0].T)

            module.register_forward_hook(hook)

draws = []
official_sample = vibevoice_codec.VibeVoiceCodec.sample


def sample(self, mean, generator=None):
    """The official draw, recorded: the generator's state is kept, the official sample() runs, and the same draws are
    made again from the kept state, to save them and to assert that they give the official latents."""
    state = generator.get_state() if generator is not None else None
    result = official_sample(self, mean, generator)
    if generator is not None:
        generator.set_state(state)
    scale = torch.randn(mean.shape[0], 1, 1, device=mean.device, dtype=mean.dtype, generator=generator)
    noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
    assert torch.equal(mean + scale * (self.config.latent_std / 0.8) * noise, result)
    draws.append((array(mean[0]), array(scale.reshape(-1)), array(noise[0]), array(result[0])))
    return result


vibevoice_codec.VibeVoiceCodec.sample = sample

prompts = []
official_build_prompt = haru_session.build_prompt


def build_prompt(*a, **k):
    capture["encoder"] = {}
    result = official_build_prompt(*a, **k)
    prompts.append(result)
    return result


haru_session.build_prompt = build_prompt

# The language model: its input rows, each layer's stream and mixer, the final norm.
rows, layers, mixers, hidden = [], [[] for _ in model.lm.layers], [[] for _ in model.lm.layers], []
official_embed = model.embed


def embed(*a, **k):
    result = official_embed(*a, **k)
    rows.append(array(result[0]))
    return result


model.embed = embed
for i, layer in enumerate(model.lm.layers):
    layer.register_forward_hook(lambda m, inputs, output, i=i: layers[i].append(array((output[0] if isinstance(output, tuple) else output)[0])))
    mixer = layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn
    mixer.register_forward_hook(lambda m, inputs, output, i=i: mixers[i].append(array((output[0] if isinstance(output, tuple) else output)[0])))
model.lm.norm.register_forward_hook(lambda m, inputs, output: hidden.append(array(output[0])))

# Sampling: the logits, the DRY penalties, the logits each token is drawn from, and the token.
logits, silent_logits, sampled, tokens, dry = [], [], [], [], []
official_logits, official_logits_fp32 = model.logits, model.logits_fp32
model.logits = lambda h: (lambda r: (logits.append(array(r[0])), r)[1])(official_logits(h))
model.logits_fp32 = lambda h, ids: (lambda r: (silent_logits.append(array(r[0])), r)[1])(official_logits_fp32(h, ids))
official_apply = DryState.apply


def apply(self, values):
    for token, penalty in self.penalties.items():
        dry.append((len(logits) - 1, token, penalty))
    return official_apply(self, values)


DryState.apply = apply
official_sample_token = haru_session.sample_token


def sample_token(values, *a, **k):
    token = official_sample_token(values, *a, **k)
    sampled.append(array(values[0]))
    tokens.append(int(token[0]))
    return token


haru_session.sample_token = sample_token

# The head: its noise, condition, layers and latent; and the latents the codec decodes with the audio of each.
head = model.head
head_noise, head_cond, head_layers, head_latents, latents, audio = [], [], [], [], [], []
official_head_sample = head.sample
layer_outputs = []
for layer in head.layers:
    layer.register_forward_hook(lambda m, inputs, output: layer_outputs.append(array(output[0])))


def head_sample(h, noise, guidance=None, steps=None):
    del layer_outputs[:]
    result = official_head_sample(h, noise, guidance, steps)
    g = torch.full((noise.shape[0],), float(config.guidance_scale if guidance is None else guidance))
    one = lambda v: torch.full((noise.shape[0],), v, dtype=noise.dtype)
    c = head.cond_proj(h.to(noise.dtype)) + head.t_embed(one(0.0)) + head.dt_embed(one(head.config.interval_embed))
    c = F.silu(c + head.h_embed(one(1.0)) + head.omega_embed((1.0 - 1.0 / g).to(noise.dtype)))
    head_noise.append(array(noise[0]))
    head_cond.append(array(c[0]))
    head_layers.append(np.stack(layer_outputs))
    head_latents.append(array(result[0]))
    return result


head.sample = head_sample
official_unscale = model.unscale_latents
model.unscale_latents = lambda x: (lambda r: (latents.append(array(r[0])), r)[1])(official_unscale(x))


codec_audio = []
official_stream_decode = session._codec.decode
session._codec.decode = lambda x: (lambda r: (codec_audio.append(array(r[0])), r)[1])(official_stream_decode(x))

first = session.start()
capture["encoder"], encoder_stages = None, capture["encoder"]
cache = session._cache
linear = [i for i, kind in enumerate(model.lm.config.layer_types) if kind == "linear_attention"]
save("delta_state", np.stack([array(cache.layers[i].recurrent_states[0][0]) for i in linear]))
save("conv_state", np.stack([array(cache.layers[i].conv_states[0][0]) for i in linear]))
assert len(draws) == 1, "the reference's latents are not drawn once"
frames = [first]
for i in range(steps):
    frames.append(session.step(user[i * frame:(i + 1) * frame]))
streamed = np.concatenate([f.audio for f in frames])

prompt = prompts[0]
save("prompt_tokens", array(prompt.tokens).astype(np.int32))
save("prompt_model", array(prompt.model_latents))
save("prompt_user", array(prompt.user_latents))
save("prompt_model_mask", array(prompt.model_mask).astype(np.int32))
save("prompt_user_mask", array(prompt.user_mask).astype(np.int32))
for label, value in encoder_stages.items():
    save(f"codec_encoder_{label}", value)
means, scale, noise, result = draws[0]
save("reference_means", means)
save("reference_scale", scale)
save("reference_noise", noise)
save("reference_latents", result)
assert np.array_equal(saved["codec_encoder_head"], means)
from haru.prompt import silence_latent  # noqa: E402

save("silence_latent", array(silence_latent(model.codec)))
user_draws = draws[1:]
assert len(user_draws) == steps
save("user_means", np.concatenate([d[0] for d in user_draws]))
save("user_scale", np.concatenate([d[1] for d in user_draws]))
save("user_noise", np.concatenate([d[2] for d in user_draws]))
save("user_latents", np.concatenate([d[3] for d in user_draws]))

save("embeddings", np.concatenate(rows))
assert saved["embeddings"].shape[0] == prompt.length + steps
save("lm_layers", np.stack([np.concatenate(x) for x in layers]))
save("lm_mixer", np.stack([np.concatenate(x) for x in mixers]))
save("lm_hidden", np.concatenate(hidden))
save("logits", np.stack(logits))
save("silent_logits", np.stack(silent_logits))
save("dry", np.array(dry, dtype=np.float32).reshape(-1, 3))
save("sampled_logits", np.stack(sampled))
save("tokens", np.array(tokens, dtype=np.int32))
assert tokens == [f.token for f in frames]
save("head_noise", np.stack(head_noise))
save("head_cond", np.stack(head_cond))
save("head_layers", np.stack(head_layers))
save("head_latents", np.stack(head_latents))
save("latents", np.stack(latents))
save("codec_audio", np.concatenate(codec_audio))
save("audio", streamed)

# The same latents decoded whole, with the decoder's stages.
capture["decoder"] = {}
with torch.no_grad():
    whole = array(model.codec.decode(torch.from_numpy(saved["latents"])[None])[0])
for label, value in capture["decoder"].items():
    save(f"codec_decoder_{label}", value)
capture["decoder"] = None

for name, value in saved.items():
    np.save(os.path.join(args.out, name + ".npy"), value)
meta = {
    "model": {key: pins.MODEL[key] for key in ("repository", "revision")},
    "code": pins.CODE,
    "torch": torch.__version__,
    "fleurs": {"revision": FLEURS_REVISION, "reference": REFERENCE[0], "user": USER[0]},
    "card": system_prompt,
    "seed": args.seed,
    "frames": args.frames,
    "speak_first": args.speak_first,
    "guidance_scale": config.guidance_scale,
    "temperature": config.temperature,
    "top_k": config.top_k,
    "top_p": config.top_p,
    "sample_rate": SAMPLE_RATE,
    "frame_samples": frame,
    "prompt_rows": prompt.length,
    "text": [f.text for f in frames],
    "whole_decode_max_difference": float(np.abs(whole - saved["codec_audio"]).max()),
    "shapes": {name: list(value.shape) for name, value in saved.items()},
}
with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print(f"{args.out}: {prompt.length} prompt rows, {steps} steps, text {''.join(meta['text']).strip()!r}, "
      f"whole decode differs by {meta['whole_decode_max_difference']:.3g}")
