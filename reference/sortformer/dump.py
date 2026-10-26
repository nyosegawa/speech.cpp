"""Runs Nemotron 3 Diarization through NeMo on the CPU in float32 and saves the tensors the C++ port is checked against.

usage: uv run python dump.py <out dir> [input...] [--configs <name,...>] [--data <folder>]

Each input, a name from inputs.py (every one when none is given), read from the clips under <folder> (by default
~/speech-bench-data), goes to <out dir>/<name>/<input>/, and its diarization with each configuration (by default
every streaming configuration of the model card, pins.configurations()) to <out dir>/<name>/<input>/<configuration>/,
<name> being the model's, Nemotron-3-Diarization. The official forward() runs the whole input: the
preprocessor's features of all of it, then forward_streaming(), which cuts the features into chunks and, for each,
pre-encodes the chunk and its right context, encodes them after the speaker cache and the FIFO queue, upsamples and
scores the frames, and updates the cache and the queue (SortformerModules.streaming_update()). Wrappers around those
calls record what each took and gave. The speaker cache and the FIFO queue hold copies of pre-encoded frames, or the
learnable silence embedding, so the state is saved as the index of the 80 ms frame each row is a copy of, -1 for the
silence embedding, with the frames' embeddings saved once; each copy is asserted to be the row it names bit for bit.

<input>/audio.npy        the samples, [N]
<input>/features.npy     the preprocessor's log-mel features of the whole input, [T, 128], T = N // 160
<configuration>/
  pre_encode.npy         the pre-encoded embedding of each 80 ms frame, as the chunk it belongs to computed it, [F, 512]
  rc_pre_encode.npy      each step's right context, pre-encoded with its chunk, [sum of rc frames, 512]
  step_preds.npy         each step's 80 ms speaker probabilities of the speaker cache, the FIFO queue, the chunk and its
                         right context, the average of eight 10 ms ones, which streaming_update() takes, [sum, 8]
  cache_index.npy        the speaker cache after each step, as frame indices, [sum], and its probabilities, [sum, 8]
  cache_preds.npy
  fifo_index.npy         the FIFO queue after each step, likewise
  fifo_preds.npy
  total_preds.npy        forward()'s speaker probabilities, one row every 10 ms, [T, 8]
  step<k>/               for some steps k, the stages of the step from its encoder's input:
    hidden.npy             the encoder's output after its final norm, [n, 512]
    proj.npy               encoder_proj's output, [n, 192]
    upsampled.npy          upsample_hidden()'s output, [8 n, 192]
    preds_hr.npy           the speaker probabilities every 10 ms, [8 n, 8]
    layers.npy             for one of them, the step after the first compression of the cache or else the last, the
                           embedding norm's and each layer's output, [32, n, 512]
  meta.json              the configuration, each step's geometry (features, frames, state lengths before and after,
                         whether the cache was compressed) and the segments postprocessing gives total_preds with
                         DiarizeConfig's defaults, asserted to be what diarize() gives the input
  segments.json          the segments the postprocessing gives total_preds with each set of POSTPROCESSING, per speaker

meta.json at <input>/ holds the versions, the pin and the input's pieces.
"""

import argparse
import importlib.metadata
import json
import os

import numpy as np
import torch
from nemo.collections.asr.parts.utils.speaker_utils import generate_diarization_output_lines
from nemo.collections.asr.parts.utils.vad_utils import load_postprocessing_from_yaml, predlist_to_timestamps

import inputs
from pins import MODELS, NEMO, NEMO_COMMIT, configurations, restore

# The settings of the postprocessing, PostProcessingParams', whose segments are dumped: the defaults; hysteresis, and
# thresholds the other way round, which toggle between them; padding that joins segments; dropping short segments and
# filling short pauses; and all of them.
POSTPROCESSING = {
    "default": {},
    "hysteresis": {"onset": 0.7, "offset": 0.3},
    "toggling": {"onset": 0.4, "offset": 0.6},
    "padded": {"pad_onset": 0.1, "pad_offset": 0.25},
    "filtered": {"min_duration_on": 0.3, "min_duration_off": 0.4},
    "all": {"onset": 0.6, "offset": 0.4, "pad_onset": 0.05, "pad_offset": 0.15, "min_duration_on": 0.2, "min_duration_off": 0.6},
}

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("inputs", nargs="*", choices=[[]] + sorted(inputs.INPUTS))
parser.add_argument("--configs")
parser.add_argument("--data", default=os.path.expanduser("~/speech-bench-data"))
args = parser.parse_args()
PIN = MODELS["Nemotron-3-Diarization"]
CONFIGS = configurations(PIN)[0]
configs = args.configs.split(",") if args.configs else sorted(CONFIGS, key=lambda name: -CONFIGS[name]["chunk_len"])
for c in configs:
    if c not in CONFIGS:
        raise SystemExit(f"{c} is not a configuration; give one of {', '.join(CONFIGS)}")

model = restore(PIN)
sm = model.sortformer_modules
factor = model.encoder.subsampling_factor
assert model.high_resolution and model.output_subsampling_factor == 1 and model.streaming_mode
assert sm.use_learnable_sil_emb and sm.chunk_left_context == 0
silence = sm.learnable_sil_emb.detach().numpy()


def save(folder, name, array):
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, name + ".npy"), np.ascontiguousarray(array))


class Recorder:
    """What the wrapped calls of one forward() take and give, step by step."""

    def __init__(self, encoder_steps):
        self.encoder_steps = encoder_steps
        self.step = -1
        self.steps = []
        self.frames = {}
        self.table = []
        self.rc_rows = []
        self.step_preds, self.cache_index, self.cache_preds, self.fifo_index, self.fifo_preds = [], [], [], [], []
        self.stages = {}
        self.layers = None

    def index(self, row):
        """The frame a row of the state is a copy of, or -1 for the silence embedding."""
        key = row.tobytes()
        if key in self.frames:
            return self.frames[key]
        if np.array_equal(row, silence):
            return -1
        raise AssertionError(f"step {self.step}: a row of the state is a copy of no frame")


rec = None
pre_encode = model._call_pre_encode
encoder_forward = model.encoder.forward
upsample_hidden = sm.upsample_hidden
speaker_logits = sm.forward_speaker_logits
streaming_update = sm.streaming_update


def wrapped_pre_encode(features, lengths):
    out, out_lengths = pre_encode(features, lengths)
    rec.step += 1
    rec.steps.append({"features": int(features.shape[1]), "frames": int(out.shape[1]), "valid_frames": int(out_lengths[0])})
    rec.pre = out[0].numpy()
    return out, out_lengths


def wrapped_encoder(audio_signal, length, bypass_pre_encode=False):
    assert bypass_pre_encode
    out, out_length = encoder_forward(audio_signal=audio_signal, length=length, bypass_pre_encode=True)
    if rec.step in rec.encoder_steps:
        rec.stages[rec.step] = {"hidden": out[0].transpose(0, 1).numpy().copy()}
    rec.steps[-1]["encoder_frames"] = int(audio_signal.shape[1])
    return out, out_length


def wrapped_upsample(hidden):
    out = upsample_hidden(hidden)
    if rec.step in rec.encoder_steps:
        rec.stages[rec.step]["proj"] = hidden[0].numpy().copy()
        rec.stages[rec.step]["upsampled"] = out[0].numpy().copy()
    return out


def wrapped_logits(hidden):
    out = speaker_logits(hidden)
    if rec.step in rec.encoder_steps:
        rec.stages[rec.step]["preds_hr"] = torch.sigmoid(out)[0].numpy().copy()
    return out


def wrapped_update(streaming_state, chunk, preds, lc=0, rc=0):
    step = rec.steps[-1]
    first = sum(s["chunk_frames"] for s in rec.steps[:-1])
    n = chunk.shape[1] - lc - rc
    step.update({"first_frame": first, "chunk_frames": n, "rc_frames": rc, "lc_frames": lc})
    step["cache_before"] = int(streaming_state.spkcache.shape[1])
    step["fifo_before"] = int(streaming_state.fifo.shape[1])
    step["compressed_before"] = bool(streaming_state.spkcache_compressed)
    assert np.array_equal(chunk[0].numpy(), rec.pre)
    for i in range(n):
        row = rec.pre[lc + i]
        rec.frames.setdefault(row.tobytes(), first + i)
        rec.table.append(row)
    rec.rc_rows.extend(rec.pre[lc + n :])
    rec.step_preds.append(preds[0].numpy().copy())
    state, chunk_preds = streaming_update(streaming_state, chunk, preds, lc=lc, rc=rc)
    cache = state.spkcache[0].numpy()
    fifo = state.fifo[0].numpy()
    rec.cache_index.append(np.array([rec.index(r) for r in cache], dtype=np.int32))
    rec.fifo_index.append(np.array([rec.index(r) for r in fifo], dtype=np.int32))
    rec.cache_preds.append(state.spkcache_preds[0].numpy().copy() if state.spkcache_preds is not None else np.zeros((0, sm.n_spk), np.float32))
    rec.fifo_preds.append(state.fifo_preds[0].numpy().copy())
    step["cache_after"] = int(cache.shape[0])
    step["fifo_after"] = int(fifo.shape[0])
    step["compressed_after"] = bool(state.spkcache_compressed)
    return state, chunk_preds


model._call_pre_encode = wrapped_pre_encode
model.encoder.forward = wrapped_encoder
sm.upsample_hidden = wrapped_upsample
sm.forward_speaker_logits = wrapped_logits
sm.streaming_update = wrapped_update


def layer_hook(index):
    def hook(module, inputs_, output):
        if rec.layers is not None and rec.step == rec.layer_step:
            rec.layers[index] = output[0].numpy().copy()

    return hook


model.encoder.embed_norm.register_forward_hook(layer_hook(0))
for i, layer in enumerate(model.encoder.layers):
    layer.register_forward_hook(layer_hook(i + 1))


def segments(total_preds, settings=None):
    """The segments of the official postprocessing with `settings` or its defaults, per speaker and as diarize() writes them."""
    params = load_postprocessing_from_yaml(None)
    for key, value in (settings or {}).items():
        setattr(params, key, value)
    per_speaker = predlist_to_timestamps(
        batch_preds_list=[total_preds.unsqueeze(0)],
        audio_rttm_map_dict={"input": {"offset": 0.0}},
        cfg_vad_params=params,
        unit_10ms_frame_count=model.output_subsampling_factor,
        bypass_postprocessing=False,
    )[0]
    lines = generate_diarization_output_lines(speaker_timestamps=per_speaker, model_spk_num=len(per_speaker))
    return per_speaker, lines


for name in args.inputs or sorted(inputs.INPUTS):
    audio = inputs.make(args.data, name)
    folder = os.path.join(args.out_dir, PIN["repository"].split("/")[1], name)
    save(folder, "audio", audio)
    signal = torch.from_numpy(audio).unsqueeze(0)
    length = torch.tensor([len(audio)])
    features, feature_length = model.process_signal(audio_signal=signal, audio_signal_length=length)
    features = features[0, :, : int(feature_length[0])].transpose(0, 1).numpy()
    save(folder, "features", features)
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(
            {
                "nemo": NEMO,
                "nemo_commit": NEMO_COMMIT,
                "torch": torch.__version__,
                "numpy": importlib.metadata.version("numpy"),
                "pin": PIN,
                "input": inputs.INPUTS[name],
                "clips": {c: inputs.CLIPS[c] for c in {p["clip"] for p in inputs.INPUTS[name]["pieces"] if "clip" in p}},
                "fleurs_revision": inputs.FLEURS_REVISION,
                "samples": len(audio),
                "sample_rate": inputs.SAMPLE_RATE,
                "features": int(features.shape[0]),
            },
            f,
            indent=2,
        )

    for config in configs:
        for key, value in CONFIGS[config].items():
            setattr(sm, key, value)
        model._check_streaming_parameters()
        n_steps = -(-features.shape[0] // (sm.chunk_len * factor))
        # The first two steps, before any state; the steps around the first compression of the cache, found in a first
        # run; and the last, which holds the partial frame at the end.
        rec = Recorder(set())
        model.forward(audio_signal=signal, audio_signal_length=length)
        compressions = [i for i, s in enumerate(rec.steps) if s["compressed_after"] and not s["compressed_before"]]
        chosen = {0, 1, n_steps - 1} | ({compressions[0], compressions[0] + 1} if compressions else set())
        chosen = {s for s in chosen if 0 <= s < n_steps}
        rec = Recorder(chosen)
        rec.layers = [None] * (len(model.encoder.layers) + 1)
        rec.layer_step = compressions[0] + 1 if compressions and compressions[0] + 1 < n_steps else max(chosen)
        total = model.forward(audio_signal=signal, audio_signal_length=length)[0]
        assert len(rec.steps) == n_steps

        out = os.path.join(folder, config)
        save(out, "pre_encode", np.stack(rec.table))
        save(out, "rc_pre_encode", np.stack(rec.rc_rows) if rec.rc_rows else np.zeros((0, sm.fc_d_model), np.float32))
        save(out, "step_preds", np.concatenate(rec.step_preds))
        save(out, "cache_index", np.concatenate(rec.cache_index))
        save(out, "cache_preds", np.concatenate(rec.cache_preds))
        save(out, "fifo_index", np.concatenate(rec.fifo_index))
        save(out, "fifo_preds", np.concatenate(rec.fifo_preds))
        save(out, "total_preds", total.numpy())
        for step, stages in rec.stages.items():
            for stage, array in stages.items():
                save(os.path.join(out, f"step{step}"), stage, array)
        save(os.path.join(out, f"step{rec.layer_step}"), "layers", np.stack(rec.layers))

        per_speaker, lines = segments(total)
        recorded, rec = rec, Recorder(set())
        official = model.diarize(audio=[audio], sample_rate=inputs.SAMPLE_RATE, batch_size=1, verbose=False)[0]
        assert official == lines, f"{name} {config}: diarize() gives other segments than forward() and its postprocessing"
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(
                {
                    "config": CONFIGS[config],
                    "factor": factor,
                    "speakers": sm.n_spk,
                    "steps": recorded.steps,
                    "encoder_steps": sorted(chosen),
                    "layer_step": recorded.layer_step,
                    "segments": per_speaker,
                    "lines": lines,
                },
                f,
                indent=1,
            )
        with open(os.path.join(out, "segments.json"), "w") as f:
            json.dump({k: {"settings": v, "segments": segments(total, v)[0]} for k, v in POSTPROCESSING.items()}, f, indent=1)
        print(f"{name} {config}: {n_steps} steps, {len(lines)} segments, compressions at {compressions[:3]}")
