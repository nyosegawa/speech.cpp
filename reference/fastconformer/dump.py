"""Runs a pinned NeMo FastConformer on the CPU in float32 and saves the tensors the C++ port is checked against.

usage: uv run python dump.py <model> <out dir> <audio.wav>...

Each WAVE file, mono at the model's sample rate, goes to <out dir>/<model>/<file name without .wav>/. The
stages are those of the official forward pass and TDT greedy decoding in evaluation (no dither), recorded with
hooks:

  audio          the samples as read, [N]
  features       the normalized log-mel features of the valid frames, [frames, mels]
  subsampled     the subsampling's output, before the encoder scales it by sqrt(d_model), [T, d_model]
  layers         the output of each conformer layer, [layers, T, d_model]
  encoded        the encoder's output, [T, d_model]
  pred_labels    the label of each call of the prediction network in decoding order, the blank first, [n]
  pred_output    the prediction network's output for each, its state carried from one call to the next, [n, hidden]
  joint_frames   the encoder frame of each evaluation of the joint, [k]
  joint_steps    the index in pred_labels of the prediction network's output each evaluation used, [k]
  joint_output   the joint's output for each, a log-softmax over the tokens, the blank and the durations together
                 as the joint returns it on the CPU, [k, vocabulary + 1 + durations]
  ids            the token ids of the greedy decoding, [m]

text.txt holds the text in UTF-8 without a newline, and meta.json the same text and the lengths. The text of the
stages is asserted to equal what the official transcribe() returns with the model's default decoding.
"""

import argparse
import json
import os

import numpy as np
import soundfile
import torch

from pins import MODELS, NEMO, restore

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
parser.add_argument("audio", nargs="+")
args = parser.parse_args()

pin = MODELS[args.model]
model = restore(pin)
sample_rate = int(model.cfg.preprocessor.sample_rate)
# A hybrid checkpoint has a CTC head beside its TDT decoder; transcribe() decodes with TDT while cur_decoder is "rnnt".
assert getattr(model, "cur_decoder", "rnnt") == "rnnt" and model.cfg.decoding.model_type == "tdt"

# The decoding calls the prediction network once per label it feeds, projects each output for the joint, and
# evaluates the joint on one encoder frame at a time; these wrappers record every call.
calls = {}
predict, project_prednet, joint_after_projection = model.decoder.predict, model.joint.project_prednet, model.joint.joint_after_projection


def recording_predict(y, state, *a, **k):
    assert y.shape == (1, 1)
    out = predict(y, state, *a, **k)
    calls["pred"].append((int(y[0, 0]), out[0][0, 0].clone()))
    return out


def recording_project_prednet(g):
    out = project_prednet(g)
    calls["projected"].append(out.clone())
    return out


def recording_joint(f, g):
    out = joint_after_projection(f, g)
    calls["joint"].append((f[0, 0].clone(), g.clone(), out[0, 0, 0].clone()))
    return out


model.decoder.predict, model.joint.project_prednet, model.joint.joint_after_projection = (
    recording_predict, recording_project_prednet, recording_joint)

for path in args.audio:
    samples, rate = soundfile.read(path, dtype="float32")
    assert rate == sample_rate and samples.ndim == 1, f"{path} is not mono at {sample_rate} Hz"
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(args.out_dir, args.model, name)
    os.makedirs(out, exist_ok=True)
    saved = {"audio": samples}
    # transcribe() ends with the encoder's unfreeze(), which puts it in training: dropout would run, and every
    # batch norm would update its statistics, changing the model for all that follows.
    model.eval()

    signal = torch.from_numpy(samples)[None]
    features, frames = model.preprocessor(input_signal=signal, length=torch.tensor([signal.shape[1]]))
    frames = int(frames[0])
    saved["features"] = features[0, :, :frames].T

    hooked = {"layers": []}
    hooks = [model.encoder.pre_encode.register_forward_hook(lambda m, a, o: hooked.update(subsampled=o[0]))]
    hooks += [layer.register_forward_hook(lambda m, a, o: hooked["layers"].append(o)) for layer in model.encoder.layers]
    encoded, length = model.encoder(audio_signal=features, length=torch.tensor([frames]))
    for h in hooks:
        h.remove()
    t = int(length[0])
    saved["subsampled"] = hooked["subsampled"][0, :t]
    saved["layers"] = torch.stack([layer[0, :t] for layer in hooked["layers"]])
    saved["encoded"] = encoded[0, :, :t].T
    assert torch.equal(saved["layers"][-1], saved["encoded"])

    calls.update(pred=[], projected=[], joint=[])
    hypothesis = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=encoded, encoded_lengths=length, return_hypotheses=True)[0]
    assert len(calls["pred"]) == len(calls["projected"])
    saved["pred_labels"] = np.array([label for label, _ in calls["pred"]], dtype=np.int32)
    saved["pred_output"] = torch.stack([g for _, g in calls["pred"]])
    # Each evaluation's frame and prediction output are found by equality with the projected encoder output and
    # the projected prediction outputs, the very tensors the decoding indexes and carries.
    projected = model.joint.project_encoder(encoded.transpose(1, 2))[0, :t]
    joint_frames, joint_steps = [], []
    for f, g, _ in calls["joint"]:
        frame = [i for i in range(t) if torch.equal(projected[i], f)]
        step = [i for i, p in enumerate(calls["projected"]) if torch.equal(p, g)]
        assert len(frame) == 1 and step, "a joint evaluation matches no single frame or no prediction output"
        joint_frames.append(frame[0])
        joint_steps.append(step[-1])
    saved["joint_frames"] = np.array(joint_frames, dtype=np.int32)
    saved["joint_steps"] = np.array(joint_steps, dtype=np.int32)
    saved["joint_output"] = torch.stack([o for _, _, o in calls["joint"]])
    ids = [int(i) for i in hypothesis.y_sequence.tolist()]
    saved["ids"] = np.array(ids, dtype=np.int32)
    text = hypothesis.text
    assert model.decoding.decode_tokens_to_str_with_strip_punctuation(ids) == text
    official = model.transcribe([path], batch_size=1, verbose=False)[0].text
    assert text == official, f"the stages give {text!r}, transcribe() gives {official!r}"

    for key, value in saved.items():
        array = value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        np.save(os.path.join(out, f"{key}.npy"), np.ascontiguousarray(array).astype(np.int32 if array.dtype.kind in "iu" else np.float32))
    meta = {
        "nemo": NEMO, "model": pin, "audio": os.path.basename(path), "samples": int(samples.shape[0]),
        "seconds": samples.shape[0] / sample_rate, "frames": frames, "encoded_frames": t,
        "prediction_steps": len(saved["pred_labels"]), "joint_evaluations": len(saved["joint_frames"]), "text": text,
    }
    with open(os.path.join(out, "text.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(text)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("nemo", "model")}, ensure_ascii=False))
