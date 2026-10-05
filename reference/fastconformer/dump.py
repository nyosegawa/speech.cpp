"""Runs a pinned NeMo FastConformer on the CPU in float32 and saves the tensors the C++ port is checked against.

usage: uv run python dump.py <model> <out dir> <audio.wav>...

Each WAVE file, mono at the model's sample rate, goes to <out dir>/<model>/<file name without .wav>/. The
stages are those of the official forward pass in evaluation (no dither), recorded with hooks:

  audio          the samples as read, [N]
  features       the normalized log-mel features of the valid frames, [frames, mels]
  subsampled     the subsampling's output, before the encoder scales it by sqrt(d_model), [T, d_model]
  layers         the output of each conformer layer, [layers, T, d_model]
  encoded        the encoder's output, [T, d_model]
  ctc_log_probs  the CTC head's log-probabilities, the blank last, [T, vocabulary + 1]
  ctc_ids        the greedy CTC decode's token ids after merging repeats and dropping blanks, [n]

ctc_text.txt holds the CTC text in UTF-8 without a newline, and meta.json the same text, the TDT text of the same model for reference, and the lengths. The CTC text
of the stages is asserted to equal what the official transcribe() returns with the CTC decoder.
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


def transcribe(path, decoder_type):
    model.change_decoding_strategy(decoder_type=decoder_type, verbose=False)
    return model.transcribe([path], batch_size=1, verbose=False)[0].text


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

    log_probs = model.ctc_decoder(encoder_output=encoded)
    saved["ctc_log_probs"] = log_probs[0, :t]
    hypothesis = model.ctc_decoding.ctc_decoder_predictions_tensor(log_probs, decoder_lengths=length, return_hypotheses=True)[0]
    ids = [int(i) for i in hypothesis.y_sequence[:t].tolist()]
    blank = model.ctc_decoding.blank_id
    merged = [p for i, p in enumerate(ids) if p != blank and (i == 0 or p != ids[i - 1])]
    saved["ctc_ids"] = np.array(merged, dtype=np.int32)
    text = hypothesis.text
    assert model.ctc_decoding.decode_tokens_to_str_with_strip_punctuation(merged) == text
    official = transcribe(path, "ctc")
    assert text == official, f"the stages give {text!r}, transcribe() gives {official!r}"

    for key, value in saved.items():
        array = value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        np.save(os.path.join(out, f"{key}.npy"), np.ascontiguousarray(array).astype(np.int32 if array.dtype.kind in "iu" else np.float32))
    meta = {
        "nemo": NEMO, "model": pin, "audio": os.path.basename(path), "samples": int(samples.shape[0]),
        "seconds": samples.shape[0] / sample_rate, "frames": frames, "encoded_frames": t,
        "ctc_text": text, "tdt_text": transcribe(path, "rnnt"),
    }
    with open(os.path.join(out, "ctc_text.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(text)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("nemo", "model")}, ensure_ascii=False))
