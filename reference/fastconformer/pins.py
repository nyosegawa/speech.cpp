"""The pinned checkpoints every script in this folder runs, and the version of NeMo pyproject.toml pins."""

import hashlib
import importlib.metadata
import os

import torch
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import ASRModel

NEMO = importlib.metadata.version("nemo-toolkit")
MODELS = {
    "parakeet-tdt_ctc-0.6b-ja": {
        "repository": "nvidia/parakeet-tdt_ctc-0.6b-ja",
        "revision": "44edb27eea9317daf89333e75eb830db4b1cc298",
        "file": "parakeet-tdt_ctc-0.6b-ja.nemo",
        "size": 2490951680,
        "sha256": "a2ed2aab9c82cce4e9b699dca669184d7525c7b5e1eb8c3f7b789421bc3637e8",
    },
}


def checkpoint(pin):
    """The local path of a pinned .nemo checkpoint, downloaded on first use and checked against its pin."""
    path = hf_hub_download(pin["repository"], pin["file"], revision=pin["revision"])
    size = os.path.getsize(path)
    if size != pin["size"]:
        raise SystemExit(f"{path} has {size} bytes, not the pinned {pin['size']}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != pin["sha256"]:
        raise SystemExit(f"{path} has the SHA-256 {digest.hexdigest()}, not the pinned {pin['sha256']}")
    return path


def restore(pin):
    """The official model of a pinned checkpoint, on the CPU in float32 and in evaluation."""
    torch.set_grad_enabled(False)
    model = ASRModel.restore_from(checkpoint(pin), map_location="cpu")
    return model.float().eval()
