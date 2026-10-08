"""The pinned model every script in this folder runs: the 16 kHz model of the silero-vad package pyproject.toml pins.

The package ships its weights inside it, so the pin is the file's path in the package, its size and its SHA-256, checked
before the model is loaded, and the commit of the package's release tag on GitHub, whose src/silero_vad/data/ holds the
same file (git blob 5c6988d663950a93a5f0d6c38c2fe024653ec552).
"""

import hashlib
import importlib.metadata
import os

import torch
import silero_vad

PACKAGE = "silero-vad"
VERSION = "6.2.3"
MODEL = {
    "repository": "snakers4/silero-vad",
    # The commit of the tag v6.2.3.
    "revision": "5cd7945676eb32225748052e2e6a0580e4686a08",
    "file": "data/silero_vad.jit",
    "size": 2272526,
    "sha256": "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720",
}


def model_path():
    """The path of the pinned model file inside the installed package, checked against the pin."""
    installed = importlib.metadata.version(PACKAGE)
    if installed != VERSION:
        raise SystemExit(f"{PACKAGE} {installed} is installed, not the pinned {VERSION}; run uv sync")
    path = os.path.join(os.path.dirname(silero_vad.__file__), *MODEL["file"].split("/"))
    size = os.path.getsize(path)
    if size != MODEL["size"]:
        raise SystemExit(f"{path} has {size} bytes, not the pinned {MODEL['size']}")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != MODEL["sha256"]:
        raise SystemExit(f"{path} has the SHA-256 {digest}, not the pinned {MODEL['sha256']}")
    return path


def load():
    """The official model as load_silero_vad() loads it, on the CPU in float32 and in evaluation, after the pin is checked."""
    model_path()
    torch.set_grad_enabled(False)
    return silero_vad.load_silero_vad()
