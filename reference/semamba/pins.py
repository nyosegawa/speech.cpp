"""The pinned checkpoints every script in this folder runs, and the official code that runs them.

NVIDIA NemotronLabs RE-USE and RE-USE-laughter keep their code beside their weights on Hugging Face: the SEMamba modules,
the STFT, a utility, the training recipe that inference reads its settings from, and the scripts. Both repositories
hold the same modules and recipe, byte for byte, so the code is RE-USE's at its pinned revision, and each checkpoint is
its own repository's model.safetensors. Every file is checked against its size and SHA-256 before it is used.
"""

import hashlib
import os
import sys

import yaml
from huggingface_hub import hf_hub_download

RECIPE = "recipes/USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml"
CODE = {
    "repository": "nvidia/NVIDIA-NemotronLabs-RE-USE",
    "revision": "31f82f2efad1095034492f236e8dac9239ba713b",
    "files": {
        "models/codec_module_time_d4.py": (6059, "02bd1d478f36a9b38dce7d44368805298f42798ea239304580f1ccffafe766a0"),
        "models/generator_SEMamba_time_d4.py": (3311, "beebaf2d9ab7b83c101f46246ae7a7a8a2d2f57b6683025d3712c64c1b7d6817"),
        "models/mamba_block2_SEMamba.py": (2733, "97397af2d7e0f5e6530aa8c4a45b4f0f0030b4b56d8c9922a9b0bc25f8b5962e"),
        "models/stfts.py": (3303, "b82383c1ec7d7e9fe9bca1f93b3c0afe9609e191c38efc16f705e7145198d835"),
        "utils/util.py": (1061, "85960c6eb94033f37f9f77676a0cf2e7285ce12b11c24ba77577fc354766d0bf"),
        "inference_chunk.py": (6411, "df0fd3e983536f1a49440ab61ec7fb5cea2e878c58ea1aee15e182769646db60"),
        RECIPE: (2216, "06aefc68eb8a5e8bbe6faf51e3bb8a596a82842e9993cbda42d21477b5dcf168"),
    },
}

MODELS = {
    "NVIDIA-NemotronLabs-RE-USE": {
        "repository": "nvidia/NVIDIA-NemotronLabs-RE-USE",
        "revision": "31f82f2efad1095034492f236e8dac9239ba713b",
        "size": 38592940,
        "sha256": "06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1",
    },
    "NVIDIA-NemotronLabs-RE-USE-laughter": {
        "repository": "nvidia/NVIDIA-NemotronLabs-RE-USE-laughter",
        "revision": "1b1521dd659608e1e063ada83e0acfe1da473a8b",
        "size": 38592940,
        "sha256": "35c2a4e284515275d2e49a8e8cb5cd392ae67a1f1d345de1c12a0202c7b6b610",
    },
}

# The settings of the official chunked inference, which inference_chunk.sh passes to inference_chunk.py.
CHUNK_SIZE_IN_SECONDS = 5.0
HOP_LENGTH_PORTION = 0.5
# The rates the model card lists as its inputs, from 8 to 48 kHz.
SAMPLE_RATES = (8000, 48000)


def download(repository, revision, name, size, sha256):
    """The local path of a pinned file of a repository, downloaded on first use and checked against its pin."""
    path = hf_hub_download(repository, name, revision=revision)
    if os.path.getsize(path) != size:
        raise SystemExit(f"{path} has {os.path.getsize(path)} bytes, not the pinned {size}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise SystemExit(f"{path} has the SHA-256 {digest.hexdigest()}, not the pinned {sha256}")
    return path


def code_dir():
    """The folder of the pinned official code, each file checked, put first on the import path."""
    paths = {name: download(CODE["repository"], CODE["revision"], name, *pin) for name, pin in CODE["files"].items()}
    root = paths["utils/util.py"][: -len("utils/util.py")]
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def recipe():
    """The training recipe the official inference reads its settings from."""
    with open(os.path.join(code_dir(), RECIPE)) as f:
        return yaml.safe_load(f)


def weights(name):
    """The local path of a pinned checkpoint's model.safetensors."""
    pin = MODELS[name]
    return download(pin["repository"], pin["revision"], "model.safetensors", pin["size"], pin["sha256"])


def load(name, device):
    """The official SEMamba of a pinned checkpoint on `device`, in float32 and in evaluation, with its recipe."""
    import torch
    from safetensors.torch import load_file

    code_dir()
    from models.generator_SEMamba_time_d4 import SEMamba

    cfg = recipe()
    model = SEMamba(cfg)
    model.load_state_dict(load_file(weights(name)), strict=True)
    torch.set_grad_enabled(False)
    return model.to(device).float().eval(), cfg
