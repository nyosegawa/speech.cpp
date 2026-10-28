"""The pinned checkpoint every script in this folder runs, and the commits of NeMo Speech and Automodel pyproject.toml
pins.

Nemotron 3.5 Transcribe's model.safetensors holds the speech LLM in BF16, but for the CTC timestamp head's tensors and
the Mamba layers' A_log, D and dt_bias, which it holds in F32, and the timestamp head's configuration in its metadata.
config.json is the SpeechLM's configuration, the parallel expert encoder's and the language model's among it, and
llm_backbone/config.json the language model's, as Transformers reads it.

usage: uv run python pins.py    downloads every pinned file into the Hugging Face cache and checks it
"""

import hashlib
import importlib.metadata
import os

from huggingface_hub import hf_hub_download

NEMO = importlib.metadata.version("nemo-toolkit")
NEMO_COMMIT = "040574f7ffa3bb0ae4b92aa8424ecea454d03c91"
AUTOMODEL_COMMIT = "309a10098e3ea3d9b79d4b96fa397db40908b751"
MODELS = {
    "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B": {
        "repository": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
        "revision": "ef905575a106fdad904f3b0ce661dccad654607d",
        "files": {
            "model.safetensors": (67496013734, "03142dde5176c1d4df21ee829642745a1d670ea4d79e1a4684c64d53195acff2"),
            "config.json": (431098, "5470c2642bbb32572a45aa417a06ad226f4a10c78e7f3eab163b84ff2707343a"),
            "generation_config.json": (56, "d48f72c695fa8a239373070371a404a51d778d30018a1768307a2d0209549e43"),
            "tokenizer.json": (17078144, "d071004cb4450d3aa31f87aab979cd09ffc99c4e709e6cf72c324346cd948d0d"),
            "tokenizer_config.json": (11782, "e11d0bb1b107a182a6db567822b605db852edcb01e87368bd8e28c62c2991d15"),
            "llm_backbone/config.json": (2403, "8f11ffe191b8ade26ff9b43196a2cfcc356f6ee73cfb3c6a6336968f178611d0"),
        },
    },
}


def download(pin, name):
    """The local path of a pinned file of the repository, downloaded on first use and checked against its pin.

    Hashing the weights reads 67.5 GB, minutes on a network volume, so a file whose hash matched is recorded beside it
    with its size and modification time, and is hashed again only when either has changed.
    """
    size, sha256 = pin["files"][name]
    path = hf_hub_download(pin["repository"], name, revision=pin["revision"])
    stat = os.stat(path)
    if stat.st_size != size:
        raise SystemExit(f"{path} has {stat.st_size} bytes, not the pinned {size}")
    record = os.path.realpath(path) + ".verified"
    checked = f"{sha256} {stat.st_size} {stat.st_mtime_ns}\n"
    if os.path.exists(record):
        with open(record) as f:
            if f.read() == checked:
                return path
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise SystemExit(f"{path} has the SHA-256 {digest.hexdigest()}, not the pinned {sha256}")
    with open(record, "w") as f:
        f.write(checked)
    return path


def snapshot(pin):
    """The folder of the pinned revision, every pinned file in it checked; the files outside the pin are not read."""
    paths = {name: download(pin, name) for name in pin["files"]}
    return os.path.dirname(paths["config.json"])


if __name__ == "__main__":
    for name, pin in MODELS.items():
        print(name, snapshot(pin))
