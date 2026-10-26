"""The pinned checkpoint every script in this folder runs, and the commit of NeMo pyproject.toml pins.

Nemotron 3 Diarization's .nemo holds its weights in BF16 and its configuration; NeMo's main branch at the pinned commit
runs it, since nemo-toolkit 3.0.0 has neither its high-resolution output nor its learnable silence embedding.
"""

import hashlib
import importlib.metadata
import json
import os

import torch
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import SortformerEncLabelModel

NEMO = importlib.metadata.version("nemo-toolkit")
NEMO_COMMIT = "d32dd550ffb016cd7b4133bb2bc69f15ba866361"
MODELS = {
    "Nemotron-3-Diarization": {
        "repository": "nvidia/Nemotron-3-Diarization",
        "revision": "f667ed73aee57d40cc39428eb768b4fd87a0a29e",
        "file": "Nemotron-3-Diarization.nemo",
        "size": 198676480,
        "sha256": "867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d",
        # The configuration of the repository's transformers port, which holds the streaming configurations of the
        # model card's table that the .nemo does not: the offline one, the cache's settings of the others, and the
        # chunks of each of those, by the names its processor gives them.
        "configs": {
            "config.json": {"size": 1458, "sha256": "dd795ec07ffa09acf18d231aa3fa8642aa5c521760953754e02559a7413b6873"},
            "processor_config.json": {"size": 623, "sha256": "91726746788a024a2f0f2f699a6c3667f0d74e6a0e9288fc44fe3018e23af68e"},
        },
    },
}


def download(pin, name, size, sha256):
    """The local path of a pinned file of the repository, downloaded on first use and checked against its pin."""
    path = hf_hub_download(pin["repository"], name, revision=pin["revision"])
    if os.path.getsize(path) != size:
        raise SystemExit(f"{path} has {os.path.getsize(path)} bytes, not the pinned {size}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise SystemExit(f"{path} has the SHA-256 {digest.hexdigest()}, not the pinned {sha256}")
    return path


def checkpoint(pin):
    """The local path of a pinned .nemo checkpoint."""
    return download(pin, pin["file"], pin["size"], pin["sha256"])


def config(pin, name):
    """The JSON of one of the pinned configuration files."""
    entry = pin["configs"][name]
    with open(download(pin, name, entry["size"], entry["sha256"])) as f:
        return json.load(f)


def restore(pin):
    """The official model of a pinned checkpoint, on the CPU in float32 and in evaluation."""
    if NEMO_COMMIT[:9] not in NEMO:
        raise SystemExit(f"nemo-toolkit {NEMO} is installed, not the pinned commit {NEMO_COMMIT}; run uv sync")
    torch.set_grad_enabled(False)
    model = SortformerEncLabelModel.restore_from(checkpoint(pin), map_location="cpu")
    return model.float().eval()


def configurations(pin):
    """
    The streaming configurations of the model card's table, in 80 ms frames, by the names the repository's transformers
    processor gives the three that stream (processor_config.json's streaming_modes, with config.json's streaming_config)
    and "offline" for the fourth, which the card's diarize() runs (config.json's own chunk_length, chunk_right_context,
    fifo_length and speaker_cache_update_period); the speaker cache has the same length in each. The processor's
    streaming_mode is the one that streams by default.
    """
    model, processor = config(pin, "config.json"), config(pin, "processor_config.json")
    streaming = model["streaming_config"]
    out = {
        "offline": {
            "spkcache_len": streaming["speaker_cache_length"],
            "fifo_len": model["fifo_length"],
            "chunk_len": model["chunk_length"],
            "chunk_right_context": model["chunk_right_context"],
            "spkcache_update_period": model["speaker_cache_update_period"],
        }
    }
    for name, (chunk, right_context) in processor["streaming_modes"].items():
        out[name] = {
            "spkcache_len": streaming["speaker_cache_length"],
            "fifo_len": streaming["fifo_length"],
            "chunk_len": chunk,
            "chunk_right_context": right_context,
            "spkcache_update_period": streaming["speaker_cache_update_period"],
        }
    assert processor["streaming_mode"] in out and processor["subsampling_factor"] == model["audio_config"]["subsampling_factor"]
    return out, processor["streaming_mode"]
