"""The pinned checkpoints convert.py converts."""

from huggingface_hub import snapshot_download

MODELS = {
    "0.6b": {"repository": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "revision": "85e237c12c027371202489a0ec509ded67b5e4b5"},
    "1.7b": {"repository": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "revision": "0c0e3051f131929182e2c023b9537f8b1c68adfe"},
}


def snapshot(pin):
    """The local folder of a pinned Hugging Face repository, downloaded on first use."""
    return snapshot_download(pin["repository"], revision=pin["revision"])
