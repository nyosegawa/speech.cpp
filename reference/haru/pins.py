"""The pinned checkpoint every script in this folder reads, each file checked against its size and SHA-256, and the commit
of the code pyproject.toml pins."""

import hashlib
import importlib.metadata
import json
import os

from huggingface_hub import snapshot_download

_installed = json.loads(importlib.metadata.distribution("haru-speech").read_text("direct_url.json"))
CODE = {"repository": _installed["url"], "commit": _installed["vcs_info"]["commit_id"]}
MODEL = {
    "repository": "higurashi-ai/Haru-4B",
    "revision": "8d6e83bcd3701a76fc27a764965665da1df7da78",
    "files": {
        "config.json": (2735, "ff29b2dab25780d33fe1ea8efc3d1364d31e7bef95b0b08514b935305ba2f427"),
        "model.safetensors": (10540113510, "ad2c3534ef6bd88525e22e5da3ac4fb739368406bf3b1c852cbafbade647c0fa"),
        "tokenizer.json": (19989325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
        "tokenizer_config.json": (496, "7df4f800f74978c0ede927b5938f137ec112f872059adf76ddb9918f3281668a"),
        "README.md": (4113, "c39e053e4b08954c1f809a2e77430c6e3878e35eb46a1e514c845de953bd8f93"),
    },
}


def check(path, size, sha256):
    """Fails unless the file at `path` has the pinned size and SHA-256."""
    actual = os.path.getsize(path)
    if actual != size:
        raise SystemExit(f"{path} has {actual} bytes, not the pinned {size}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise SystemExit(f"{path} has the SHA-256 {digest.hexdigest()}, not the pinned {sha256}")


def snapshot():
    """The local folder of the pinned checkpoint, downloaded and checked on first use; a folder already checked holds a
    file that says so, since hashing 10.5 GB takes a while."""
    folder = snapshot_download(MODEL["repository"], revision=MODEL["revision"], allow_patterns=list(MODEL["files"]))
    stamp = os.path.join(folder, ".pins-checked")
    if not os.path.exists(stamp):
        for name, (size, sha256) in MODEL["files"].items():
            check(os.path.join(folder, name), size, sha256)
        with open(stamp, "w") as f:
            f.write(MODEL["revision"] + "\n")
    return folder


if __name__ == "__main__":
    print(snapshot())
