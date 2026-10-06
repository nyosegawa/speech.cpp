"""The pinned checkpoints and test audio the scripts in this folder read, each file checked against its size and SHA-256."""

import hashlib
import os

from huggingface_hub import hf_hub_download, snapshot_download

MODELS = {
    "Qwen3-ASR-0.6B": {
        "repository": "Qwen/Qwen3-ASR-0.6B",
        "revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        "weights": {
            "model.safetensors": (1876091704, "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"),
        },
    },
    "Qwen3-ASR-1.7B": {
        "repository": "Qwen/Qwen3-ASR-1.7B",
        "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5",
        "weights": {
            "model-00001-of-00002.safetensors": (4220320824, "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6"),
            "model-00002-of-00002.safetensors": (478200688, "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc"),
        },
    },
}

FLEURS = {"repository": "google/fleurs", "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd"}
# The test split of each configuration: its TSV and its archive of 16 kHz WAVE files, as (bytes, SHA-256).
FLEURS_TEST = {
    "ja_jp": {
        "tsv": (361174, "5dd9643511437414681ad3f23508596c621cdf78978724a09f1f06fefe9d300b"),
        "audio": (448762391, "5de465fa7aaafc4e2c13aba44771550b8cd2dd29bb9b265daeb6d92ca8e0c136"),
    },
    "en_us": {
        "tsv": (367864, "74c046239374deeb60fa63f258f907388093a32bcaa3140965f70ef05c79f7ca"),
        "audio": (289851356, "d9c2e37b41aacd41bc283554a0a82b5476b36887049774ecb2819dcaaa55a356"),
    },
    "cmn_hans_cn": {
        "tsv": (491487, "5734461648f816181d7dab5fc79204b18c4b9bc2cd5138225b25c72d18385d21"),
        "audio": (525346466, "09d19ad18f5d7e91076880807e866cd16abd924c7052b55f71cdae91714fc166"),
    },
    "de_de": {
        "tsv": (559542, "82fab72c58a347345675c238f2492bb997105a4ab45d69e2a2b34ed29082fa97"),
        "audio": (568734559, "e86b42dfcdef749926cd92135045f87c25966c09e50d01c401adb04ee7d8628f"),
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


def snapshot(pin):
    """The local folder of a pinned checkpoint with its configuration, tokenizer and weights, downloaded on first use."""
    folder = snapshot_download(pin["repository"], revision=pin["revision"], allow_patterns=["*.json", "*.txt", "*.safetensors"])
    for name, (size, sha256) in pin["weights"].items():
        check(os.path.join(folder, name), size, sha256)
    return folder


def fleurs_test(config):
    """The local paths of the TSV and the audio archive of a FLEURS configuration's test split, downloaded on first use."""
    paths = {}
    for kind, name in (("tsv", "test.tsv"), ("audio", "audio/test.tar.gz")):
        paths[kind] = hf_hub_download(
            FLEURS["repository"], f"data/{config}/{name}", repo_type="dataset", revision=FLEURS["revision"])
        check(paths[kind], *FLEURS_TEST[config][kind])
    return paths
