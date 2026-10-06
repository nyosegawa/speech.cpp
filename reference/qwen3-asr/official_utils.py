"""qwen-asr's inference/utils.py, the official code of what transformers' Qwen3-ASR leaves out."""

import importlib.util
import os


def qwen_asr_utils():
    """qwen-asr's inference/utils.py. It is loaded from its file because importing the package runs its __init__,
    which imports qwen-asr's own copy of the model, written for transformers 4.57.6 and failing on 5.18."""
    package = importlib.util.find_spec("qwen_asr").submodule_search_locations[0]
    spec = importlib.util.spec_from_file_location("qwen_asr_inference_utils", os.path.join(package, "inference", "utils.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
