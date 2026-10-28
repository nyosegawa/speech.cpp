"""Builds Nemotron 3.5 Transcribe from its official modules on the CPU in float32.

SALMAutomodel.from_pretrained(), the official entry point, builds its language model through NeMo Automodel, whose Mamba
layers run mamba_ssm's Triton kernels and nothing else, so it cannot run on a CPU. load() builds what from_pretrained()
builds, in its order, but for the language model: SALMAutomodel itself from the checkpoint's configuration (its
tokenizer, prompt format and generate()), the perception module with the parallel expert encoder through
setup_speech_encoder(), and the language model as Transformers' NemotronHForCausalLM, whose PyTorch paths run anywhere,
from the checkpoint's llm.* tensors renamed by Automodel's own NemotronV3StateDictAdapter and merged as Transformers'
conversion of NemotronH checkpoints merges them. Every tensor of those goes into exactly one parameter or buffer, and
every parameter and persistent buffer is loaded, but for the MTP layers', which Transformers' model does not have and
generation does not use.

The CTC timestamp head is left out: from_pretrained() attaches it with read_ctc_timestamp_config(), which at the pinned
commit takes F32, BF16, F16 and U8 tensors alone and stops on the head's upsample_factor, an I64 scalar, and
CTCTimestampCheckpoint, which refuses any tensor of the head that is not floating point. timestamp_head() builds the
head's decoder, TransformerCTCDecoder, from the checkpoint's metadata and loads its tensors.

Transformers 5.18's NemotronH clamps each Mamba layer's time step from below at the configuration's time_step_min, 0.001
(NemotronHMamba2Mixer.time_step_limit), where Automodel's mixer and vLLM 0.32's MambaMixer2 take the checkpoint's
time_step_limit, (0, inf), and clamp nothing; language_model() gives the mixers the checkpoint's limit. Without it the
first Mamba layer's output on the dumps' requests is 30 to 33 dB from the one without the clamp.

Transformers 5.18's NemotronHForCausalLM refuses generate(inputs_embeds=...), which SALMAutomodel.generate() calls, so
the caller passes the language model's generation in (see dump.py).
"""

import os
import re

import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from transformers import AutoConfig, NemotronHForCausalLM

from nemo.collections.speechlm2.models.salm_automodel import SALMAutomodel
from nemo.collections.speechlm2.parts.ctc_timestamp_utils import TransformerCTCDecoder
from nemo.collections.speechlm2.parts.hf_hub import _inject_local_artifact_paths
from nemo.collections.speechlm2.parts.pretrained import setup_speech_encoder, update_perception_output_dim
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM as AutomodelNemotronH

WEIGHTS = "model.safetensors"
EXPERT = re.compile(r"^(model\.layers\.\d+\.mixer\.experts)\.(\d+)\.(up_proj|down_proj)\.weight$")


def language_model_state(path, config, layers=None):
    """The checkpoint's language model tensors in float32 under Transformers' NemotronH names.

    Automodel's adapter renames each native tensor to the HF checkpoint's names and splits the merged experts into one
    tensor per expert; Transformers' conversion of a NemotronH checkpoint then renames backbone.* to model.* and stacks
    each layer's experts along a first dimension. With `layers`, only the first that many layers are read.
    """
    with torch.device("meta"):
        adapter = AutomodelNemotronH(config, backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch",
                                                                   experts="torch", dispatcher="torch")).state_dict_adapter
    state, experts = {}, {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if not key.startswith("llm.") or key.startswith("llm.mtp.") or key.endswith("._extra_state"):
                continue
            native = key[len("llm."):]
            m = re.match(r"^model\.layers\.(\d+)\.", native)
            if layers is not None and m and int(m.group(1)) >= layers:
                continue
            for hf_key, tensor in adapter.convert_single_tensor_to_hf(native, f.get_tensor(key).float()):
                name = re.sub(r"^backbone\.", "model.", hf_key)
                expert = EXPERT.match(name)
                if expert:
                    experts.setdefault(f"{expert.group(1)}.{expert.group(3)}", {})[int(expert.group(2))] = tensor
                else:
                    state[name] = tensor
    for name, by_index in experts.items():
        if sorted(by_index) != list(range(config.n_routed_experts)):
            raise SystemExit(f"{name} has experts {sorted(by_index)}, not 0 to {config.n_routed_experts - 1}")
        state[name] = torch.stack([by_index[i] for i in range(config.n_routed_experts)])
    return state


def language_model(snapshot, layers=None):
    """Transformers' NemotronHForCausalLM in float32 with the checkpoint's weights, all of them or its first `layers`."""
    config = AutoConfig.from_pretrained(os.path.join(snapshot, "llm_backbone"))
    if layers is not None:
        config.layers_block_type = config.layers_block_type[:layers]
        config.num_hidden_layers = layers
    config.dtype = torch.float32
    with torch.device("meta"):
        llm = NemotronHForCausalLM(config)
    state = language_model_state(os.path.join(snapshot, WEIGHTS), config, layers)
    expected = set(llm.state_dict())
    if set(state) != expected:
        raise SystemExit(f"the language model's tensors differ from the model's: missing {sorted(expected - set(state))[:8]}, "
                         f"unexpected {sorted(set(state) - expected)[:8]}")
    llm.load_state_dict(state, strict=True, assign=True)
    limit = tuple(float(v) for v in config.time_step_limit)
    if limit != (0.0, float("inf")):
        raise SystemExit(f"the checkpoint's time_step_limit is {limit}, which the reference was not written for")
    for layer in llm.model.layers:
        if hasattr(layer.mixer, "time_step_limit"):
            layer.mixer.time_step_limit = limit
    # SALMAutomodel reads the embedding table as llm.model.embed_tokens, Automodel's name for Transformers' embeddings.
    llm.model.embed_tokens = llm.model.embeddings
    for name, tensor in list(llm.named_parameters()) + list(llm.named_buffers()):
        if tensor.is_meta:
            raise SystemExit(f"{name} of the language model was not loaded")
    return llm.eval()


def load(snapshot, layers=None, with_language_model=True):
    """SALMAutomodel in float32 on the CPU, as from_pretrained() builds it from the snapshot folder of the pinned revision,
    with Transformers' language model; `layers` keeps only the first that many layers of it, for trying on less memory,
    and without `with_language_model` it has none, its perception module projecting to the width its configuration
    gives."""
    weights = os.path.join(snapshot, WEIGHTS)
    cfg = OmegaConf.to_container(OmegaConf.load(os.path.join(snapshot, "config.json")))
    cfg["trust_remote_code"] = False
    _inject_local_artifact_paths(cfg, snapshot, {})
    cfg["pretrained_weights"] = False
    cfg["init_configure_model"] = False
    model = SALMAutomodel(cfg)
    if with_language_model:
        model.llm = language_model(snapshot, layers)
    setup_speech_encoder(model, pretrained_weights=False)
    if with_language_model:
        update_perception_output_dim(model)
    perception = {}
    with safe_open(weights, framework="pt") as f:
        for key in f.keys():
            if key.startswith("perception."):
                perception[key[len("perception."):]] = f.get_tensor(key).float()
    model.perception.load_state_dict(perception, strict=True)
    model.float().eval()
    return model


def timestamp_head(snapshot):
    """The CTC timestamp head's decoder in float32, its configuration from the checkpoint's metadata, every tensor of
    ctc_timestamp.decoder.* loaded; and the metadata's decoder and tokenizer configurations."""
    import json

    with safe_open(os.path.join(snapshot, WEIGHTS), framework="pt") as f:
        metadata = f.metadata()
        decoder_config = json.loads(metadata["ctc_timestamp_decoder_config"])
        tokenizer_config = json.loads(metadata["ctc_timestamp_tokenizer_config"])
        prefix = "ctc_timestamp.decoder."
        state = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    decoder = TransformerCTCDecoder(**decoder_config)
    decoder.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()}, strict=True)
    return decoder.float().eval(), decoder_config, tokenizer_config
