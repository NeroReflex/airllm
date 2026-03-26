"""AirLLM support for GPT-OSS models (for example `unsloth/gpt-oss-20b`)."""

import torch
from transformers import GenerationConfig

from .airllm_base import AirLLMBaseModel


class AirLLMGPTOss(AirLLMBaseModel):
    """
    AirLLM handler for GPT-OSS models.
    
    Examples:
    - unsloth/gpt-oss-20b (20B parameters)
    - unsloth/gpt-oss-120b (120B parameters, when available)
    """

    def __init__(self, *args, **kwargs):
        # GPT-OSS MXFP4 expert path in transformers is calibrated for bf16.
        # If dtype is left at the base default (fp16), routed MoE matmuls can
        # fail with dtype mismatch errors inside torch.bmm.
        kwargs.setdefault("dtype", torch.bfloat16)
        super().__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def get_config_trust_remote_code(self):
        return False

    def get_tokenizer_trust_remote_code(self):
        return False

    def get_model_trust_remote_code(self):
        return False

    def set_experts_implementation(self, implementation: str):
        pass

    def get_generation_config(self):
        return GenerationConfig()

    def init_model(self):
        # Reuse base init pipeline; just force eager attention for GPT-OSS.
        self.config.attn_implementation = "eager"
        # MXFP4 swizzled weights are stored as custom triton tensor objects that
        # cannot be evicted with layer.to("meta") between AirLLM streaming passes.
        # Forcing dequantize=True causes the quantizer to convert blocks+scales to
        # plain BF16 nn.Parameters, which AirLLM's memory eviction handles correctly.
        q_cfg = getattr(self.config, "quantization_config", None)
        if q_cfg is not None and getattr(q_cfg, "quant_type", None) in ("mxfp4", "fp4"):
            q_cfg.dequantize = True
        try:
            super().init_model()
        except ImportError as exc:
            if "kernels" in str(exc) or "MXFP4" in str(exc) or "mxfp4" in str(exc):
                raise ImportError(
                    "GPT-OSS models require the `kernels` package for MXFP4 expert loading. "
                    "Install it with `pip install kernels` or `uv sync --extra gpt-oss`."
                ) from exc
            raise

    def set_layer_names_dict(self):
        """GPT-OSS follows the standard transformers base-model layout."""
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
        }
