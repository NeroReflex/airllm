"""AirLLM support for GPT-OSS models (for example `unsloth/gpt-oss-20b`)."""

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
        super().__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def set_experts_implementation(self, implementation: str):
        pass

    def get_generation_config(self):
        return GenerationConfig()

    def init_model(self):
        # Reuse base init pipeline; just force eager attention for GPT-OSS.
        self.config.attn_implementation = "eager"
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
