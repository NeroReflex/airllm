"""AirLLM support for GPT-OSS models (for example `unsloth/gpt-oss-20b`)."""

import torch
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM
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

    def _move_rotary_emb_to_device(self):
        try:
            self.model.model.rotary_emb.to(
                device=self.running_device,
                dtype=self.running_dtype,
            )
        except Exception:
            pass

    def init_model(self):
        try:
            self.model = None
            self.config.attn_implementation = "eager"
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(
                    self.config,
                    attn_implementation="eager",
                    trust_remote_code=True,
                )
            self._init_strategy = "gpt_oss_eager"
            self._finalize_model_init()
            self._move_rotary_emb_to_device()
        except ImportError as exc:
            if "kernels" in str(exc) or "MXFP4" in str(exc) or "mxfp4" in str(exc):
                raise ImportError(
                    "GPT-OSS models require the `kernels` package for MXFP4 expert loading. "
                    "Install it with `pip install kernels` or `uv sync --extra gpt-oss`."
                ) from exc
            raise

    def _init_model_fast(self):
        if getattr(self, "_init_strategy", None) == "gpt_oss_eager":
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(
                    self.config,
                    attn_implementation="eager",
                    trust_remote_code=True,
                )
            self._finalize_model_init()
            self._move_rotary_emb_to_device()
            return
        super()._init_model_fast()

    def get_pos_emb_args(self, len_p, len_s):
        position_ids = torch.arange(
            len_p,
            len_p + len_s,
            dtype=torch.long,
            device=self.running_device,
        ).unsqueeze(0)
        hidden_states = torch.zeros(
            1,
            len_s,
            self.config.hidden_size,
            dtype=self.running_dtype,
            device=self.running_device,
        )
        with torch.no_grad():
            cos, sin = self.model.model.rotary_emb(hidden_states, position_ids)
        return {"position_embeddings": (cos, sin)}

    def set_layer_names_dict(self):
        """GPT-OSS follows the standard transformers base-model layout."""
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
        }
