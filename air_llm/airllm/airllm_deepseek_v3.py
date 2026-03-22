from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, GenerationConfig

from .airllm_base import AirLLMBaseModel
from .utils import clean_memory


class AirLLMDeepseekV3(AirLLMBaseModel):
    """
    AirLLM handler for DeepSeek-V3 style CausalLM models.

    The unsloth DeepSeek-V3.1-Terminus config is remote-code based and misses a
    few attributes expected by transformers' built-in DeepseekV3 model class.
    We normalize these aliases and instantiate with trust_remote_code=False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()

    def _normalize_deepseek_config(self):
        cfg = self.config

        if getattr(cfg, "qk_head_dim", None) is None:
            cfg.qk_head_dim = int(getattr(cfg, "qk_nope_head_dim", 0)) + int(
                getattr(cfg, "qk_rope_head_dim", 0)
            )

        if getattr(cfg, "head_dim", None) is None:
            cfg.head_dim = int(getattr(cfg, "qk_rope_head_dim", 0))

        if getattr(cfg, "num_local_experts", None) is None:
            cfg.num_local_experts = int(getattr(cfg, "n_routed_experts", 0))

        rope_params = getattr(cfg, "rope_parameters", None)
        if isinstance(rope_params, dict):
            for key in ("factor", "beta_fast", "beta_slow"):
                val = rope_params.get(key)
                if isinstance(val, int):
                    rope_params[key] = float(val)

    def init_model(self):
        self.model = None
        self.hf_quantizer = None
        self._normalize_deepseek_config()

        try:
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(
                    self.config, trust_remote_code=False
                )
        except Exception as e:
            clean_memory(self.running_device)
            raise RuntimeError(
                f"Failed to build DeepSeek-V3 model skeleton: {e}"
            ) from e

        self._init_strategy = "deepseek_v3_builtin"
        self._finalize_model_init()
