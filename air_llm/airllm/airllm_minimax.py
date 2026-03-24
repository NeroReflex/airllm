from transformers import GenerationConfig

from .airllm_base import AirLLMBaseModel


class AirLLMMinimax(AirLLMBaseModel):
    """AirLLM handler for MiniMax M2-family causal LM checkpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            'embed': 'model.embed_tokens',
            'layer_prefix': 'model.layers',
            'norm': 'model.norm',
            'lm_head': 'lm_head',
        }

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()

    def init_model(self):
        # Prefer eager attention for widest compatibility on commodity systems.
        self.config.attn_implementation = 'eager'
        super().init_model()
