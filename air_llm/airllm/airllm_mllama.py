import torch
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer
from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration

from .airllm_base import AirLLMBaseModel
from .utils import clean_memory


class AirLLMMllama(AirLLMBaseModel):
    """AirLLM handler for text-only inference on Mllama conditional-generation models.

    This backend targets the decoder weights exposed by vision-capable checkpoints
    such as Llama 3.2 Vision. It does not yet process image inputs.
    """

    def set_layer_names_dict(self):
        self.layer_names_dict = {
            "embed": "language_model.model.embed_tokens",
            "layer_prefix": "language_model.model.layers",
            "norm": "language_model.model.norm",
            "lm_head": "language_model.lm_head",
        }

    def get_use_better_transformer(self):
        return False

    def init_model(self):
        self.model = None
        self.hf_quantizer = None

        print("Mllama: building conditional-generation skeleton...")
        try:
            with init_empty_weights():
                self.model = MllamaForConditionalGeneration(self.config)
        except Exception as e:
            clean_memory()
            raise RuntimeError(
                f"Failed to build Mllama model skeleton: {e}\n"
                "Ensure the installed transformers version includes Mllama support."
            ) from e

        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(
                quantization_config, pre_quantized=True
            )
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)

        # Mirror the checkpoint naming scheme so the generic split/load path can
        # continue to use language_model.* prefixes unchanged.
        self.model.language_model = torch.nn.Module()
        self.model.language_model.model = self.model.model.language_model
        self.model.language_model.lm_head = self.model.lm_head

        # Expose the text decoder rotary embedding where the base forward path expects it.
        self.model.model.rotary_emb = self.model.model.language_model.rotary_emb

        self.model.eval()
        self.model.tie_weights()
        self.set_layers_from_layer_names()

        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model,
                buffer_name,
                self.running_device,
                value=buffer,
                dtype=self.running_dtype,
            )

        try:
            self.model.model.rotary_emb.to(
                device=self.running_device, dtype=self.running_dtype
            )
        except Exception:
            pass

    def set_layers_from_layer_names(self):
        decoder = self.model.model.language_model
        self.layers = [decoder.embed_tokens]
        self.layers.extend(list(decoder.layers))
        self.layers.append(decoder.norm)
        self.layers.append(self.model.lm_head)

    def move_layer_to_device(self, state_dict):
        return super().move_layer_to_device(state_dict)

    def should_skip_layer(self, layer):
        return type(layer).__name__ == "MllamaCrossAttentionDecoderLayer"