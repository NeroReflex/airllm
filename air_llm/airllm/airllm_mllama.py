import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor
from transformers.quantizers import AutoHfQuantizer
from transformers.models.mllama.modeling_mllama import (
    MllamaForConditionalGeneration,
    _prepare_cross_attention_mask,
)

from .airllm_base import AirLLMBaseModel
from .utils import clean_memory, load_layer


class AirLLMMllama(AirLLMBaseModel):
    """AirLLM handler for Mllama conditional-generation models.

    Supports text-only generation and image-conditioned generation by preparing
    cross-attention states from the vision encoder on demand.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = None

    def get_processor(self):
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_local_path,
                token=self.hf_token,
                trust_remote_code=True,
            )
        return self.processor

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
        self.model.vision_model = self.model.model.vision_model
        self.model.multi_modal_projector = self.model.model.multi_modal_projector

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

    def should_skip_layer(self, layer, **kwargs):
        if type(layer).__name__ != "MllamaCrossAttentionDecoderLayer":
            return False
        return kwargs.get("cross_attention_states") is None

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        cross_attention_states = kwargs.get("cross_attention_states")
        cross_attention_mask = kwargs.get("cross_attention_mask")
        full_text_row_masked_out_mask = kwargs.get("full_text_row_masked_out_mask")

        if cross_attention_states is None and kwargs.get("pixel_values") is not None:
            cross_attention_states, cross_attention_mask, full_text_row_masked_out_mask = (
                self._prepare_cross_attention_inputs(
                    pixel_values=kwargs.get("pixel_values"),
                    aspect_ratio_ids=kwargs.get("aspect_ratio_ids"),
                    aspect_ratio_mask=kwargs.get("aspect_ratio_mask"),
                    cross_attention_mask=kwargs.get("cross_attention_mask"),
                )
            )

        if cross_attention_states is not None:
            model_inputs["cross_attention_states"] = cross_attention_states
        if cross_attention_mask is not None:
            cross_attention_mask, full_text_row_masked_out_mask = self._align_processed_cross_attention_masks(
                cross_attention_mask,
                full_text_row_masked_out_mask,
                input_ids.shape[1],
            )
            model_inputs["cross_attention_mask"] = cross_attention_mask
        if full_text_row_masked_out_mask is not None:
            model_inputs["full_text_row_masked_out_mask"] = full_text_row_masked_out_mask

        return model_inputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        aspect_ratio_mask=None,
        aspect_ratio_ids=None,
        cross_attention_mask=None,
        cross_attention_states=None,
        full_text_row_masked_out_mask=None,
    ):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "pixel_values": pixel_values,
            "aspect_ratio_mask": aspect_ratio_mask,
            "aspect_ratio_ids": aspect_ratio_ids,
            "cross_attention_mask": cross_attention_mask,
            "cross_attention_states": cross_attention_states,
            "full_text_row_masked_out_mask": full_text_row_masked_out_mask,
        }

        if kwargs.get("cross_attention_states") is None and kwargs.get("pixel_values") is not None:
            cross_attention_states, cross_attention_mask, full_text_row_masked_out_mask = (
                self._prepare_cross_attention_inputs(
                    pixel_values=kwargs.get("pixel_values"),
                    aspect_ratio_ids=kwargs.get("aspect_ratio_ids"),
                    aspect_ratio_mask=kwargs.get("aspect_ratio_mask"),
                    cross_attention_mask=kwargs.get("cross_attention_mask"),
                )
            )
            kwargs["cross_attention_states"] = cross_attention_states
            kwargs["cross_attention_mask"] = cross_attention_mask
            kwargs["full_text_row_masked_out_mask"] = full_text_row_masked_out_mask
            kwargs.pop("pixel_values", None)
            kwargs.pop("aspect_ratio_ids", None)
            kwargs.pop("aspect_ratio_mask", None)

        if kwargs.get("cross_attention_mask") is not None and kwargs.get("input_ids") is not None:
            kwargs["cross_attention_mask"], kwargs["full_text_row_masked_out_mask"] = (
                self._align_processed_cross_attention_masks(
                    kwargs.get("cross_attention_mask"),
                    kwargs.get("full_text_row_masked_out_mask"),
                    kwargs["input_ids"].shape[1],
                )
            )

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return super().forward(**kwargs)

    def _align_processed_cross_attention_masks(
        self,
        cross_attention_mask,
        full_text_row_masked_out_mask,
        target_seq_len,
    ):
        if cross_attention_mask is None:
            return cross_attention_mask, full_text_row_masked_out_mask

        # Processed mask shape is [batch, 1, seq_len, num_vision_tokens].
        if cross_attention_mask.dim() == 4 and cross_attention_mask.shape[2] < target_seq_len:
            pad_len = target_seq_len - cross_attention_mask.shape[2]
            cross_attention_mask = F.pad(cross_attention_mask, (0, 0, 0, pad_len, 0, 0, 0, 0), value=0.0)

            if full_text_row_masked_out_mask is not None:
                full_text_row_masked_out_mask = F.pad(
                    full_text_row_masked_out_mask,
                    (0, 0, 0, pad_len, 0, 0, 0, 0),
                    value=0.0,
                )

        return cross_attention_mask, full_text_row_masked_out_mask

    def _ensure_multimodal_split_files(self):
        checkpoint_dir = self.checkpoint_path
        vision_split_path = os.path.join(checkpoint_dir, "vision_model.safetensors")
        projector_split_path = os.path.join(checkpoint_dir, "multi_modal_projector.safetensors")
        if os.path.exists(vision_split_path) and os.path.exists(projector_split_path):
            return

        index_path = os.path.join(self.model_local_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Expected model.safetensors.index.json in {self.model_local_path} for Mllama vision support"
            )

        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        vision_keys = [k for k in weight_map.keys() if k.startswith("vision_model.")]
        projector_keys = [k for k in weight_map.keys() if k.startswith("multi_modal_projector.")]

        if not vision_keys or not projector_keys:
            raise RuntimeError("Failed to locate Mllama vision/projector keys in model index")

        shard_to_keys = defaultdict(list)
        for key in vision_keys + projector_keys:
            shard_to_keys[weight_map[key]].append(key)

        vision_state = {}
        projector_state = {}

        for shard_file, keys in shard_to_keys.items():
            shard_path = os.path.join(self.model_local_path, shard_file)
            shard_state = load_file(shard_path)
            for key in keys:
                tensor = shard_state[key]
                if key.startswith("vision_model."):
                    vision_state[key] = tensor
                elif key.startswith("multi_modal_projector."):
                    projector_state[key] = tensor

        save_file(vision_state, vision_split_path)
        save_file(projector_state, projector_split_path)

    def _prepare_cross_attention_inputs(
        self,
        pixel_values,
        aspect_ratio_ids,
        aspect_ratio_mask,
        cross_attention_mask,
    ):
        if aspect_ratio_ids is None:
            raise ValueError("aspect_ratio_ids is required when pixel_values is provided")

        self._ensure_multimodal_split_files()

        vision_state = load_layer(self.checkpoint_path, "vision_model")
        projector_state = load_layer(self.checkpoint_path, "multi_modal_projector")
        vision_layers = self.move_layer_to_device(vision_state)
        projector_layers = self.move_layer_to_device(projector_state)

        pixel_values = pixel_values.to(self.running_device, dtype=self.running_dtype)
        aspect_ratio_ids = aspect_ratio_ids.to(self.running_device)
        if aspect_ratio_mask is not None:
            aspect_ratio_mask = aspect_ratio_mask.to(self.running_device)

        with torch.no_grad():
            vision_outputs = self.model.model.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
            )
            cross_attention_states = vision_outputs.last_hidden_state
            cross_attention_states = self.model.model.multi_modal_projector(
                cross_attention_states
            ).reshape(
                -1,
                cross_attention_states.shape[-2],
                self.model.model.hidden_size,
            )

            full_text_row_masked_out_mask = None
            if cross_attention_mask is not None:
                cross_attention_mask = cross_attention_mask.to(self.running_device)
                cross_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
                    cross_attention_mask,
                    num_vision_tokens=self.model.model.vision_model.num_patches,
                    dtype=self.running_dtype,
                )

        for layer_name in vision_layers + projector_layers:
            set_module_tensor_to_device(self.model, layer_name, "meta")
        clean_memory(self.running_device)

        return cross_attention_states, cross_attention_mask, full_text_row_masked_out_mask