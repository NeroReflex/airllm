"""
AirLLM support for Qwen3 dense and Qwen3 MoE model families.

Supports:
  - Qwen3ForCausalLM          (e.g. Qwen3-8B, Qwen3-32B)
  - Qwen3MoeForCausalLM       (e.g. Qwen3-30B-A3B)
  - Qwen3_5MoeForConditionalGeneration  (e.g. Qwen3.5-397B multimodal wrapper)

Original implementation by masterx / MasterX1582.
"""

import json
import os

import torch
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, GenerationConfig
from transformers.quantizers import AutoHfQuantizer

from .airllm_base import AirLLMBaseModel
from .utils import clean_memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_qwen35_moe_skeleton(config):
    """
    Directly instantiate Qwen3_5MoeForConditionalGeneration.
    AutoModelForCausalLM.from_config fails for this model because the outer
    multimodal config does not expose vocab_size at the top level (it lives on
    config.text_config).
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration,
    )
    return Qwen3_5MoeForConditionalGeneration(config)


def _is_qwen35_moe_multimodal(config_or_path):
    """
    Returns True for Qwen3_5MoeForConditionalGeneration (397B multimodal).
    Accepts either a config object or a filesystem path string.
    """
    if isinstance(config_or_path, str):
        import json
        import os
        cfg_path = os.path.join(config_or_path, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                data = json.load(f)
            archs = data.get('architectures', []) or []
        else:
            return False
    else:
        archs = getattr(config_or_path, 'architectures', []) or []
    return any('Qwen3_5Moe' in a for a in archs)


def _get_architectures(config_or_path):
    if isinstance(config_or_path, str):
        cfg_path = os.path.join(config_or_path, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                data = json.load(f)
            return data.get('architectures', []) or []
        try:
            cfg = AutoConfig.from_pretrained(config_or_path, trust_remote_code=True)
            return getattr(cfg, 'architectures', []) or []
        except Exception:
            return []
    return getattr(config_or_path, 'architectures', []) or []


def _is_qwen35_dense_conditional(config_or_path):
    return any('Qwen3_5ForConditionalGeneration' in a for a in _get_architectures(config_or_path))


def _build_qwen35_dense_skeleton(config):
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )
    return Qwen3_5ForConditionalGeneration(config)


# ---------------------------------------------------------------------------
# Qwen3 MoE (30B-A3B and 397B variants)
# ---------------------------------------------------------------------------

class AirLLMQwen3Moe(AirLLMBaseModel):
    """
    AirLLM handler for Qwen3 MoE models.

    Handles two variants:
    - Qwen3MoeForCausalLM (e.g. Qwen3-30B-A3B): standard model.layers.* layout
    - Qwen3_5MoeForConditionalGeneration (e.g. Qwen3.5-397B): multimodal wrapper
      with layers nested under model.language_model.*, includes SSM layers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def set_experts_implementation(self, implementation: str):
        pass

    def get_generation_config(self):
        return GenerationConfig()

    def set_layer_names_dict(self):
        # self.config is not yet set at call time — use the raw path string.
        path = getattr(self, 'model_local_path_or_repo_id', None) or ''
        if _is_qwen35_moe_multimodal(path):
            # Qwen3.5-397B: layers nested under language_model
            self.layer_names_dict = {
                'embed': 'model.language_model.embed_tokens',
                'layer_prefix': 'model.language_model.layers',
                'norm': 'model.language_model.norm',
                'lm_head': 'lm_head',
            }
        else:
            # Qwen3-30B-A3B: standard flat layout
            self.layer_names_dict = {
                'embed': 'model.embed_tokens',
                'layer_prefix': 'model.layers',
                'norm': 'model.norm',
                'lm_head': 'lm_head',
            }

    def init_model(self):
        self.model = None
        self.hf_quantizer = None

        if _is_qwen35_moe_multimodal(self.config):
            # Qwen3.5-397B: multimodal wrapper — AutoModelForCausalLM.from_config
            # fails because vocab_size lives on config.text_config.
            print("Qwen3.5 MoE: building multimodal conditional-generation skeleton...")
            try:
                with init_empty_weights():
                    self.model = _build_qwen35_moe_skeleton(self.config)
            except Exception as e:
                clean_memory()
                raise RuntimeError(
                    f"Failed to build Qwen3.5 MoE model skeleton: {e}\n"
                    "Ensure transformers>=4.57.0 is installed."
                ) from e

            quantization_config = getattr(self.config, "quantization_config", None)
            if quantization_config is not None:
                self.hf_quantizer = AutoHfQuantizer.from_config(
                    quantization_config, pre_quantized=True
                )
                device_map = self.hf_quantizer.update_device_map(None)
                self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)

            self.model.eval()
            self.model.tie_weights()
            self.set_layers_from_layer_names()

            for buffer_name, buffer in self.model.named_buffers():
                set_module_tensor_to_device(
                    self.model, buffer_name, self.running_device,
                    value=buffer, dtype=self.running_dtype,
                )

            # Move rotary_emb to device so get_pos_emb_args can call it.
            try:
                self.model.model.language_model.rotary_emb.to(
                    device=self.running_device, dtype=self.running_dtype
                )
            except Exception:
                pass

        else:
            # Qwen3-30B-A3B and similar: standard CausalLM skeleton.
            print("Qwen3 MoE: building standard CausalLM skeleton...")
            super().init_model()
            try:
                self.model.model.rotary_emb.to(
                    device=self.running_device, dtype=self.running_dtype
                )
            except Exception:
                pass

    def _get_rotary_emb(self):
        """Return the rotary embedding module, regardless of model nesting."""
        try:
            return self.model.model.language_model.rotary_emb  # 397B multimodal
        except AttributeError:
            pass
        try:
            return self.model.model.rotary_emb  # 30B standard
        except AttributeError:
            return None

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        # Both Qwen3 MoE variants handle causal masking internally.
        # AirLLM's 4D causal mask causes shape errors — return nothing.
        return {}

    def get_pos_emb_args(self, len_p, len_s):
        """Compute position_embeddings=(cos, sin) via the model's rotary_emb."""
        try:
            rotary_emb = self._get_rotary_emb()
            if rotary_emb is None:
                return {}
            head_dim = rotary_emb.inv_freq.shape[0] * 2
            x = torch.zeros(
                1, len_s, head_dim,
                dtype=self.running_dtype, device=self.running_device,
            )
            position_ids = torch.arange(
                len_p, len_p + len_s,
                dtype=torch.long, device=self.running_device,
            ).unsqueeze(0)
            with torch.no_grad():
                cos, sin = rotary_emb(x, position_ids)
            return {'position_embeddings': (cos, sin)}
        except Exception as e:
            print(f"get_pos_emb_args failed: {e}", flush=True)
            return {}

    def run_layer(self, layer, seq, **kwargs):
        """
        Run one transformer layer.  SSM (linear_attn) layers only exist in
        Qwen3.5-397B; they are numerically unstable in fp16/bf16 so we upcast
        them to float32 for the forward pass.
        """
        kwargs.pop('use_cache', None)
        kwargs.pop('position_ids', None)

        layer_type = getattr(layer, 'layer_type', None)
        if layer_type == 'linear_attention':
            # SSM layer: run in float32
            orig_dtype = seq.dtype
            layer.float()
            seq = seq.float()
            if 'position_embeddings' in kwargs and kwargs['position_embeddings'] is not None:
                cos, sin = kwargs['position_embeddings']
                kwargs['position_embeddings'] = (cos.float(), sin.float())
            with torch.no_grad():
                out = layer(seq, **kwargs)
            layer.to(orig_dtype)
            if isinstance(out, torch.Tensor):
                return out.to(orig_dtype), (out,)
            return out[0].to(orig_dtype), out
        else:
            out = layer(seq, **kwargs)
            if isinstance(out, torch.Tensor):
                return out, (out,)
            return out[0], out

    def move_layer_to_device(self, state_dict):
        """
        Override to handle transformers 5.x fused Qwen3MoeExperts.

        Pre-fused tensors (gate_up_proj, down_proj) are injected directly onto
        the Qwen3MoeExperts module, bypassing set_module_tensor_to_device which
        cannot traverse the fused expert module.  Non-expert keys are delegated
        to the base implementation.
        """
        import re

        fused_expert_re = re.compile(
            r'^(.*\.mlp\.experts)\.(gate_up_proj|down_proj)$'
        )

        remaining = {}
        expert_tensors = {}

        for k, v in state_dict.items():
            m = fused_expert_re.match(k)
            if m:
                experts_path, attr = m.group(1), m.group(2)
                expert_tensors.setdefault(experts_path, {})[attr] = v
            else:
                remaining[k] = v

        for experts_path, attrs in expert_tensors.items():
            module = self.model
            for part in experts_path.split('.'):
                module = getattr(module, part)
            for attr, tensor in attrs.items():
                setattr(module, attr, torch.nn.Parameter(
                    tensor.to(device=self.running_device, dtype=self.running_dtype),
                    requires_grad=False,
                ))

        if remaining:
            return super().move_layer_to_device(remaining)
        return []


# ---------------------------------------------------------------------------
# Qwen3 dense (non-MoE)
# ---------------------------------------------------------------------------

class AirLLMQwen3(AirLLMBaseModel):
    """
    AirLLM handler for Qwen3 dense models (Qwen3ForCausalLM).
    Examples: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-14B, Qwen3-32B
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()

    def set_layer_names_dict(self):
        path = getattr(self, 'model_local_path_or_repo_id', None) or ''
        if _is_qwen35_dense_conditional(path):
            self.layer_names_dict = {
                'embed': 'model.language_model.embed_tokens',
                'layer_prefix': 'model.language_model.layers',
                'norm': 'model.language_model.norm',
                'lm_head': 'lm_head',
            }
        else:
            self.layer_names_dict = {
                'embed': 'model.embed_tokens',
                'layer_prefix': 'model.layers',
                'norm': 'model.norm',
                'lm_head': 'lm_head',
            }

    def init_model(self):
        self.model = None
        self.hf_quantizer = None

        if _is_qwen35_dense_conditional(self.config):
            print("Qwen3.5 dense: building conditional-generation skeleton...")
            try:
                with init_empty_weights():
                    self.model = _build_qwen35_dense_skeleton(self.config)
            except Exception as e:
                clean_memory()
                raise RuntimeError(
                    f"Failed to build Qwen3.5 dense model skeleton: {e}\n"
                    "Ensure transformers>=5.3.0 is installed."
                ) from e

            quantization_config = getattr(self.config, "quantization_config", None)
            if quantization_config is not None:
                self.hf_quantizer = AutoHfQuantizer.from_config(
                    quantization_config, pre_quantized=True
                )
                device_map = self.hf_quantizer.update_device_map(None)
                self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)

            self.model.eval()
            self.model.tie_weights()
            self.set_layers_from_layer_names()

            for buffer_name, buffer in self.model.named_buffers():
                set_module_tensor_to_device(
                    self.model, buffer_name, self.running_device,
                    value=buffer, dtype=self.running_dtype,
                )

            try:
                self.model.model.language_model.rotary_emb.to(
                    device=self.running_device, dtype=self.running_dtype
                )
            except Exception:
                pass
            self._init_strategy = 'qwen35_dense'
        else:
            super().init_model()

    def _init_model_fast(self):
        if getattr(self, '_init_strategy', None) == 'qwen35_dense':
            with init_empty_weights():
                self.model = _build_qwen35_dense_skeleton(self.config)

            quantization_config = getattr(self.config, "quantization_config", None)
            if quantization_config is not None:
                self.hf_quantizer = AutoHfQuantizer.from_config(
                    quantization_config, pre_quantized=True
                )
                device_map = self.hf_quantizer.update_device_map(None)
                self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)

            self.model.eval()
            self.model.tie_weights()
            self.set_layers_from_layer_names()

            for buffer_name, buffer in self.model.named_buffers():
                set_module_tensor_to_device(
                    self.model, buffer_name, self.running_device,
                    value=buffer, dtype=self.running_dtype,
                )
            return
        super()._init_model_fast()

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        if _is_qwen35_dense_conditional(self.config):
            return {}
        return super().get_attention_mask_args(full_attention_mask, len_p, len_s)

    def get_pos_emb_args(self, len_p, len_s):
        if not _is_qwen35_dense_conditional(self.config):
            return super().get_pos_emb_args(len_p, len_s)
        try:
            rotary_emb = self.model.model.language_model.rotary_emb
            head_dim = rotary_emb.inv_freq.shape[0] * 2
            x = torch.zeros(
                1, len_s, head_dim,
                dtype=self.running_dtype, device=self.running_device,
            )
            position_ids = torch.arange(
                len_p, len_p + len_s,
                dtype=torch.long, device=self.running_device,
            ).unsqueeze(0)
            with torch.no_grad():
                cos, sin = rotary_emb(x, position_ids)
            return {'position_embeddings': (cos, sin)}
        except Exception as e:
            print(f"Qwen3.5 dense get_pos_emb_args failed: {e}", flush=True)
            return {}
