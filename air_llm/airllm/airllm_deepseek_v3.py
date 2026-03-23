from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers import AutoModelForCausalLM, GenerationConfig

from .airllm_base import AirLLMBaseModel
from .utils import clean_memory


class AirLLMDeepseekV3(AirLLMBaseModel):
    """
    AirLLM handler for DeepSeek-V3 / DeepSeek-V3.1 CausalLM models.

    These models ship as custom remote code (unsloth-style repos).  The remote
    ``modeling_deepseek.py`` imports ``is_torch_fx_available`` which was removed
    in newer transformers.  We patch it back before the import so the remote
    code loads cleanly.  Using the remote model class is important because it
    stores experts as an indexed ``nn.ModuleList`` (matching the per-expert
    ``mlp.experts.{i}.gate/up/down_proj.weight`` checkpoint layout), whereas the
    built-in transformers ``DeepseekV3ForCausalLM`` uses a fused ``FP8Expert``
    module that does not match the checkpoint format.

    The model weights are block-quantised FP8 (``float8_e4m3fn``).  The FP8
    quantizer's ``preprocess_model`` cannot traverse the remote expert
    ``ModuleList``; we therefore skip it entirely and dequantise each weight
    shard on-the-fly inside ``move_layer_to_device`` using the per-block
    ``weight_scale_inv`` tensors stored alongside every FP8 weight.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False

    def get_generation_config(self):
        return GenerationConfig()

    @staticmethod
    def _patch_transformers_compat():
        """Re-add ``is_torch_fx_available`` removed from newer transformers.

        The unsloth DeepSeek remote modeling code imports this at module level.
        We add a shim so the import succeeds before the remote code is executed.
        """
        import transformers.utils.import_utils as _tiu
        if not hasattr(_tiu, "is_torch_fx_available"):
            try:
                import torch.fx  # noqa: F401 – just verify it exists
                _tiu.is_torch_fx_available = lambda: True
            except ImportError:
                _tiu.is_torch_fx_available = lambda: False

    def _normalize_deepseek_config(self):
        """Patch config attributes that exist on the native DeepseekV3Config but
        are absent from the unsloth remote-code config object."""
        cfg = self.config

        if getattr(cfg, "qk_head_dim", None) is None:
            cfg.qk_head_dim = int(getattr(cfg, "qk_nope_head_dim", 0)) + int(
                getattr(cfg, "qk_rope_head_dim", 0)
            )

        if getattr(cfg, "head_dim", None) is None:
            cfg.head_dim = int(getattr(cfg, "qk_rope_head_dim", 0))

        if getattr(cfg, "num_local_experts", None) is None:
            cfg.num_local_experts = int(getattr(cfg, "n_routed_experts", 0))

        # rope_interleave is read on every attention layer forward – must exist.
        if not hasattr(cfg, "rope_interleave"):
            cfg.rope_interleave = True

        rope_params = getattr(cfg, "rope_parameters", None)
        if isinstance(rope_params, dict):
            for key in ("factor", "beta_fast", "beta_slow"):
                val = rope_params.get(key)
                if isinstance(val, int):
                    rope_params[key] = float(val)

    def _finalize_model_init_deepseek(self):
        """Like ``_finalize_model_init`` but skips the FP8 quantizer.

        The HF FP8 quantizer's ``preprocess_model`` tries to replace the remote
        model's expert ``ModuleList`` with a fused ``FP8Expert``; that traversal
        fails with ``AttributeError: FP8Expert has no attribute '0'``.  We skip
        the quantizer entirely and dequantise FP8 weights manually in
        ``move_layer_to_device``.
        """
        self.hf_quantizer = None  # no quantizer; we handle FP8 manually below

        self.model.eval()
        self.model.tie_weights()
        self.set_layers_from_layer_names()

        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model, buffer_name, self.running_device,
                value=buffer, dtype=self.running_dtype,
            )

    def init_model(self):
        self.model = None
        self.hf_quantizer = None

        # Must happen before the remote modeling module is imported.
        self._patch_transformers_compat()
        self._normalize_deepseek_config()

        try:
            with init_empty_weights():
                # trust_remote_code=True is required: the remote model uses an
                # indexed ModuleList for MoE experts which matches the per-expert
                # checkpoint layout.
                self.model = AutoModelForCausalLM.from_config(
                    self.config, trust_remote_code=True
                )
        except Exception as e:
            clean_memory(self.running_device)
            raise RuntimeError(
                f"Failed to build DeepSeek-V3 model skeleton: {e}"
            ) from e

        self._init_strategy = "deepseek_v3_remote"
        self._finalize_model_init_deepseek()

    def _init_model_fast(self):
        """Fast model re-creation for repeated forward passes."""
        self._patch_transformers_compat()
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(
                self.config, trust_remote_code=True
            )
        self._finalize_model_init_deepseek()

    def move_layer_to_device(self, state_dict):
        """Dequantise block-FP8 weights, then delegate to the base loader.

        The checkpoint stores weights as ``float8_e4m3fn`` with per-block scale
        factors in companion ``*_scale_inv`` tensors (shape
        ``[ceil(out/128), ceil(in/128)]``).  Dequantisation:
            ``w_bf16 = w_fp8.to(float32) * scale_inv_expanded``
        We dequantise here (once per shard, on CPU) so the base class can load
        clean bfloat16/float16 tensors via ``set_module_tensor_to_device``,
        which works with standard ``nn.Linear`` expert modules.
        """
        import torch

        # Base class already filters rotary_emb; we also filter weight_scale_inv
        # keys (consumed below during dequantisation).
        state_dict = {
            k: v for k, v in state_dict.items() if "rotary_emb" not in k
        }

        dequantised = {}
        consumed = set()

        for key, tensor in state_dict.items():
            if key in consumed:
                continue
            if tensor.dtype == torch.float8_e4m3fn:
                scale_inv_key = key + "_scale_inv"
                if scale_inv_key in state_dict:
                    scale_inv = state_dict[scale_inv_key]  # [out_blocks, in_blocks] float32
                    out_size, in_size = tensor.shape
                    # Tile scale_inv to full weight dimensions (block size = 128)
                    scale_full = (
                        scale_inv
                        .repeat_interleave(128, dim=0)[:out_size, :]
                        .repeat_interleave(128, dim=1)[:, :in_size]
                    )
                    dequantised[key] = (
                        tensor.to(torch.float32).mul_(scale_full).to(self.running_dtype)
                    )
                    consumed.add(key)
                    consumed.add(scale_inv_key)
                else:
                    # No scale available – plain cast (should rarely happen)
                    dequantised[key] = tensor.to(self.running_dtype)
                    consumed.add(key)

        # Merge: dequantised FP8 weights + remaining non-FP8 tensors
        merged = {**{k: v for k, v in state_dict.items() if k not in consumed}, **dequantised}

        # Delegate: the base class sets each tensor onto the model with
        # set_module_tensor_to_device (hf_quantizer is None so no bnb path).
        # Bypass the base's own rotary_emb filter (already applied above).
        from accelerate.utils.modeling import set_module_tensor_to_device as _smtd
        layers = []
        for param_name, param in merged.items():
            layers.append(param_name)
            _smtd(
                self.model, param_name, self.running_device,
                value=param, dtype=self.running_dtype,
            )
        return layers
