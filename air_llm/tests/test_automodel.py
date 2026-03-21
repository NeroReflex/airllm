import sys
import unittest

#sys.path.insert(0, '../airllm')

from ..airllm.auto_model import AutoModel



class TestAutoModel(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

    def test_auto_model_should_return_correct_model(self):
        mapping_dict = {
            'garage-bAInd/Platypus2-7B': 'AirLLMLlama2',
            'Qwen/Qwen-7B': 'AirLLMQWen',
            'internlm/internlm-chat-7b': 'AirLLMInternLM',
            'THUDM/chatglm3-6b-base': 'AirLLMChatGLM',
            'baichuan-inc/Baichuan2-7B-Base': 'AirLLMBaichuan',
            'mistralai/Mistral-7B-Instruct-v0.1': 'AirLLMMistral',
            'mistralai/Mixtral-8x7B-v0.1': 'AirLLMMixtral'
        }


        for k,v in mapping_dict.items():
            module, cls = AutoModel.get_module_class(k)
            self.assertEqual(cls, v, f"expecting {v}")

    def test_qwen3_architecture_detection(self):
        """Validate Qwen3 dense and MoE architectures are dispatched correctly.

        Uses unittest.mock to avoid network access: the real get_module_class
        calls AutoConfig.from_pretrained, so we inject a fake config object.
        """
        from unittest.mock import patch, MagicMock

        cases = [
            ("Qwen3ForCausalLM",    "AirLLMQwen3"),
            ("Qwen3MoeForCausalLM", "AirLLMQwen3Moe"),
            ("Qwen3_5MoeForConditionalGeneration", "AirLLMQwen3Moe"),
            ("GptOssForCausalLM", "AirLLMGPTOss"),
        ]

        for arch, expected_cls in cases:
            fake_config = MagicMock()
            fake_config.architectures = [arch]
            with patch(
                "airllm.auto_model.AutoConfig.from_pretrained",
                return_value=fake_config,
            ):
                _, cls = AutoModel.get_module_class("fake/model-id")
            self.assertEqual(
                cls, expected_cls,
                f"Architecture '{arch}' should map to {expected_cls}, got {cls}",
            )


class TestAirLLMGPTOss(unittest.TestCase):
    """Unit tests for AirLLMGPTOss that don't require downloading the model."""

    def _bare(self):
        """Return a bare AirLLMGPTOss instance without calling __init__."""
        import torch
        from ..airllm.airllm_gpt_oss import AirLLMGPTOss
        obj = object.__new__(AirLLMGPTOss)
        obj.running_device = "cpu"
        obj.running_dtype = torch.float16
        return obj

    def test_layer_names_dict(self):
        obj = self._bare()
        obj.set_layer_names_dict()
        self.assertEqual(obj.layer_names_dict["embed"], "model.embed_tokens")
        self.assertEqual(obj.layer_names_dict["layer_prefix"], "model.layers")
        self.assertEqual(obj.layer_names_dict["norm"], "model.norm")
        self.assertEqual(obj.layer_names_dict["lm_head"], "lm_head")

    def test_get_use_better_transformer_returns_false(self):
        obj = self._bare()
        self.assertFalse(obj.get_use_better_transformer())

    def test_automodel_routes_gpt_oss_architecture(self):
        from unittest.mock import patch, MagicMock
        fake_config = MagicMock()
        fake_config.architectures = ["GptOssForCausalLM"]
        with patch("airllm.auto_model.AutoConfig.from_pretrained", return_value=fake_config):
            _, cls = AutoModel.get_module_class("unsloth/gpt-oss-20b")
        self.assertEqual(cls, "AirLLMGPTOss")

    def test_init_model_uses_eager_attention(self):
        """init_model must load with attn_implementation='eager'."""
        from unittest.mock import patch, MagicMock
        from ..airllm.airllm_gpt_oss import AirLLMGPTOss
        obj = self._bare()
        obj.config = MagicMock()
        obj.model = None
        mock_model = MagicMock()
        mock_model.model.rotary_emb = None
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        with patch("airllm.airllm_gpt_oss.init_empty_weights", return_value=mock_ctx), \
             patch("airllm.airllm_gpt_oss.AutoModelForCausalLM.from_config",
                   return_value=mock_model) as mock_from_config, \
             patch.object(obj, "_finalize_model_init"), \
             patch.object(obj, "_move_rotary_emb_to_device"):
            obj.init_model()
        mock_from_config.assert_called_once()
        self.assertEqual(mock_from_config.call_args.kwargs.get("attn_implementation"), "eager")

    def test_get_pos_emb_args_returns_position_embeddings(self):
        """get_pos_emb_args must return {'position_embeddings': (cos, sin)}."""
        import torch
        from unittest.mock import MagicMock
        obj = self._bare()
        obj.config = MagicMock()
        obj.config.hidden_size = 64
        mock_cos = torch.zeros(1, 8, 32)
        mock_sin = torch.zeros(1, 8, 32)
        mock_rotary = MagicMock(return_value=(mock_cos, mock_sin))
        obj.model = MagicMock()
        obj.model.model.rotary_emb = mock_rotary
        result = obj.get_pos_emb_args(0, 8)
        self.assertIn("position_embeddings", result)
        cos, sin = result["position_embeddings"]
        self.assertIsInstance(cos, torch.Tensor)
        self.assertIsInstance(sin, torch.Tensor)
