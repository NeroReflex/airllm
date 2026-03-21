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

