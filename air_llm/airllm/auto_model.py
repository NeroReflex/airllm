import importlib
from typing import Any
from typing import Type

from transformers import AutoConfig
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from airllm import AirLLMLlamaMlx


class AutoModel:
    def __init__(self) -> None:
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_module_class(
        cls,
        pretrained_model_name_or_path: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> tuple[str, str]:
        config_kwargs = {}
        if "hf_token" in kwargs:
            print(f"using hf_token")
            config_kwargs["token"] = kwargs["hf_token"]

        try:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=False,
                **config_kwargs,
            )
        except Exception:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                **config_kwargs,
            )

        arch = config.architectures[0] if config.architectures else ""

        if "GptOssForCausalLM" in arch:
            return "airllm", "AirLLMGPTOss"
        elif "MiniMaxM2ForCausalLM" in arch:
            return "airllm", "AirLLMMinimax"
        elif "DeepseekV3ForCausalLM" in arch:
            return "airllm", "AirLLMDeepseekV3"
        # Qwen3 MoE variants (check before Qwen2/Qwen to avoid prefix matches)
        elif "Qwen3_5Moe" in arch or "Qwen3MoeForCausalLM" in arch or "Qwen3NextForCausalLM" in arch:
            return "airllm", "AirLLMQwen3Moe"
        # Qwen3 / Qwen3.5 dense
        elif "Qwen3ForCausalLM" in arch or "Qwen3_5ForConditionalGeneration" in arch:
            return "airllm", "AirLLMQwen3"
        # Qwen2
        elif "Qwen2ForCausalLM" in arch:
            return "airllm", "AirLLMQWen2"
        elif "QWen" in arch:
            return "airllm", "AirLLMQWen"
        elif "Baichuan" in arch:
            return "airllm", "AirLLMBaichuan"
        elif "Glm4Moe" in arch:
            return "airllm", "AirLLMGLM4"
        elif "ChatGLM" in arch:
            return "airllm", "AirLLMChatGLM"
        elif "InternLM" in arch:
            return "airllm", "AirLLMInternLM"
        elif "SpeechT5ForTextToSpeech" in arch:
            return "airllm", "AirLLMSpeechT5"
        elif "MllamaForConditionalGeneration" in arch or "MllamaForCausalLM" in arch:
            return "airllm", "AirLLMMllama"
        elif "Mistral" in arch:
            return "airllm", "AirLLMMistral"
        elif "Mixtral" in arch:
            return "airllm", "AirLLMMixtral"
        elif "Llama" in arch:
            return "airllm", "AirLLMLlama2"
        else:
            print(
                f"unknown artichitecture: {arch}, try to use Llama2..."
            )
            return "airllm", "AirLLMLlama2"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        if is_on_mac_os:
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        module, class_name = AutoModel.get_module_class(
            pretrained_model_name_or_path, *inputs, **kwargs
        )
        module = importlib.import_module(module)
        class_: Type[Any] = getattr(module, class_name)
        return class_(pretrained_model_name_or_path, *inputs, **kwargs)
