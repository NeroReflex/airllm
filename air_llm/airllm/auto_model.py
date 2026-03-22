import importlib
from transformers import AutoConfig
from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from airllm import AirLLMLlamaMlx


class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if "hf_token" in kwargs:
            print(f"using hf_token")
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                token=kwargs["hf_token"],
            )
        else:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True
            )

        arch = config.architectures[0] if config.architectures else ""

        if "GptOssForCausalLM" in arch:
            return "airllm", "AirLLMGPTOss"
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
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if is_on_mac_os:
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        module, cls = AutoModel.get_module_class(
            pretrained_model_name_or_path, *inputs, **kwargs
        )
        module = importlib.import_module(module)
        class_ = getattr(module, cls)
        return class_(pretrained_model_name_or_path, *inputs, **kwargs)
