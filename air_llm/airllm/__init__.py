from __future__ import annotations

from importlib import import_module
from sys import platform
from typing import Any

is_on_mac_os = platform == "darwin"

if is_on_mac_os:
    _EXPORTS = {
        "AirLLMLlamaMlx": (".airllm_llama_mlx", "AirLLMLlamaMlx"),
        "AutoModel": (".auto_model", "AutoModel"),
    }
else:
    _EXPORTS = {
        "AirLLMLlama2": (".airllm", "AirLLMLlama2"),
        "AirLLMChatGLM": (".airllm_chatglm", "AirLLMChatGLM"),
        "AirLLMGLM4": (".airllm_glm4", "AirLLMGLM4"),
        "AirLLMGPTOss": (".airllm_gpt_oss", "AirLLMGPTOss"),
        "AirLLMMinimax": (".airllm_minimax", "AirLLMMinimax"),
        "AirLLMDeepseekV3": (".airllm_deepseek_v3", "AirLLMDeepseekV3"),
        "AirLLMQWen": (".airllm_qwen", "AirLLMQWen"),
        "AirLLMQWen2": (".airllm_qwen2", "AirLLMQWen2"),
        "AirLLMQwen3Moe": (".airllm_qwen3_moe", "AirLLMQwen3Moe"),
        "AirLLMQwen3": (".airllm_qwen3_moe", "AirLLMQwen3"),
        "AirLLMMllama": (".airllm_mllama", "AirLLMMllama"),
        "AirLLMSpeechT5": (".airllm_speecht5", "AirLLMSpeechT5"),
        "AirLLMBaichuan": (".airllm_baichuan", "AirLLMBaichuan"),
        "AirLLMInternLM": (".airllm_internlm", "AirLLMInternLM"),
        "AirLLMMistral": (".airllm_mistral", "AirLLMMistral"),
        "AirLLMMixtral": (".airllm_mixtral", "AirLLMMixtral"),
        "AirLLMBaseModel": (".airllm_base", "AirLLMBaseModel"),
        "AutoModel": (".auto_model", "AutoModel"),
        "split_and_save_layers": (".utils", "split_and_save_layers"),
        "NotEnoughSpaceException": (".utils", "NotEnoughSpaceException"),
    }

__all__ = ["is_on_mac_os", *_EXPORTS.keys()]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
