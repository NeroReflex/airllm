from dataclasses import dataclass
import os


def _as_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return int(value)


@dataclass
class Settings:
    host: str = os.environ.get("AIRLLM_HOST", "0.0.0.0")
    port: int = int(os.environ.get("AIRLLM_PORT", "8000"))

    api_key: str = os.environ.get("AIRLLM_API_KEY", "")
    enforce_auth: bool = _as_bool(os.environ.get("AIRLLM_ENFORCE_AUTH"), False)

    model_id: str = os.environ.get("AIRLLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    hf_token: str = os.environ.get("HF_TOKEN", "")

    device: str = os.environ.get("AIRLLM_DEVICE", "cuda:0")
    # None means auto-infer from model config at load time.
    max_seq_len: int | None = _as_optional_int(os.environ.get("AIRLLM_MAX_SEQ_LEN"))
    max_new_tokens: int = int(os.environ.get("AIRLLM_MAX_NEW_TOKENS", "256"))
    temperature: float = float(os.environ.get("AIRLLM_TEMPERATURE", "0.2"))
    top_p: float = float(os.environ.get("AIRLLM_TOP_P", "0.95"))

    prefetching: bool = _as_bool(os.environ.get("AIRLLM_PREFETCHING"), True)
    layers_per_batch: str = os.environ.get("AIRLLM_LAYERS_PER_BATCH", "auto")

    lazy_load_model: bool = _as_bool(os.environ.get("AIRLLM_LAZY_LOAD_MODEL"), True)
    cache_dir: str = os.environ.get("AIRLLM_CACHE_DIR", os.path.expanduser("~/.cache/huggingface/hub"))
