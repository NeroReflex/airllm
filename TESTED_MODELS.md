# Tested Models

This file records models and backends that have been exercised in this workspace.

## Real Smoke-Tested Models

| Model | Backend | Status | Notes |
| --- | --- | --- | --- |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | CUDA | Passed | Existing real-model smoke test |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | CPU | Passed | Existing real-model smoke test |
| `Qwen/Qwen2.5-0.5B-Instruct` | CUDA | Passed | Existing real-model smoke test |
| `Qwen/Qwen2.5-0.5B-Instruct` | CPU | Passed | Existing real-model smoke test |
| `unsloth/gpt-oss-20b` | CUDA | Passed | Single-pass smoke test |
| `unsloth/gpt-oss-20b` | CUDA | Passed | Two-pass same-instance reuse smoke test |
| `unsloth/gpt-oss-120b` | CUDA | Passed | End-to-end run completed successfully; first run took about 27 minutes due to download/splitting |
| `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` | CUDA | Passed | End-to-end run succeeded in about 194s after adding `Qwen3_5ForConditionalGeneration` routing, nested `model.language_model.*` support, and robust shard-prefix parsing |

## Backend / Device Validation

| Backend | Status | Notes |
| --- | --- | --- |
| CUDA | Passed | Primary supported path |
| CPU | Passed | Smoke-tested with TinyLlama and Qwen2.5-0.5B |
| DirectML | Not available on this machine | Linux host; `torch-directml` not installed; tests skip cleanly |
| Intel XPU (Linux) | Code path added, runtime unavailable on this machine | `torch.xpu.is_available()` is currently false; helper/tests added and skip cleanly |
| Vulkan | Not supported by AirLLM | Vulkan runtime may exist on host, but no AirLLM execution backend uses it |

## Gated / Expensive Tests

| Test | Gate | Notes |
| --- | --- | --- |
| GPT-OSS 120B smoke | `AIRLLM_RUN_GPT_OSS_120B=1` | Intentionally gated because it can take tens of minutes and download/split a very large checkpoint |
| Qwen3.5-27B smoke | `AIRLLM_RUN_QWEN35_27B=1` | Intentionally gated because first run may need to download/split a large checkpoint |

## Notes

- Architecture routing tests exist in `air_llm/tests/test_automodel.py` for:
  - `GptOssForCausalLM`
  - `Qwen3ForCausalLM`
  - `Qwen3_5ForConditionalGeneration`
  - `Qwen3MoeForCausalLM`
  - `Qwen3_5MoeForConditionalGeneration`
- Real backend smoke tests live in `air_llm/tests/test_real_model_backends.py`.
- Compatibility fixes verified for `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled`:
  - `Qwen3_5ForConditionalGeneration` now routes to `AirLLMQwen3`
  - dense Qwen3.5 wrapper layout uses `model.language_model.*`
  - shard splitting now handles nested layer prefixes via regex matching
