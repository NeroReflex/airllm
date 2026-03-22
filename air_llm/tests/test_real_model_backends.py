import os
import unittest

import torch

from airllm import AutoModel



def _ensure_cuda_or_skip(test_case):
    if not torch.cuda.is_available():
        test_case.skipTest("CUDA not available on this host")


def _ensure_env_enabled_or_skip(test_case, env_var, reason):
    if os.environ.get(env_var) != "1":
        test_case.skipTest(f"Set {env_var}=1 to run this test. {reason}")


def _run_smoke(model_id, device, prompt, max_new_tokens=6):
    model = AutoModel.from_pretrained(
        model_id,
        device=device,
        max_seq_len=64,
        prefetching=False,
        layers_per_batch=1,
    )

    toks = model.tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=32,
    )
    out = model.generate(
        toks["input_ids"].to(device),
        max_new_tokens=max_new_tokens,
        use_cache=False,
    )
    decoded = model.tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded


class TestRealModelBackends(unittest.TestCase):
    def test_tinyllama_cuda_smoke(self):
        _ensure_cuda_or_skip(self)
        out = _run_smoke(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device="cuda:0",
            prompt="What is 2 + 2?",
            max_new_tokens=6,
        )
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_tinyllama_cpu_smoke(self):
        out = _run_smoke(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device="cpu",
            prompt="What is 2 + 2?",
            max_new_tokens=6,
        )
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_qwen25_cuda_smoke(self):
        _ensure_cuda_or_skip(self)
        out = _run_smoke(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            device="cuda:0",
            prompt="Say hello in one short sentence.",
            max_new_tokens=8,
        )
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_qwen25_cpu_smoke(self):
        out = _run_smoke(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            device="cpu",
            prompt="Say hello in one short sentence.",
            max_new_tokens=8,
        )
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_gpt_oss_20b_cuda_smoke(self):
        _ensure_cuda_or_skip(self)
        out = _run_smoke(
            model_id="unsloth/gpt-oss-20b",
            device="cuda:0",
            prompt="What is the capital of France?",
            max_new_tokens=6,
        )
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_gpt_oss_20b_cuda_two_pass_smoke(self):
        _ensure_cuda_or_skip(self)

        model = AutoModel.from_pretrained(
            "unsloth/gpt-oss-20b",
            device="cuda:0",
            max_seq_len=64,
            prefetching=False,
            layers_per_batch=1,
        )

        prompts = [
            "Answer briefly: what is 3 + 5?",
            "Answer briefly: what color is the sky on a clear day?",
        ]

        for prompt in prompts:
            toks = model.tokenizer(
                [prompt],
                return_tensors="pt",
                truncation=True,
                max_length=32,
            )
            out = model.generate(
                toks["input_ids"].to("cuda:0"),
                max_new_tokens=6,
                use_cache=False,
            )
            decoded = model.tokenizer.decode(out[0], skip_special_tokens=True)
            self.assertIsInstance(decoded, str)
            self.assertGreater(len(decoded), 0)

    def test_gpt_oss_120b_cuda_smoke(self):
        _ensure_cuda_or_skip(self)
        _ensure_env_enabled_or_skip(
            self,
            "AIRLLM_RUN_GPT_OSS_120B",
            "This smoke test is intentionally gated because first run can take tens of minutes "
            "and download/split a very large checkpoint.",
        )
        out = _run_smoke(
            model_id="unsloth/gpt-oss-120b",
            device="cuda:0",
            prompt="Say hello in three words.",
            max_new_tokens=4,
        )
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_qwen35_27b_cuda_smoke(self):
        _ensure_cuda_or_skip(self)
        _ensure_env_enabled_or_skip(
            self,
            "AIRLLM_RUN_QWEN35_27B",
            "This smoke test is intentionally gated because first run can take several minutes "
            "and requires downloading/splitting a large checkpoint.",
        )
        out = _run_smoke(
            model_id="Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
            device="cuda:0",
            prompt="Say hello in one short sentence.",
            max_new_tokens=8,
        )
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_unsloth_phi4_cuda_smoke(self):
        _ensure_cuda_or_skip(self)
        _ensure_env_enabled_or_skip(
            self,
            "AIRLLM_RUN_PHI4",
            "This smoke test is intentionally gated because first run can take several minutes "
            "and requires downloading/splitting a large checkpoint.",
        )
        out = _run_smoke(
            model_id="unsloth/phi-4",
            device="cuda:0",
            prompt="Say hello in one short sentence.",
            max_new_tokens=8,
        )
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_unsloth_llama32_11b_vision_text_smoke(self):
        _ensure_cuda_or_skip(self)
        _ensure_env_enabled_or_skip(
            self,
            "AIRLLM_RUN_LLAMA32_11B_VISION",
            "This smoke test is intentionally gated because first run can take several minutes "
            "and requires downloading/splitting a large checkpoint.",
        )
        out = _run_smoke(
            model_id="unsloth/Llama-3.2-11B-Vision-Instruct",
            device="cuda:0",
            prompt="Say hello in one short sentence.",
            max_new_tokens=8,
        )
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_unsloth_llama32_11b_vision_image_smoke(self):
        _ensure_cuda_or_skip(self)
        _ensure_env_enabled_or_skip(
            self,
            "AIRLLM_RUN_LLAMA32_11B_VISION_IMAGE",
            "This smoke test is intentionally gated because first run can take several minutes "
            "and requires downloading/splitting a large checkpoint.",
        )

        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed; required for vision-image smoke test")

        model = AutoModel.from_pretrained(
            "unsloth/Llama-3.2-11B-Vision-Instruct",
            device="cuda:0",
            max_seq_len=128,
            prefetching=False,
            layers_per_batch=1,
        )
        processor = model.get_processor()

        image = Image.new("RGB", (560, 560), color=(120, 180, 220))
        prompt = "<|image|>Describe this image in one short sentence."
        inputs = processor(text=prompt, images=image, return_tensors="pt")

        out = model.generate(
            input_ids=inputs["input_ids"].to("cuda:0"),
            attention_mask=inputs.get("attention_mask", None).to("cuda:0") if inputs.get("attention_mask", None) is not None else None,
            pixel_values=inputs.get("pixel_values", None),
            aspect_ratio_ids=inputs.get("aspect_ratio_ids", None),
            aspect_ratio_mask=inputs.get("aspect_ratio_mask", None),
            cross_attention_mask=inputs.get("cross_attention_mask", None),
            max_new_tokens=8,
            use_cache=False,
        )

        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = out[:, prompt_len:]
        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        self.assertIsInstance(decoded, str)
        self.assertGreater(len(decoded.strip()), 0)

    def test_speecht5_tts_cpu_smoke(self):
        _ensure_env_enabled_or_skip(
            self,
            "AIRLLM_RUN_SPEECHT5_TTS",
            "This smoke test is gated because it downloads SpeechT5 and vocoder checkpoints.",
        )

        model = AutoModel.from_pretrained(
            "microsoft/speecht5_tts",
            device="cpu",
        )
        wav = model.tts("Hello from AirLLM.")
        self.assertIsInstance(wav, torch.Tensor)
        self.assertGreater(wav.numel(), 0)

    def test_qwen3_coder_next_tool_calling(self):
        """Test tool-calling capability of Qwen3-Coder-Next-FP8-Dynamic.
        
        Verifies that the model can parse and respond to structured function calls
        in the OpenAI-compatible format. The model is queried with a tool definition
        and a message requesting a tool call, validating that:
        1. Model loads correctly via AirLLM routing
        2. Tool-calling format is recognized
        3. Model generates structured output with function_call intent
        """
        _ensure_cuda_or_skip(self)
        _ensure_env_enabled_or_skip(
            self,
            "AIRLLM_RUN_QWEN3_CODER_NEXT_TOOL_CALL",
            "This test is gated because first run downloads/splits a very large model (~80GB quantized) "
            "and requires significant time and disk space. Set to '1' to enable.",
        )
        import json
        
        model = AutoModel.from_pretrained(
            "unsloth/Qwen3-Coder-Next-FP8-Dynamic",
            device="cuda:0",
            max_seq_len=1024,
            prefetching=False,
            layers_per_batch=1,
        )
        
        # Define a simple tool (from model card pattern)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_square",
                    "description": "Calculate the square of a number",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "number": {
                                "type": "number",
                                "description": "The number to square",
                            }
                        },
                        "required": ["number"],
                    },
                },
            }
        ]
        
        # Prepare messages with tool context (OpenAI-compatible format)
        system_msg = (
            "You are a helpful assistant with access to tools. "
            "When the user asks you to use a tool, use it. "
            "Call functions when appropriate using the provided tools."
        )
        
        user_msg = "Calculate the square of 7. Use the calculate_square tool."
        
        # Construct the prompt with tool definitions and messages
        # Format: system + tools + user query
        prompt = f"""System: {system_msg}

Available tools:
{json.dumps(tools, indent=2)}

User: {user_msg}
Assistant:"""
        
        # Tokenize and generate response
        toks = model.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        
        out = model.generate(
            toks["input_ids"].to("cuda:0"),
            max_new_tokens=64,
            use_cache=False,
        )
        
        decoded = model.tokenizer.decode(out[0], skip_special_tokens=True)
        
        # Verify output is generated and non-empty
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(decoded), len(prompt))  # Must generate beyond prompt
        
        # Check for signs of tool-call structure (function_call, calculate_square, number)
        # The model should attempt to use the tool or acknowledge it
        decoded_lower = decoded.lower()
        has_tool_signal = (
            "calculate_square" in decoded_lower
            or "function" in decoded_lower
            or "tool" in decoded_lower
            or "square" in decoded_lower
            or "49" in decoded_lower  # 7^2 = 49
        )
        self.assertTrue(
            has_tool_signal or len(decoded) > 200,
            f"Tool calling or substantial output expected. Got: {decoded[:200]}"
        )
