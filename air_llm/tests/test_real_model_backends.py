import unittest

import torch

from airllm import AutoModel



def _ensure_cuda_or_skip(test_case):
    if not torch.cuda.is_available():
        test_case.skipTest("CUDA not available on this host")


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
