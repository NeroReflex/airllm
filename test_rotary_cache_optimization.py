#!/usr/bin/env python
"""
Test rotary embedding cache optimization.

This script:
1. Verifies outputs are identical with/without cache enabled
2. Measures inference time with cache enabled vs disabled
3. Reports speedup percentage
"""

import time
import torch
import sys
import os

# Add airllm to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'air_llm'))

from airllm import AutoModel


def test_rotary_cache(model_id, num_runs=3, num_tokens=8):
    """Test rotary cache optimization.
    
    Args:
        model_id: HuggingFace model ID
        num_runs: Number of inference runs per configuration
        num_tokens: Max tokens to generate per inference
    """
    print(f"\n{'='*70}")
    print(f"Testing Rotary Cache Optimization")
    print(f"{'='*70}")
    print(f"Model: {model_id}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Runs per config: {num_runs}")
    print(f"Max tokens: {num_tokens}")
    print(f"{'='*70}\n")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # ===== Load model once =====
    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_id,
        device=device,
        max_seq_len=128,
        prefetching=False,
        layers_per_batch=1,
    )
    print(f"✓ Model loaded: {model_id}")
    print(f"  Rotary cache enabled (default): {model.enable_rotary_cache}\n")
    
    # ===== Test prompt =====
    test_prompt = "What is machine learning? Explain briefly:"
    print(f"Test prompt: '{test_prompt}'\n")
    
    # ===== RUN 1: With cache enabled (default) =====
    print("-" * 70)
    print("RUN 1: WITH ROTARY CACHE (Default)")
    print("-" * 70)
    
    model.enable_rotary_cache = True
    times_cached = []
    outputs_cached = []
    
    for run_idx in range(num_runs):
        # Tokenize
        toks = model.tokenizer(
            [test_prompt],
            return_tensors="pt",
            truncation=True,
            max_length=32,
        )
        
        # Time inference
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        
        out = model.generate(
            toks["input_ids"].to(device),
            max_new_tokens=num_tokens,
            use_cache=False,
        )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        
        times_cached.append(elapsed)
        output_text = model.tokenizer.decode(out[0], skip_special_tokens=True)
        outputs_cached.append(output_text)
        
        print(f"  Run {run_idx + 1}/{num_runs}: {elapsed:.4f}s")
    
    avg_time_cached = sum(times_cached) / len(times_cached)
    print(f"  Average: {avg_time_cached:.4f}s")
    
    # ===== RUN 2: With cache disabled =====
    print("\n" + "-" * 70)
    print("RUN 2: WITHOUT ROTARY CACHE (Disabled)")
    print("-" * 70)
    
    model.enable_rotary_cache = False
    times_uncached = []
    outputs_uncached = []
    
    for run_idx in range(num_runs):
        # Tokenize
        toks = model.tokenizer(
            [test_prompt],
            return_tensors="pt",
            truncation=True,
            max_length=32,
        )
        
        # Time inference
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        
        out = model.generate(
            toks["input_ids"].to(device),
            max_new_tokens=num_tokens,
            use_cache=False,
        )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        
        times_uncached.append(elapsed)
        output_text = model.tokenizer.decode(out[0], skip_special_tokens=True)
        outputs_uncached.append(output_text)
        
        print(f"  Run {run_idx + 1}/{num_runs}: {elapsed:.4f}s")
    
    avg_time_uncached = sum(times_uncached) / len(times_uncached)
    print(f"  Average: {avg_time_uncached:.4f}s")
    
    # ===== Analysis =====
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # 1. Verify output correctness
    print("\n✓ Output Correctness:")
    outputs_match = outputs_cached[0] == outputs_uncached[0]
    if outputs_match:
        print(f"  ✓ Outputs MATCH (cache is transparent)")
        print(f"    With cache:    '{outputs_cached[0][:80]}...'")
        print(f"    Without cache: '{outputs_uncached[0][:80]}...'")
    else:
        print(f"  ✗ Outputs DIFFER (unexpected!)")
        print(f"    With cache:    '{outputs_cached[0][:80]}...'")
        print(f"    Without cache: '{outputs_uncached[0][:80]}...'")
    
    # 2. Performance comparison
    print(f"\n✓ Performance:")
    print(f"  With cache:    {avg_time_cached:.4f}s (avg of {num_runs} runs)")
    print(f"  Without cache: {avg_time_uncached:.4f}s (avg of {num_runs} runs)")
    
    # 3. Speedup
    if avg_time_uncached > avg_time_cached:
        speedup_pct = ((avg_time_uncached - avg_time_cached) / avg_time_uncached) * 100
        time_saved = avg_time_uncached - avg_time_cached
        print(f"  Speedup: {speedup_pct:.2f}% ({time_saved:.4f}s saved per inference)")
        status = "✓ CACHE IMPROVES PERFORMANCE"
    elif avg_time_cached > avg_time_uncached:
        slowdown_pct = ((avg_time_cached - avg_time_uncached) / avg_time_uncached) * 100
        print(f"  Slowdown: {slowdown_pct:.2f}% (cache overhead detected)")
        status = "⚠ CACHE HAS OVERHEAD"
    else:
        print(f"  No measurable difference")
        status = "─ PERFORMANCE IDENTICAL"
    
    # 4. Summary
    print(f"\n{'='*70}")
    if outputs_match:
        print(f"{status} + OUTPUTS MATCH ✓")
    else:
        print(f"{status} but OUTPUTS DIFFER ✗")
    print(f"{'='*70}\n")
    
    return {
        "model": model_id,
        "outputs_match": outputs_match,
        "time_cached": avg_time_cached,
        "time_uncached": avg_time_uncached,
        "speedup_pct": ((avg_time_uncached - avg_time_cached) / avg_time_uncached * 100) if avg_time_uncached > 0 else 0,
    }


if __name__ == "__main__":
    # Use small model for quick testing
    test_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Run test
    result = test_rotary_cache(
        model_id=test_model,
        num_runs=3,
        num_tokens=8,
    )
    
    # Exit with appropriate code
    if result["outputs_match"]:
        print("✓ Test PASSED: Outputs match and optimization is working")
        sys.exit(0)
    else:
        print("✗ Test FAILED: Outputs don't match")
        sys.exit(1)
