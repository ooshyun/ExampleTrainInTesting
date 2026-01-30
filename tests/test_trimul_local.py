#!/usr/bin/env python3
"""
Local test script for TriMul GPU kernel on small-scale inputs.
Designed to run on limited GPU resources to understand system operation.

Usage:
    python tests/test_trimul_local.py
    python tests/test_trimul_local.py --device cpu  # For no-GPU testing
    python tests/test_trimul_local.py --verbose     # Detailed output
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple
import math

import torch
from torch import nn, einsum

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "gpu_mode" / "bioml" / "trimul"))


# ============================================================================
# Small-Scale Test Configurations (for limited GPU memory)
# ============================================================================

SMALL_SCALE_TESTS = [
    # Minimal tests for system verification (< 100MB GPU memory each)
    {"seqlen": 16, "bs": 1, "dim": 64, "hiddendim": 32, "seed": 42, "nomask": True, "distribution": "normal"},
    {"seqlen": 16, "bs": 1, "dim": 64, "hiddendim": 32, "seed": 123, "nomask": False, "distribution": "normal"},
    {"seqlen": 32, "bs": 1, "dim": 64, "hiddendim": 32, "seed": 456, "nomask": True, "distribution": "normal"},
    {"seqlen": 32, "bs": 1, "dim": 128, "hiddendim": 64, "seed": 789, "nomask": False, "distribution": "normal"},
]

MEDIUM_SCALE_TESTS = [
    # Medium tests (< 500MB GPU memory each) - use if you have 4GB+ GPU
    {"seqlen": 64, "bs": 1, "dim": 128, "hiddendim": 64, "seed": 1001, "nomask": True, "distribution": "normal"},
    {"seqlen": 64, "bs": 2, "dim": 128, "hiddendim": 64, "seed": 1002, "nomask": False, "distribution": "normal"},
    {"seqlen": 128, "bs": 1, "dim": 256, "hiddendim": 128, "seed": 1003, "nomask": True, "distribution": "cauchy"},
]


# ============================================================================
# TriMul Reference Implementation (Simplified for local testing)
# ============================================================================

class TriMulReference(nn.Module):
    """Reference TriMul implementation based on AlphaFold3 architecture."""

    def __init__(self, dim: int, hidden_dim: int, device="cuda"):
        super().__init__()
        self.norm = nn.LayerNorm(dim, device=device)
        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.to_out_norm = nn.LayerNorm(hidden_dim, device=device)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, device=device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TriMul.

        Args:
            x: Input tensor [bs, seq_len, seq_len, dim]
            mask: Mask tensor [bs, seq_len, seq_len]

        Returns:
            Output tensor [bs, seq_len, seq_len, dim]
        """
        # Layer normalization
        x = self.norm(x)

        # Linear projections
        left = self.left_proj(x)
        right = self.right_proj(x)

        # Apply mask
        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        # Gate activations
        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        # Apply gates
        left = left * left_gate
        right = right * right_gate

        # Core einsum operation: '... i k d, ... j k d -> ... i j d'
        # This is the O(N³) triangular multiplication
        out = einsum('... i k d, ... j k d -> ... i j d', left, right)

        # Output normalization and projection
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class TriMulOptimized(nn.Module):
    """
    Optimized TriMul using bfloat16 for einsum.
    This is a simple optimization that can be further improved.
    """

    def __init__(self, dim: int, hidden_dim: int, device="cuda"):
        super().__init__()
        self.norm = nn.LayerNorm(dim, device=device)
        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, device=device, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, device=device, dtype=torch.float32)
        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, device=device, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, device=device, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, device=device, dtype=torch.float32)
        self.to_out_norm = nn.LayerNorm(hidden_dim, device=device)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, device=device, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x)
        right = self.right_proj(x)

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        # Use bfloat16 for einsum (key optimization)
        out = einsum('... i k d, ... j k d -> ... i j d',
                     left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


# ============================================================================
# Input Generation
# ============================================================================

def generate_input(
    seqlen: int,
    bs: int,
    dim: int,
    hiddendim: int,
    seed: int,
    nomask: bool,
    distribution: str = "normal",
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]:
    """Generate input data for TriMul testing."""

    config = {"hidden_dim": hiddendim, "dim": dim}

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Generate input tensor
    if distribution == "cauchy":
        u = torch.empty((bs, seqlen, seqlen, dim), device=device, dtype=torch.float32)
        u.uniform_(0.0, 1.0, generator=gen)
        input_tensor = 2.0 * torch.tan(math.pi * (u - 0.5))
    else:
        input_tensor = torch.randn(
            (bs, seqlen, seqlen, dim),
            device=device,
            dtype=torch.float32,
            generator=gen
        ).contiguous()

    # Generate mask
    if nomask:
        mask = torch.ones(bs, seqlen, seqlen, device=device)
    else:
        mask = torch.randint(0, 2, (bs, seqlen, seqlen), device=device, generator=gen).float()

    # Generate weights
    weights = {}
    weights["norm.weight"] = torch.randn(dim, device=device, dtype=torch.float32)
    weights["norm.bias"] = torch.randn(dim, device=device, dtype=torch.float32)
    weights["left_proj.weight"] = torch.randn(hiddendim, dim, device=device, dtype=torch.float32) / math.sqrt(hiddendim)
    weights["right_proj.weight"] = torch.randn(hiddendim, dim, device=device, dtype=torch.float32) / math.sqrt(hiddendim)
    weights["left_gate.weight"] = torch.randn(hiddendim, dim, device=device, dtype=torch.float32) / math.sqrt(hiddendim)
    weights["right_gate.weight"] = torch.randn(hiddendim, dim, device=device, dtype=torch.float32) / math.sqrt(hiddendim)
    weights["out_gate.weight"] = torch.randn(hiddendim, dim, device=device, dtype=torch.float32) / math.sqrt(hiddendim)
    weights["to_out_norm.weight"] = torch.randn(hiddendim, device=device, dtype=torch.float32)
    weights["to_out_norm.bias"] = torch.randn(hiddendim, device=device, dtype=torch.float32)
    weights["to_out.weight"] = torch.randn(dim, hiddendim, device=device, dtype=torch.float32) / math.sqrt(dim)

    return input_tensor, mask, weights, config


def load_weights_to_model(model: nn.Module, weights: Dict[str, torch.Tensor]):
    """Load pre-generated weights into a TriMul model."""
    model.norm.weight = nn.Parameter(weights['norm.weight'].clone())
    model.norm.bias = nn.Parameter(weights['norm.bias'].clone())
    model.left_proj.weight = nn.Parameter(weights['left_proj.weight'].clone())
    model.right_proj.weight = nn.Parameter(weights['right_proj.weight'].clone())
    model.left_gate.weight = nn.Parameter(weights['left_gate.weight'].clone())
    model.right_gate.weight = nn.Parameter(weights['right_gate.weight'].clone())
    model.out_gate.weight = nn.Parameter(weights['out_gate.weight'].clone())
    model.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].clone())
    model.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].clone())
    model.to_out.weight = nn.Parameter(weights['to_out.weight'].clone())


# ============================================================================
# Correctness Verification
# ============================================================================

def verify_correctness(
    output: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 2e-2,
    atol: float = 2e-2
) -> Tuple[bool, str]:
    """Verify output matches expected within tolerance."""
    if output.shape != expected.shape:
        return False, f"Shape mismatch: {output.shape} vs {expected.shape}"

    diff = torch.abs(output.to(torch.float32) - expected.to(torch.float32))
    tolerance = atol + rtol * torch.abs(expected)

    mismatched = diff > tolerance
    num_mismatched = mismatched.sum().item()

    if num_mismatched > 0:
        max_diff = diff.max().item()
        return False, f"{num_mismatched} elements mismatched. Max diff: {max_diff:.6f}"

    return True, f"All elements within tolerance. Max diff: {diff.max().item():.6f}"


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_kernel(
    model: nn.Module,
    input_tensor: torch.Tensor,
    mask: torch.Tensor,
    num_warmup: int = 3,
    num_runs: int = 10,
    device: str = "cuda"
) -> Dict[str, float]:
    """Benchmark a TriMul kernel implementation."""

    # Warmup runs
    for _ in range(num_warmup):
        _ = model(input_tensor, mask)
        if device == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    durations = []
    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        _ = model(input_tensor, mask)

        if device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            duration_ms = start_event.elapsed_time(end_event)
        else:
            duration_ms = (time.perf_counter() - start_time) * 1000

        durations.append(duration_ms)

    return {
        "mean_ms": sum(durations) / len(durations),
        "min_ms": min(durations),
        "max_ms": max(durations),
        "std_ms": (sum((d - sum(durations)/len(durations))**2 for d in durations) / len(durations)) ** 0.5
    }


# ============================================================================
# Main Test Runner
# ============================================================================

def run_single_test(
    test_config: Dict,
    device: str = "cuda",
    verbose: bool = False
) -> Tuple[bool, Dict]:
    """Run a single test case."""

    # Generate input
    input_tensor, mask, weights, config = generate_input(
        device=device,
        **test_config
    )

    # Create models
    ref_model = TriMulReference(config["dim"], config["hidden_dim"], device=device)
    opt_model = TriMulOptimized(config["dim"], config["hidden_dim"], device=device)

    # Load weights
    load_weights_to_model(ref_model, weights)
    load_weights_to_model(opt_model, weights)

    # Run reference
    with torch.no_grad():
        ref_output = ref_model(input_tensor, mask)

    # Run optimized
    with torch.no_grad():
        opt_output = opt_model(input_tensor, mask)

    # Verify correctness
    passed, message = verify_correctness(opt_output, ref_output)

    # Benchmark if verbose
    benchmark_results = {}
    if verbose:
        with torch.no_grad():
            ref_bench = benchmark_kernel(ref_model, input_tensor, mask, device=device)
            opt_bench = benchmark_kernel(opt_model, input_tensor, mask, device=device)
        benchmark_results = {
            "reference": ref_bench,
            "optimized": opt_bench,
            "speedup": ref_bench["mean_ms"] / opt_bench["mean_ms"] if opt_bench["mean_ms"] > 0 else 0
        }

    # Calculate memory usage
    input_size_mb = input_tensor.element_size() * input_tensor.nelement() / (1024 * 1024)

    return passed, {
        "config": test_config,
        "passed": passed,
        "message": message,
        "input_size_mb": input_size_mb,
        "benchmark": benchmark_results
    }


def estimate_memory_usage(seqlen: int, bs: int, dim: int, hiddendim: int) -> float:
    """Estimate GPU memory usage in MB."""
    # Input: [bs, seq_len, seq_len, dim]
    input_size = bs * seqlen * seqlen * dim * 4  # float32
    # Intermediate tensors (left, right, gates, out)
    intermediate_size = bs * seqlen * seqlen * hiddendim * 4 * 6  # 6 intermediate tensors
    # Output: [bs, seq_len, seq_len, dim]
    output_size = bs * seqlen * seqlen * dim * 4
    # Weights
    weight_size = (dim * hiddendim * 5 + dim * dim + hiddendim * 2 + dim * 2) * 4

    total_bytes = input_size + intermediate_size + output_size + weight_size
    return total_bytes / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Local TriMul GPU kernel testing")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                       help="Device to run tests on")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output with benchmarking")
    parser.add_argument("--scale", choices=["small", "medium", "all"], default="small",
                       help="Test scale: small (minimal GPU), medium (4GB+ GPU), all")
    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("=" * 70)
    print("🧪 TTT-Discover TriMul GPU Kernel Local Test")
    print("=" * 70)

    # Show device info
    if args.device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🖥️  Device: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print(f"🖥️  Device: CPU")

    # Select test cases
    if args.scale == "small":
        test_cases = SMALL_SCALE_TESTS
    elif args.scale == "medium":
        test_cases = SMALL_SCALE_TESTS + MEDIUM_SCALE_TESTS
    else:
        test_cases = SMALL_SCALE_TESTS + MEDIUM_SCALE_TESTS

    print(f"\n📋 Running {len(test_cases)} test cases (scale: {args.scale})")
    print("-" * 70)

    # Show estimated memory usage
    print("\n📊 Estimated Memory Usage:")
    for i, tc in enumerate(test_cases):
        mem_mb = estimate_memory_usage(tc["seqlen"], tc["bs"], tc["dim"], tc["hiddendim"])
        print(f"   Test {i+1}: seqlen={tc['seqlen']:4d}, bs={tc['bs']}, dim={tc['dim']:3d} → ~{mem_mb:.1f} MB")

    print("\n" + "=" * 70)
    print("🚀 Running Tests...")
    print("=" * 70)

    passed_count = 0
    failed_count = 0
    results = []

    for i, test_config in enumerate(test_cases):
        print(f"\n[Test {i+1}/{len(test_cases)}]", end=" ")
        print(f"seqlen={test_config['seqlen']}, bs={test_config['bs']}, dim={test_config['dim']}, "
              f"hiddendim={test_config['hiddendim']}, mask={not test_config['nomask']}")

        try:
            passed, result = run_single_test(test_config, args.device, args.verbose)
            results.append(result)

            if passed:
                passed_count += 1
                print(f"   ✅ PASSED - {result['message']}")
            else:
                failed_count += 1
                print(f"   ❌ FAILED - {result['message']}")

            if args.verbose and result.get("benchmark"):
                bench = result["benchmark"]
                print(f"   ⏱️  Reference: {bench['reference']['mean_ms']:.3f} ms")
                print(f"   ⏱️  Optimized: {bench['optimized']['mean_ms']:.3f} ms")
                print(f"   🚀 Speedup: {bench['speedup']:.2f}x")

        except RuntimeError as e:
            failed_count += 1
            print(f"   ❌ ERROR - {str(e)[:100]}")
            if "out of memory" in str(e).lower():
                print("   💡 Tip: Try --scale small or --device cpu")

    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    print(f"   ✅ Passed: {passed_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📈 Pass Rate: {passed_count / len(test_cases) * 100:.1f}%")

    if args.device == "cuda":
        max_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"   💾 Peak GPU Memory: {max_memory_allocated:.2f} GB")

    # System operation explanation
    print("\n" + "=" * 70)
    print("📚 System Operation Summary")
    print("=" * 70)
    print("""
    TriMul (Triangular Multiplicative Update) - AlphaFold3 핵심 연산

    1️⃣  Input: [batch, seq_len, seq_len, dim] 4D 텐서

    2️⃣  Processing Pipeline:
        ├─ LayerNorm: 입력 정규화
        ├─ Linear Projections: left/right 벡터 생성
        ├─ Masking: 마스크 적용 (단백질 구조 예측용)
        ├─ Gating: sigmoid 게이트로 정보 흐름 제어
        ├─ Einsum: '...ikd, ...jkd -> ...ijd' (O(N³) 핵심 연산)
        └─ Output Projection: 최종 출력 생성

    3️⃣  Key Optimization Point:
        - einsum 연산이 가장 compute-intensive
        - bfloat16 사용으로 속도 향상 가능
        - 추가 최적화: Triton 커널, Flash Attention 패턴 등

    4️⃣  TTT-Discover 역할:
        - 테스트 타임에 RL로 커널 최적화 코드 생성
        - 다양한 GPU 아키텍처에 맞춤 최적화
    """)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
