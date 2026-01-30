"""
GPU Kernel Discovery Task for TTT-Discover.

This module provides utilities for:
- Defining GPU kernel optimization problems
- Evaluating kernel correctness and performance
- Saving discovered kernel solutions

For small-scale local testing, this provides mock evaluation.
For production, connect to actual GPU evaluation infrastructure.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional
import json
import os
from datetime import datetime

from .state import GPUKernelState
from .utils import save_best_kernels


@dataclass
class KernelTask:
    """Definition of a GPU kernel optimization task."""

    name: str
    description: str
    target_runtime_us: float  # Target runtime in microseconds
    reference_code: str  # Reference implementation
    test_configs: list[dict]  # Test configurations
    gpu_type: str = "H100"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "target_runtime_us": self.target_runtime_us,
            "reference_code": self.reference_code,
            "test_configs": self.test_configs,
            "gpu_type": self.gpu_type,
        }


# Example kernel tasks (from the paper)
TRIMUL_TASK = KernelTask(
    name="trimul",
    description="Triangle Multiplicative Update for AlphaFold3",
    target_runtime_us=1000.0,  # Target: 1000 µs (human best: 1371 µs)
    gpu_type="H100",
    test_configs=[
        {"seqlen": 256, "bs": 2, "dim": 128, "hidden_dim": 128},
        {"seqlen": 512, "bs": 1, "dim": 128, "hidden_dim": 128},
        {"seqlen": 768, "bs": 1, "dim": 128, "hidden_dim": 128},
        {"seqlen": 1024, "bs": 1, "dim": 128, "hidden_dim": 128},
    ],
    reference_code="""
# Reference TriMul implementation (PyTorch)
import torch
from torch import nn, einsum

class TriMul(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.left_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.left_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, mask):
        x = self.norm(x)
        left = self.left_proj(x) * mask.unsqueeze(-1) * self.left_gate(x).sigmoid()
        right = self.right_proj(x) * mask.unsqueeze(-1) * self.right_gate(x).sigmoid()
        out = einsum('... i k d, ... j k d -> ... i j d', left, right)
        out = self.to_out(self.to_out_norm(out) * self.out_gate(x).sigmoid())
        return out
""",
)

MLA_DECODE_TASK = KernelTask(
    name="mla_decode",
    description="Multi-head Latent Attention Decode for DeepSeek-V3",
    target_runtime_us=1700.0,  # Target: 1700 µs
    gpu_type="H200",
    test_configs=[
        {"batch_size": 1, "seq_len": 512, "n_heads": 32},
        {"batch_size": 1, "seq_len": 1024, "n_heads": 32},
        {"batch_size": 1, "seq_len": 2048, "n_heads": 32},
    ],
    reference_code="""
# Reference MLA Decode implementation
# See DeepSeek-V3 paper for details
""",
)


class MockKernelEvaluator:
    """
    Mock evaluator for testing without GPU infrastructure.

    Simulates kernel evaluation with realistic reward distributions.
    """

    def __init__(
        self,
        task: KernelTask,
        noise_scale: float = 0.1,
        improvement_bias: float = 0.01,
    ):
        """
        Args:
            task: Kernel task definition
            noise_scale: Scale of random noise in evaluations
            improvement_bias: Slight bias toward better solutions over time
        """
        self.task = task
        self.noise_scale = noise_scale
        self.improvement_bias = improvement_bias
        self._eval_count = 0
        self._best_runtime = task.target_runtime_us * 2  # Start worse than target

    def evaluate(self, code: str, parent_state: GPUKernelState) -> dict[str, Any]:
        """
        Evaluate a kernel implementation (mock).

        Returns:
            Dictionary with:
            - score: Higher is better (inversely related to runtime)
            - runtime_us: Simulated runtime
            - correctness: 1.0 if "correct", 0.0 otherwise
            - metrics: Additional evaluation metrics
        """
        import numpy as np

        self._eval_count += 1

        # Check for obvious errors
        if not code.strip() or "def " not in code:
            return {
                "score": 0.0,
                "runtime_us": float("inf"),
                "correctness": 0.0,
                "observation": "Invalid code: missing function definition",
                "metrics": {"error": "parse_error"},
            }

        # Check for triton kernel
        has_triton = "@triton.jit" in code
        has_kernel = "def custom_kernel" in code or "def kernel" in code

        if not has_kernel:
            return {
                "score": 0.0,
                "runtime_us": float("inf"),
                "correctness": 0.0,
                "observation": "Missing custom_kernel function",
                "metrics": {"error": "missing_kernel"},
            }

        # Simulate correctness (higher if has triton, based on code quality)
        correctness_prob = 0.3  # Base probability
        if has_triton:
            correctness_prob += 0.4
        if "tl.load" in code and "tl.store" in code:
            correctness_prob += 0.2

        correct = np.random.random() < correctness_prob

        if not correct:
            return {
                "score": 0.0,
                "runtime_us": float("inf"),
                "correctness": 0.0,
                "observation": "Kernel failed correctness check",
                "metrics": {"correctness_prob": correctness_prob},
            }

        # Simulate runtime based on code characteristics
        base_runtime = self.task.target_runtime_us * 1.5  # Start above target

        # Better runtime if parent was good
        # Handle both State and GPUKernelState (check for runtime_us attribute)
        parent_runtime_us = getattr(parent_state, 'runtime_us', None)
        if parent_runtime_us and parent_runtime_us < float("inf"):
            base_runtime = min(base_runtime, parent_runtime_us * 1.1)

        # Improvement factors based on code
        improvement = 1.0
        if "BLOCK_" in code:
            improvement *= 0.95  # Good: explicit blocking
        if "num_warps" in code:
            improvement *= 0.97  # Good: warp configuration
        if "float16" in code or "fp16" in code:
            improvement *= 0.98  # Good: mixed precision
        if len(code) > 1000:
            improvement *= 0.99  # Longer code often more optimized

        # Gradual improvement bias
        improvement *= (1 - self.improvement_bias * self._eval_count / 100)

        # Add noise
        noise = np.random.normal(0, self.noise_scale * base_runtime)
        runtime = max(
            self.task.target_runtime_us * 0.5,  # Lower bound
            base_runtime * improvement + noise,
        )

        # Track best
        if runtime < self._best_runtime:
            self._best_runtime = runtime

        # Score: inverse of runtime (higher is better)
        # Normalized so target runtime gives score ~1.0
        score = self.task.target_runtime_us / runtime

        return {
            "score": score,
            "runtime_us": runtime,
            "correctness": 1.0,
            "observation": f"Runtime: {runtime:.2f} µs (target: {self.task.target_runtime_us} µs)",
            "metrics": {
                "runtime_us": runtime,
                "target_runtime_us": self.task.target_runtime_us,
                "gap_us": runtime - self.task.target_runtime_us,
                "improvement_vs_parent": (
                    (parent_runtime_us - runtime) if parent_runtime_us else None
                ),
                "has_triton": has_triton,
                "code_length": len(code),
            },
        }

    def get_stats(self) -> dict:
        """Get evaluation statistics."""
        return {
            "eval_count": self._eval_count,
            "best_runtime": self._best_runtime,
            "target_runtime": self.task.target_runtime_us,
            "gap_to_target": self._best_runtime - self.task.target_runtime_us,
        }


def create_kernel_evaluation_fn(
    task: KernelTask,
    mock: bool = True,
) -> Callable[[str, GPUKernelState], dict]:
    """
    Create evaluation function for kernel optimization.

    Args:
        task: Kernel task definition
        mock: Use mock evaluation (True) or real GPU evaluation (False)

    Returns:
        Evaluation function(code, parent_state) -> result dict
    """
    if mock:
        evaluator = MockKernelEvaluator(task)

        def evaluate(code: str, parent_state: GPUKernelState) -> dict:
            return evaluator.evaluate(code, parent_state)

        return evaluate
    else:
        # For real evaluation, would connect to GPU infrastructure
        raise NotImplementedError(
            "Real GPU evaluation requires connection to Modal/GPU infrastructure. "
            "See tasks/gpu_mode/task.py for the production implementation."
        )


def create_improvement_prompt(
    task: KernelTask,
    current_state: GPUKernelState,
) -> str:
    """
    Create improvement prompt for LLM.

    This prompt guides the LLM to improve upon the current solution.
    """
    prompt = f"""You are an expert Triton engineer optimizing GPU kernels.

TASK: {task.name}
{task.description}

TARGET: {task.target_runtime_us:.2f} µs

CURRENT CODE:
```python
{current_state.code if current_state.code else "# No previous attempt"}
```

"""

    if current_state.runtime_us and current_state.runtime_us < float("inf"):
        gap = current_state.runtime_us - task.target_runtime_us
        prompt += f"""CURRENT RUNTIME: {current_state.runtime_us:.2f} µs
GAP TO TARGET: {gap:.2f} µs

"""

    prompt += """Your goal: Write an optimized Triton kernel that achieves faster runtime.

Tips:
- Use appropriate BLOCK sizes for the problem dimensions
- Consider mixed precision (fp16/bf16) where possible
- Fuse operations to reduce memory bandwidth
- Configure num_warps appropriately

Write your solution in a ```python ``` code block.
Define a function `custom_kernel(data)` as the entry point.
"""

    return prompt


def save_kernel_discovery_results(
    solutions: list[dict],
    task: KernelTask,
    output_dir: str,
    top_k: int = 20,
):
    """
    Save kernel discovery results to disk.

    Creates:
    - Individual kernel files for top solutions
    - Summary JSON with all metrics
    - Task definition file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save task definition
    task_path = os.path.join(output_dir, "task.json")
    with open(task_path, "w") as f:
        json.dump(task.to_dict(), f, indent=2)

    # Save best kernels as Python files
    kernels_dir = os.path.join(output_dir, "kernels")
    save_best_kernels(solutions, kernels_dir, top_k=top_k)

    # Save comprehensive results
    results = {
        "task": task.to_dict(),
        "saved_at": datetime.now().isoformat(),
        "total_evaluations": len(solutions),
        "top_k_saved": min(top_k, len(solutions)),
    }

    # Add statistics
    valid_solutions = [s for s in solutions if s.get("reward", 0) > 0]
    if valid_solutions:
        runtimes = [
            s["metrics"]["runtime_us"]
            for s in valid_solutions
            if s.get("metrics", {}).get("runtime_us")
        ]
        if runtimes:
            results["statistics"] = {
                "best_runtime_us": min(runtimes),
                "worst_runtime_us": max(runtimes),
                "mean_runtime_us": sum(runtimes) / len(runtimes),
                "target_runtime_us": task.target_runtime_us,
                "num_beat_target": sum(1 for r in runtimes if r < task.target_runtime_us),
            }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nKernel discovery results saved to: {output_dir}")
    print(f"  - Task: {task_path}")
    print(f"  - Kernels: {kernels_dir}")
    print(f"  - Results: {results_path}")

    return results
