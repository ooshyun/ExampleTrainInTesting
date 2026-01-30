"""
Tests for GPU Kernel Task utilities.
"""
import numpy as np
import pytest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from ttt_discover.state import GPUKernelState
from ttt_discover.gpu_kernel_task import (
    KernelTask,
    MockKernelEvaluator,
    TRIMUL_TASK,
    MLA_DECODE_TASK,
    create_improvement_prompt,
    save_kernel_discovery_results,
)


class TestKernelTask:
    """Tests for KernelTask."""

    def test_trimul_task_definition(self):
        """Test TriMul task is properly defined."""
        assert TRIMUL_TASK.name == "trimul"
        assert TRIMUL_TASK.target_runtime_us > 0
        assert len(TRIMUL_TASK.test_configs) > 0
        assert TRIMUL_TASK.gpu_type == "H100"

    def test_mla_decode_task_definition(self):
        """Test MLA decode task is properly defined."""
        assert MLA_DECODE_TASK.name == "mla_decode"
        assert MLA_DECODE_TASK.target_runtime_us > 0
        assert MLA_DECODE_TASK.gpu_type == "H200"

    def test_task_to_dict(self):
        """Test task serialization."""
        d = TRIMUL_TASK.to_dict()
        assert "name" in d
        assert "target_runtime_us" in d
        assert "test_configs" in d


class TestMockKernelEvaluator:
    """Tests for MockKernelEvaluator."""

    def test_init(self):
        """Test evaluator initialization."""
        evaluator = MockKernelEvaluator(TRIMUL_TASK)
        assert evaluator._eval_count == 0
        assert evaluator._best_runtime > TRIMUL_TASK.target_runtime_us

    def test_evaluate_invalid_code(self):
        """Test evaluation of invalid code."""
        evaluator = MockKernelEvaluator(TRIMUL_TASK)
        parent = GPUKernelState(timestep=0, value=-2000, runtime_us=2000)

        result = evaluator.evaluate("", parent)
        assert result["score"] == 0.0
        assert result["correctness"] == 0.0

    def test_evaluate_missing_kernel(self):
        """Test evaluation of code without custom_kernel."""
        evaluator = MockKernelEvaluator(TRIMUL_TASK)
        parent = GPUKernelState(timestep=0, value=-2000, runtime_us=2000)

        code = "import torch\ndef foo(): pass"
        result = evaluator.evaluate(code, parent)
        assert result["score"] == 0.0
        assert "missing_kernel" in str(result.get("metrics", {}))

    def test_evaluate_valid_kernel(self):
        """Test evaluation of valid-looking kernel code."""
        evaluator = MockKernelEvaluator(TRIMUL_TASK)
        parent = GPUKernelState(timestep=0, value=-2000, runtime_us=2000)

        code = """
import triton
import triton.language as tl

@triton.jit
def kernel():
    x = tl.load(ptr)
    tl.store(out, x)

def custom_kernel(data):
    return kernel(data)
"""
        # Run multiple times since evaluation has randomness
        scores = []
        for _ in range(10):
            result = evaluator.evaluate(code, parent)
            scores.append(result["score"])

        # Should have some successful evaluations
        assert max(scores) > 0

    def test_improvement_over_parent(self):
        """Test that good kernels can improve over parent."""
        evaluator = MockKernelEvaluator(TRIMUL_TASK, improvement_bias=0.02)

        # Start with poor parent
        parent = GPUKernelState(
            timestep=0,
            value=-3000,  # 3000µs
            runtime_us=3000,
        )

        code = """
import triton
import triton.language as tl

@triton.jit
def kernel(BLOCK_M: tl.constexpr):
    x = tl.load(ptr)
    tl.store(out, x)

def custom_kernel(data):
    return kernel[grid](data, BLOCK_M=128, num_warps=8)
"""

        # Evaluate many times
        improvements = []
        for _ in range(20):
            result = evaluator.evaluate(code, parent)
            if result["correctness"] > 0:
                runtime = result["runtime_us"]
                if runtime < parent.runtime_us:
                    improvements.append(parent.runtime_us - runtime)

        # Should see some improvements
        assert len(improvements) > 0

    def test_get_stats(self):
        """Test statistics retrieval."""
        evaluator = MockKernelEvaluator(TRIMUL_TASK)
        parent = GPUKernelState(timestep=0, value=-2000, runtime_us=2000)

        code = "def custom_kernel(data): pass"
        for _ in range(5):
            evaluator.evaluate(code, parent)

        stats = evaluator.get_stats()
        assert stats["eval_count"] == 5
        assert "best_runtime" in stats
        assert "target_runtime" in stats


class TestImprovementPrompt:
    """Tests for improvement prompt generation."""

    def test_prompt_generation(self):
        """Test basic prompt generation."""
        state = GPUKernelState(
            timestep=0,
            code="# Previous code",
            value=-1500,
            runtime_us=1500,
        )

        prompt = create_improvement_prompt(TRIMUL_TASK, state)

        assert TRIMUL_TASK.name in prompt
        assert "Previous code" in prompt
        assert str(TRIMUL_TASK.target_runtime_us) in prompt

    def test_prompt_with_gap(self):
        """Test prompt includes gap information."""
        state = GPUKernelState(
            timestep=0,
            code="# Code",
            value=-1500,
            runtime_us=1500,
        )

        prompt = create_improvement_prompt(TRIMUL_TASK, state)

        assert "1500" in prompt  # Current runtime
        assert "GAP" in prompt.upper() or "gap" in prompt.lower()

    def test_prompt_without_runtime(self):
        """Test prompt when no previous runtime."""
        state = GPUKernelState(
            timestep=-1,
            code="",
            value=None,
            runtime_us=None,
        )

        prompt = create_improvement_prompt(TRIMUL_TASK, state)

        assert "No previous attempt" in prompt or "TARGET" in prompt.upper()


class TestSaveKernelResults:
    """Tests for saving kernel discovery results."""

    def test_save_results(self, tmp_path):
        """Test saving kernel results."""
        solutions = [
            {
                "step": i,
                "code": f"# Kernel {i}\ndef custom_kernel(data): pass",
                "reward": 0.5 + i * 0.1,
                "metrics": {"runtime_us": 1500 - i * 100},
            }
            for i in range(5)
        ]

        output_dir = str(tmp_path / "results")
        results = save_kernel_discovery_results(
            solutions, TRIMUL_TASK, output_dir, top_k=3
        )

        assert os.path.exists(output_dir)
        assert os.path.exists(os.path.join(output_dir, "task.json"))
        assert os.path.exists(os.path.join(output_dir, "results.json"))
        assert os.path.exists(os.path.join(output_dir, "kernels"))

    def test_save_kernels_ranked(self, tmp_path):
        """Test kernels are saved in ranked order."""
        solutions = [
            {"code": "# Low", "reward": 0.1, "step": 0, "metrics": {}},
            {"code": "# High", "reward": 0.9, "step": 1, "metrics": {}},
            {"code": "# Mid", "reward": 0.5, "step": 2, "metrics": {}},
        ]

        output_dir = str(tmp_path / "results")
        save_kernel_discovery_results(solutions, TRIMUL_TASK, output_dir, top_k=3)

        kernels_dir = os.path.join(output_dir, "kernels")
        files = os.listdir(kernels_dir)

        # Should have summary and kernel files
        assert "summary.json" in files

        # Rank 1 should be highest score
        rank1_files = [f for f in files if "rank01" in f]
        assert len(rank1_files) == 1
        assert "0.9" in rank1_files[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
