#!/usr/bin/env python3
"""
TTT-Discover Demo - Small-scale demonstration of the TTT-Discover algorithm.

This script demonstrates:
1. PUCT-based state sampling (exploitation + exploration)
2. Entropic objective (focus on MAX reward, not mean)
3. Test-time training loop

Run with: python run_ttt_discover_demo.py

For GPU kernel optimization demo, pass --task gpu_kernel
"""
import argparse
import os
import sys
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ttt_discover.trainer import TTTDiscoverTrainer, TrainerConfig
from ttt_discover.state import State, GPUKernelState
from ttt_discover.gpu_kernel_task import (
    TRIMUL_TASK,
    MockKernelEvaluator,
    create_improvement_prompt,
    save_kernel_discovery_results,
)
from ttt_discover.utils import print_training_summary, analyze_training_progress


def run_simple_demo():
    """
    Run a simple demo with a mock evaluation function.

    This demonstrates the core TTT-Discover concepts without
    requiring any GPU infrastructure.
    """
    print("=" * 70)
    print("TTT-Discover Simple Demo")
    print("Demonstrating: PUCT sampling + Entropic objective")
    print("=" * 70)

    # Simple evaluation: reward based on "code quality"
    def mock_evaluate(code: str, parent_state: State) -> dict:
        # Simulate improvement: longer code, certain patterns = better
        base_reward = len(code) / 200.0

        # Bonus for "good patterns"
        if "def " in code:
            base_reward += 0.1
        if "for " in code or "while " in code:
            base_reward += 0.1
        if "return " in code:
            base_reward += 0.1

        # Build on parent success
        if parent_state.value and np.random.random() < 0.3:
            base_reward = max(base_reward, parent_state.value + 0.02)

        # Add noise
        reward = max(0, base_reward + np.random.normal(0, 0.05))

        return {
            "score": min(reward, 1.0),
            "metrics": {"code_length": len(code)},
            "observation": f"Evaluated: score={reward:.4f}",
        }

    # Create initial state
    initial_state = State(
        timestep=-1,
        code="""# Initial solution
def solution():
    result = 0
    for i in range(10):
        result += i
    return result
""",
        value=0.15,
    )

    # Configure trainer
    config = TrainerConfig(
        num_steps=20,
        rollouts_per_step=8,
        entropic_beta=2.0,  # Focus on max rewards
        puct_c=1.0,  # Exploration constant
        max_buffer_size=100,
        log_dir="./logs/simple_demo",
        save_every=5,
        log_every=1,
    )

    # Create trainer
    trainer = TTTDiscoverTrainer(
        config=config,
        initial_state=initial_state,
        evaluate_fn=mock_evaluate,
    )

    # Run training
    result = trainer.train()

    # Print summary
    print_training_summary(result["history"], trainer.all_solutions)

    return result


def run_gpu_kernel_demo():
    """
    Run a GPU kernel optimization demo with mock evaluation.

    This demonstrates how TTT-Discover would work for GPU kernel
    optimization (like TriMul from the paper) without needing
    actual GPU infrastructure.
    """
    print("=" * 70)
    print("TTT-Discover GPU Kernel Demo")
    print(f"Task: {TRIMUL_TASK.name} - {TRIMUL_TASK.description}")
    print(f"Target: {TRIMUL_TASK.target_runtime_us:.2f} µs")
    print("=" * 70)

    # Create mock evaluator
    evaluator = MockKernelEvaluator(TRIMUL_TASK)

    # Wrapper for trainer
    def evaluate_kernel(code: str, parent_state: GPUKernelState) -> dict:
        return evaluator.evaluate(code, parent_state)

    # Mock code generator (simulates LLM)
    def mock_generate(state: GPUKernelState) -> str:
        """Generate mock Triton kernel code."""
        base_code = state.code or ""

        # Generate variations
        block_sizes = [64, 128, 256]
        num_warps = [4, 8]

        block_m = np.random.choice(block_sizes)
        block_n = np.random.choice(block_sizes)
        warps = np.random.choice(num_warps)

        code = f'''"""
Optimized TriMul kernel - Variation {np.random.randint(10000)}
BLOCK_M={block_m}, BLOCK_N={block_n}, num_warps={warps}
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _trimul_kernel(
    x_ptr, y_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_yk, stride_yn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        y = tl.load(y_ptr + offs_k[:, None] * stride_yk + offs_n[None, :] * stride_yn)
        acc += tl.dot(x, y)
        offs_k += BLOCK_K

    out = acc.to(tl.float16)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out)

def custom_kernel(data):
    input_tensor, mask, weights, config = data
    # Implementation using _trimul_kernel
    return input_tensor  # Placeholder
'''
        return code

    # Create initial state
    initial_state = GPUKernelState(
        timestep=-1,
        code="# No previous kernel",
        value=-2000.0,  # Performance is -runtime, so this is 2000µs
        runtime_us=2000.0,
        correctness=0.0,
        gpu_type=TRIMUL_TASK.gpu_type,
        task_name=TRIMUL_TASK.name,
    )

    # Configure trainer
    config = TrainerConfig(
        num_steps=30,
        rollouts_per_step=8,
        entropic_beta=2.0,
        puct_c=1.5,
        max_buffer_size=200,
        log_dir="./logs/gpu_kernel_demo",
        save_every=5,
        log_every=1,
    )

    # Create trainer
    trainer = TTTDiscoverTrainer(
        config=config,
        initial_state=initial_state,
        evaluate_fn=evaluate_kernel,
        generate_fn=mock_generate,
    )

    # Run training
    result = trainer.train()

    # Save results
    output_dir = f"./results/gpu_kernels/{TRIMUL_TASK.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_kernel_discovery_results(
        trainer.all_solutions,
        TRIMUL_TASK,
        output_dir,
        top_k=20,
    )

    # Print summary
    print_training_summary(result["history"], trainer.all_solutions)

    # Print evaluator stats
    print("\nEvaluator Statistics:")
    eval_stats = evaluator.get_stats()
    print(f"  Total evaluations: {eval_stats['eval_count']}")
    print(f"  Best runtime: {eval_stats['best_runtime']:.2f} µs")
    print(f"  Target runtime: {eval_stats['target_runtime']:.2f} µs")
    print(f"  Gap to target: {eval_stats['gap_to_target']:.2f} µs")

    return result


def run_entropic_comparison():
    """
    Compare entropic vs standard advantage estimation.

    This demonstrates why the entropic objective is important
    for discovery problems.
    """
    print("=" * 70)
    print("Entropic vs Standard Advantage Comparison")
    print("=" * 70)

    from ttt_discover.entropic_loss import EntropicAdvantageEstimator

    # Simulate rewards from a batch
    rewards = np.array([0.1, 0.15, 0.12, 0.18, 0.14, 0.85, 0.11, 0.13])
    print(f"\nRewards: {rewards}")
    print(f"Max reward: {rewards.max():.3f} (index {rewards.argmax()})")
    print(f"Mean reward: {rewards.mean():.3f}")

    # Standard advantage (uniform weights)
    print("\n--- Standard RL (uniform weights) ---")
    uniform_weight = 1.0 / len(rewards)
    baseline = rewards.mean()
    standard_adv = rewards - baseline
    print(f"Weight on best solution: {uniform_weight:.3f}")
    print(f"Advantage of best: {standard_adv[rewards.argmax()]:.3f}")
    print("All solutions contribute equally to gradient")

    # Entropic advantage
    print("\n--- Entropic (focus on max) ---")
    estimator = EntropicAdvantageEstimator(beta=2.0)
    entropic_adv, stats = estimator.compute_advantages(rewards)

    print(f"Adaptive temperature: {stats['temperature']:.4f}")
    print(f"Weight on best solution: {stats['max_weight']:.3f}")
    print(f"Weight entropy: {stats['weight_entropy']:.3f}")
    print(f"Advantage of best: {entropic_adv[rewards.argmax()]:.3f}")
    print("Best solution dominates gradient → faster convergence to optimum")

    # Show weight comparison
    print("\n--- Weight comparison ---")
    entropic_weights = estimator.compute_entropic_weights(rewards)
    print("Index | Reward | Uniform | Entropic")
    print("-" * 40)
    for i in range(len(rewards)):
        print(f"  {i}   | {rewards[i]:.3f}  | {uniform_weight:.3f}   | {entropic_weights[i]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="TTT-Discover Demo")
    parser.add_argument(
        "--task",
        choices=["simple", "gpu_kernel", "comparison"],
        default="simple",
        help="Demo type to run",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    if args.task == "simple":
        run_simple_demo()
    elif args.task == "gpu_kernel":
        run_gpu_kernel_demo()
    elif args.task == "comparison":
        run_entropic_comparison()


if __name__ == "__main__":
    main()
