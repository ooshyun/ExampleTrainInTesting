"""
Utility functions for TTT-Discover.

Provides logging, saving, and analysis utilities for:
- Training logs
- GPU kernel solutions
- Performance analysis
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from typing import Any, Optional
import numpy as np


def save_training_log(history: list[dict], path: str):
    """
    Save training history to JSON file.

    Args:
        history: List of step statistics
        path: Output file path
    """
    # Add metadata
    data = {
        "metadata": {
            "saved_at": datetime.now().isoformat(),
            "num_steps": len(history),
        },
        "history": history,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_serializer)


def load_training_log(path: str) -> dict:
    """Load training history from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_kernel_solutions(
    solutions: list[dict],
    path: str,
    include_code: bool = True,
):
    """
    Save GPU kernel solutions to file.

    Args:
        solutions: List of solution dictionaries
        path: Output file path
        include_code: Whether to include full code (can be large)
    """
    # Optionally truncate code
    if not include_code:
        solutions = [
            {k: (v[:100] + "..." if k == "code" and isinstance(v, str) and len(v) > 100 else v)
             for k, v in sol.items()}
            for sol in solutions
        ]

    data = {
        "metadata": {
            "saved_at": datetime.now().isoformat(),
            "num_solutions": len(solutions),
            "include_full_code": include_code,
        },
        "solutions": solutions,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_serializer)


def load_kernel_solutions(path: str) -> list[dict]:
    """Load kernel solutions from file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("solutions", [])


def save_best_kernels(
    solutions: list[dict],
    output_dir: str,
    top_k: int = 10,
):
    """
    Save top-k best kernel solutions as separate Python files.

    Args:
        solutions: List of solution dictionaries with 'code' and 'reward'
        output_dir: Directory to save kernel files
        top_k: Number of top solutions to save
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sort by reward
    sorted_solutions = sorted(
        [s for s in solutions if s.get("reward") is not None],
        key=lambda x: x["reward"],
        reverse=True,
    )

    # Save top-k
    for i, sol in enumerate(sorted_solutions[:top_k]):
        filename = f"kernel_rank{i+1:02d}_score{sol['reward']:.4f}.py"
        filepath = os.path.join(output_dir, filename)

        header = f'''"""
GPU Kernel Solution - Rank {i+1}
Score: {sol['reward']:.4f}
Step: {sol.get('step', 'unknown')}
Parent Value: {sol.get('parent_value', 'unknown')}
"""

'''
        code = sol.get("code", "# No code")

        with open(filepath, "w") as f:
            f.write(header + code)

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    summary = {
        "saved_at": datetime.now().isoformat(),
        "num_kernels": min(top_k, len(sorted_solutions)),
        "top_scores": [s["reward"] for s in sorted_solutions[:top_k]],
        "kernels": [
            {
                "rank": i + 1,
                "score": s["reward"],
                "step": s.get("step"),
                "filename": f"kernel_rank{i+1:02d}_score{s['reward']:.4f}.py",
            }
            for i, s in enumerate(sorted_solutions[:top_k])
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {len(sorted_solutions[:top_k])} best kernels to {output_dir}")
    return summary


def analyze_training_progress(history: list[dict]) -> dict:
    """
    Analyze training progress from history.

    Returns summary statistics and trends.
    """
    if not history:
        return {"error": "No history to analyze"}

    rewards_mean = [h["reward_mean"] for h in history if "reward_mean" in h]
    rewards_max = [h["reward_max"] for h in history if "reward_max" in h]
    best_values = [h["best_value"] for h in history if h.get("best_value") is not None]

    analysis = {
        "num_steps": len(history),
        "reward_mean": {
            "initial": rewards_mean[0] if rewards_mean else None,
            "final": rewards_mean[-1] if rewards_mean else None,
            "improvement": (rewards_mean[-1] - rewards_mean[0]) if len(rewards_mean) >= 2 else 0,
        },
        "reward_max": {
            "initial": rewards_max[0] if rewards_max else None,
            "final": rewards_max[-1] if rewards_max else None,
            "global_max": max(rewards_max) if rewards_max else None,
        },
        "best_value": {
            "initial": best_values[0] if best_values else None,
            "final": best_values[-1] if best_values else None,
            "global_best": max(best_values) if best_values else None,
        },
    }

    # Compute improvement trend
    if len(best_values) >= 10:
        early = np.mean(best_values[:5])
        late = np.mean(best_values[-5:])
        analysis["improvement_trend"] = late - early

    return analysis


def format_runtime(runtime_us: float) -> str:
    """Format runtime in appropriate units."""
    if runtime_us < 1000:
        return f"{runtime_us:.2f} µs"
    elif runtime_us < 1_000_000:
        return f"{runtime_us/1000:.2f} ms"
    else:
        return f"{runtime_us/1_000_000:.2f} s"


def _json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def print_training_summary(history: list[dict], solutions: list[dict]):
    """Print a formatted training summary."""
    analysis = analyze_training_progress(history)

    print("\n" + "=" * 70)
    print("TTT-DISCOVER TRAINING SUMMARY")
    print("=" * 70)

    print(f"\nTraining Progress:")
    print(f"  Steps completed: {analysis['num_steps']}")
    print(f"  Total solutions: {len(solutions)}")

    print(f"\nReward Statistics:")
    rm = analysis.get("reward_mean", {})
    print(f"  Mean reward: {rm.get('initial', 0):.4f} → {rm.get('final', 0):.4f}")

    bv = analysis.get("best_value", {})
    print(f"  Best value: {bv.get('initial', 0):.4f} → {bv.get('final', 0):.4f}")
    print(f"  Global best: {bv.get('global_best', 0):.4f}")

    if "improvement_trend" in analysis:
        trend = analysis["improvement_trend"]
        direction = "↑" if trend > 0 else "↓" if trend < 0 else "→"
        print(f"  Improvement trend: {direction} {abs(trend):.4f}")

    # Top solutions
    if solutions:
        sorted_sols = sorted(
            [s for s in solutions if s.get("reward") is not None],
            key=lambda x: x["reward"],
            reverse=True,
        )[:5]

        print(f"\nTop 5 Solutions:")
        for i, sol in enumerate(sorted_sols):
            print(f"  {i+1}. Score: {sol['reward']:.4f} (step {sol.get('step', '?')})")

    print("=" * 70)
