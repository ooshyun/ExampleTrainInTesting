"""
Entropic Advantage Estimator for TTT-Discover.

Key insight: Standard RL optimizes E[R] (expected reward), but discovery
problems want to find the SINGLE BEST solution, not average performance.

The entropic objective from the paper:

J_β(θ) = E_s [log E_a [e^{β(s)R(s,a)}]]

∇_θ J_β(θ) = E_{s,a} [w_β(s)(a) ∇_θ log π_θ(a|s)]

where:
    w_β(s)(a) = e^{β(s)R(s,a)} / E[e^{β(s)R(s,a)}]  (entropic weights)

Advantage with KL penalty:
    A(a; s) = w_β(s)(a) - 1 - λ log(π_θ(a|s) / π_θ₀(a|s))

where -1 is the baseline since E[w_β(s)] = 1.

As β → ∞, the entropic objective tends to the max.
"""
import numpy as np
from typing import Optional


class EntropicAdvantageEstimator:
    """
    Computes advantages using the entropic objective from TTT-Discover.

    Paper formula:
        w_β(s)(a) = exp(β(s) * R(s,a)) / E[exp(β(s) * R)]
        A(a; s) = w_β(s)(a) - 1 - λ * log(π_θ / π_θ₀)

    The baseline is -1 since E[w] = 1.

    Adaptive temperature β(s) is set by constraining KL divergence
    of the induced policy (Appendix A.1 of paper).
    """

    def __init__(
        self,
        beta: float = 2.0,
        min_temperature: float = 0.01,
        max_temperature: float = 10.0,
        kl_penalty_coef: float = 0.0,
    ):
        """
        Args:
            beta: Controls how much to emphasize high rewards (higher = more focus on max)
            min_temperature: Minimum adaptive temperature (1/β_max)
            max_temperature: Maximum adaptive temperature (1/β_min)
            kl_penalty_coef: λ for KL penalty term
        """
        self.beta = beta
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.kl_penalty_coef = kl_penalty_coef

    def compute_adaptive_temperature(self, rewards: np.ndarray) -> float:
        """
        Compute adaptive temperature ς(s) based on reward distribution.

        Formula: ς = β * std(R) / sqrt(N)

        This ensures:
        - Low variance rewards → low ς → sharp focus on best
        - High variance rewards → high ς → broader exploration
        """
        if len(rewards) <= 1:
            return self.max_temperature

        std = np.std(rewards)
        n = len(rewards)

        # ς = β * σ / √N
        temperature = self.beta * std / np.sqrt(n)

        # Clamp to reasonable range
        temperature = np.clip(temperature, self.min_temperature, self.max_temperature)

        return float(temperature)

    def compute_entropic_weights(
        self,
        rewards: np.ndarray,
        temperature: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute entropic weights: softmax(R / ς).

        High-reward solutions get exponentially higher weights.
        """
        if temperature is None:
            temperature = self.compute_adaptive_temperature(rewards)

        # Prevent division by zero
        temperature = max(temperature, 1e-8)

        # Softmax with temperature scaling
        scaled = rewards / temperature
        # Numerical stability: subtract max
        scaled = scaled - np.max(scaled)
        exp_scaled = np.exp(scaled)
        weights = exp_scaled / (np.sum(exp_scaled) + 1e-8)

        return weights

    def compute_advantages(
        self,
        rewards: np.ndarray,
        log_prob_ratios: Optional[np.ndarray] = None,
        baseline: Optional[float] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Compute entropic advantages for policy gradient.

        Paper formula:
            A(a; s) = w_β(s)(a) - 1 - λ * log(π_θ / π_θ₀)

        where:
            - w_β(s)(a) = exp(β*R) / E[exp(β*R)]  (entropic weight)
            - -1 is the baseline since E[w] = 1
            - λ is the KL penalty coefficient

        Args:
            rewards: Array of rewards for each rollout
            log_prob_ratios: Optional log(π_θ / π_θ₀) for KL penalty
            baseline: Optional baseline (default: uses -1 as per paper, since E[w]=1)

        Returns:
            advantages: Weighted advantages for each rollout
            stats: Dictionary with diagnostic statistics
        """
        rewards = np.asarray(rewards, dtype=np.float64)

        if len(rewards) == 0:
            return np.array([]), {"temperature": 0.0, "max_weight": 0.0}

        # Compute adaptive temperature
        temperature = self.compute_adaptive_temperature(rewards)

        # Compute entropic weights: w = exp(β*R) / E[exp(β*R)]
        weights = self.compute_entropic_weights(rewards, temperature)

        # Paper formula: A(a; s) = w_β(s)(a) - 1 - λ * log(π_θ / π_θ₀)
        # The -1 baseline comes from E[w] = 1
        advantages = weights - 1.0

        # Add KL penalty if provided
        if log_prob_ratios is not None and self.kl_penalty_coef > 0:
            kl_penalty = self.kl_penalty_coef * np.asarray(log_prob_ratios)
            advantages = advantages - kl_penalty

        # Normalize for stable training (optional, can be disabled)
        if len(advantages) > 1 and np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        stats = {
            "temperature": temperature,
            "max_weight": float(np.max(weights)),
            "min_weight": float(np.min(weights)),
            "weight_entropy": float(-np.sum(weights * np.log(weights + 1e-8))),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "reward_max": float(np.max(rewards)),
            "reward_min": float(np.min(rewards)),
            "baseline": -1.0,  # Paper uses -1 since E[w] = 1
        }

        return advantages, stats


def demonstrate_entropic_vs_standard():
    """
    Demonstrate the difference between standard and entropic advantages.

    Standard RL treats all rewards equally weighted.
    Entropic RL focuses on the highest rewards.
    """
    # Example: 8 rollouts with varying rewards
    rewards = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.8, 0.12, 0.18])

    print("=" * 60)
    print("Standard vs Entropic Advantage Comparison")
    print("=" * 60)
    print(f"\nRewards: {rewards}")
    print(f"Best reward: {np.max(rewards):.3f} (index {np.argmax(rewards)})")

    # Standard advantage (equal weights)
    baseline = np.mean(rewards)
    standard_adv = rewards - baseline
    print(f"\nStandard Advantages (equal weights):")
    print(f"  Baseline: {baseline:.3f}")
    print(f"  Advantages: {standard_adv}")
    print(f"  Weight on best: {1/len(rewards):.3f} (uniform)")

    # Entropic advantage
    estimator = EntropicAdvantageEstimator(beta=2.0)
    entropic_adv, stats = estimator.compute_advantages(rewards)

    print(f"\nEntropic Advantages (focus on max):")
    print(f"  Temperature ς: {stats['temperature']:.4f}")
    print(f"  Weight on best: {stats['max_weight']:.3f}")
    print(f"  Weight entropy: {stats['weight_entropy']:.3f}")
    print(f"  Advantages: {entropic_adv}")

    print("\n" + "=" * 60)
    print("Key insight: Entropic weights focus training on the best solution!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_entropic_vs_standard()
