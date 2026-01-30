"""
Tests for Entropic Advantage Estimator.

The entropic objective is key to TTT-Discover's success:
- Standard RL: optimizes E[R] (expected reward)
- Entropic RL: optimizes weighted sum focusing on MAX reward
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from ttt_discover.entropic_loss import EntropicAdvantageEstimator


class TestEntropicAdvantageEstimator:
    """Tests for EntropicAdvantageEstimator."""

    def test_init(self):
        """Test estimator initialization."""
        estimator = EntropicAdvantageEstimator(beta=2.0)
        assert estimator.beta == 2.0
        assert estimator.min_temperature > 0
        assert estimator.max_temperature > estimator.min_temperature

    def test_adaptive_temperature_single_reward(self):
        """Test temperature computation with single reward."""
        estimator = EntropicAdvantageEstimator()
        rewards = np.array([0.5])
        temp = estimator.compute_adaptive_temperature(rewards)
        # Single reward should give max temperature
        assert temp == estimator.max_temperature

    def test_adaptive_temperature_low_variance(self):
        """Test temperature with low variance rewards."""
        estimator = EntropicAdvantageEstimator(beta=2.0)
        # All rewards nearly identical
        rewards = np.array([0.5, 0.51, 0.49, 0.5, 0.5])
        temp = estimator.compute_adaptive_temperature(rewards)
        # Low variance should give low temperature
        assert temp < 1.0

    def test_adaptive_temperature_high_variance(self):
        """Test temperature with high variance rewards."""
        estimator = EntropicAdvantageEstimator(beta=2.0)
        # High variance rewards
        rewards = np.array([0.1, 0.9, 0.2, 0.8, 0.5])
        temp = estimator.compute_adaptive_temperature(rewards)
        # High variance should give higher temperature
        assert temp > 0.1

    def test_entropic_weights_sum_to_one(self):
        """Test that entropic weights sum to 1."""
        estimator = EntropicAdvantageEstimator()
        rewards = np.array([0.1, 0.3, 0.5, 0.8, 0.2])
        weights = estimator.compute_entropic_weights(rewards)

        assert len(weights) == len(rewards)
        assert abs(np.sum(weights) - 1.0) < 1e-6

    def test_entropic_weights_favor_high_rewards(self):
        """Test that high rewards get higher weights."""
        estimator = EntropicAdvantageEstimator()
        rewards = np.array([0.1, 0.5, 0.9])
        weights = estimator.compute_entropic_weights(rewards)

        # Highest reward should have highest weight
        assert weights[2] > weights[1] > weights[0]

    def test_entropic_weights_with_low_temperature(self):
        """Test weights become sharper with lower temperature."""
        estimator = EntropicAdvantageEstimator()
        rewards = np.array([0.1, 0.5, 0.9])

        weights_low_temp = estimator.compute_entropic_weights(rewards, temperature=0.1)
        weights_high_temp = estimator.compute_entropic_weights(rewards, temperature=1.0)

        # Low temperature should concentrate more on max
        assert weights_low_temp[2] > weights_high_temp[2]

    def test_compute_advantages_empty(self):
        """Test advantages with empty rewards."""
        estimator = EntropicAdvantageEstimator()
        advantages, stats = estimator.compute_advantages(np.array([]))

        assert len(advantages) == 0
        assert stats["temperature"] == 0.0

    def test_compute_advantages_shape(self):
        """Test advantages have correct shape."""
        estimator = EntropicAdvantageEstimator()
        rewards = np.array([0.1, 0.3, 0.5, 0.8, 0.2])
        advantages, stats = estimator.compute_advantages(rewards)

        assert len(advantages) == len(rewards)
        assert "temperature" in stats
        assert "max_weight" in stats

    def test_compute_advantages_normalized(self):
        """Test advantages are normalized."""
        estimator = EntropicAdvantageEstimator()
        rewards = np.array([0.1, 0.3, 0.5, 0.8, 0.2, 0.6, 0.7, 0.4])
        advantages, _ = estimator.compute_advantages(rewards)

        # Should be roughly zero mean and unit variance
        assert abs(np.mean(advantages)) < 0.1
        assert 0.5 < np.std(advantages) < 1.5

    def test_stats_contain_expected_keys(self):
        """Test stats dictionary has expected keys."""
        estimator = EntropicAdvantageEstimator()
        rewards = np.array([0.1, 0.3, 0.5, 0.8])
        _, stats = estimator.compute_advantages(rewards)

        expected_keys = [
            "temperature", "max_weight", "min_weight", "weight_entropy",
            "reward_mean", "reward_std", "reward_max", "reward_min", "baseline"
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_entropic_vs_standard_comparison(self):
        """Test that entropic weights differ from uniform."""
        estimator = EntropicAdvantageEstimator(beta=2.0)
        rewards = np.array([0.1, 0.2, 0.3, 0.5, 0.9])  # One high outlier

        weights = estimator.compute_entropic_weights(rewards)
        uniform_weight = 1.0 / len(rewards)

        # Highest reward should have more than uniform weight
        assert weights[4] > uniform_weight
        # Lowest reward should have less than uniform weight
        assert weights[0] < uniform_weight

    def test_baseline_option(self):
        """Test baseline follows paper formula (A = w - 1, so baseline = -1)."""
        estimator = EntropicAdvantageEstimator()
        rewards = np.array([0.5, 0.6, 0.7])

        _, stats_default = estimator.compute_advantages(rewards)

        # Paper formula: A(a;s) = w_β(s)(a) - 1
        # The baseline is -1 since E[w] = 1
        assert stats_default["baseline"] == -1.0


class TestEntropicWeightsMathematical:
    """Mathematical property tests for entropic weights."""

    def test_weights_are_probabilities(self):
        """Test weights are valid probabilities."""
        estimator = EntropicAdvantageEstimator()
        for _ in range(10):
            rewards = np.random.randn(20) * 0.5 + 0.5
            weights = estimator.compute_entropic_weights(rewards)

            assert np.all(weights >= 0), "Weights should be non-negative"
            assert abs(np.sum(weights) - 1.0) < 1e-6, "Weights should sum to 1"

    def test_monotonicity(self):
        """Test weight monotonicity with reward ordering."""
        estimator = EntropicAdvantageEstimator()
        # Strictly increasing rewards
        rewards = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        weights = estimator.compute_entropic_weights(rewards, temperature=0.5)

        # Weights should be strictly increasing
        for i in range(len(weights) - 1):
            assert weights[i] < weights[i+1], f"weights[{i}] should be < weights[{i+1}]"

    def test_temperature_limiting_behavior(self):
        """Test extreme temperature limits."""
        estimator = EntropicAdvantageEstimator()
        rewards = np.array([0.1, 0.5, 0.9])

        # Very low temperature: should concentrate on max
        weights_cold = estimator.compute_entropic_weights(rewards, temperature=0.001)
        assert weights_cold[2] > 0.99, "Cold temperature should concentrate on max"

        # Very high temperature: should approach uniform
        weights_hot = estimator.compute_entropic_weights(rewards, temperature=100.0)
        expected_uniform = 1.0 / len(rewards)
        assert all(abs(w - expected_uniform) < 0.1 for w in weights_hot), \
            "Hot temperature should approach uniform"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
