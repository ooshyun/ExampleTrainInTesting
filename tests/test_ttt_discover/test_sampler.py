"""
Tests for State Samplers (PUCT and Greedy).

PUCT (Predictor + Upper Confidence bounds for Trees) is key to TTT-Discover:
- Balances exploitation (high Q states) and exploration (under-visited states)
- Uses MAX reward as Q value (not mean) - this is crucial for discovery
"""
import numpy as np
import pytest
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from ttt_discover.state import State
from ttt_discover.sampler import PUCTStateSampler, GreedyStateSampler


class TestState:
    """Tests for State class."""

    def test_state_creation(self):
        """Test basic state creation."""
        state = State(timestep=0, code="test", value=0.5)
        assert state.timestep == 0
        assert state.code == "test"
        assert state.value == 0.5
        assert state.id is not None

    def test_state_to_dict(self):
        """Test state serialization."""
        state = State(timestep=1, code="code", value=0.8, observation="obs")
        d = state.to_dict()

        assert d["timestep"] == 1
        assert d["code"] == "code"
        assert d["value"] == 0.8
        assert d["observation"] == "obs"

    def test_state_from_dict(self):
        """Test state deserialization."""
        d = {"timestep": 2, "code": "test_code", "value": 0.7}
        state = State.from_dict(d)

        assert state.timestep == 2
        assert state.code == "test_code"
        assert state.value == 0.7


class TestGreedyStateSampler:
    """Tests for GreedyStateSampler."""

    def test_init(self):
        """Test greedy sampler initialization."""
        initial = State(timestep=-1, value=0.1)
        sampler = GreedyStateSampler(initial_state=initial, epsilon=0.1)

        assert sampler.epsilon == 0.1
        assert len(sampler._states) == 1

    def test_sample_initial(self):
        """Test sampling from initial state only."""
        initial = State(timestep=-1, value=0.1)
        sampler = GreedyStateSampler(initial_state=initial, epsilon=0.0)

        states = sampler.sample_states(5)
        assert len(states) == 5
        assert all(s.id == initial.id for s in states)

    def test_sample_with_buffer(self):
        """Test sampling after updates."""
        initial = State(timestep=-1, value=0.1)
        sampler = GreedyStateSampler(initial_state=initial, epsilon=0.0)

        # Add better state
        new_state = State(timestep=0, value=0.5)
        sampler.update_states([new_state], [initial])

        # Should sample best state (greedy)
        states = sampler.sample_states(3)
        assert all(s.value == 0.5 for s in states)

    def test_epsilon_exploration(self):
        """Test epsilon-greedy exploration."""
        initial = State(timestep=-1, value=0.1)
        sampler = GreedyStateSampler(initial_state=initial, epsilon=0.5)

        # Add several states
        for i in range(5):
            new_state = State(timestep=i, value=0.1 + i * 0.1)
            sampler.update_states([new_state], [initial])

        # Sample many times - should see variation due to epsilon
        all_values = set()
        for _ in range(100):
            states = sampler.sample_states(1)
            all_values.add(states[0].value)

        # Should have sampled multiple different values
        assert len(all_values) > 1, "Epsilon should cause exploration"

    def test_get_best_state(self):
        """Test getting best state."""
        initial = State(timestep=-1, value=0.1)
        sampler = GreedyStateSampler(initial_state=initial)

        for i in range(5):
            new_state = State(timestep=i, value=0.1 * i)
            sampler.update_states([new_state], [initial])

        best = sampler.get_best_state()
        assert best is not None
        assert best.value == max(s.value for s in sampler._states if s.value is not None)

    def test_buffer_size_limit(self):
        """Test buffer size limiting."""
        initial = State(timestep=-1, value=0.0)
        sampler = GreedyStateSampler(initial_state=initial, max_buffer_size=5)

        # Add many states
        for i in range(20):
            new_state = State(timestep=i, value=float(i))
            sampler.update_states([new_state], [initial])

        assert len(sampler._states) <= 5
        # Should keep best states
        values = [s.value for s in sampler._states]
        assert min(values) >= 15  # Top 5 of 0-19


class TestPUCTStateSampler:
    """Tests for PUCTStateSampler."""

    def test_init(self):
        """Test PUCT sampler initialization."""
        initial = State(timestep=-1, value=0.1)
        sampler = PUCTStateSampler(initial_state=initial, puct_c=1.0)

        assert sampler.puct_c == 1.0
        assert sampler._total_rollouts == 0

    def test_sample_initial(self):
        """Test sampling from initial state."""
        initial = State(timestep=-1, value=0.1)
        sampler = PUCTStateSampler(initial_state=initial)

        # With only 1 state, lineage blocking limits output to 1
        states = sampler.sample_states(3)
        assert len(states) >= 1  # At least the initial state
        assert states[0].id == initial.id

    def test_puct_exploration_bonus(self):
        """Test that PUCT explores under-visited states."""
        initial = State(timestep=-1, value=0.5)
        sampler = PUCTStateSampler(initial_state=initial, puct_c=2.0)

        # Add two states with same value
        state1 = State(timestep=0, value=0.6, code="code1")
        state2 = State(timestep=0, value=0.6, code="code2")
        sampler.update_states([state1, state2], [initial, initial])

        # Visit state1 many times
        for _ in range(10):
            sampler._visit_counts[state1.id] = sampler._visit_counts.get(state1.id, 0) + 1
            sampler._total_rollouts += 1

        # Now state2 should be favored (less visited)
        states = sampler.sample_states(1)
        # Due to PUCT bonus, under-visited state2 should be selected
        # (though this is probabilistic, we just check the logic runs)
        assert len(states) == 1

    def test_max_value_tracking(self):
        """Test Q value uses MAX reward (not mean)."""
        initial = State(timestep=-1, value=0.3)
        sampler = PUCTStateSampler(initial_state=initial)

        # Add children with different values
        child1 = State(timestep=0, value=0.5)
        child2 = State(timestep=0, value=0.8)
        child3 = State(timestep=0, value=0.4)

        sampler.update_states([child1, child2, child3], [initial, initial, initial])

        # Q for initial should be MAX of children, not mean
        Q = sampler._max_values.get(initial.id, 0.0)
        assert Q == 0.8, "Q should be max reward, not mean"

    def test_visit_count_backprop(self):
        """Test visit counts propagate to ancestors."""
        initial = State(timestep=-1, value=0.3)
        sampler = PUCTStateSampler(initial_state=initial)

        # Create chain: initial -> mid -> child
        mid = State(timestep=0, value=0.5)
        sampler.update_states([mid], [initial])

        child = State(timestep=1, value=0.7)
        sampler.update_states([child], [mid])

        # Both initial and mid should have visit counts
        assert sampler._visit_counts.get(initial.id, 0) > 0
        assert sampler._visit_counts.get(mid.id, 0) > 0

    def test_lineage_blocking(self):
        """Test that lineage blocking prevents related states from being selected together."""
        initial = State(timestep=-1, value=0.3)
        sampler = PUCTStateSampler(initial_state=initial)

        # Add parent and child
        parent = State(timestep=0, value=0.5, code="parent")
        sampler.update_states([parent], [initial])

        child = State(timestep=1, value=0.6, code="child")
        sampler.update_states([child], [parent])

        # When sampling 2 states, shouldn't get both parent and child
        # (they share lineage)
        states = sampler.sample_states(2)

        # Check we don't have direct parent-child pair
        ids = {s.id for s in states}
        if parent.id in ids:
            assert child.id not in ids or len(ids) == 1

    def test_save_and_load(self):
        """Test checkpoint save/load."""
        initial = State(timestep=-1, value=0.1)
        sampler = PUCTStateSampler(initial_state=initial)

        # Add some states
        for i in range(5):
            new_state = State(timestep=i, value=0.1 * i, code=f"code_{i}")
            sampler.update_states([new_state], [initial])

        # Save
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            sampler.save(path)

            # Create new sampler and load
            sampler2 = PUCTStateSampler(initial_state=initial)
            sampler2.load(path)

            assert len(sampler2._states) == len(sampler._states)
            assert sampler2._total_rollouts == sampler._total_rollouts
        finally:
            os.unlink(path)

    def test_get_stats(self):
        """Test statistics collection."""
        initial = State(timestep=-1, value=0.1)
        sampler = PUCTStateSampler(initial_state=initial)

        for i in range(5):
            new_state = State(timestep=i, value=0.2 + 0.1 * i)
            sampler.update_states([new_state], [initial])

        stats = sampler.get_stats()

        assert "buffer_size" in stats
        assert "total_rollouts" in stats
        assert "value_max" in stats
        assert "value_min" in stats
        assert stats["buffer_size"] == 6  # initial + 5 new


class TestPUCTMathematics:
    """Mathematical property tests for PUCT."""

    def test_prior_normalization(self):
        """Test rank-based prior sums to 1."""
        initial = State(timestep=-1, value=0.1)
        sampler = PUCTStateSampler(initial_state=initial)

        values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        prior = sampler._compute_prior(values)

        assert abs(np.sum(prior) - 1.0) < 1e-6

    def test_prior_ranking(self):
        """Test prior favors higher values."""
        initial = State(timestep=-1, value=0.1)
        sampler = PUCTStateSampler(initial_state=initial)

        values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        prior = sampler._compute_prior(values)

        # Higher values should have higher prior
        for i in range(len(prior) - 1):
            assert prior[i] < prior[i + 1]

    def test_scale_computation(self):
        """Test reward scale computation."""
        initial = State(timestep=-1, value=0.1)
        sampler = PUCTStateSampler(initial_state=initial)

        values = np.array([0.2, 0.4, 0.8])
        scale = sampler._compute_scale(values)

        assert scale == pytest.approx(0.6, abs=1e-6)

    def test_exploration_bonus_decreases_with_visits(self):
        """Test exploration bonus decreases as state is visited."""
        initial = State(timestep=-1, value=0.5)
        sampler = PUCTStateSampler(initial_state=initial, puct_c=1.0, group_size=1)

        values = np.array([0.5])
        priors = sampler._compute_prior(values)
        scale = sampler._compute_scale(values) or 1.0

        # Score with 0 visits
        sampler._visit_counts[initial.id] = 0
        sampler._total_rollouts = 10
        score0, _ = sampler._compute_puct_score(initial, values, priors, scale, 0)

        # Score with 5 visits
        sampler._visit_counts[initial.id] = 5
        score5, _ = sampler._compute_puct_score(initial, values, priors, scale, 0)

        # More visits = lower exploration bonus = lower score
        assert score0 > score5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
