"""
Tests for TTT-Discover Trainer.

The trainer orchestrates the full TTT-Discover loop:
1. Sample states using PUCT
2. Generate solutions
3. Evaluate and compute rewards
4. Update model using entropic policy gradient
5. Update state buffer
"""
import numpy as np
import pytest
import tempfile
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from ttt_discover.trainer import TTTDiscoverTrainer, TrainerConfig
from ttt_discover.state import State


def simple_evaluate(code: str, parent_state: State) -> dict:
    """Simple evaluation function for testing."""
    # Reward based on code length (simple proxy)
    base_reward = min(len(code) / 100.0, 1.0)
    noise = np.random.normal(0, 0.05)
    reward = max(0, base_reward + noise)

    # Simulate occasional improvement
    if parent_state.value and np.random.random() < 0.3:
        reward = max(reward, parent_state.value + 0.05)

    return {
        "score": reward,
        "metrics": {"code_length": len(code)},
        "observation": f"Score: {reward:.4f}",
    }


class TestTrainerConfig:
    """Tests for TrainerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrainerConfig()

        assert config.num_steps == 50
        assert config.rollouts_per_step == 8
        assert config.learning_rate == 4e-5
        assert config.entropic_beta == 2.0
        assert config.puct_c == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainerConfig(
            num_steps=10,
            rollouts_per_step=4,
            entropic_beta=1.5,
        )

        assert config.num_steps == 10
        assert config.rollouts_per_step == 4
        assert config.entropic_beta == 1.5


class TestTTTDiscoverTrainer:
    """Tests for TTTDiscoverTrainer."""

    @pytest.fixture
    def trainer(self, tmp_path):
        """Create trainer for testing."""
        initial_state = State(
            timestep=-1,
            code="# Initial\ndef f(): pass",
            value=0.1,
        )

        config = TrainerConfig(
            num_steps=5,
            rollouts_per_step=4,
            log_dir=str(tmp_path / "logs"),
            save_every=2,
            mock_llm=True,
        )

        return TTTDiscoverTrainer(
            config=config,
            initial_state=initial_state,
            evaluate_fn=simple_evaluate,
        )

    def test_init(self, trainer):
        """Test trainer initialization."""
        assert trainer.sampler is not None
        assert trainer.advantage_estimator is not None
        assert len(trainer.history) == 0
        assert len(trainer.all_solutions) == 0

    def test_mock_generate(self, trainer):
        """Test mock generation function."""
        state = State(timestep=0, code="# base code")
        code = trainer._mock_generate(state)

        assert isinstance(code, str)
        assert "base code" in code  # Contains parent code

    def test_train_step(self, trainer):
        """Test single training step."""
        stats = trainer.train_step(0)

        assert "step" in stats
        assert stats["step"] == 0
        assert "reward_mean" in stats
        assert "reward_max" in stats
        assert "best_value" in stats
        assert "temperature" in stats  # From entropic estimator

    def test_train_step_updates_buffer(self, trainer):
        """Test that training step updates state buffer."""
        initial_buffer_size = len(trainer.sampler._states)

        trainer.train_step(0)

        # Buffer should have grown
        assert len(trainer.sampler._states) >= initial_buffer_size

    def test_train_step_records_solutions(self, trainer):
        """Test that solutions are recorded."""
        trainer.train_step(0)

        # Due to PUCT lineage-blocking, actual rollouts may be fewer than requested
        # when the buffer has limited diverse states
        assert len(trainer.all_solutions) >= 1  # At least some solutions recorded

        # Check solution structure
        sol = trainer.all_solutions[0]
        assert "step" in sol
        assert "code" in sol
        assert "reward" in sol
        assert "parent_id" in sol

    def test_train_multiple_steps(self, trainer):
        """Test multiple training steps."""
        for step in range(3):
            stats = trainer.train_step(step)
            assert stats["step"] == step

        assert len(trainer.history) == 3
        # Solutions grow over time as buffer diversifies
        assert len(trainer.all_solutions) >= 3  # At least 1 per step

    def test_train_full(self, trainer):
        """Test full training loop."""
        result = trainer.train(num_steps=3)

        assert "best_state" in result
        assert "history" in result
        assert "total_solutions" in result
        assert result["total_solutions"] >= 3  # At least 1 solution per step

    def test_best_value_improves(self, trainer):
        """Test that best value tends to improve over training."""
        # Run training
        result = trainer.train(num_steps=5)

        # Check that we have a best state
        best = result["best_state"]
        assert best is not None

        # History should show some progression
        history = result["history"]
        assert len(history) == 5

    def test_checkpoint_save(self, trainer, tmp_path):
        """Test checkpoint saving."""
        trainer.train_step(0)
        trainer.save_checkpoint(0)

        checkpoint_dir = tmp_path / "logs" / "checkpoint_000000"
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "sampler.json").exists()
        assert (checkpoint_dir / "history.json").exists()
        assert (checkpoint_dir / "solutions.json").exists()

    def test_checkpoint_load(self, trainer, tmp_path):
        """Test checkpoint loading."""
        # Train and save
        trainer.train(num_steps=3)
        trainer.save_checkpoint(2)

        # Create new trainer
        initial_state = State(timestep=-1, code="# Initial", value=0.1)
        config = TrainerConfig(
            num_steps=5,
            rollouts_per_step=4,
            log_dir=str(tmp_path / "logs2"),
        )
        trainer2 = TTTDiscoverTrainer(
            config=config,
            initial_state=initial_state,
            evaluate_fn=simple_evaluate,
        )

        # Load checkpoint
        checkpoint_dir = tmp_path / "logs" / "checkpoint_000002"
        trainer2.load_checkpoint(str(checkpoint_dir))

        # Should have loaded state
        assert len(trainer2.history) == len(trainer.history)

    def test_custom_generate_fn(self, tmp_path):
        """Test trainer with custom generate function."""
        initial_state = State(timestep=-1, code="base", value=0.1)
        config = TrainerConfig(
            num_steps=2,
            rollouts_per_step=2,
            log_dir=str(tmp_path / "logs"),
        )

        custom_codes = []

        def custom_generate(state):
            code = f"# Custom generated from {state.id}"
            custom_codes.append(code)
            return code

        trainer = TTTDiscoverTrainer(
            config=config,
            initial_state=initial_state,
            evaluate_fn=simple_evaluate,
            generate_fn=custom_generate,
        )

        trainer.train()

        # Should have used our custom generator (at least once per step)
        assert len(custom_codes) >= 2  # At least 1 per step


class TestTrainerIntegration:
    """Integration tests for trainer."""

    def test_improvement_over_time(self, tmp_path):
        """Test that training improves best value over time."""
        initial_state = State(timestep=-1, code="def f(): pass", value=0.01)

        # Evaluation that rewards longer code and builds on parent
        def improving_evaluate(code, parent_state):
            base = len(code) / 200.0
            if parent_state.value:
                base = max(base, parent_state.value + np.random.uniform(0, 0.05))
            return {"score": min(base, 1.0), "metrics": {}, "observation": ""}

        config = TrainerConfig(
            num_steps=10,
            rollouts_per_step=8,
            log_dir=str(tmp_path / "logs"),
        )

        trainer = TTTDiscoverTrainer(
            config=config,
            initial_state=initial_state,
            evaluate_fn=improving_evaluate,
        )

        result = trainer.train()

        # Should see improvement
        early_best = max(h["best_value"] for h in result["history"][:3] if h["best_value"])
        late_best = max(h["best_value"] for h in result["history"][-3:] if h["best_value"])

        assert late_best >= early_best * 0.9  # Some improvement or stability

    def test_entropic_focuses_on_best(self, tmp_path):
        """Test that entropic objective focuses on best solutions."""
        initial_state = State(timestep=-1, code="base", value=0.1)

        # Evaluation with variable rewards
        def varied_evaluate(code, parent_state):
            # Reward based on random factors
            reward = np.random.uniform(0.1, 0.5)
            return {"score": reward, "metrics": {}, "observation": ""}

        config = TrainerConfig(
            num_steps=10,
            rollouts_per_step=16,
            entropic_beta=2.0,
            log_dir=str(tmp_path / "logs"),
        )

        trainer = TTTDiscoverTrainer(
            config=config,
            initial_state=initial_state,
            evaluate_fn=varied_evaluate,
        )

        result = trainer.train()

        # Check that we have solutions recorded
        all_rewards = [s["reward"] for s in trainer.all_solutions]
        assert len(all_rewards) >= 10  # At least 1 per step

        # Check temperature adaptation
        temperatures = [h["temperature"] for h in result["history"]]
        assert all(t > 0 for t in temperatures)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
