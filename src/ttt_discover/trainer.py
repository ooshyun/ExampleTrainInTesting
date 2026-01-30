"""
TTT-Discover Trainer - Small-scale implementation for understanding the system.

This is a simplified trainer that demonstrates the core TTT-Discover loop:

1. Sample states from buffer using PUCT
2. Generate solutions from LLM (or mock for testing)
3. Evaluate solutions and get rewards
4. Update model using entropic policy gradient
5. Update state buffer with new states
6. Repeat

Key components:
- Entropic objective: Focus on maximum rewards, not mean
- PUCT state reuse: Start from promising previous states
- Online RL: Model improves during test time
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Any
import json
import os
import time
import numpy as np

from .state import State, GPUKernelState
from .sampler import PUCTStateSampler, StateSampler
from .entropic_loss import EntropicAdvantageEstimator
from .utils import save_training_log, save_kernel_solutions
from .lora_updater import create_lora_updater, LoRAConfig


@dataclass
class TrainerConfig:
    """
    Configuration for TTT-Discover trainer.

    Paper hyperparameters (Table 9):
    - Model: gpt-oss-120b
    - LoRA rank: 32
    - Steps: 50
    - Batch: 512 rollouts (8 groups × 64 rollouts)
    - Learning rate: ~4e-5 (typical for LoRA)
    - Max tokens: 26000 (prompt + thinking) + response
    - Context window: 32768
    """

    # Training (paper: 50 steps, 512 rollouts = 8 groups × 64)
    num_steps: int = 50  # Total training steps
    rollouts_per_step: int = 8  # Solutions per step (paper: 512, use smaller for testing)
    num_groups: int = 1  # Number of groups (paper: 8, each with 64 rollouts)
    learning_rate: float = 4e-5  # LoRA learning rate
    lora_rank: int = 32  # LoRA rank (paper: 32)

    # Entropic objective (paper: adaptive β per initial state)
    entropic_beta: float = 2.0  # Temperature scaling for entropic weights
    kl_penalty_coef: float = 0.0  # λ for KL penalty: A = w - 1 - λ*log(π/π₀)

    # PUCT sampler (paper formula: Q + c*P*sqrt(1+T)/(1+n))
    puct_c: float = 1.0  # Exploration constant c
    max_buffer_size: int = 1000  # Max states in reuse buffer
    group_size: int = 64  # G in paper formula (paper: 64)

    # Logging
    log_dir: str = "./logs/ttt_discover"
    save_every: int = 5
    log_every: int = 1

    # LLM Configuration
    mock_llm: bool = False  # Use mock LLM for testing (no GPU needed)
    model_name: str = "gpt2"  # HuggingFace model name (paper: gpt-oss-120b)
    local_model_path: str | None = None  # Local model path (avoids HF rate limits)

    # Generation settings (paper: 26000 tokens for prompt+thinking)
    max_new_tokens: int = 512  # Max tokens to generate (paper uses token forcing at 26000)
    max_context_tokens: int = 32768  # Context window (paper: 32768 for gpt-oss-120b)
    temperature: float = 1.0  # Sampling temperature

    # Advanced settings
    importance_sampling: bool = True  # Correct for sampler/learner mismatch (paper: yes)
    off_policy_steps: int = 0  # Paper: 0 (1 gradient step per batch)


class TTTDiscoverTrainer:
    """
    Simplified TTT-Discover trainer for educational purposes.

    This implementation demonstrates the core algorithm without
    requiring actual GPU training infrastructure.
    """

    def __init__(
        self,
        config: TrainerConfig,
        initial_state: State,
        evaluate_fn: Callable[[str, State], dict[str, Any]],
        generate_fn: Optional[Callable[[State], str]] = None,
    ):
        """
        Args:
            config: Training configuration
            initial_state: Initial state for exploration
            evaluate_fn: Function(code, parent_state) -> {"score": float, "metrics": dict}
            generate_fn: Function(state) -> code string (optional, uses mock if None)
        """
        self.config = config
        self.evaluate_fn = evaluate_fn

        # Create sampler
        self.sampler = PUCTStateSampler(
            initial_state=initial_state,
            max_buffer_size=config.max_buffer_size,
            puct_c=config.puct_c,
            group_size=config.group_size,
        )

        # Create entropic advantage estimator
        self.advantage_estimator = EntropicAdvantageEstimator(
            beta=config.entropic_beta,
        )

        # Create LoRA updater for weight updates
        self.lora_updater = create_lora_updater(
            model_name=config.model_name,
            use_mock=config.mock_llm,
            learning_rate=config.learning_rate,
            lora_config=LoRAConfig(rank=config.lora_rank),
            local_model_path=config.local_model_path,
        )

        # Set generate function
        if generate_fn is not None:
            self.generate_fn = generate_fn
        elif config.mock_llm:
            self.generate_fn = self._mock_generate
        else:
            self.generate_fn = self._llm_generate

        # Training history
        self.history: list[dict] = []
        self.all_solutions: list[dict] = []  # Store all evaluated solutions

        # Track prompts/completions for LoRA update
        self._current_prompts: list[str] = []
        self._current_completions: list[str] = []

        # Create log directory
        os.makedirs(config.log_dir, exist_ok=True)

    def _mock_generate(self, state: State) -> str:
        """Mock LLM generation for testing."""
        # Generate slightly perturbed version of parent code
        base_code = state.code or "# Initial code\ndef solution():\n    pass"

        # Add random variation
        variation = np.random.randint(1000)
        return f"{base_code}\n# Variation {variation}"

    def _llm_generate(self, state: State) -> str:
        """Generate code using the actual LLM with LoRA."""
        # Create prompt from state
        prompt = self._create_prompt(state)

        # Generate using LoRA model
        completion = self.lora_updater.generate(
            prompt=prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )

        return completion

    def _create_prompt(self, state: State) -> str:
        """Create prompt for code generation."""
        base_prompt = """You are an expert programmer. Given the following code, generate an improved version.

Previous code:
```
{code}
```

Previous score: {value}

Generate improved code that achieves a higher score. Output only the code, no explanations.

Improved code:
```
"""
        code = state.code or "# No previous code"
        value = state.value if state.value is not None else "N/A"

        return base_prompt.format(code=code, value=value)

    def train_step(self, step: int) -> dict:
        """
        Execute one training step.

        Returns:
            Dictionary with step statistics
        """
        step_start = time.time()

        # 1. Sample states from buffer
        parent_states = self.sampler.sample_states(self.config.rollouts_per_step)

        # 2. Generate solutions (track prompts for LoRA update)
        generated_codes = []
        self._current_prompts = []
        self._current_completions = []

        for parent in parent_states:
            prompt = self._create_prompt(parent)
            code = self.generate_fn(parent)
            generated_codes.append(code)

            # Store for weight update
            self._current_prompts.append(prompt)
            self._current_completions.append(code)

        # 3. Evaluate solutions
        rewards = []
        new_states = []
        metrics_list = []

        for code, parent in zip(generated_codes, parent_states):
            result = self.evaluate_fn(code, parent)

            reward = result.get("score", 0.0)
            rewards.append(reward)
            metrics_list.append(result.get("metrics", {}))

            # Create new state
            new_state = State(
                timestep=step,
                code=code,
                value=reward,
                observation=result.get("observation", ""),
            )
            new_states.append(new_state)

            # Store solution
            self.all_solutions.append({
                "step": step,
                "code": code,
                "reward": reward,
                "parent_id": parent.id,
                "parent_value": parent.value,
                "metrics": metrics_list[-1],
            })

        rewards = np.array(rewards)

        # 4. Compute entropic advantages
        advantages, adv_stats = self.advantage_estimator.compute_advantages(rewards)

        # 5. Update state buffer
        self.sampler.update_states(new_states, parent_states, step=step)

        # 6. Update LoRA weights using policy gradient
        # θ_new = θ_old + lr * Σ_i advantage_i * ∇log π(a_i|s_i)
        lora_stats = {}
        if len(self._current_prompts) > 0 and np.any(advantages != 0):
            lora_stats = self.lora_updater.train_on_batch(
                prompts=self._current_prompts,
                completions=self._current_completions,
                advantages=advantages,
                kl_penalty_coef=self.config.kl_penalty_coef,
            )

        # Collect statistics
        sampler_stats = self.sampler.get_stats()
        best_state = self.sampler.get_best_state()

        step_stats = {
            "step": step,
            "time": time.time() - step_start,
            "reward_mean": float(np.mean(rewards)),
            "reward_max": float(np.max(rewards)),
            "reward_min": float(np.min(rewards)),
            "reward_std": float(np.std(rewards)),
            "best_value": best_state.value if best_state else None,
            "lora_loss": lora_stats.get("loss", 0.0),
            "lora_updates": lora_stats.get("update_count", 0),
            **adv_stats,
            **sampler_stats,
        }

        self.history.append(step_stats)

        return step_stats

    def train(self, num_steps: Optional[int] = None) -> dict:
        """
        Run full training loop.

        Returns:
            Final statistics and best solution
        """
        num_steps = num_steps or self.config.num_steps

        print("=" * 60)
        print("TTT-Discover Training")
        print("=" * 60)
        print(f"Steps: {num_steps}")
        print(f"Rollouts per step: {self.config.rollouts_per_step}")
        print(f"PUCT c: {self.config.puct_c}")
        print(f"Entropic beta: {self.config.entropic_beta}")
        print("=" * 60)

        for step in range(num_steps):
            stats = self.train_step(step)

            if step % self.config.log_every == 0:
                print(
                    f"Step {step:3d} | "
                    f"Reward: {stats['reward_mean']:.4f} ± {stats['reward_std']:.4f} | "
                    f"Max: {stats['reward_max']:.4f} | "
                    f"Best: {stats['best_value']:.4f} | "
                    f"Buffer: {stats['buffer_size']} | "
                    f"Time: {stats['time']:.2f}s"
                )

            if (step + 1) % self.config.save_every == 0:
                self.save_checkpoint(step)

        # Final save
        self.save_checkpoint(num_steps - 1)

        # Get best solution
        best_state = self.sampler.get_best_state()

        return {
            "best_state": best_state,
            "history": self.history,
            "total_solutions": len(self.all_solutions),
        }

    def save_checkpoint(self, step: int):
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.config.log_dir, f"checkpoint_{step:06d}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save sampler state
        self.sampler.save(os.path.join(checkpoint_dir, "sampler.json"))

        # Save training history
        save_training_log(self.history, os.path.join(checkpoint_dir, "history.json"))

        # Save all solutions
        save_kernel_solutions(
            self.all_solutions,
            os.path.join(checkpoint_dir, "solutions.json"),
        )

        # Save best solution separately
        best_state = self.sampler.get_best_state()
        if best_state:
            with open(os.path.join(checkpoint_dir, "best_solution.json"), "w") as f:
                json.dump(best_state.to_dict(), f, indent=2)

        # Save LoRA weights
        lora_path = os.path.join(checkpoint_dir, "lora_weights")
        self.lora_updater.save_lora(lora_path)

        print(f"  → Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        self.sampler.load(os.path.join(checkpoint_path, "sampler.json"))

        history_path = os.path.join(checkpoint_path, "history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                data = json.load(f)
                # Extract history list from wrapped format
                self.history = data.get("history", data) if isinstance(data, dict) else data

        solutions_path = os.path.join(checkpoint_path, "solutions.json")
        if os.path.exists(solutions_path):
            with open(solutions_path, "r") as f:
                data = json.load(f)
                # Extract solutions list from wrapped format
                self.all_solutions = data.get("solutions", data) if isinstance(data, dict) else data

        # Load LoRA weights if available
        lora_path = os.path.join(checkpoint_path, "lora_weights")
        if os.path.exists(lora_path):
            self.lora_updater.load_lora(lora_path)

        print(f"Loaded checkpoint from {checkpoint_path}")


def demo_training():
    """
    Demonstrate TTT-Discover training with a simple mock environment.

    This shows how the algorithm works without requiring actual GPU resources.
    """
    # Simple evaluation function: reward = how close to target pattern
    def evaluate(code: str, parent_state: State) -> dict:
        # Mock: reward based on code length and randomness
        base_reward = len(code) / 100.0
        noise = np.random.normal(0, 0.1)
        reward = max(0, base_reward + noise)

        return {
            "score": reward,
            "metrics": {"code_length": len(code)},
            "observation": f"Evaluated: score={reward:.4f}",
        }

    # Create initial state
    initial_state = State(
        timestep=-1,
        code="# Initial GPU kernel\ndef custom_kernel(data):\n    return data",
        value=0.1,
    )

    # Create trainer
    config = TrainerConfig(
        num_steps=10,
        rollouts_per_step=8,
        log_dir="./logs/demo_ttt",
        mock_llm=True,
    )

    trainer = TTTDiscoverTrainer(
        config=config,
        initial_state=initial_state,
        evaluate_fn=evaluate,
    )

    # Run training
    result = trainer.train()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total solutions evaluated: {result['total_solutions']}")
    print(f"Best value found: {result['best_state'].value:.4f}")
    print(f"Best code:\n{result['best_state'].code[:200]}...")


if __name__ == "__main__":
    demo_training()
