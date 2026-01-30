"""
LoRA Weight Updater for TTT-Discover.

Implements policy gradient updates with entropic weighting for LoRA parameters.
Based on the TTT-Discover paper's approach:
    θ_new = θ_old + lr * Σ_i advantage_i * ∇log π(a_i|s_i)

Requirements:
    pip install torch transformers peft
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Callable
from dataclasses import dataclass
import numpy as np

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapter."""
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.1
    target_modules: list[str] | None = None  # None = auto-detect

    def __post_init__(self):
        if self.target_modules is None:
            # Common target modules for transformer models
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class LoRAUpdater:
    """
    LoRA weight updater using policy gradient with entropic advantages.

    This implements the core weight update from TTT-Discover:
    - Compute log probabilities for generated solutions
    - Weight by entropic advantages (focus on MAX reward)
    - Update LoRA parameters via gradient descent

    Example:
        updater = LoRAUpdater(model_name="gpt2", lora_config=LoRAConfig(rank=32))

        # During training
        loss = updater.compute_policy_loss(
            prompts=["Optimize this kernel..."],
            completions=["import triton..."],
            advantages=[0.8, 0.2, 0.0, ...]
        )
        updater.update_step(loss)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        lora_config: Optional[LoRAConfig] = None,
        learning_rate: float = 4e-5,
        device: str = "auto",
        local_model_path: Optional[str] = None,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers and peft are required for LoRA updates. "
                "Install with: pip install transformers peft"
            )

        self.model_name = model_name
        self.lora_config = lora_config or LoRAConfig()
        self.learning_rate = learning_rate

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        model_path = local_model_path or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "cuda" else None,
        )

        # Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config.rank,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            target_modules=self.lora_config.target_modules,
        )
        self.model = get_peft_model(self.base_model, peft_config)

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        # Setup optimizer (only for LoRA parameters)
        lora_params = [p for n, p in self.model.named_parameters() if "lora" in n.lower()]
        self.optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)

        # For KL penalty (optional)
        self.reference_model = None

        # Statistics
        self._update_count = 0
        self._total_loss = 0.0

    def compute_log_probs(
        self,
        prompts: list[str],
        completions: list[str],
    ) -> torch.Tensor:
        """
        Compute log probabilities of completions given prompts.

        Returns:
            log_probs: Shape (batch_size,) - sum of log probs for each completion
        """
        log_probs_list = []

        for prompt, completion in zip(prompts, completions):
            # Tokenize
            full_text = prompt + completion
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)

            # Get model logits
            with torch.no_grad() if not self.model.training else torch.enable_grad():
                outputs = self.model(full_ids)
                logits = outputs.logits  # (1, seq_len, vocab_size)

            # Get log probs for completion tokens only
            prompt_len = prompt_ids.shape[1]
            completion_logits = logits[0, prompt_len-1:-1]  # Shift by 1 for next-token prediction
            completion_ids = full_ids[0, prompt_len:]

            # Compute log softmax
            log_probs = F.log_softmax(completion_logits, dim=-1)

            # Get log prob of actual tokens
            token_log_probs = log_probs.gather(1, completion_ids.unsqueeze(1)).squeeze(1)

            # Sum log probs (or could use mean)
            total_log_prob = token_log_probs.sum()
            log_probs_list.append(total_log_prob)

        return torch.stack(log_probs_list)

    def compute_policy_loss(
        self,
        prompts: list[str],
        completions: list[str],
        advantages: np.ndarray | list[float],
        kl_penalty_coef: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute policy gradient loss with entropic advantages.

        Loss = -Σ_i advantage_i * log π(completion_i | prompt_i)

        Args:
            prompts: List of input prompts
            completions: List of generated completions
            advantages: Entropic advantages (higher = more weight)
            kl_penalty_coef: KL divergence penalty coefficient (optional)

        Returns:
            loss: Scalar tensor
        """
        self.model.train()

        # Convert advantages to tensor
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # Compute log probabilities
        log_probs = self.compute_log_probs(prompts, completions)

        # Policy gradient loss: -advantage * log_prob
        policy_loss = -(advantages * log_probs).sum()

        # Optional KL penalty
        kl_loss = torch.tensor(0.0, device=self.device)
        if kl_penalty_coef > 0 and self.reference_model is not None:
            with torch.no_grad():
                ref_log_probs = self._compute_reference_log_probs(prompts, completions)
            kl_div = (log_probs - ref_log_probs).mean()
            kl_loss = kl_penalty_coef * kl_div

        total_loss = policy_loss + kl_loss

        return total_loss

    def _compute_reference_log_probs(
        self,
        prompts: list[str],
        completions: list[str],
    ) -> torch.Tensor:
        """Compute log probs from reference (frozen) model for KL penalty."""
        if self.reference_model is None:
            return torch.zeros(len(prompts), device=self.device)

        # Similar to compute_log_probs but using reference model
        # Implementation would mirror compute_log_probs
        raise NotImplementedError("Reference model KL not yet implemented")

    def update_step(self, loss: torch.Tensor):
        """Perform a single optimization step."""
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        self._update_count += 1
        self._total_loss += loss.item()

    def train_on_batch(
        self,
        prompts: list[str],
        completions: list[str],
        advantages: np.ndarray,
        kl_penalty_coef: float = 0.0,
    ) -> dict:
        """
        Complete training step: compute loss and update weights.

        Returns:
            stats: Dictionary with loss and other metrics
        """
        loss = self.compute_policy_loss(
            prompts=prompts,
            completions=completions,
            advantages=advantages,
            kl_penalty_coef=kl_penalty_coef,
        )

        self.update_step(loss)

        return {
            "loss": loss.item(),
            "update_count": self._update_count,
            "avg_loss": self._total_loss / max(1, self._update_count),
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        """Generate completion using the current model."""
        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        completion = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return completion

    def save_lora(self, path: str):
        """Save LoRA adapter weights."""
        self.model.save_pretrained(path)
        print(f"LoRA weights saved to {path}")

    def load_lora(self, path: str):
        """Load LoRA adapter weights."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, path)
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        print(f"LoRA weights loaded from {path}")

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            "update_count": self._update_count,
            "total_loss": self._total_loss,
            "avg_loss": self._total_loss / max(1, self._update_count),
            "device": str(self.device),
            "lora_rank": self.lora_config.rank,
        }


class MockLoRAUpdater:
    """
    Mock LoRA updater for testing without GPU/transformers.

    Simulates the interface of LoRAUpdater but doesn't do actual updates.
    """

    def __init__(self, **kwargs):
        self._update_count = 0
        self._total_loss = 0.0
        self.learning_rate = kwargs.get("learning_rate", 4e-5)

    def compute_policy_loss(
        self,
        prompts: list[str],
        completions: list[str],
        advantages: np.ndarray,
        **kwargs,
    ) -> float:
        """Mock loss computation."""
        # Simulate loss based on advantages
        loss = -np.sum(advantages * np.random.uniform(0.5, 1.0, len(advantages)))
        return loss

    def update_step(self, loss: float):
        """Mock update step."""
        self._update_count += 1
        self._total_loss += abs(loss) if isinstance(loss, float) else abs(loss.item())

    def train_on_batch(
        self,
        prompts: list[str],
        completions: list[str],
        advantages: np.ndarray,
        **kwargs,
    ) -> dict:
        """Mock training step."""
        loss = self.compute_policy_loss(prompts, completions, advantages)
        self.update_step(loss)

        return {
            "loss": abs(loss),
            "update_count": self._update_count,
            "avg_loss": self._total_loss / max(1, self._update_count),
        }

    def generate(self, prompt: str, **kwargs) -> str:
        """Mock generation."""
        return f"# Mock generated code\n{prompt[:50]}..."

    def save_lora(self, path: str):
        """Mock save."""
        print(f"[Mock] Would save LoRA to {path}")

    def load_lora(self, path: str):
        """Mock load."""
        print(f"[Mock] Would load LoRA from {path}")

    def get_stats(self) -> dict:
        return {
            "update_count": self._update_count,
            "total_loss": self._total_loss,
            "avg_loss": self._total_loss / max(1, self._update_count),
            "device": "mock",
            "lora_rank": 32,
        }


def create_lora_updater(
    model_name: str = "gpt2",
    use_mock: bool = False,
    **kwargs,
) -> LoRAUpdater | MockLoRAUpdater:
    """
    Factory function to create LoRA updater.

    Args:
        model_name: HuggingFace model name
        use_mock: If True, return MockLoRAUpdater (no GPU needed)
        **kwargs: Additional arguments for LoRAUpdater

    Returns:
        LoRAUpdater or MockLoRAUpdater instance
    """
    if use_mock or not HAS_TRANSFORMERS:
        return MockLoRAUpdater(**kwargs)

    try:
        return LoRAUpdater(model_name=model_name, **kwargs)
    except Exception as e:
        print(f"Failed to create LoRAUpdater: {e}")
        print("Falling back to MockLoRAUpdater")
        return MockLoRAUpdater(**kwargs)
