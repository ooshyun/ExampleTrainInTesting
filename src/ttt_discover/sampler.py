"""
State samplers for TTT-Discover.

The key innovation of TTT-Discover is PUCT-based state reuse:
- Instead of always starting from scratch, we start from promising previous states
- PUCT (Predictor + Upper Confidence bounds for Trees) balances:
  - Exploitation: states that led to good results (high Q)
  - Exploration: states that haven't been tried much (high uncertainty bonus)

Formula:
  score(i) = Q(i) + c * scale * P(i) * sqrt(1 + T/G) / (1 + n[i]/G)

where:
  Q(i) = max reward reachable from state i (not mean!)
  P(i) = prior probability (rank-based)
  T = total rollouts
  n[i] = visits to state i
  G = group size
  c = exploration constant
  scale = max(R) - min(R)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import json
import os
import threading
from typing import Optional
import numpy as np

from .state import State, state_from_dict


class StateSampler(ABC):
    """Abstract base for state sampling strategies."""

    @abstractmethod
    def sample_states(self, num_states: int) -> list[State]:
        """Sample states to start rollouts from."""
        pass

    @abstractmethod
    def update_states(
        self,
        new_states: list[State],
        parent_states: list[State],
        step: Optional[int] = None,
    ):
        """Update sampler with new states from rollouts."""
        pass

    @abstractmethod
    def get_best_state(self) -> Optional[State]:
        """Return the best state found so far."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save sampler state to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load sampler state from disk."""
        pass


class GreedyStateSampler(StateSampler):
    """
    Epsilon-greedy sampler.

    With probability epsilon, sample a random state from buffer.
    Otherwise, sample the best state by value.
    """

    def __init__(
        self,
        initial_state: State,
        max_buffer_size: int = 100,
        epsilon: float = 0.1,
    ):
        self.initial_state = initial_state
        self.max_buffer_size = max_buffer_size
        self.epsilon = epsilon
        self._states: list[State] = [initial_state]
        self._lock = threading.Lock()

    def sample_states(self, num_states: int) -> list[State]:
        with self._lock:
            if not self._states:
                return [self.initial_state] * num_states

            result = []
            for _ in range(num_states):
                if np.random.random() < self.epsilon and len(self._states) > 1:
                    # Random exploration
                    idx = np.random.randint(len(self._states))
                    result.append(self._states[idx])
                else:
                    # Greedy: best state
                    result.append(self._states[0])  # Sorted by value
            return result

    def update_states(
        self,
        new_states: list[State],
        parent_states: list[State],
        step: Optional[int] = None,
    ):
        with self._lock:
            # Set parent info
            for child, parent in zip(new_states, parent_states):
                if parent.value is not None:
                    child.parent_values = [parent.value] + parent.parent_values
                child.parents = [{"id": parent.id, "timestep": parent.timestep}] + parent.parents

            # Add new states
            self._states.extend(new_states)

            # Sort by value (descending) and keep top-k
            self._states.sort(
                key=lambda s: s.value if s.value is not None else float("-inf"),
                reverse=True,
            )
            self._states = self._states[:self.max_buffer_size]

    def get_best_state(self) -> Optional[State]:
        with self._lock:
            return self._states[0] if self._states else None

    def save(self, path: str):
        with self._lock:
            data = {
                "states": [s.to_dict() for s in self._states],
                "initial_state": self.initial_state.to_dict(),
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        with self._lock:
            self._states = [state_from_dict(s) for s in data.get("states", [])]
            if not self._states:
                self._states = [self.initial_state]


class PUCTStateSampler(StateSampler):
    """
    PUCT-based state sampler (main innovation of TTT-Discover).

    Key differences from standard MCTS PUCT:
    - Q(i) = MAX reward reachable from state i (not mean)
    - Focus on finding the single best solution

    This encourages:
    - Exploitation of promising branches
    - Exploration of under-visited states
    - Avoiding branches that have been thoroughly explored
    """

    def __init__(
        self,
        initial_state: State,
        max_buffer_size: int = 1000,
        puct_c: float = 1.0,
        group_size: int = 64,
    ):
        self.initial_state = initial_state
        self.max_buffer_size = max_buffer_size
        self.puct_c = puct_c
        self.group_size = group_size

        self._states: list[State] = [initial_state]
        self._lock = threading.Lock()

        # PUCT statistics
        self._visit_counts: dict[str, int] = {}  # n[i]
        self._max_values: dict[str, float] = {}  # Q[i] = max reward from state i
        self._total_rollouts: int = 0  # T

        # For diagnostics
        self._last_scores: list[tuple[State, float, dict]] = []

    def _compute_prior(self, values: np.ndarray) -> np.ndarray:
        """Rank-based prior: P(i) ∝ (N - rank_i)."""
        if len(values) == 0:
            return np.array([])

        # Higher values = higher prior
        ranks = np.argsort(np.argsort(-values))
        weights = (len(values) - ranks).astype(np.float64)
        return weights / weights.sum()

    def _compute_scale(self, values: np.ndarray) -> float:
        """Reward scale for bonus normalization."""
        if len(values) == 0:
            return 1.0
        return max(np.max(values) - np.min(values), 1e-6)

    def _compute_puct_score(
        self,
        state: State,
        values: np.ndarray,
        priors: np.ndarray,
        scale: float,
        idx: int,
    ) -> tuple[float, dict]:
        """
        Compute PUCT score for a state.

        score = Q + c * scale * P * sqrt(1 + T/G) / (1 + n/G)
        """
        n = self._visit_counts.get(state.id, 0)
        value = state.value if state.value is not None else float("-inf")

        # Q = max reward from this state (or current value if unexplored)
        Q = self._max_values.get(state.id, value)

        # Exploration bonus
        T = self._total_rollouts
        G = self.group_size
        P = priors[idx]

        sqrt_T = np.sqrt(1.0 + T / G)
        exploration_bonus = self.puct_c * scale * P * sqrt_T / (1.0 + n / G)

        score = Q + exploration_bonus

        stats = {
            "Q": Q,
            "n": n,
            "P": P,
            "bonus": exploration_bonus,
            "score": score,
        }

        return score, stats

    def sample_states(self, num_states: int) -> list[State]:
        with self._lock:
            if not self._states:
                return [self.initial_state] * num_states

            # Get values for all states
            values = np.array([
                s.value if s.value is not None else float("-inf")
                for s in self._states
            ])

            # Compute priors and scale
            priors = self._compute_prior(values)
            scale = self._compute_scale(values)

            # Compute PUCT scores
            scored = []
            for i, state in enumerate(self._states):
                score, stats = self._compute_puct_score(
                    state, values, priors, scale, i
                )
                scored.append((state, score, stats))

            # Sort by score (descending)
            scored.sort(key=lambda x: x[1], reverse=True)

            # Select top states, avoiding duplicates from same lineage
            selected = []
            selected_lineages = set()

            for state, score, stats in scored:
                # Get lineage (ancestors)
                lineage = {state.id}
                for p in state.parents:
                    if p.get("id"):
                        lineage.add(p["id"])

                # Skip if overlapping lineage
                if lineage & selected_lineages:
                    continue

                selected.append((state, score, stats))
                selected_lineages.update(lineage)

                if len(selected) >= num_states:
                    break

            # Store for diagnostics
            self._last_scores = selected[:num_states]

            return [s[0] for s in selected[:num_states]]

    def update_states(
        self,
        new_states: list[State],
        parent_states: list[State],
        step: Optional[int] = None,
    ):
        if not new_states:
            return

        with self._lock:
            # Update PUCT statistics
            for child, parent in zip(new_states, parent_states):
                if child.value is None:
                    continue

                # Update max value (Q) for parent and ancestors
                pid = parent.id
                child_value = float(child.value)

                # Update Q for parent
                self._max_values[pid] = max(
                    self._max_values.get(pid, float("-inf")),
                    child_value,
                )

                # Increment visit counts for parent and ancestors
                self._visit_counts[pid] = self._visit_counts.get(pid, 0) + 1
                for p in parent.parents:
                    if p.get("id"):
                        aid = p["id"]
                        self._visit_counts[aid] = self._visit_counts.get(aid, 0) + 1

                self._total_rollouts += 1

                # Set parent info on child
                if parent.value is not None:
                    child.parent_values = [parent.value] + parent.parent_values
                child.parents = [{"id": pid, "timestep": parent.timestep}] + parent.parents

            # Add valid new states to buffer
            existing_ids = {s.id for s in self._states}
            for state in new_states:
                if state.value is not None and state.id not in existing_ids:
                    self._states.append(state)
                    existing_ids.add(state.id)

            # Prune buffer if too large (keep best by value)
            if len(self._states) > self.max_buffer_size:
                self._states.sort(
                    key=lambda s: s.value if s.value is not None else float("-inf"),
                    reverse=True,
                )
                self._states = self._states[:self.max_buffer_size]

    def get_best_state(self) -> Optional[State]:
        with self._lock:
            if not self._states:
                return None
            best = max(
                self._states,
                key=lambda s: s.value if s.value is not None else float("-inf"),
            )
            return best

    def get_stats(self) -> dict:
        """Get sampler statistics for logging."""
        with self._lock:
            values = [s.value for s in self._states if s.value is not None]
            return {
                "buffer_size": len(self._states),
                "total_rollouts": self._total_rollouts,
                "value_mean": float(np.mean(values)) if values else 0.0,
                "value_max": float(np.max(values)) if values else 0.0,
                "value_min": float(np.min(values)) if values else 0.0,
                "value_std": float(np.std(values)) if len(values) > 1 else 0.0,
            }

    def save(self, path: str):
        with self._lock:
            data = {
                "states": [s.to_dict() for s in self._states],
                "initial_state": self.initial_state.to_dict(),
                "visit_counts": self._visit_counts,
                "max_values": self._max_values,
                "total_rollouts": self._total_rollouts,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        with self._lock:
            self._states = [state_from_dict(s) for s in data.get("states", [])]
            self._visit_counts = data.get("visit_counts", {})
            self._max_values = data.get("max_values", {})
            self._total_rollouts = data.get("total_rollouts", 0)
            if not self._states:
                self._states = [self.initial_state]
