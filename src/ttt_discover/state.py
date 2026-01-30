"""
State management for TTT-Discover.

States represent the current point in the exploration tree. Each state contains:
- The generated code/solution
- Performance metrics (higher value = better)
- Parent information for lineage tracking
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import uuid
import json


@dataclass
class State:
    """Base class for exploration states."""

    timestep: int = -1
    code: str = ""
    value: Optional[float] = None
    observation: str = ""

    # Lineage tracking for PUCT
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_values: list[float] = field(default_factory=list)
    parents: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestep": self.timestep,
            "code": self.code,
            "value": self.value,
            "observation": self.observation,
            "id": self.id,
            "parent_values": self.parent_values,
            "parents": self.parents,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "State":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GPUKernelState(State):
    """State for GPU kernel optimization tasks."""

    runtime_us: Optional[float] = None  # Runtime in microseconds (lower = better)
    correctness: float = 0.0  # 1.0 if kernel passes all tests
    gpu_type: str = "unknown"
    task_name: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "runtime_us": self.runtime_us,
            "correctness": self.correctness,
            "gpu_type": self.gpu_type,
            "task_name": self.task_name,
        })
        return d


@dataclass
class MathState(State):
    """State for mathematical discovery tasks (e.g., Erdős problems)."""

    construction: Optional[list[float]] = None
    bound: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "construction": self.construction,
            "bound": self.bound,
        })
        return d


def state_from_dict(data: dict[str, Any]) -> State:
    """Factory function to create appropriate State subclass from dict."""
    if "runtime_us" in data:
        return GPUKernelState(**{k: v for k, v in data.items() if k in GPUKernelState.__dataclass_fields__})
    elif "construction" in data:
        return MathState(**{k: v for k, v in data.items() if k in MathState.__dataclass_fields__})
    return State.from_dict(data)
