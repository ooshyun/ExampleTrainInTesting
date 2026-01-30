"""
Unit tests for TriMul GPU kernel implementation.
Run with: pytest tests/test_trimul_unit.py -v
"""

import pytest
import torch
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from torch import nn, einsum


# ============================================================================
# Test Utilities
# ============================================================================

def generate_test_input(
    seqlen: int,
    bs: int,
    dim: int,
    hiddendim: int,
    seed: int,
    nomask: bool,
    distribution: str = "normal",
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]:
    """Generate input data for TriMul testing."""
    config = {"hidden_dim": hiddendim, "dim": dim}

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if distribution == "cauchy":
        u = torch.empty((bs, seqlen, seqlen, dim), device=device, dtype=torch.float32)
        u.uniform_(0.0, 1.0, generator=gen)
        input_tensor = 2.0 * torch.tan(math.pi * (u - 0.5))
    else:
        input_tensor = torch.randn(
            (bs, seqlen, seqlen, dim),
            device=device,
            dtype=torch.float32,
            generator=gen
        ).contiguous()

    if nomask:
        mask = torch.ones(bs, seqlen, seqlen, device=device)
    else:
        mask = torch.randint(0, 2, (bs, seqlen, seqlen), device=device, generator=gen).float()

    weights = {
        "norm.weight": torch.randn(dim, device=device, dtype=torch.float32),
        "norm.bias": torch.randn(dim, device=device, dtype=torch.float32),
        "left_proj.weight": torch.randn(hiddendim, dim, device=device, dtype=torch.float32) / math.sqrt(hiddendim),
        "right_proj.weight": torch.randn(hiddendim, dim, device=device, dtype=torch.float32) / math.sqrt(hiddendim),
        "left_gate.weight": torch.randn(hiddendim, dim, device=device, dtype=torch.float32) / math.sqrt(hiddendim),
        "right_gate.weight": torch.randn(hiddendim, dim, device=device, dtype=torch.float32) / math.sqrt(hiddendim),
        "out_gate.weight": torch.randn(hiddendim, dim, device=device, dtype=torch.float32) / math.sqrt(hiddendim),
        "to_out_norm.weight": torch.randn(hiddendim, device=device, dtype=torch.float32),
        "to_out_norm.bias": torch.randn(hiddendim, device=device, dtype=torch.float32),
        "to_out.weight": torch.randn(dim, hiddendim, device=device, dtype=torch.float32) / math.sqrt(dim),
    }

    return input_tensor, mask, weights, config


class TriMulReference(nn.Module):
    """Reference TriMul implementation."""

    def __init__(self, dim: int, hidden_dim: int, device="cuda"):
        super().__init__()
        self.norm = nn.LayerNorm(dim, device=device)
        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.to_out_norm = nn.LayerNorm(hidden_dim, device=device)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, device=device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask
        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()
        left = left * left_gate
        right = right * right_gate
        out = einsum('... i k d, ... j k d -> ... i j d', left, right)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def load_weights_to_model(model: nn.Module, weights: Dict[str, torch.Tensor]):
    """Load weights into model."""
    model.norm.weight = nn.Parameter(weights['norm.weight'].clone())
    model.norm.bias = nn.Parameter(weights['norm.bias'].clone())
    model.left_proj.weight = nn.Parameter(weights['left_proj.weight'].clone())
    model.right_proj.weight = nn.Parameter(weights['right_proj.weight'].clone())
    model.left_gate.weight = nn.Parameter(weights['left_gate.weight'].clone())
    model.right_gate.weight = nn.Parameter(weights['right_gate.weight'].clone())
    model.out_gate.weight = nn.Parameter(weights['out_gate.weight'].clone())
    model.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].clone())
    model.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].clone())
    model.to_out.weight = nn.Parameter(weights['to_out.weight'].clone())


# ============================================================================
# Basic Shape Tests
# ============================================================================

class TestTriMulShapes:
    """Test TriMul input/output shapes."""

    @pytest.mark.parametrize("bs,seqlen,dim,hiddendim", [
        (1, 16, 64, 32),
        (2, 16, 64, 32),
        (1, 32, 128, 64),
        (2, 32, 128, 64),
    ])
    def test_output_shape(self, device, bs, seqlen, dim, hiddendim):
        """Test that output shape matches expected dimensions."""
        input_tensor, mask, weights, config = generate_test_input(
            seqlen=seqlen, bs=bs, dim=dim, hiddendim=hiddendim,
            seed=42, nomask=True, device=device
        )

        model = TriMulReference(dim, hiddendim, device=device)
        load_weights_to_model(model, weights)

        with torch.no_grad():
            output = model(input_tensor, mask)

        assert output.shape == (bs, seqlen, seqlen, dim), \
            f"Expected shape {(bs, seqlen, seqlen, dim)}, got {output.shape}"

    def test_mask_dimensions(self, device, small_scale_config):
        """Test mask dimension handling."""
        cfg = small_scale_config
        input_tensor, mask, weights, config = generate_test_input(
            device=device, **cfg
        )

        # Mask should be [bs, seqlen, seqlen]
        assert mask.shape == (cfg["bs"], cfg["seqlen"], cfg["seqlen"])

        # After unsqueeze, should be [bs, seqlen, seqlen, 1]
        mask_expanded = mask.unsqueeze(-1)
        assert mask_expanded.shape == (cfg["bs"], cfg["seqlen"], cfg["seqlen"], 1)


# ============================================================================
# Correctness Tests
# ============================================================================

class TestTriMulCorrectness:
    """Test TriMul correctness and numerical stability."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789])
    def test_determinism(self, device, small_scale_config, seed):
        """Test that same model with same input produces deterministic results."""
        cfg = small_scale_config.copy()
        cfg["seed"] = seed

        input_tensor, mask, weights, config = generate_test_input(device=device, **cfg)

        model = TriMulReference(config["dim"], config["hidden_dim"], device=device)
        load_weights_to_model(model, weights)

        # Run the same model twice with same input
        with torch.no_grad():
            out1 = model(input_tensor, mask)
            out2 = model(input_tensor, mask)

        assert torch.allclose(out1, out2), "Same model with same input should produce identical results"

    @pytest.mark.parametrize("nomask", [True, False])
    def test_mask_effect(self, device, small_scale_config, nomask):
        """Test that masking produces valid results."""
        cfg = small_scale_config.copy()
        cfg["nomask"] = nomask

        input_tensor, mask, weights, config = generate_test_input(device=device, **cfg)

        model = TriMulReference(config["dim"], config["hidden_dim"], device=device)
        load_weights_to_model(model, weights)

        with torch.no_grad():
            output = model(input_tensor, mask)

        # Output should be finite
        assert torch.isfinite(output).all(), "Output contains non-finite values"

    def test_einsum_equivalence(self, device):
        """Test that einsum produces correct results."""
        bs, seq, dim = 2, 8, 16

        left = torch.randn(bs, seq, seq, dim, device=device)
        right = torch.randn(bs, seq, seq, dim, device=device)

        # Einsum method
        out_einsum = einsum('... i k d, ... j k d -> ... i j d', left, right)

        # Manual loop (reference)
        out_manual = torch.zeros(bs, seq, seq, dim, device=device)
        for b in range(bs):
            for i in range(seq):
                for j in range(seq):
                    for k in range(seq):
                        out_manual[b, i, j] += left[b, i, k, :] * right[b, j, k, :]

        assert torch.allclose(out_einsum, out_manual, rtol=1e-4, atol=1e-4), \
            "Einsum should match manual computation"

    @pytest.mark.parametrize("distribution", ["normal", "cauchy"])
    def test_distribution_stability(self, device, small_scale_config, distribution):
        """Test numerical stability with different input distributions."""
        cfg = small_scale_config.copy()
        cfg["distribution"] = distribution

        input_tensor, mask, weights, config = generate_test_input(device=device, **cfg)

        model = TriMulReference(config["dim"], config["hidden_dim"], device=device)
        load_weights_to_model(model, weights)

        with torch.no_grad():
            output = model(input_tensor, mask)

        # Check for NaN/Inf
        assert not torch.isnan(output).any(), f"NaN in output with {distribution} distribution"
        assert not torch.isinf(output).any(), f"Inf in output with {distribution} distribution"


# ============================================================================
# Performance Tests
# ============================================================================

class TestTriMulPerformance:
    """Test TriMul performance characteristics."""

    @pytest.mark.gpu
    def test_cuda_synchronization(self, device):
        """Test CUDA synchronization behavior."""
        if device != "cuda":
            pytest.skip("CUDA not available")

        cfg = {"seqlen": 16, "bs": 1, "dim": 64, "hiddendim": 32,
               "seed": 42, "nomask": True, "distribution": "normal"}

        input_tensor, mask, weights, config = generate_test_input(device=device, **cfg)

        model = TriMulReference(config["dim"], config["hidden_dim"], device=device)
        load_weights_to_model(model, weights)

        # Ensure CUDA operations complete
        with torch.no_grad():
            output = model(input_tensor, mask)
            torch.cuda.synchronize()

        assert output is not None

    @pytest.mark.slow
    @pytest.mark.parametrize("seqlen", [16, 32, 64])
    def test_scaling_behavior(self, device, seqlen):
        """Test memory and compute scaling with sequence length."""
        cfg = {"seqlen": seqlen, "bs": 1, "dim": 64, "hiddendim": 32,
               "seed": 42, "nomask": True, "distribution": "normal"}

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        input_tensor, mask, weights, config = generate_test_input(device=device, **cfg)

        model = TriMulReference(config["dim"], config["hidden_dim"], device=device)
        load_weights_to_model(model, weights)

        with torch.no_grad():
            output = model(input_tensor, mask)

        if device == "cuda":
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
            # Memory should scale roughly as O(N²) for sequence length N
            # This is a sanity check, not a strict assertion
            assert peak_memory < seqlen * seqlen * 10, f"Memory usage too high: {peak_memory} MB"


# ============================================================================
# Edge Cases
# ============================================================================

class TestTriMulEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_element(self, device):
        """Test with minimal sequence length."""
        cfg = {"seqlen": 1, "bs": 1, "dim": 32, "hiddendim": 16,
               "seed": 42, "nomask": True, "distribution": "normal"}

        input_tensor, mask, weights, config = generate_test_input(device=device, **cfg)

        model = TriMulReference(config["dim"], config["hidden_dim"], device=device)
        load_weights_to_model(model, weights)

        with torch.no_grad():
            output = model(input_tensor, mask)

        assert output.shape == (1, 1, 1, 32)

    def test_all_zeros_mask(self, device, small_scale_config):
        """Test with all-zeros mask."""
        cfg = small_scale_config
        input_tensor, _, weights, config = generate_test_input(device=device, **cfg)

        # Create all-zeros mask
        mask = torch.zeros(cfg["bs"], cfg["seqlen"], cfg["seqlen"], device=device)

        model = TriMulReference(config["dim"], config["hidden_dim"], device=device)
        load_weights_to_model(model, weights)

        with torch.no_grad():
            output = model(input_tensor, mask)

        # With all-zeros mask, output should still be finite
        assert torch.isfinite(output).all()

    def test_all_ones_mask(self, device, small_scale_config):
        """Test with all-ones mask."""
        cfg = small_scale_config
        input_tensor, _, weights, config = generate_test_input(device=device, **cfg)

        # Create all-ones mask
        mask = torch.ones(cfg["bs"], cfg["seqlen"], cfg["seqlen"], device=device)

        model = TriMulReference(config["dim"], config["hidden_dim"], device=device)
        load_weights_to_model(model, weights)

        with torch.no_grad():
            output = model(input_tensor, mask)

        assert torch.isfinite(output).all()


# ============================================================================
# Integration Tests
# ============================================================================

class TestTriMulIntegration:
    """Integration tests for TriMul."""

    @pytest.mark.integration
    def test_multiple_forward_passes(self, device, small_scale_config):
        """Test multiple forward passes with same model."""
        cfg = small_scale_config
        input_tensor, mask, weights, config = generate_test_input(device=device, **cfg)

        model = TriMulReference(config["dim"], config["hidden_dim"], device=device)
        load_weights_to_model(model, weights)

        outputs = []
        with torch.no_grad():
            for _ in range(3):
                output = model(input_tensor, mask)
                outputs.append(output.clone())

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i]), \
                "Multiple forward passes should produce identical results"

    @pytest.mark.integration
    def test_batch_consistency(self, device):
        """Test that batch processing is consistent with single samples."""
        dim, hiddendim, seqlen = 64, 32, 16

        # Generate single sample
        input_single, mask_single, weights, config = generate_test_input(
            seqlen=seqlen, bs=1, dim=dim, hiddendim=hiddendim,
            seed=42, nomask=True, device=device
        )

        model = TriMulReference(dim, hiddendim, device=device)
        load_weights_to_model(model, weights)

        with torch.no_grad():
            out_single = model(input_single, mask_single)

        # Generate batch (duplicate same sample)
        input_batch = input_single.repeat(2, 1, 1, 1)
        mask_batch = mask_single.repeat(2, 1, 1)

        with torch.no_grad():
            out_batch = model(input_batch, mask_batch)

        # Batch results should match single sample results
        assert torch.allclose(out_single[0], out_batch[0], rtol=1e-4, atol=1e-4)
        assert torch.allclose(out_single[0], out_batch[1], rtol=1e-4, atol=1e-4)
