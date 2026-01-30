"""
Pytest configuration and fixtures for TTT-Discover tests.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Model storage location for large files
MODEL_STORAGE_PATH = Path("/mnt/sda1/model/ttt-discover")


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@pytest.fixture(scope="session")
def model_storage_path():
    """Get the model storage path, creating it if necessary."""
    MODEL_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    return MODEL_STORAGE_PATH


@pytest.fixture(scope="session")
def gpu_info(device):
    """Get GPU information if available."""
    if device == "cuda":
        return {
            "name": torch.cuda.get_device_name(0),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "compute_capability": torch.cuda.get_device_capability(0),
        }
    return {"name": "CPU", "total_memory_gb": 0, "compute_capability": None}


@pytest.fixture
def small_scale_config():
    """Small-scale test configuration for limited GPU memory."""
    return {
        "seqlen": 16,
        "bs": 1,
        "dim": 64,
        "hiddendim": 32,
        "seed": 42,
        "nomask": True,
        "distribution": "normal"
    }


@pytest.fixture
def medium_scale_config():
    """Medium-scale test configuration."""
    return {
        "seqlen": 64,
        "bs": 1,
        "dim": 128,
        "hiddendim": 64,
        "seed": 123,
        "nomask": True,
        "distribution": "normal"
    }


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "integration: marks integration tests")


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if no GPU available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
