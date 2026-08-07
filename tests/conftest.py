"""Shared pytest configuration."""

import sys
from pathlib import Path

import pytest
import torch

# Make `import src` work when pytest is run from anywhere in the repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def deterministic():
    """Every test starts from the same RNG state."""
    torch.manual_seed(0)


@pytest.fixture
def vocab_size():
    return 37


@pytest.fixture
def dims():
    """Small dimensions so the whole suite runs in seconds on a CPU."""
    return {
        "batch": 2,
        "seq_len": 9,
        "d_model": 32,
        "num_heads": 4,
        "num_layers": 2,
        "max_seq_len": 24,
    }
