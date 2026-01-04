"""
Pytest configuration and shared fixtures for WDVA tests.

Copyright 2025 Enclave
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_key_hex() -> str:
    """Generate a sample encryption key in hex format."""
    import secrets
    return secrets.token_bytes(32).hex()


@pytest.fixture
def sample_key_bytes() -> bytes:
    """Generate a sample encryption key as bytes."""
    import secrets
    return secrets.token_bytes(32)


@pytest.fixture
def mock_weights() -> dict:
    """Create mock adapter weights for testing.

    Note: These are simple tensors for testing. Real adapter weights
    would be PyTorch tensors from a trained model.
    """
    try:
        import torch
        return {
            "layer.0.weight": torch.randn(64, 64),
            "layer.0.bias": torch.randn(64),
            "layer.1.weight": torch.randn(32, 64),
            "layer.1.bias": torch.randn(32),
        }
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture
def encrypted_adapter_path(temp_dir: Path, sample_key_bytes: bytes, mock_weights: dict) -> Path:
    """Create an encrypted adapter file for testing."""
    from wdva.crypto import EncryptedAdapter

    adapter = EncryptedAdapter(sample_key_bytes)
    output_path = temp_dir / "test_adapter.wdva"
    adapter.encrypt_weights(mock_weights, output_path)
    return output_path
