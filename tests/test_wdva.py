"""
Tests for the main WDVA class.

Copyright 2025 Enclave
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wdva import WDVA
from wdva.wdva import TrainingResult, WDVAStatus


class TestWDVAInit:
    """Tests for WDVA initialization."""

    def test_init_default_state(self) -> None:
        """New WDVA instance should be in READY state."""
        wdva = WDVA()
        assert wdva.status == WDVAStatus.READY
        assert not wdva.is_loaded
        assert wdva.adapter_path is None

    def test_init_with_model_name(self) -> None:
        """Should accept custom model name."""
        wdva = WDVA(model_name="custom/model")
        assert wdva.model_name == "custom/model"

    def test_init_default_model(self) -> None:
        """Should use default model when none specified."""
        wdva = WDVA()
        assert wdva.model_name == WDVA.DEFAULT_MODEL

    def test_repr_without_adapter(self) -> None:
        """repr should show no adapter when none loaded."""
        wdva = WDVA()
        repr_str = repr(wdva)
        assert "no adapter" in repr_str
        assert "ready" in repr_str


class TestWDVALoad:
    """Tests for WDVA.load() method."""

    def test_load_nonexistent_file_raises(self, sample_key_hex: str) -> None:
        """Loading nonexistent file should raise FileNotFoundError."""
        wdva = WDVA()
        with pytest.raises(FileNotFoundError):
            wdva.load("/nonexistent/adapter.wdva", sample_key_hex)

    def test_load_invalid_key_length_raises(self, temp_dir: Path) -> None:
        """Loading with wrong key length should raise ValueError."""
        # Create a dummy file
        adapter_path = temp_dir / "test.wdva"
        adapter_path.write_text("{}")

        wdva = WDVA()
        with pytest.raises(ValueError, match="64 hex characters"):
            wdva.load(str(adapter_path), "short_key")

    def test_load_invalid_key_format_raises(self, temp_dir: Path) -> None:
        """Loading with invalid hex should raise ValueError."""
        adapter_path = temp_dir / "test.wdva"
        adapter_path.write_text("{}")

        # Create 64-char string that's not valid hex
        invalid_hex = "g" * 64

        wdva = WDVA()
        with pytest.raises(ValueError, match="Invalid encryption key format"):
            wdva.load(str(adapter_path), invalid_hex)

    def test_load_sets_status_to_loaded(
        self, temp_dir: Path, sample_key_hex: str
    ) -> None:
        """Successful load should set status to LOADED."""
        adapter_path = temp_dir / "test.wdva"
        adapter_path.write_text("{}")

        wdva = WDVA()
        wdva.load(str(adapter_path), sample_key_hex)

        assert wdva.status == WDVAStatus.LOADED
        assert wdva.is_loaded
        assert wdva.adapter_path == adapter_path


class TestWDVAUnload:
    """Tests for WDVA.unload() method."""

    def test_unload_clears_adapter(self, temp_dir: Path, sample_key_hex: str) -> None:
        """Unload should clear adapter and return to READY state."""
        adapter_path = temp_dir / "test.wdva"
        adapter_path.write_text("{}")

        wdva = WDVA()
        wdva.load(str(adapter_path), sample_key_hex)
        assert wdva.is_loaded

        wdva.unload()

        assert wdva.status == WDVAStatus.READY
        assert not wdva.is_loaded
        assert wdva.adapter_path is None


class TestWDVADelete:
    """Tests for WDVA.delete() method."""

    def test_delete_clears_key_and_adapter(
        self, temp_dir: Path, sample_key_hex: str
    ) -> None:
        """Delete should clear encryption key and adapter."""
        adapter_path = temp_dir / "test.wdva"
        adapter_path.write_text("{}")

        wdva = WDVA()
        wdva.load(str(adapter_path), sample_key_hex)

        wdva.delete()

        assert wdva.status == WDVAStatus.READY
        assert not wdva.is_loaded
        assert wdva._encryption_key is None

    def test_delete_idempotent(self) -> None:
        """Delete should be safe to call multiple times."""
        wdva = WDVA()
        wdva.delete()
        wdva.delete()  # Should not raise
        assert wdva.status == WDVAStatus.READY


class TestWDVATrain:
    """Tests for WDVA.train() method."""

    def test_train_returns_training_result(self, temp_dir: Path) -> None:
        """Train should return TrainingResult."""
        wdva = WDVA()
        result = wdva.train(
            documents=["doc1.txt"],
            output_path=temp_dir / "adapter.wdva"
        )

        assert isinstance(result, TrainingResult)
        assert result.num_documents == 1
        assert len(result.encryption_key_hex) == 64

    def test_train_empty_documents_raises(self) -> None:
        """Train with empty document list should raise ValueError."""
        wdva = WDVA()
        with pytest.raises(ValueError, match="At least one document"):
            wdva.train(documents=[])

    def test_train_creates_output_dir(self, temp_dir: Path) -> None:
        """Train should create output directory if needed."""
        output_path = temp_dir / "subdir" / "adapter.wdva"

        wdva = WDVA()
        result = wdva.train(documents=["doc.txt"], output_path=output_path)

        assert result.adapter_path == str(output_path)


class TestWDVAExportKey:
    """Tests for WDVA.export_key() method."""

    def test_export_key_when_loaded(
        self, temp_dir: Path, sample_key_hex: str
    ) -> None:
        """export_key should return key when adapter is loaded."""
        adapter_path = temp_dir / "test.wdva"
        adapter_path.write_text("{}")

        wdva = WDVA()
        wdva.load(str(adapter_path), sample_key_hex)

        exported = wdva.export_key()
        assert exported == sample_key_hex

    def test_export_key_when_not_loaded_raises(self) -> None:
        """export_key should raise when no adapter loaded."""
        wdva = WDVA()
        with pytest.raises(ValueError, match="No encryption key"):
            wdva.export_key()


class TestWDVAGenerateKey:
    """Tests for WDVA.generate_key() static method."""

    def test_generate_key_returns_tuple(self) -> None:
        """generate_key should return (bytes, hex) tuple."""
        key_bytes, key_hex = WDVA.generate_key()

        assert isinstance(key_bytes, bytes)
        assert isinstance(key_hex, str)
        assert len(key_bytes) == 32
        assert len(key_hex) == 64

    def test_generate_key_hex_matches_bytes(self) -> None:
        """Hex string should match bytes."""
        key_bytes, key_hex = WDVA.generate_key()
        assert key_bytes.hex() == key_hex


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_training_result_immutable(self) -> None:
        """TrainingResult should be frozen (immutable)."""
        result = TrainingResult(
            adapter_path="/path/to/adapter.wdva",
            encryption_key_hex="a" * 64,
            num_documents=5,
            num_samples=100,
        )

        with pytest.raises(AttributeError):
            result.adapter_path = "/new/path"  # type: ignore

    def test_training_result_repr_hides_key(self) -> None:
        """repr should not expose full encryption key."""
        result = TrainingResult(
            adapter_path="/path/to/adapter.wdva",
            encryption_key_hex="a" * 64,
            num_documents=5,
            num_samples=100,
        )

        repr_str = repr(result)
        assert "a" * 64 not in repr_str
        assert "adapter_path" in repr_str


class TestWDVAQuery:
    """Tests for WDVA query methods."""

    def test_query_without_load_raises(self) -> None:
        """query() should raise when no adapter loaded."""
        wdva = WDVA()
        with pytest.raises(ValueError, match="No adapter loaded"):
            wdva.query("test prompt")
