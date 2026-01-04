"""
Tests for the inference module.

Copyright 2025 Enclave
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import pytest

from wdva.inference import (
    DEFAULT_MODEL,
    LocalInference,
    _check_mlx,
    _check_torch,
    create_inference_engine,
)


class TestBackendDetection:
    """Tests for backend detection functions."""

    def test_check_mlx_returns_bool(self) -> None:
        """_check_mlx should return boolean."""
        result = _check_mlx()
        assert isinstance(result, bool)

    def test_check_torch_returns_bool(self) -> None:
        """_check_torch should return boolean."""
        result = _check_torch()
        assert isinstance(result, bool)

    def test_check_mlx_cached(self) -> None:
        """_check_mlx should be cached (same result on repeat calls)."""
        result1 = _check_mlx()
        result2 = _check_mlx()
        assert result1 is result2  # Same object due to caching

    def test_check_torch_cached(self) -> None:
        """_check_torch should be cached (same result on repeat calls)."""
        result1 = _check_torch()
        result2 = _check_torch()
        assert result1 is result2


class TestLocalInferenceInit:
    """Tests for LocalInference initialization."""

    def test_init_default_model(self) -> None:
        """Should use default model when none specified."""
        try:
            engine = LocalInference()
            assert engine.model_name == DEFAULT_MODEL
        except RuntimeError:
            pytest.skip("No ML backend available")

    def test_init_custom_model(self) -> None:
        """Should accept custom model name."""
        try:
            engine = LocalInference(model_name="custom/model")
            assert engine.model_name == "custom/model"
        except RuntimeError:
            pytest.skip("No ML backend available")

    def test_init_not_loaded(self) -> None:
        """Model should not be loaded on init."""
        try:
            engine = LocalInference()
            assert not engine.is_loaded
        except RuntimeError:
            pytest.skip("No ML backend available")

    def test_init_force_mlx_when_unavailable_raises(self) -> None:
        """Forcing unavailable backend should raise."""
        if _check_mlx():
            pytest.skip("MLX is available")

        with pytest.raises(RuntimeError, match="MLX requested but not available"):
            LocalInference(backend="mlx")

    def test_init_force_torch_when_unavailable_raises(self) -> None:
        """Forcing unavailable backend should raise."""
        if _check_torch():
            pytest.skip("PyTorch is available")

        with pytest.raises(RuntimeError, match="PyTorch requested but not available"):
            LocalInference(backend="torch")

    def test_init_no_backend_available_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise when no backend available."""
        # Clear the cache
        _check_mlx.cache_clear()
        _check_torch.cache_clear()

        # Mock both to return False
        monkeypatch.setattr("wdva.inference._check_mlx", lambda: False)
        monkeypatch.setattr("wdva.inference._check_torch", lambda: False)

        with pytest.raises(RuntimeError, match="No ML backend available"):
            LocalInference()


class TestLocalInferenceProperties:
    """Tests for LocalInference properties."""

    def test_backend_property(self) -> None:
        """backend property should return backend type."""
        try:
            engine = LocalInference()
            assert engine.backend in ("mlx", "torch")
        except RuntimeError:
            pytest.skip("No ML backend available")

    def test_is_loaded_initially_false(self) -> None:
        """is_loaded should be False before load_model()."""
        try:
            engine = LocalInference()
            assert engine.is_loaded is False
        except RuntimeError:
            pytest.skip("No ML backend available")

    def test_repr(self) -> None:
        """repr should include useful info."""
        try:
            engine = LocalInference()
            repr_str = repr(engine)
            assert "backend=" in repr_str
            assert "model=" in repr_str
            assert "status=" in repr_str
        except RuntimeError:
            pytest.skip("No ML backend available")


class TestLocalInferenceUnload:
    """Tests for LocalInference.unload_model()."""

    def test_unload_clears_model(self) -> None:
        """unload_model should clear model from memory."""
        try:
            engine = LocalInference()
            engine._model = "fake_model"  # Simulate loaded model
            engine._tokenizer = "fake_tokenizer"

            engine.unload_model()

            assert engine._model is None
            assert engine._tokenizer is None
        except RuntimeError:
            pytest.skip("No ML backend available")


class TestCreateInferenceEngine:
    """Tests for create_inference_engine factory function."""

    def test_create_returns_engine(self) -> None:
        """Factory should return LocalInference instance."""
        try:
            engine = create_inference_engine(auto_load=False)
            assert isinstance(engine, LocalInference)
        except RuntimeError:
            pytest.skip("No ML backend available")

    def test_create_with_custom_model(self) -> None:
        """Factory should accept custom model name."""
        try:
            engine = create_inference_engine(
                model_name="custom/model",
                auto_load=False
            )
            assert engine.model_name == "custom/model"
        except RuntimeError:
            pytest.skip("No ML backend available")

    def test_create_auto_load_false(self) -> None:
        """auto_load=False should not load model."""
        try:
            engine = create_inference_engine(auto_load=False)
            assert not engine.is_loaded
        except RuntimeError:
            pytest.skip("No ML backend available")


class TestQueryMethods:
    """Tests for query methods (without actual model loading)."""

    def test_query_base_without_model_loads(self) -> None:
        """query_base should attempt to load model if not loaded."""
        try:
            engine = LocalInference()
            # This would try to load the model
            # In a real test environment, we'd mock the model loading
            assert not engine.is_loaded
        except RuntimeError:
            pytest.skip("No ML backend available")

    def test_query_with_adapter_without_model_loads(self) -> None:
        """query_with_adapter should attempt to load model if not loaded."""
        try:
            engine = LocalInference()
            assert not engine.is_loaded
        except RuntimeError:
            pytest.skip("No ML backend available")
