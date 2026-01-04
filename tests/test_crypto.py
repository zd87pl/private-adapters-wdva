"""
Tests for the crypto module.

Copyright 2025 Enclave
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wdva.crypto import (
    KEY_SIZE,
    NONCE_SIZE,
    SALT_SIZE,
    EncryptedAdapter,
    generate_encryption_key,
)


class TestKeyGeneration:
    """Tests for key generation functions."""

    def test_generate_key_returns_correct_length(self) -> None:
        """Generated key should be exactly KEY_SIZE bytes."""
        key = EncryptedAdapter.generate_key()
        assert len(key) == KEY_SIZE

    def test_generate_key_returns_bytes(self) -> None:
        """Generated key should be bytes type."""
        key = EncryptedAdapter.generate_key()
        assert isinstance(key, bytes)

    def test_generate_key_is_random(self) -> None:
        """Each generated key should be unique."""
        keys = [EncryptedAdapter.generate_key() for _ in range(100)]
        assert len(set(keys)) == 100

    def test_generate_encryption_key_returns_tuple(self) -> None:
        """Module-level function should return (bytes, hex) tuple."""
        key_bytes, key_hex = generate_encryption_key()
        assert isinstance(key_bytes, bytes)
        assert isinstance(key_hex, str)
        assert len(key_bytes) == KEY_SIZE
        assert len(key_hex) == KEY_SIZE * 2

    def test_key_hex_roundtrip(self) -> None:
        """Key should survive bytes -> hex -> bytes conversion."""
        original = EncryptedAdapter.generate_key()
        hex_str = EncryptedAdapter.key_to_hex(original)
        recovered = EncryptedAdapter.key_from_hex(hex_str)
        assert original == recovered


class TestKeyValidation:
    """Tests for key validation."""

    def test_key_to_hex_rejects_non_bytes(self) -> None:
        """key_to_hex should reject non-bytes input."""
        with pytest.raises(TypeError, match="must be bytes"):
            EncryptedAdapter.key_to_hex("not bytes")  # type: ignore

    def test_key_from_hex_rejects_non_string(self) -> None:
        """key_from_hex should reject non-string input."""
        with pytest.raises(TypeError, match="must be str"):
            EncryptedAdapter.key_from_hex(b"not string")  # type: ignore

    def test_key_from_hex_rejects_invalid_hex(self) -> None:
        """key_from_hex should reject invalid hex characters."""
        with pytest.raises(ValueError, match="Invalid hex string"):
            EncryptedAdapter.key_from_hex("not_valid_hex_string_here!")

    def test_key_from_hex_rejects_wrong_length(self) -> None:
        """key_from_hex should reject hex strings of wrong length."""
        short_hex = "abcd1234"  # Too short
        with pytest.raises(ValueError, match="Invalid key length"):
            EncryptedAdapter.key_from_hex(short_hex)


class TestEncryptedAdapterInit:
    """Tests for EncryptedAdapter initialization."""

    def test_init_with_valid_key(self, sample_key_bytes: bytes) -> None:
        """Should initialize successfully with valid key."""
        adapter = EncryptedAdapter(sample_key_bytes)
        assert adapter is not None

    def test_init_rejects_short_key(self) -> None:
        """Should reject keys shorter than KEY_SIZE."""
        with pytest.raises(ValueError, match=f"exactly {KEY_SIZE} bytes"):
            EncryptedAdapter(b"too_short")

    def test_init_rejects_long_key(self) -> None:
        """Should reject keys longer than KEY_SIZE."""
        long_key = b"x" * (KEY_SIZE + 10)
        with pytest.raises(ValueError, match=f"exactly {KEY_SIZE} bytes"):
            EncryptedAdapter(long_key)

    def test_init_rejects_non_bytes(self) -> None:
        """Should reject non-bytes key."""
        with pytest.raises(TypeError, match="must be bytes"):
            EncryptedAdapter("string_key")  # type: ignore

    def test_init_with_compression_disabled(self, sample_key_bytes: bytes) -> None:
        """Should allow disabling compression."""
        adapter = EncryptedAdapter(sample_key_bytes, enable_compression=False)
        assert not adapter._enable_compression


class TestEncryptDecrypt:
    """Tests for encryption and decryption."""

    @pytest.fixture
    def adapter(self, sample_key_bytes: bytes) -> EncryptedAdapter:
        """Create an EncryptedAdapter instance."""
        return EncryptedAdapter(sample_key_bytes)

    def test_encrypt_creates_file(
        self, adapter: EncryptedAdapter, mock_weights: dict, temp_dir: Path
    ) -> None:
        """Encryption should create output file."""
        output_path = temp_dir / "test.wdva"
        adapter.encrypt_weights(mock_weights, output_path)
        assert output_path.exists()

    def test_encrypted_file_is_json(
        self, adapter: EncryptedAdapter, mock_weights: dict, temp_dir: Path
    ) -> None:
        """Encrypted file should be valid JSON."""
        output_path = temp_dir / "test.wdva"
        adapter.encrypt_weights(mock_weights, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "ciphertext" in data
        assert "salt" in data
        assert "nonce" in data
        assert "tag" in data
        assert "algorithm" in data

    def test_decrypt_recovers_weights(
        self, adapter: EncryptedAdapter, mock_weights: dict, temp_dir: Path
    ) -> None:
        """Decryption should recover original weights."""
        output_path = temp_dir / "test.wdva"
        adapter.encrypt_weights(mock_weights, output_path)

        recovered = adapter.decrypt_weights(output_path)

        assert set(recovered.keys()) == set(mock_weights.keys())
        for key in mock_weights:
            assert recovered[key].shape == mock_weights[key].shape

    def test_decrypt_with_wrong_key_fails(
        self, adapter: EncryptedAdapter, mock_weights: dict, temp_dir: Path
    ) -> None:
        """Decryption with wrong key should fail."""
        output_path = temp_dir / "test.wdva"
        adapter.encrypt_weights(mock_weights, output_path)

        # Create new adapter with different key
        wrong_key = EncryptedAdapter.generate_key()
        wrong_adapter = EncryptedAdapter(wrong_key)

        with pytest.raises(ValueError, match="wrong key or data has been tampered"):
            wrong_adapter.decrypt_weights(output_path)

    def test_decrypt_detects_tampering(
        self, adapter: EncryptedAdapter, mock_weights: dict, temp_dir: Path
    ) -> None:
        """Decryption should detect tampered ciphertext."""
        output_path = temp_dir / "test.wdva"
        adapter.encrypt_weights(mock_weights, output_path)

        # Tamper with the ciphertext
        with open(output_path) as f:
            data = json.load(f)

        # Modify ciphertext (base64 encoded)
        ciphertext = data["ciphertext"]
        tampered = ciphertext[:-4] + "XXXX"
        data["ciphertext"] = tampered

        with open(output_path, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError):
            adapter.decrypt_weights(output_path)

    def test_decrypt_nonexistent_file_fails(self, adapter: EncryptedAdapter) -> None:
        """Decryption should fail for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            adapter.decrypt_weights("/nonexistent/path.wdva")

    def test_metadata_included_in_package(
        self, adapter: EncryptedAdapter, mock_weights: dict, temp_dir: Path
    ) -> None:
        """Custom metadata should be included in encrypted package."""
        output_path = temp_dir / "test.wdva"
        custom_meta = {"custom_field": "custom_value", "version": "test"}

        package = adapter.encrypt_weights(
            mock_weights, output_path, metadata=custom_meta
        )

        assert package["metadata"]["custom_field"] == "custom_value"


class TestEncryptedPackageStructure:
    """Tests for encrypted package format."""

    def test_package_has_required_fields(
        self, sample_key_bytes: bytes, mock_weights: dict, temp_dir: Path
    ) -> None:
        """Encrypted package should have all required fields."""
        adapter = EncryptedAdapter(sample_key_bytes)
        output_path = temp_dir / "test.wdva"

        package = adapter.encrypt_weights(mock_weights, output_path)

        required_fields = {"salt", "nonce", "ciphertext", "tag", "metadata", "algorithm", "kdf"}
        assert required_fields <= set(package.keys())

    def test_algorithm_is_xchacha20_poly1305(
        self, sample_key_bytes: bytes, mock_weights: dict, temp_dir: Path
    ) -> None:
        """Algorithm should be XChaCha20-Poly1305."""
        adapter = EncryptedAdapter(sample_key_bytes)
        output_path = temp_dir / "test.wdva"

        package = adapter.encrypt_weights(mock_weights, output_path)

        assert package["algorithm"] == "XChaCha20-Poly1305"

    def test_kdf_is_hkdf_sha256(
        self, sample_key_bytes: bytes, mock_weights: dict, temp_dir: Path
    ) -> None:
        """KDF should be HKDF-SHA256."""
        adapter = EncryptedAdapter(sample_key_bytes)
        output_path = temp_dir / "test.wdva"

        package = adapter.encrypt_weights(mock_weights, output_path)

        assert package["kdf"] == "HKDF-SHA256"
