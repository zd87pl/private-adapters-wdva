"""
Encrypted Adapter - Core encryption/decryption for WDVA.

This module handles the cryptographic operations that make WDVA secure:
- XChaCha20-Poly1305 authenticated encryption
- HKDF key derivation
- Secure serialization/deserialization

Security Notes:
- Uses 256-bit keys (32 bytes) for XChaCha20-Poly1305
- Uses 192-bit nonces (24 bytes) for XChaCha20
- Uses HKDF-SHA256 for key derivation with random salt
- Authenticated encryption prevents tampering
- AAD (Additional Authenticated Data) binds metadata to ciphertext

Copyright 2025 Enclave
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import io
import json
import logging
import os
import secrets
from base64 import b64decode, b64encode
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

logger = logging.getLogger(__name__)

# =============================================================================
# Dependency Availability Checks
# =============================================================================

_CRYPTO_AVAILABLE = False
_COMPRESSION_AVAILABLE = False
_SAFETENSORS_AVAILABLE = False

try:
    from Crypto.Cipher import ChaCha20_Poly1305
    from Crypto.Random import get_random_bytes
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    _CRYPTO_AVAILABLE = True
except ImportError:
    pass

try:
    import zstandard as zstd
    _COMPRESSION_AVAILABLE = True
except ImportError:
    pass

try:
    from safetensors.torch import load as safetensors_load
    from safetensors.torch import save as safetensors_save
    _SAFETENSORS_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Type Definitions
# =============================================================================

class EncryptedPackage(TypedDict):
    """Structure of an encrypted adapter file."""
    salt: str  # Base64-encoded
    nonce: str  # Base64-encoded
    ciphertext: str  # Base64-encoded
    tag: str  # Base64-encoded
    metadata: Dict[str, Any]
    algorithm: str
    kdf: str


# =============================================================================
# Constants
# =============================================================================

KEY_SIZE = 32  # 256 bits
NONCE_SIZE = 24  # 192 bits for XChaCha20
SALT_SIZE = 16  # 128 bits
TAG_SIZE = 16  # 128 bits

ENCRYPTION_INFO = b"wdva-encryption-v1"
CURRENT_VERSION = "1.0"
ALGORITHM = "XChaCha20-Poly1305"
KDF_ALGORITHM = "HKDF-SHA256"


# =============================================================================
# EncryptedAdapter Class
# =============================================================================

class EncryptedAdapter:
    """
    Encrypted adapter manager for WDVA.
    
    Handles encryption and decryption of DoRA adapter weights using
    XChaCha20-Poly1305 authenticated encryption.
    
    Security Properties:
    - Confidentiality: Encrypted data is unreadable without key
    - Integrity: Authentication tag detects any tampering
    - Authenticity: AAD binds metadata to ciphertext
    
    Example:
        >>> key = EncryptedAdapter.generate_key()
        >>> adapter = EncryptedAdapter(key)
        >>> adapter.encrypt_weights(weights, "adapter.wdva")
        >>> decrypted = adapter.decrypt_weights("adapter.wdva")
    """
    
    __slots__ = ('_encryption_key', '_enable_compression', '_compressor', '_decompressor')
    
    def __init__(
        self,
        encryption_key: bytes,
        *,
        enable_compression: bool = True
    ) -> None:
        """
        Initialize encrypted adapter manager.
        
        Args:
            encryption_key: 32-byte encryption key (use generate_key() to create)
            enable_compression: Whether to compress before encryption (default: True)
            
        Raises:
            ImportError: If required cryptography libraries are not available
            ValueError: If encryption key is not exactly 32 bytes
        """
        if not _CRYPTO_AVAILABLE:
            raise ImportError(
                "Cryptography libraries not available. "
                "Install with: pip install pycryptodome cryptography"
            )
        
        if not isinstance(encryption_key, bytes):
            raise TypeError(f"encryption_key must be bytes, got {type(encryption_key).__name__}")
        
        if len(encryption_key) != KEY_SIZE:
            raise ValueError(
                f"Encryption key must be exactly {KEY_SIZE} bytes, got {len(encryption_key)}"
            )
        
        self._encryption_key = encryption_key
        self._enable_compression = enable_compression and _COMPRESSION_AVAILABLE
        
        # Initialize compressor/decompressor if compression is enabled
        self._compressor: Optional[Any] = None
        self._decompressor: Optional[Any] = None
        
        if self._enable_compression:
            self._compressor = zstd.ZstdCompressor(level=3)
            self._decompressor = zstd.ZstdDecompressor()
        
        logger.debug("Initialized EncryptedAdapter (compression=%s)", self._enable_compression)
    
    def _derive_key(self, salt: bytes, info: bytes = ENCRYPTION_INFO) -> bytes:
        """
        Derive encryption key using HKDF-SHA256.
        
        This ensures that even with the same master key, different adapters
        have different encryption keys (due to random salt).
        
        Args:
            salt: Random salt (should be unique per adapter)
            info: Application-specific info string
            
        Returns:
            Derived 32-byte key
        """
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=salt,
            info=info,
        )
        return hkdf.derive(self._encryption_key)
    
    def encrypt_weights(
        self,
        weights: Dict[str, Any],
        output_path: str | Path,
        *,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EncryptedPackage:
        """
        Encrypt adapter weights and save to file.
        
        Args:
            weights: Dictionary of adapter weights (PyTorch tensors)
            output_path: Path to save encrypted adapter
            metadata: Optional metadata to include (stored in plaintext but authenticated)
            
        Returns:
            Dictionary with encryption metadata
            
        Raises:
            ImportError: If safetensors is not available
            OSError: If file cannot be written
        """
        if not _SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors not available. Install with: pip install safetensors"
            )
        
        output_path = Path(output_path)
        
        # Serialize weights to safetensors format (in memory)
        buffer = io.BytesIO()
        safetensors_save(weights, buffer)
        serialized_data = buffer.getvalue()
        original_size = len(serialized_data)
        
        # Compress if enabled
        if self._enable_compression and self._compressor is not None:
            data_to_encrypt = self._compressor.compress(serialized_data)
            compressed_size = len(data_to_encrypt)
            compression_ratio = (1 - compressed_size / original_size) * 100
            logger.debug(
                "Compressed: %d -> %d bytes (%.1f%% reduction)",
                original_size, compressed_size, compression_ratio
            )
        else:
            data_to_encrypt = serialized_data
        
        # Generate random salt and derive encryption key
        salt = get_random_bytes(SALT_SIZE)
        derived_key = self._derive_key(salt)
        
        # Generate random nonce (24 bytes for XChaCha20)
        nonce = get_random_bytes(NONCE_SIZE)
        
        # Create cipher
        cipher = ChaCha20_Poly1305.new(key=derived_key, nonce=nonce)
        
        # Prepare metadata (stored in plaintext but authenticated)
        encryption_metadata: Dict[str, Any] = {
            'version': CURRENT_VERSION,
            'num_tensors': len(weights),
            'original_size': original_size,
            'compressed': self._enable_compression,
        }
        if metadata:
            encryption_metadata.update(metadata)
        
        # Add metadata as AAD (Authenticated Associated Data)
        # This ensures metadata cannot be tampered with
        aad = json.dumps(encryption_metadata, sort_keys=True, separators=(',', ':')).encode('utf-8')
        cipher.update(aad)
        
        # Encrypt and generate authentication tag
        ciphertext, tag = cipher.encrypt_and_digest(data_to_encrypt)
        
        # Package encrypted data
        encrypted_package: EncryptedPackage = {
            'salt': b64encode(salt).decode('ascii'),
            'nonce': b64encode(nonce).decode('ascii'),
            'ciphertext': b64encode(ciphertext).decode('ascii'),
            'tag': b64encode(tag).decode('ascii'),
            'metadata': encryption_metadata,
            'algorithm': ALGORITHM,
            'kdf': KDF_ALGORITHM,
        }
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(encrypted_package, f, indent=2)
        
        # Set file permissions (readable by all, writable by owner)
        try:
            os.chmod(output_path, 0o644)
        except OSError:
            pass  # May fail on some platforms, non-critical
        
        logger.info("Encrypted adapter saved to %s (%d tensors)", output_path, len(weights))
        return encrypted_package
    
    def decrypt_weights(self, encrypted_path: str | Path) -> Dict[str, Any]:
        """
        Decrypt adapter weights from file.
        
        This loads weights into memory only—they are never persisted to disk.
        
        Args:
            encrypted_path: Path to encrypted adapter file
            
        Returns:
            Dictionary of decrypted weights (PyTorch tensors)
            
        Raises:
            FileNotFoundError: If encrypted file doesn't exist
            ValueError: If decryption fails (wrong key or tampered data)
            json.JSONDecodeError: If file is not valid JSON
        """
        encrypted_path = Path(encrypted_path)
        
        if not encrypted_path.exists():
            raise FileNotFoundError(f"Encrypted adapter not found: {encrypted_path}")
        
        # Load encrypted package
        with open(encrypted_path, 'r', encoding='utf-8') as f:
            encrypted_package: EncryptedPackage = json.load(f)
        
        # Validate structure
        required_fields = {'salt', 'nonce', 'ciphertext', 'tag', 'metadata', 'algorithm', 'kdf'}
        missing_fields = required_fields - set(encrypted_package.keys())
        if missing_fields:
            raise ValueError(f"Invalid encrypted adapter: missing fields {missing_fields}")
        
        # Check algorithm compatibility
        if encrypted_package['algorithm'] != ALGORITHM:
            raise ValueError(
                f"Unsupported algorithm: {encrypted_package['algorithm']} "
                f"(expected {ALGORITHM})"
            )
        
        # Check version (with forward compatibility)
        version = encrypted_package.get('metadata', {}).get('version', CURRENT_VERSION)
        if version != CURRENT_VERSION:
            logger.warning("Version mismatch: file is v%s, code is v%s", version, CURRENT_VERSION)
        
        # Derive decryption key
        salt = b64decode(encrypted_package['salt'])
        derived_key = self._derive_key(salt)
        
        # Create cipher for decryption
        nonce = b64decode(encrypted_package['nonce'])
        cipher = ChaCha20_Poly1305.new(key=derived_key, nonce=nonce)
        
        # Re-add AAD for authentication verification
        aad = json.dumps(
            encrypted_package['metadata'],
            sort_keys=True,
            separators=(',', ':')
        ).encode('utf-8')
        cipher.update(aad)
        
        # Decrypt and verify authentication tag
        ciphertext = b64decode(encrypted_package['ciphertext'])
        tag = b64decode(encrypted_package['tag'])
        
        try:
            decrypted_data = cipher.decrypt_and_verify(ciphertext, tag)
        except ValueError as e:
            raise ValueError(
                "Decryption failed: wrong key or data has been tampered with. "
                f"Original error: {e}"
            ) from e
        
        # Decompress if necessary
        is_compressed = encrypted_package['metadata'].get('compressed', False)
        if is_compressed:
            if not _COMPRESSION_AVAILABLE:
                raise ImportError(
                    "zstandard not available but adapter is compressed. "
                    "Install with: pip install zstandard"
                )
            if self._decompressor is None:
                self._decompressor = zstd.ZstdDecompressor()
            serialized_data = self._decompressor.decompress(decrypted_data)
        else:
            serialized_data = decrypted_data
        
        # Deserialize from safetensors (in-memory only)
        if not _SAFETENSORS_AVAILABLE:
            raise ImportError(
                "safetensors not available. Install with: pip install safetensors"
            )
        
        weights = safetensors_load(serialized_data)
        
        num_tensors = len(weights)
        logger.info("Decrypted %d tensors from %s", num_tensors, encrypted_path)
        
        return weights
    
    @staticmethod
    def generate_key() -> bytes:
        """
        Generate a cryptographically secure random encryption key.
        
        Returns:
            32-byte random key suitable for XChaCha20-Poly1305
            
        Example:
            >>> key = EncryptedAdapter.generate_key()
            >>> print(f"Key (hex): {key.hex()}")
        """
        return secrets.token_bytes(KEY_SIZE)
    
    @staticmethod
    def key_to_hex(key: bytes) -> str:
        """
        Convert key bytes to hex string for storage.
        
        Args:
            key: Key bytes
            
        Returns:
            Hex-encoded key string
        """
        if not isinstance(key, bytes):
            raise TypeError(f"key must be bytes, got {type(key).__name__}")
        return key.hex()
    
    @staticmethod
    def key_from_hex(key_hex: str) -> bytes:
        """
        Convert hex string back to key bytes.
        
        Args:
            key_hex: Hex-encoded key string
            
        Returns:
            Key bytes
            
        Raises:
            ValueError: If hex string is invalid or wrong length
        """
        if not isinstance(key_hex, str):
            raise TypeError(f"key_hex must be str, got {type(key_hex).__name__}")
        
        try:
            key = bytes.fromhex(key_hex)
        except ValueError as e:
            raise ValueError(f"Invalid hex string: {e}") from e
        
        if len(key) != KEY_SIZE:
            raise ValueError(
                f"Invalid key length: expected {KEY_SIZE} bytes, got {len(key)}"
            )
        
        return key
    
    def __repr__(self) -> str:
        return f"EncryptedAdapter(compression={self._enable_compression})"


# =============================================================================
# Module-level convenience functions
# =============================================================================

def generate_encryption_key() -> tuple[bytes, str]:
    """
    Generate a new encryption key.
    
    Returns:
        Tuple of (key_bytes, key_hex)
        
    Example:
        >>> key_bytes, key_hex = generate_encryption_key()
        >>> print(f"Store this securely: {key_hex}")
    """
    key = EncryptedAdapter.generate_key()
    return key, key.hex()
