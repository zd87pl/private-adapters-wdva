"""
WDVA Exceptions - Custom exception hierarchy.

This module defines the exception hierarchy for WDVA, providing
specific exception types for different error conditions.

Copyright 2025 Enclave
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations


class WDVAError(Exception):
    """
    Base exception for all WDVA errors.

    All WDVA-specific exceptions inherit from this class, allowing
    callers to catch all WDVA errors with a single except clause.

    Example:
        >>> try:
        ...     wdva.query("test")
        ... except WDVAError as e:
        ...     print(f"WDVA error: {e}")
    """

    pass


class EncryptionError(WDVAError):
    """
    Raised when encryption operations fail.

    This includes failures during:
    - Key derivation
    - Cipher initialization
    - Data encryption
    - File writing

    Example:
        >>> try:
        ...     adapter.encrypt_weights(weights, path)
        ... except EncryptionError as e:
        ...     print(f"Encryption failed: {e}")
    """

    pass


class DecryptionError(WDVAError):
    """
    Raised when decryption operations fail.

    This includes failures during:
    - File reading
    - Key derivation
    - Authentication tag verification
    - Data decryption
    - Decompression

    Example:
        >>> try:
        ...     weights = adapter.decrypt_weights(path)
        ... except DecryptionError as e:
        ...     print(f"Decryption failed: {e}")
    """

    pass


class KeyError(WDVAError):
    """
    Raised for encryption key-related errors.

    This includes:
    - Invalid key format
    - Invalid key length
    - Key derivation failures

    Note: This shadows the built-in KeyError, but is only used within
    WDVA context. Import explicitly if needed alongside dict operations.

    Example:
        >>> try:
        ...     key = EncryptedAdapter.key_from_hex("invalid")
        ... except wdva.exceptions.KeyError as e:
        ...     print(f"Key error: {e}")
    """

    pass


class AdapterError(WDVAError):
    """
    Raised for adapter-related errors.

    This includes:
    - Adapter file not found
    - Invalid adapter format
    - Adapter loading failures

    Example:
        >>> try:
        ...     wdva.load("nonexistent.wdva", key)
        ... except AdapterError as e:
        ...     print(f"Adapter error: {e}")
    """

    pass


class InferenceError(WDVAError):
    """
    Raised when inference operations fail.

    This includes:
    - Model loading failures
    - Backend initialization errors
    - Generation failures

    Example:
        >>> try:
        ...     response = wdva.query("test")
        ... except InferenceError as e:
        ...     print(f"Inference failed: {e}")
    """

    pass


class BackendError(InferenceError):
    """
    Raised when ML backend operations fail.

    This includes:
    - No backend available
    - Backend initialization failure
    - Backend-specific errors

    Example:
        >>> try:
        ...     engine = LocalInference(backend="mlx")
        ... except BackendError as e:
        ...     print(f"Backend error: {e}")
    """

    pass


class ModelError(InferenceError):
    """
    Raised when model operations fail.

    This includes:
    - Model download failure
    - Model loading failure
    - Model format errors

    Example:
        >>> try:
        ...     engine.load_model()
        ... except ModelError as e:
        ...     print(f"Model error: {e}")
    """

    pass


class ConfigurationError(WDVAError):
    """
    Raised for configuration-related errors.

    This includes:
    - Invalid configuration values
    - Missing required configuration
    - Incompatible configuration combinations

    Example:
        >>> try:
        ...     wdva = WDVA(model_name="")
        ... except ConfigurationError as e:
        ...     print(f"Configuration error: {e}")
    """

    pass
