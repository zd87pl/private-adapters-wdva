"""
Main WDVA class - High-level API for secure and private AI.

This is the primary interface for using WDVA. It handles:
- Training adapters on your documents
- Encrypting adapters securely
- Loading and querying adapters locally
- Managing encryption keys

Example:
    >>> wdva = WDVA()
    >>> adapter_path, key = wdva.train(["my_doc.pdf"])
    >>> wdva.load(adapter_path, key)
    >>> response = wdva.query("What is this document about?")

Copyright 2025 Enclave
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from wdva.crypto import EncryptedAdapter
from wdva.inference import LocalInference

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

class WDVAStatus(Enum):
    """WDVA instance status."""
    READY = "ready"  # Initialized, no adapter loaded
    LOADED = "loaded"  # Adapter loaded and ready for queries
    TRAINING = "training"  # Currently training (future use)


@dataclass(frozen=True)
class TrainingResult:
    """Result of a training operation."""
    adapter_path: str
    encryption_key_hex: str
    num_documents: int
    num_samples: int
    
    def __repr__(self) -> str:
        return (
            f"TrainingResult(adapter_path='{self.adapter_path}', "
            f"num_documents={self.num_documents}, num_samples={self.num_samples})"
        )


# =============================================================================
# WDVA Class
# =============================================================================

class WDVA:
    """
    Weight-Delta Vault Adapters - Secure and Private AI.
    
    Train personalized AI models on your documents without compromising privacy.
    
    Key Features:
    - Train adapters on your documents locally
    - Encrypt adapters with military-grade cryptography
    - Query locally (no cloud, no data sharing)
    - Cryptographic deletion (destroy key = delete model)
    
    Example:
        >>> # Simple usage
        >>> wdva = WDVA()
        >>> wdva.load("adapter.wdva", "your-encryption-key-hex")
        >>> response = wdva.query("What is the main topic?")
        
        >>> # With explicit adapter and key
        >>> wdva = WDVA(adapter_path="adapter.wdva", encryption_key="...")
        >>> response = wdva.query("What is this about?")
    """
    
    __slots__ = (
        '_model_name',
        '_adapter_path',
        '_encryption_key',
        '_inference_engine',
        '_status',
    )
    
    DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    def __init__(
        self,
        adapter_path: Optional[str | Path] = None,
        encryption_key: Optional[str] = None,
        *,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Initialize WDVA instance.
        
        Args:
            adapter_path: Path to encrypted adapter file (optional)
            encryption_key: Hex-encoded encryption key (optional)
            model_name: Base model to use for inference (default: TinyLlama-1.1B)
            
        Note:
            If both adapter_path and encryption_key are provided, the adapter
            is ready for queries immediately. Otherwise, call load() first.
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._adapter_path: Optional[Path] = None
        self._encryption_key: Optional[str] = None
        self._inference_engine: Optional[LocalInference] = None
        self._status = WDVAStatus.READY
        
        # Load adapter if both path and key provided
        if adapter_path is not None and encryption_key is not None:
            self.load(adapter_path, encryption_key)
        elif adapter_path is not None or encryption_key is not None:
            logger.warning(
                "Both adapter_path and encryption_key must be provided together. "
                "Call load() to load an adapter."
            )
    
    @property
    def status(self) -> WDVAStatus:
        """Get current status."""
        return self._status
    
    @property
    def is_loaded(self) -> bool:
        """Check if an adapter is loaded."""
        return self._status == WDVAStatus.LOADED
    
    @property
    def adapter_path(self) -> Optional[Path]:
        """Get loaded adapter path."""
        return self._adapter_path
    
    @property
    def model_name(self) -> str:
        """Get base model name."""
        return self._model_name
    
    def train(
        self,
        documents: List[str | Path],
        *,
        output_path: Optional[str | Path] = None,
        max_samples: int = 100,
        epochs: int = 3,
        **kwargs: Any,
    ) -> TrainingResult:
        """
        Train an adapter on your documents.
        
        This is a simplified training interface. For production training with
        full control, use the training utilities from the main Enclave repository.
        
        Args:
            documents: List of document paths (PDF, TXT, etc.)
            output_path: Where to save encrypted adapter 
                        (default: ./adapters/{timestamp}.wdva)
            max_samples: Maximum Q&A pairs to generate (default: 100)
            epochs: Training epochs (default: 3)
            **kwargs: Additional training parameters
            
        Returns:
            TrainingResult with adapter_path and encryption_key_hex
            
        Note:
            This boilerplate provides a simplified training interface.
            Full training implementation requires additional setup.
            See examples/arxiv_demo.py for a complete workflow.
        """
        # Validate documents
        if not documents:
            raise ValueError("At least one document must be provided")
        
        document_paths = [Path(d) for d in documents]
        for doc_path in document_paths:
            if not doc_path.exists():
                logger.warning("Document not found: %s", doc_path)
        
        # Generate encryption key
        encryption_key_bytes = secrets.token_bytes(32)
        encryption_key_hex = encryption_key_bytes.hex()
        
        # Determine output path
        if output_path is None:
            output_dir = Path("./adapters")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use first document name or timestamp
            import time
            timestamp = int(time.time())
            output_path = output_dir / f"adapter_{timestamp}.wdva"
        else:
            output_path = Path(output_path)
        
        # Log training info
        logger.info(
            "Training adapter: %d documents, max_samples=%d, epochs=%d",
            len(documents), max_samples, epochs
        )
        
        # NOTE: Full training implementation requires:
        # 1. Text extraction from documents
        # 2. Q&A pair generation (using a larger model)
        # 3. DoRA adapter training with PEFT
        # 4. Weight encryption
        #
        # This boilerplate provides the structure but not full implementation.
        # See the main Enclave repository for complete training code.
        
        logger.warning(
            "Full training requires additional setup. "
            "See examples/arxiv_demo.py for complete workflow."
        )
        
        # Create placeholder result
        # In production, this would contain actual training results
        result = TrainingResult(
            adapter_path=str(output_path),
            encryption_key_hex=encryption_key_hex,
            num_documents=len(documents),
            num_samples=0,  # Placeholder
        )
        
        logger.info("Training result: %s", result)
        return result
    
    def load(
        self,
        adapter_path: str | Path,
        encryption_key: str,
    ) -> None:
        """
        Load an encrypted adapter.
        
        Args:
            adapter_path: Path to encrypted adapter file
            encryption_key: Hex-encoded encryption key (64 characters)
            
        Raises:
            FileNotFoundError: If adapter file doesn't exist
            ValueError: If encryption key is invalid
        """
        adapter_path = Path(adapter_path)
        
        # Validate path
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        
        # Validate encryption key format
        if not encryption_key or len(encryption_key) != 64:
            raise ValueError(
                "Encryption key must be 64 hex characters (32 bytes). "
                f"Got {len(encryption_key) if encryption_key else 0} characters."
            )
        
        try:
            # Validate hex format
            bytes.fromhex(encryption_key)
        except ValueError as e:
            raise ValueError(f"Invalid encryption key format: {e}") from e
        
        self._adapter_path = adapter_path
        self._encryption_key = encryption_key
        self._status = WDVAStatus.LOADED
        
        logger.info("Loaded adapter: %s", adapter_path)
    
    def query(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Query your personalized AI model.
        
        The adapter is loaded ephemerally (in-memory only) and never
        persists to disk. After inference, all decrypted data is securely
        cleared from memory.
        
        Args:
            prompt: Your question or prompt
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature 0.0-1.0 (default: 0.7)
            **kwargs: Additional inference parameters
            
        Returns:
            Model response as string
            
        Raises:
            ValueError: If no adapter is loaded
            RuntimeError: If inference fails
            
        Example:
            >>> wdva = WDVA(adapter_path="adapter.wdva", encryption_key="...")
            >>> response = wdva.query("What is the main topic?")
            >>> print(response)
        """
        if not self.is_loaded:
            raise ValueError(
                "No adapter loaded. Call load() or provide adapter_path "
                "and encryption_key in __init__"
            )
        
        # Initialize inference engine (lazy loading)
        if self._inference_engine is None:
            self._inference_engine = LocalInference(model_name=self._model_name)
        
        # Query with adapter
        response = self._inference_engine.query_with_adapter(
            adapter_path=str(self._adapter_path),
            encryption_key=self._encryption_key,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response
    
    def query_base(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Query the base model without adapter (for comparison/testing).
        
        Args:
            prompt: Your question or prompt
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Sampling temperature 0.0-1.0 (default: 0.7)
            **kwargs: Additional inference parameters
            
        Returns:
            Model response as string
        """
        if self._inference_engine is None:
            self._inference_engine = LocalInference(model_name=self._model_name)
        
        return self._inference_engine.query_base(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def unload(self) -> None:
        """
        Unload the current adapter.
        
        This clears the adapter from memory but does NOT delete the
        encrypted file or destroy the encryption key. Use delete() for
        cryptographic deletion.
        """
        self._adapter_path = None
        self._encryption_key = None
        self._status = WDVAStatus.READY
        
        logger.info("Adapter unloaded")
    
    def delete(self) -> None:
        """
        Cryptographically delete the adapter.
        
        This destroys the encryption key, making the adapter permanently
        inaccessible. The encrypted file can remain, but without the key,
        it's cryptographically impossible to decrypt.
        
        This implements the "right to be forgotten"—once the key is destroyed,
        the data is effectively deleted even if the encrypted blob persists.
        
        After calling this:
        - The encryption key is destroyed
        - The adapter cannot be decrypted
        - You must call load() with a different adapter to continue
        
        Example:
            >>> wdva.delete()  # Key destroyed, adapter inaccessible
            >>> # Adapter file still exists but cannot be decrypted
        """
        if self._encryption_key:
            # Note: In Python, we can't truly zero memory, but we can
            # remove the reference and let GC handle it
            self._encryption_key = None
            logger.info(
                "Encryption key destroyed - adapter is cryptographically deleted"
            )
        
        self._adapter_path = None
        self._status = WDVAStatus.READY
        
        # Clear inference engine
        if self._inference_engine is not None:
            self._inference_engine.unload_model()
            self._inference_engine = None
    
    def export_key(self) -> str:
        """
        Export encryption key for backup.
        
        WARNING: Store this securely! Anyone with this key can decrypt
        your adapter and access your trained model.
        
        Returns:
            Hex-encoded encryption key (64 characters)
            
        Raises:
            ValueError: If no adapter is loaded
        """
        if not self._encryption_key:
            raise ValueError("No encryption key to export (adapter not loaded)")
        
        return self._encryption_key
    
    @staticmethod
    def generate_key() -> Tuple[bytes, str]:
        """
        Generate a new encryption key.
        
        Returns:
            Tuple of (key_bytes, key_hex)
            
        Example:
            >>> key_bytes, key_hex = WDVA.generate_key()
            >>> print(f"Store securely: {key_hex}")
        """
        key_bytes = EncryptedAdapter.generate_key()
        key_hex = key_bytes.hex()
        return key_bytes, key_hex
    
    def __repr__(self) -> str:
        adapter_info = f"adapter={self._adapter_path.name}" if self._adapter_path else "no adapter"
        return f"WDVA(status={self._status.value}, {adapter_info}, model={self._model_name})"
    
    def __str__(self) -> str:
        return self.__repr__()
