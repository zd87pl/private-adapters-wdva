"""
WDVA: Weight-Delta Vault Adapters

Secure and Private AI for Everyone

Your data. Your model. Your privacy. Zero compromise.

Copyright 2025 Enclave
Licensed under the Apache License, Version 2.0
"""

from wdva.wdva import WDVA
from wdva.crypto import EncryptedAdapter
from wdva.inference import LocalInference

__version__ = "1.0.0"
__author__ = "Enclave Team"
__license__ = "Apache-2.0"

__all__ = [
    "WDVA",
    "EncryptedAdapter",
    "LocalInference",
    "__version__",
]

# LangChain integration (lazy import to avoid dependency requirement)
def __getattr__(name: str):
    """Lazy import LangChain components."""
    langchain_exports = {
        "WDVALLM",
        "WDVAChatModel",
        "WDVADocumentLoader",
        "WDVAEmbeddings",
        "WDVACallbackHandler",
        "create_wdva_chain",
        "create_conversational_chain",
    }

    if name in langchain_exports:
        from wdva import langchain_integration
        return getattr(langchain_integration, name)

    raise AttributeError(f"module 'wdva' has no attribute '{name}'")
