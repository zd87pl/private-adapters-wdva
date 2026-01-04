"""
WDVA: Weight-Delta Vault Adapters

Secure and Private AI for Everyone

Your data. Your model. Your privacy. Zero compromise.

Copyright 2025 Enclave
Licensed under the Apache License, Version 2.0
"""

from wdva.crypto import EncryptedAdapter
from wdva.exceptions import (
    AdapterError,
    BackendError,
    ConfigurationError,
    DecryptionError,
    EncryptionError,
    InferenceError,
    ModelError,
    WDVAError,
)
from wdva.inference import LocalInference
from wdva.wdva import WDVA, TrainingResult, WDVAStatus

__version__ = "1.0.0"
__author__ = "Enclave Team"
__email__ = "team@enclave.ai"
__license__ = "Apache-2.0"

__all__ = [
    # Main API
    "WDVA",
    "EncryptedAdapter",
    "LocalInference",
    # Types
    "WDVAStatus",
    "TrainingResult",
    # Exceptions
    "WDVAError",
    "EncryptionError",
    "DecryptionError",
    "AdapterError",
    "InferenceError",
    "BackendError",
    "ModelError",
    "ConfigurationError",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]
