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
