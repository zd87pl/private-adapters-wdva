# WDVA: Secure & Private AI for Everyone

[![Tests](https://github.com/enclave-ai/wdva-boilerplate/actions/workflows/test.yml/badge.svg)](https://github.com/enclave-ai/wdva-boilerplate/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/wdva.svg)](https://pypi.org/project/wdva/)
[![Python](https://img.shields.io/pypi/pyversions/wdva.svg)](https://pypi.org/project/wdva/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Your data. Your model. Your privacy. Zero compromise.**

Weight-Delta Vault Adapters (WDVA) enables **personalized AI without sacrificing privacy**. Train a model on your documents, encrypt it, and run it entirely on your device—no cloud, no data sharing, no compromise.

## The Promise

**Traditional AI:**
- Share your data with cloud providers
- Models trained on your data stored on servers you don't control
- No way to delete your data once it's trained
- Privacy vs. Personalization tradeoff

**WDVA Approach:**
- Your data stays encrypted, always
- Train once, run anywhere—even offline
- Cryptographic "right to be forgotten" (delete the key = delete the model)
- Personalization without privacy compromise

## Quick Start

```bash
# Install from PyPI
pip install wdva

# Or install from source
pip install -e ".[torch]"  # PyTorch backend
pip install -e ".[mlx]"    # MLX backend (Apple Silicon)

# Run examples
python examples/simple_demo.py
python examples/arxiv_demo.py --paper-id 2502.13171
```

## What is WDVA?

**Weight-Delta Vault Adapters** is a privacy-preserving AI personalization technique:

1. **Train** a small adapter (DoRA) on your documents
2. **Encrypt** the adapter with military-grade cryptography (XChaCha20-Poly1305)
3. **Store** encrypted adapter anywhere (cloud, local, USB drive)
4. **Decrypt** and load ephemerally (in-memory only) when needed
5. **Delete** instantly by destroying the encryption key

### Key Properties

| Property | Description |
|----------|-------------|
| **Zero-knowledge** | Server never sees your data or decrypted model |
| **Portable** | Encrypted adapter is small (~20MB) and portable |
| **Fast** | Load adapter in milliseconds, switch between users instantly |
| **Deletable** | Cryptographic deletion—destroy key = model is gone forever |
| **Local-first** | Run entirely on your device, no internet required |

## Use Cases

- **Personal Knowledge Base**: Train on your notes, documents, emails
- **Private Research**: Query research papers without sharing them
- **Medical Records**: Personalized health AI without exposing sensitive data
- **Legal Documents**: Private legal research assistant
- **Code Documentation**: Personal coding assistant trained on your codebase

## Examples

### arXiv Paper Assistant

Train on a research paper and query it privately:

```python
from wdva import WDVA
from examples.arxiv_demo import download_and_train

# Download paper and train adapter
adapter_path, key = download_and_train("2502.13171")

# Create WDVA instance
wdva = WDVA(adapter_path=adapter_path, encryption_key=key)

# Query privately (runs entirely locally)
response = wdva.query("What is the main contribution of this paper?")
print(response)
```

### Simple Document Training

```python
from wdva import WDVA

# Train on your documents
wdva = WDVA()
result = wdva.train(
    documents=["doc1.txt", "doc2.pdf"],
)

# Load and query
wdva.load(result.adapter_path, result.encryption_key_hex)
answer = wdva.query("What did I write about privacy?")

# Cryptographic deletion
wdva.delete()  # Key destroyed, adapter permanently inaccessible
```

## API Reference

### WDVA Class

```python
from wdva import WDVA, WDVAStatus

# Initialize
wdva = WDVA(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Train adapter
result = wdva.train(documents=["doc.pdf"], max_samples=100)

# Load encrypted adapter
wdva.load(adapter_path, encryption_key)

# Query with adapter
response = wdva.query("Your question", max_tokens=512)

# Query base model (no adapter)
response = wdva.query_base("Your question")

# Unload adapter (keeps file)
wdva.unload()

# Cryptographic deletion (destroys key)
wdva.delete()

# Export key for backup
key_hex = wdva.export_key()

# Generate new key
key_bytes, key_hex = WDVA.generate_key()
```

### EncryptedAdapter Class

```python
from wdva import EncryptedAdapter

# Generate key
key = EncryptedAdapter.generate_key()

# Create adapter manager
adapter = EncryptedAdapter(key, enable_compression=True)

# Encrypt weights
adapter.encrypt_weights(weights_dict, "adapter.wdva")

# Decrypt weights (in-memory only)
weights = adapter.decrypt_weights("adapter.wdva")
```

### LocalInference Class

```python
from wdva import LocalInference

# Create inference engine
engine = LocalInference(backend="mlx")  # or "torch"

# Load model
engine.load_model()

# Query
response = engine.query_base("Hello!")

# Unload
engine.unload_model()
```

## Architecture

```
+---------------------------------------+
|         Your Documents                |
|    (PDFs, Notes, Emails, etc.)        |
+------------------+--------------------+
                   |
                   v
+---------------------------------------+
|      DoRA Training (Local)            |
|    Generates small adapter (~20MB)    |
+------------------+--------------------+
                   |
                   v
+---------------------------------------+
|    Encryption (XChaCha20-Poly1305)    |
|    Creates encrypted adapter blob     |
+------------------+--------------------+
                   |
                   v
+---------------------------------------+
|    Encrypted Adapter Storage          |
|  (Cloud, Local, USB - doesn't matter) |
+------------------+--------------------+
                   |
                   v
+---------------------------------------+
|    Ephemeral Loading (Your Device)    |
|  Decrypt -> Load -> Query -> Delete   |
|      (All in memory, never on disk)   |
+---------------------------------------+
```

## Security

WDVA uses industry-standard cryptography:

| Component | Algorithm | Notes |
|-----------|-----------|-------|
| Encryption | XChaCha20-Poly1305 | AEAD with 256-bit keys |
| Key Derivation | HKDF-SHA256 | Per-adapter salt |
| Serialization | SafeTensors | Secure tensor format |
| Compression | Zstandard | Optional, 30-50% reduction |

See [SECURITY.md](SECURITY.md) for the full security policy.

## Extending WDVA

See [docs/EXTENDING.md](docs/EXTENDING.md) for:
- Custom data sources
- Different model backends
- Custom encryption schemes
- Integration with existing systems

## Documentation

- [CONCEPT.md](docs/CONCEPT.md) - Understanding WDVA
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Technical deep dive
- [EXTENDING.md](docs/EXTENDING.md) - How to extend WDVA
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [CHANGELOG.md](CHANGELOG.md) - Version history

## Installation

### From PyPI

```bash
pip install wdva
```

### From Source

```bash
git clone https://github.com/enclave-ai/wdva-boilerplate.git
cd wdva-boilerplate
pip install -e ".[dev]"
```

### Backend Installation

```bash
# Apple Silicon (recommended for Mac)
pip install wdva[mlx]

# PyTorch (universal)
pip install wdva[torch]

# All optional dependencies
pip install wdva[all]
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

Apache 2.0 provides:
- Explicit patent grants (important for AI/ML projects)
- Patent retaliation protection
- Permissive (similar to MIT)
- Common in AI/ML open source projects

## Acknowledgments

WDVA builds on:
- **DoRA** (Decomposed Low-Rank Adaptation) for efficient fine-tuning
- **XChaCha20-Poly1305** for authenticated encryption
- **Small Language Models** (TinyLlama, Llama-3.2-1B) for local inference

---

**Privacy is not a feature—it's a fundamental right. WDVA makes it possible.**
