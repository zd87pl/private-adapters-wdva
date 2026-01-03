# WDVA Project Summary

## What is WDVA?

**Weight-Delta Vault Adapters (WDVA)** is a privacy-preserving AI personalization technique that enables you to:

1. **Train** personalized AI models on your documents
2. **Encrypt** the trained adapter with military-grade cryptography
3. **Store** the encrypted adapter anywhere (cloud, local, USB)
4. **Query** locally without any data leaving your device
5. **Delete** instantly by destroying the encryption key

## The Problem WDVA Solves

Traditional AI personalization requires a painful tradeoff:

| Approach | Personalization | Privacy | Control |
|----------|-----------------|---------|---------|
| Cloud AI | ✅ Yes | ❌ No | ❌ No |
| Local AI | ❌ No (generic) | ✅ Yes | ✅ Yes |
| **WDVA** | ✅ Yes | ✅ Yes | ✅ Yes |

WDVA gives you personalized AI **without** sacrificing privacy or control.

## How It Works

```
Your Documents → Train Adapter → Encrypt → Store Anywhere
                                              ↓
                     Query ← Decrypt (in-memory) ← Load
```

### Key Innovation

Instead of encrypting your data (which the AI can't learn from), WDVA encrypts the **trained model weights**. This means:

- Your documents are processed locally and never shared
- The trained adapter is encrypted before storage
- Querying decrypts the adapter **in memory only**
- After each query, decrypted weights are cleared

## Technical Architecture

### Components

| Component | File | Purpose |
|-----------|------|---------|
| WDVA | `wdva/wdva.py` | High-level API |
| Crypto | `wdva/crypto.py` | XChaCha20-Poly1305 encryption |
| Inference | `wdva/inference.py` | Local model inference |

### Security Stack

- **Encryption**: XChaCha20-Poly1305 (256-bit key, 192-bit nonce)
- **Key Derivation**: HKDF-SHA256 with random salt
- **Serialization**: SafeTensors (secure, fast, portable)
- **Compression**: Zstandard (optional, 30-50% size reduction)

### ML Stack

- **Training**: DoRA adapters via PEFT (in full implementation)
- **Inference**: MLX (Apple Silicon) or PyTorch (CPU/CUDA)
- **Base Models**: TinyLlama-1.1B, Llama-3.2-1B, Qwen2.5-1.5B

## Project Structure

```
wdva-boilerplate/
├── wdva/                    # Core package
│   ├── __init__.py         # Package exports
│   ├── wdva.py             # High-level API
│   ├── crypto.py           # Encryption/decryption
│   └── inference.py        # Local inference engine
├── examples/                # Example scripts
│   ├── simple_demo.py      # Basic usage demo
│   └── arxiv_demo.py       # arXiv paper assistant
├── docs/                    # Documentation
│   ├── CONCEPT.md          # Understanding WDVA
│   ├── ARCHITECTURE.md     # Technical deep dive
│   └── EXTENDING.md        # Customization guide
├── README.md               # Main documentation
├── QUICKSTART.md           # Getting started guide
├── LICENSE                 # Apache 2.0
├── setup.py                # Package setup
└── requirements.txt        # Dependencies
```

## API Overview

### Basic Usage

```python
from wdva import WDVA

# Initialize and load
wdva = WDVA(adapter_path="adapter.wdva", encryption_key="...")

# Query
response = wdva.query("What is this about?")

# Delete (cryptographic)
wdva.delete()
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `WDVA` | Main interface for training, loading, querying |
| `EncryptedAdapter` | Low-level encryption/decryption |
| `LocalInference` | Model loading and generation |

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

Apache 2.0 is ideal for AI/ML projects because it provides:
- ✅ Explicit patent grants (protects users from patent claims)
- ✅ Patent retaliation clause (if you sue, you lose rights)
- ✅ Still very permissive (like MIT)
- ✅ Common in AI/ML open source projects (TensorFlow, PyTorch, etc.)

## Getting Started

```bash
# Clone
git clone https://github.com/enclave-ai/wdva-boilerplate.git
cd wdva-boilerplate

# Install
pip install -e .

# Run demo
python examples/simple_demo.py
```

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

## Contact

- **Repository**: https://github.com/enclave-ai/wdva-boilerplate
- **Issues**: https://github.com/enclave-ai/wdva-boilerplate/issues

---

**Privacy is not a feature—it's a fundamental right. WDVA makes it possible.**
