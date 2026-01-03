# WDVA Quick Start Guide

Get up and running with WDVA in under 5 minutes.

## Prerequisites

- Python 3.9 or later
- 4GB+ RAM recommended
- macOS, Linux, or Windows

## Installation

### Option 1: From PyPI (when published)

```bash
pip install wdva
```

### Option 2: From Source

```bash
git clone https://github.com/enclave-ai/wdva-boilerplate.git
cd wdva-boilerplate
pip install -e .
```

### Install Backend (Choose One)

**Apple Silicon (M1/M2/M3/M4):** Fastest performance
```bash
pip install mlx mlx-lm
```

**CPU/CUDA:** Works everywhere
```bash
pip install torch transformers
```

### Install Optional Dependencies

```bash
# Compression (recommended, 30-50% smaller files)
pip install zstandard

# Faster downloads
pip install hf-transfer

# For arXiv example
pip install requests PyPDF2
```

## Hello World

```python
from wdva import WDVA

# Initialize
wdva = WDVA()
print(f"Initialized: {wdva}")

# Query the base model (no training needed)
response = wdva.query_base("What is 2 + 2?")
print(f"Response: {response}")
```

## Basic Workflow

### 1. Train an Adapter

```python
from wdva import WDVA

wdva = WDVA()
result = wdva.train(
    documents=["document1.pdf", "document2.txt"],
    max_samples=100
)

print(f"Adapter saved: {result.adapter_path}")
print(f"Encryption key: {result.encryption_key_hex}")

# IMPORTANT: Store the encryption key securely!
```

### 2. Load and Query

```python
from wdva import WDVA

# Load your trained adapter
wdva = WDVA(
    adapter_path="path/to/adapter.wdva",
    encryption_key="your-64-character-hex-key"
)

# Query your personalized AI
response = wdva.query("What is the main topic of my documents?")
print(response)
```

### 3. Cryptographic Deletion

```python
# When you want to "forget" the adapter
wdva.delete()

# The encryption key is destroyed
# The adapter file can't be decrypted anymore
# This is true "right to be forgotten"
```

## Run the Examples

### Simple Demo
```bash
python examples/simple_demo.py
```

### arXiv Paper Assistant
```bash
# Download and process a paper
python examples/arxiv_demo.py --paper-id 2502.13171

# With a query
python examples/arxiv_demo.py --paper-id 2502.13171 --query "What is the main contribution?"
```

## Key Concepts

### Encryption Key

- **64 hex characters** (32 bytes)
- Store it securely (password manager, HSM, etc.)
- **Never commit it to version control**
- Without the key, the adapter is cryptographically inaccessible

### Adapter File (`.wdva`)

- Contains encrypted model weights
- Safe to store anywhere (cloud, USB, etc.)
- Useless without the encryption key
- Typically 10-50 MB depending on training

### Privacy Guarantees

- Your documents never leave your device
- The server never sees decrypted data
- Querying happens locally
- Delete the key = delete the model

## Next Steps

- Read [CONCEPT.md](docs/CONCEPT.md) to understand how WDVA works
- Read [ARCHITECTURE.md](docs/ARCHITECTURE.md) for technical details
- Read [EXTENDING.md](docs/EXTENDING.md) to customize for your use case

## Troubleshooting

### "Cryptography libraries not available"
```bash
pip install pycryptodome cryptography
```

### "No ML backend available"
```bash
# For Mac (Apple Silicon)
pip install mlx mlx-lm

# For other platforms
pip install torch transformers
```

### "Model download is slow"
```bash
pip install hf-transfer
```
Then downloads will be 10-100x faster.

### "safetensors not available"
```bash
pip install safetensors
```

## Support

- GitHub Issues: Report bugs and request features
- Documentation: See the `docs/` folder
- Examples: See the `examples/` folder
