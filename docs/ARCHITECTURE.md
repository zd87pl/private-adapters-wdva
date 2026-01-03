# WDVA Architecture

## Overview

WDVA consists of three core components:

1. **Encryption Layer** - Securely encrypts/decrypts adapters
2. **Adapter Layer** - Manages DoRA adapter weights
3. **Inference Layer** - Runs inference locally

## Component Details

### 1. Encryption Layer (`wdva/crypto.py`)

**Purpose**: Encrypt adapter weights so they can be stored anywhere securely.

**Key Features**:
- **XChaCha20-Poly1305**: Authenticated encryption (prevents tampering)
- **HKDF-SHA256**: Key derivation (separates storage key from master secret)
- **Optional Compression**: Reduces storage size by 30-50%
- **SafeTensors**: Fast, secure serialization format

**Flow**:
```
Adapter Weights → Serialize → Compress → Encrypt → Encrypted Blob
```

**Security Properties**:
- **Confidentiality**: Encrypted data is unreadable without key
- **Integrity**: Authentication tag prevents tampering
- **Authenticity**: AAD (associated data) binds metadata to ciphertext

### 2. Adapter Layer (via `wdva/wdva.py` and `wdva/crypto.py`)

**Purpose**: Manage DoRA (Decomposed Low-Rank Adaptation) adapter weights.

**DoRA Overview**:
- DoRA is a parameter-efficient fine-tuning technique
- Only trains ~1% of model parameters
- Creates small adapters (~20MB vs full model ~2GB)
- Can be merged with base model at inference time

**Adapter Structure**:
```
DoRA Adapter:
  - Base model: TinyLlama-1.1B (shared by all users)
  - Adapter weights: User-specific (encrypted via crypto.py)
  - Merged model: Base + Adapter (ephemeral, in-memory only)
```

**Note**: Full adapter training requires the PEFT library. This boilerplate
provides the encryption and inference layers; see the main Enclave repository
for complete training implementation.

### 3. Inference Layer (`wdva/inference.py`)

**Purpose**: Run inference locally with encrypted adapters.

**Supported Backends**:
- **MLX** (Apple Silicon): Fastest on M1/M2/M3 Macs
- **PyTorch** (CPU/CUDA): Works everywhere

**Ephemeral Loading**:
1. Load base model (cached locally)
2. Decrypt adapter (in-memory only)
3. Merge adapter with base model (in-memory only)
4. Run inference
5. Clear adapter from memory
6. Base model remains cached

**Memory Management**:
- Adapter weights are never persisted to disk after decryption
- Memory is cleared after inference
- Base model can be cached for performance

## Data Flow

### Training Flow

```
Your Documents
    ↓
Text Extraction
    ↓
Q&A Generation
    ↓
DoRA Training (Local)
    ↓
Adapter Weights
    ↓
Encryption
    ↓
Encrypted Adapter (can store anywhere)
```

### Inference Flow

```
Encrypted Adapter
    ↓
Decrypt (in-memory)
    ↓
Load Base Model (cached)
    ↓
Merge Adapter + Base (in-memory)
    ↓
Run Inference
    ↓
Clear Adapter from Memory
    ↓
Return Response
```

## Security Architecture

### Threat Model

WDVA protects against:
- ✅ **Data Theft**: Encrypted adapters are unreadable without key
- ✅ **Tampering**: Authentication tags detect modifications
- ✅ **Surveillance**: Server never sees decrypted data
- ✅ **Data Retention**: Cryptographic deletion (destroy key = delete model)

WDVA does NOT protect against:
- ❌ **Key Theft**: If attacker gets your encryption key, they can decrypt
- ❌ **Malware**: If your device is compromised, attacker can access decrypted data
- ❌ **Side-Channel Attacks**: Advanced attacks on hardware (out of scope)

### Encryption Details

**Algorithm**: XChaCha20-Poly1305
- **Key Size**: 256 bits (32 bytes)
- **Nonce Size**: 192 bits (24 bytes)
- **Tag Size**: 128 bits (16 bytes)

**Key Derivation**: HKDF-SHA256
- **Salt**: 128 bits (16 bytes), random per adapter
- **Info**: Application-specific ("wdva-encryption-v1")
- **Output**: 256 bits (32 bytes)

**Compression**: Zstandard (optional)
- **Level**: 3 (balanced speed/size)
- **Typical Reduction**: 30-50%

## Performance Characteristics

### Storage
- **Base Model**: ~2GB (cached locally, shared)
- **Adapter**: ~20MB (encrypted, portable)
- **Total per User**: ~20MB (vs ~2GB for full model)

### Speed
- **Training**: ~1-5 minutes (depends on data size)
- **Encryption**: <1 second
- **Decryption**: <1 second
- **Inference**: ~100-500ms per query (depends on hardware)

### Memory
- **Base Model**: ~2GB RAM (cached)
- **Adapter**: ~20MB RAM (ephemeral)
- **Peak Memory**: ~2.1GB (during inference)

## Extensibility Points

1. **Custom Data Sources**: Implement document loaders
2. **Different Models**: Swap base model (any HuggingFace causal LM)
3. **Custom Encryption**: Implement different encryption schemes
4. **Cloud Storage**: Integrate with cloud storage providers
5. **Multi-User**: Support multiple adapters per device

See [EXTENDING.md](EXTENDING.md) for details.

