# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue,
please report it responsibly.

### How to Report

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email: **security@enclave.ai**

Include in your report:

1. **Description**: Clear description of the vulnerability
2. **Reproduction**: Step-by-step instructions to reproduce the issue
3. **Impact**: Your assessment of the potential impact
4. **Suggested Fix**: If you have ideas for remediation

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 5 business days
- **Resolution Timeline**: Communicated after assessment

### What to Expect

1. We will acknowledge receipt of your report
2. We will investigate and validate the issue
3. We will work on a fix and coordinate disclosure
4. We will credit you in the security advisory (unless you prefer anonymity)

## Security Design

WDVA is designed with security as a primary concern. Here is our security architecture:

### Cryptographic Stack

| Component | Algorithm | Key Size | Notes |
|-----------|-----------|----------|-------|
| Encryption | XChaCha20-Poly1305 | 256-bit | AEAD (authenticated encryption) |
| Key Derivation | HKDF-SHA256 | 256-bit | Per-adapter salt prevents key reuse |
| Nonce | Random | 192-bit | Unique per encryption operation |
| Authentication | Poly1305 | 128-bit | Detects tampering |

### Security Properties

- **Confidentiality**: Encrypted adapters cannot be read without the key
- **Integrity**: Authentication tag detects any modification
- **Authenticity**: AAD binds metadata to ciphertext
- **Forward Secrecy**: Per-adapter key derivation with unique salts

### Threat Model

WDVA protects against:

- Unauthorized access to adapter weights
- Tampering with encrypted adapters
- Metadata manipulation
- Key reuse vulnerabilities

WDVA does NOT protect against:

- Key theft (compromised key = compromised adapter)
- Device-level compromise (malware with memory access)
- Side-channel attacks (timing, power analysis)
- Quantum computing attacks (future consideration)

### Memory Security

Python cannot guarantee secure memory erasure. For high-security deployments:

- Consider using the `cryptography` library's `secure_zero_memory()` for sensitive data
- Use hardware security modules (HSMs) for key storage
- Deploy in trusted execution environments (TEEs) where available

### Best Practices

1. **Key Storage**: Never store encryption keys in plaintext
2. **Key Rotation**: Consider rotating keys periodically
3. **Access Control**: Limit who can access encryption keys
4. **Audit Logging**: Log adapter access for security monitoring
5. **Secure Deletion**: Destroy keys when no longer needed

## Security Audits

This project has not yet undergone a formal security audit. If you are interested
in sponsoring an audit, please contact team@enclave.ai.

## Dependencies

We carefully select dependencies with security in mind:

| Dependency | Purpose | Security Notes |
|------------|---------|----------------|
| pycryptodome | XChaCha20-Poly1305 | Well-audited cryptography library |
| cryptography | HKDF | Python Cryptographic Authority, regularly audited |
| safetensors | Tensor serialization | Designed to be safe against pickle vulnerabilities |

## Vulnerability Disclosure Policy

We follow responsible disclosure:

1. Reporter notifies us privately
2. We acknowledge and investigate
3. We develop and test a fix
4. We coordinate release timing with reporter
5. We publish security advisory with credit to reporter
6. Reporter may publish their own writeup after patch release
