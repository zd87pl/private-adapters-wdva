# Changelog

All notable changes to WDVA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite for crypto, wdva, and inference modules
- GitHub Actions CI/CD pipeline
- SECURITY.md with vulnerability reporting guidelines
- CONTRIBUTING.md with development guidelines
- Issue and PR templates
- Custom exception hierarchy for better error handling
- pyproject.toml for modern Python packaging

### Changed
- Replaced setup.py with pyproject.toml (PEP 621)
- Improved thread safety in backend detection
- Fixed memory clearing in inference module

### Fixed
- Thread-unsafe global state in backend detection
- Broken `dir()` check for local variable cleanup
- Inline import moved to module level

## [1.0.0] - 2025-01-04

### Added
- Initial release of WDVA (Weight-Delta Vault Adapters)
- Core `WDVA` class for high-level API
- `EncryptedAdapter` class for XChaCha20-Poly1305 encryption
- `LocalInference` engine with MLX and PyTorch backends
- HKDF-SHA256 key derivation with per-adapter salts
- Zstandard compression support
- SafeTensors serialization for secure weight storage
- Comprehensive documentation (README, CONCEPT, ARCHITECTURE, EXTENDING)
- Example scripts (simple_demo, arxiv_demo)
- Apache 2.0 license

### Security
- Military-grade encryption with XChaCha20-Poly1305
- Authenticated encryption with AAD for metadata integrity
- Ephemeral adapter loading (never persisted to disk)
- Cryptographic deletion support

[Unreleased]: https://github.com/enclave-ai/wdva-boilerplate/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/enclave-ai/wdva-boilerplate/releases/tag/v1.0.0
