# Contributing to WDVA

Thank you for your interest in contributing to WDVA! This document provides
guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and
inclusive environment. We expect all contributors to:

- Be respectful and considerate in all interactions
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

Before creating a bug report:

1. Search existing issues to avoid duplicates
2. Collect information about the bug:
   - Stack trace
   - OS, Python version, and WDVA version
   - Steps to reproduce
   - Expected vs actual behavior

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) when
creating issues.

### Suggesting Features

We welcome feature suggestions! Please:

1. Search existing issues and discussions first
2. Clearly describe the problem your feature solves
3. Provide example usage if possible
4. Consider security implications

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

### Security Vulnerabilities

**Do NOT open public issues for security vulnerabilities.**

Please report security issues privately to security@enclave.ai. See
[SECURITY.md](SECURITY.md) for details.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- One of: MLX (macOS Apple Silicon) or PyTorch

### Installation

```bash
# Clone the repository
git clone https://github.com/enclave-ai/wdva-boilerplate.git
cd wdva-boilerplate

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev,torch]"  # or [dev,mlx] on Apple Silicon
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=wdva --cov-report=term-missing

# Run specific test file
pytest tests/test_crypto.py

# Run tests matching a pattern
pytest tests/ -k "test_encrypt"
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
ruff format wdva/ tests/

# Lint code
ruff check wdva/ tests/

# Type checking
mypy wdva/

# Run all checks
ruff format wdva/ tests/ && ruff check wdva/ tests/ && mypy wdva/
```

## Pull Request Process

### Before Submitting

1. **Create an issue first** for significant changes
2. **Fork the repository** and create a feature branch
3. **Follow the coding standards** (see below)
4. **Write tests** for new functionality
5. **Update documentation** as needed
6. **Run all checks** locally

### Branch Naming

Use descriptive branch names:

- `feature/add-hsm-support`
- `fix/memory-leak-inference`
- `docs/improve-quickstart`
- `refactor/simplify-crypto`

### Commit Messages

Write clear, concise commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain the problem this commit solves and why this approach
was chosen.

- Bullet points are okay
- Use present tense ("Add feature" not "Added feature")

Fixes #123
```

### Pull Request Guidelines

1. Fill out the PR template completely
2. Link related issues
3. Ensure CI passes
4. Request review from maintainers
5. Respond to feedback promptly

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints for all public functions
- Maximum line length: 100 characters
- Use `ruff` for formatting and linting

### Documentation

- Write docstrings for all public modules, classes, and functions
- Use Google-style docstrings
- Include examples in docstrings where helpful
- Keep README and docs up to date

### Security

- Never log sensitive data (keys, passwords, etc.)
- Use `secrets` module for cryptographic randomness
- Validate all inputs
- Follow the principle of least privilege
- Consider timing attacks for security-sensitive comparisons

### Testing

- Write tests for all new functionality
- Aim for high test coverage
- Test edge cases and error conditions
- Use descriptive test names
- Keep tests focused and independent

### Example Code

```python
"""
Module docstring explaining purpose.

Copyright 2025 Enclave
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def example_function(
    required_param: str,
    *,
    optional_param: Optional[int] = None,
) -> bool:
    """
    Brief description of what this function does.

    Args:
        required_param: Description of this parameter.
        optional_param: Description of this parameter.

    Returns:
        Description of return value.

    Raises:
        ValueError: When required_param is empty.

    Example:
        >>> result = example_function("test", optional_param=42)
        >>> print(result)
        True
    """
    if not required_param:
        raise ValueError("required_param cannot be empty")

    logger.debug("Processing: %s", required_param)
    return True
```

## Project Structure

```
wdva-boilerplate/
├── wdva/               # Main package
│   ├── __init__.py     # Package exports
│   ├── wdva.py         # High-level API
│   ├── crypto.py       # Encryption/decryption
│   ├── inference.py    # Local inference engine
│   ├── exceptions.py   # Custom exceptions
│   └── py.typed        # PEP 561 marker
├── tests/              # Test suite
├── examples/           # Example scripts
├── docs/               # Documentation
├── .github/            # GitHub configuration
├── pyproject.toml      # Project configuration
└── README.md           # Main readme
```

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml` and `wdva/__init__.py`
2. Update CHANGELOG.md
3. Create a git tag: `git tag v1.0.1`
4. Push tag: `git push origin v1.0.1`
5. GitHub Actions will build and publish to PyPI

## Getting Help

- Open a [Discussion](https://github.com/enclave-ai/wdva-boilerplate/discussions)
- Check existing issues and documentation
- Email team@enclave.ai for private inquiries

## License

By contributing to WDVA, you agree that your contributions will be licensed
under the Apache License 2.0.

---

Thank you for contributing to WDVA!
