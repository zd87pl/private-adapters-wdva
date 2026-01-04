#!/usr/bin/env python3
"""
Simple WDVA Demo - Minimal Example

This demonstrates the core WDVA concept in the simplest possible way.

Usage:
    python examples/simple_demo.py

Copyright 2025 Enclave
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from wdva import WDVA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def print_header() -> None:
    """Print demo header."""
    print()
    print("=" * 70)
    print("  WDVA: Secure & Private AI for Everyone")
    print("  Simple Demo")
    print("=" * 70)
    print()


def print_section(title: str) -> None:
    """Print section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


def main() -> int:
    """Run the simple demo."""
    print_header()
    
    print("This demo shows the core WDVA workflow:")
    print()
    print("  1. Initialize WDVA")
    print("  2. Train an adapter on documents (simplified)")
    print("  3. Load the encrypted adapter")
    print("  4. Query your personalized AI")
    print("  5. Demonstrate cryptographic deletion")
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Initialize
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Step 1: Initialize WDVA")
    
    print("Creating WDVA instance...")
    wdva = WDVA()
    print(f"[OK] Initialized: {wdva}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Train (simplified)
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Step 2: Train Adapter (Simplified)")
    
    print("Training adapter on documents...")
    print("  (In production, this trains on actual document content)")
    print()
    
    result = wdva.train(
        documents=["document1.txt", "document2.pdf"],
        max_samples=50
    )
    
    print("[OK] Training result:")
    print(f"    Adapter path: {result.adapter_path}")
    print(f"    Encryption key: {result.encryption_key_hex[:16]}... (keep secret!)")
    print(f"    Documents: {result.num_documents}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Load adapter
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Step 3: Load Encrypted Adapter")
    
    print("Loading adapter with encryption key...")
    
    # Note: This would work with a real trained adapter
    # For this demo, we just demonstrate the API
    try:
        wdva.load(result.adapter_path, result.encryption_key_hex)
        print(f"[OK] Adapter loaded: {wdva}")
    except FileNotFoundError:
        print("[WARN] Adapter file not found (expected in demo)")
        print("  In production, load() requires an actual trained adapter file.")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Query (with base model for demo)
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Step 4: Query Your AI")
    
    print("Querying the base model (without adapter for demo)...")
    print()
    
    try:
        response = wdva.query_base(
            "What is 2 + 2? Answer briefly.",
            max_tokens=50,
            temperature=0.1
        )
        print(f"  Question: What is 2 + 2?")
        print(f"  Response: {response}")
    except Exception as e:
        print(f"[WARN] Query failed: {e}")
        print("  (Model loading may require additional dependencies)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Cryptographic deletion
    # ─────────────────────────────────────────────────────────────────────────
    print_section("Step 5: Cryptographic Deletion")
    
    print("Demonstrating 'right to be forgotten'...")
    print()
    print("Before deletion:")
    print(f"  Status: {wdva.status.value}")
    
    wdva.delete()
    
    print()
    print("After deletion:")
    print(f"  Status: {wdva.status.value}")
    print("  [OK] Encryption key destroyed")
    print("  [OK] Adapter is now cryptographically inaccessible")
    print("  [OK] Even if the file exists, it cannot be decrypted")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Key Takeaways")
    print("=" * 70)
    print()
    print("  • Your data stays encrypted, always")
    print("  • Train once, run anywhere—even offline")
    print("  • Cryptographic 'right to be forgotten'")
    print("  • Personalization without privacy compromise")
    print()
    print("  For full training, see: examples/arxiv_demo.py")
    print()
    print("=" * 70)
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
